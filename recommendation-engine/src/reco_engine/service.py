"""Service layer: orchestrates candidate generation -> scoring -> ranking
behind one clean entry point, so callers (a REST API, a CLI, a batch
job) don't need to know how the pipeline is wired internally.

This is the piece a pure algorithmic prototype is usually missing: a
single place that owns the end-to-end flow, talks to persistence, and
exposes a stable interface independent of how the underlying
components change.
"""

from __future__ import annotations

from typing import Any

from .candidate_gen import CandidateGenerator
from .config import Settings, get_settings
from .evaluator import RecommendationEvaluator
from .logging_config import get_logger
from .persistence import Repository
from .scorer import RecommendationScorer, ScoredItem
from .similarity import SimilarityCalculator

logger = get_logger(__name__)

# Recency/popularity are properties of the item itself, not of a given
# user, so real systems typically compute/cache these per item (e.g. a
# nightly batch job) rather than per request. Kept in-process here to
# stay dependency-free; swap for a real feature store in production.
RecencyLookup = dict[str, float]
PopularityLookup = dict[str, float]


class RecommendationService:
    """Wires the four components together into one callable pipeline.

    Also owns a small in-process cache for the one calculation that
    would otherwise be repeated across every candidate scored in a
    single request: a user's tag profile (the union of tags across
    their whole history). Without it, `_personalized_relevance` rebuilds
    that profile from scratch for every candidate item; with it, the
    profile is built once per user and reused until that user's history
    actually changes.

    The cache is correctness-preserving by construction: it is
    invalidated for a user the instant `record_interaction` changes
    that user's history, so a stale profile can never be served.
    """

    def __init__(
        self,
        repo: Repository,
        recency: RecencyLookup | None = None,
        popularity: PopularityLookup | None = None,
        settings: Settings | None = None,
    ):
        self.repo = repo
        self.settings = settings or get_settings()
        self.recency = recency or {}
        self.popularity = popularity or {}

        self.similarity = SimilarityCalculator()
        self.generator = CandidateGenerator(repo, max_candidates=self.settings.max_candidates)
        self.evaluator = RecommendationEvaluator()
        self.scorer = self._build_default_scorer()

        # user_id -> frozenset of tags derived from their interaction
        # history. Invalidated per-user on `record_interaction`.
        self._profile_cache: dict[str, frozenset[str]] = {}

    # ------------------------------------------------------------------
    def _get_user_profile(self, user_id: str) -> frozenset[str]:
        """Return (and cache) the union of tags across a user's history.

        A cache hit avoids re-reading every historical item's features
        and rebuilding the union on every single candidate scored in a
        `get_recommendations` call — a real, measurable saving once a
        user has any meaningful history and the candidate pool isn't tiny.
        """
        cached = self._profile_cache.get(user_id)
        if cached is not None:
            return cached

        history = self.repo.get_user_history(user_id)
        profile: frozenset[str] = frozenset()
        if history:
            tags: set[str] = set()
            for historical_item in history:
                tags |= self.repo.get_item_features(historical_item)
            profile = frozenset(tags)

        self._profile_cache[user_id] = profile
        return profile

    def _invalidate_user_cache(self, user_id: str) -> None:
        """Drop any cached profile for `user_id` so the next lookup
        recomputes it from the (now updated) history."""
        self._profile_cache.pop(user_id, None)

    # ------------------------------------------------------------------
    def _personalized_relevance(self, user_id: str, item_id: str, context: dict[str, Any]) -> float:
        """Relevance = overlap between the user's tag profile (built from
        their own history) and the candidate item's tags. This is what
        makes scoring depend on user_id — without it every user with an
        overlapping candidate pool gets identical rankings."""
        profile = self._get_user_profile(user_id)
        if not profile:
            return 0.0
        item_tags = self.repo.get_item_features(item_id)
        if not item_tags:
            return 0.0
        return self.similarity.jaccard_similarity(profile, item_tags)

    def _build_default_scorer(self) -> RecommendationScorer:
        scorer = RecommendationScorer()
        weights = self.settings.default_weights
        scorer.add_scorer("relevance", self._personalized_relevance, weights["relevance"])
        scorer.add_scorer("recency", lambda u, i, ctx: self.recency.get(i, 0.0), weights["recency"])
        scorer.add_scorer("popularity", lambda u, i, ctx: self.popularity.get(i, 0.0), weights["popularity"])
        return scorer

    # ------------------------------------------------------------------
    def get_recommendations(self, user_id: str, k: int | None = None) -> list[ScoredItem]:
        """Full pipeline for one user: generate candidates, score, rank."""
        k = k if k is not None else self.settings.default_top_k
        candidates = self.generator.hybrid_candidates(user_id)
        ranked = self.scorer.rank_candidates(user_id, candidates, limit=k)
        logger.info("get_recommendations user=%s -> %d results", user_id, len(ranked))
        return ranked

    def record_interaction(self, user_id: str, item_id: str) -> None:
        """Persist a new interaction so future recommendations reflect it.

        Invalidates this user's cached profile and the generator's
        cached popularity ranking (both derived from history that this
        call just changed) so nothing stale is ever served afterward.
        """
        self.repo.record_interaction(user_id, item_id)
        self._invalidate_user_cache(user_id)
        self.generator.invalidate_cache()

    def evaluate(
        self, user_ids: list[str], ground_truth: dict[str, list[str]], k: int | None = None
    ) -> dict[str, float]:
        """Run the pipeline for a batch of users and score it against
        known ground truth (e.g. items they engaged with next)."""
        k = k if k is not None else self.settings.default_top_k
        recommendations = {
            uid: [item.item_id for item in self.get_recommendations(uid, k)]
            for uid in user_ids
        }
        return self.evaluator.evaluate_all(recommendations, ground_truth, k)
