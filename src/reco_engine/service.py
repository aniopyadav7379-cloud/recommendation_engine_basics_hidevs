"""Service layer: orchestrates candidate generation -> scoring -> ranking
behind one clean entry point, so callers (a REST API, a CLI, a batch
job) don't need to know how the pipeline is wired internally.

This is the piece a pure algorithmic prototype is usually missing: a
single place that owns the end-to-end flow, talks to persistence, and
exposes a stable interface independent of how the underlying
components change.
"""

from typing import Any, Dict, List, Optional

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
RecencyLookup = Dict[str, float]
PopularityLookup = Dict[str, float]


class RecommendationService:
    """Wires the four components together into one callable pipeline."""

    def __init__(
        self,
        repo: Repository,
        recency: Optional[RecencyLookup] = None,
        popularity: Optional[PopularityLookup] = None,
        settings: Optional[Settings] = None,
    ):
        self.repo = repo
        self.settings = settings or get_settings()
        self.recency = recency or {}
        self.popularity = popularity or {}

        self.similarity = SimilarityCalculator()
        self.generator = CandidateGenerator(repo, max_candidates=self.settings.max_candidates)
        self.evaluator = RecommendationEvaluator()
        self.scorer = self._build_default_scorer()

    # ------------------------------------------------------------------
    def _personalized_relevance(self, user_id: str, item_id: str, context: Dict[str, Any]) -> float:
        """Relevance = overlap between the user's tag profile (built from
        their own history) and the candidate item's tags. This is what
        makes scoring depend on user_id — without it every user with an
        overlapping candidate pool gets identical rankings."""
        history = self.repo.get_user_history(user_id)
        if not history:
            return 0.0
        profile = set()
        for historical_item in history:
            profile |= self.repo.get_item_features(historical_item)
        item_tags = self.repo.get_item_features(item_id)
        if not profile or not item_tags:
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
    def get_recommendations(self, user_id: str, k: Optional[int] = None) -> List[ScoredItem]:
        """Full pipeline for one user: generate candidates, score, rank."""
        k = k or self.settings.default_top_k
        candidates = self.generator.hybrid_candidates(user_id)
        ranked = self.scorer.rank_candidates(user_id, candidates, limit=k)
        logger.info("get_recommendations user=%s -> %d results", user_id, len(ranked))
        return ranked

    def record_interaction(self, user_id: str, item_id: str) -> None:
        """Persist a new interaction so future recommendations reflect it."""
        self.repo.record_interaction(user_id, item_id)

    def evaluate(
        self, user_ids: List[str], ground_truth: Dict[str, List[str]], k: Optional[int] = None
    ) -> Dict[str, float]:
        """Run the pipeline for a batch of users and score it against
        known ground truth (e.g. items they engaged with next)."""
        k = k or self.settings.default_top_k
        recommendations = {
            uid: [item.item_id for item in self.get_recommendations(uid, k)]
            for uid in user_ids
        }
        return self.evaluator.evaluate_all(recommendations, ground_truth, k)
