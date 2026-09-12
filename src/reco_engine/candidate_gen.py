"""Component 2: Candidate Generator.

Refactored around two design patterns:

- Repository pattern (persistence.py): strategies read data through
  `Repository`, never touch raw dicts, so swapping storage backends
  needs zero changes here.
- Strategy pattern: each generation approach (collaborative,
  content-based, popularity) is its own class implementing
  `CandidateStrategy`. Adding a new strategy (e.g. a trained-model
  based one, tomorrow) means writing one new class and registering it
  — `CandidateGenerator` itself never needs to change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter

from .exceptions import InvalidInputError
from .logging_config import get_logger
from .persistence import Repository
from .similarity import SimilarityCalculator

logger = get_logger(__name__)


class CandidateStrategy(ABC):
    """A pluggable candidate-generation approach."""

    name: str = "base"

    @abstractmethod
    def generate(self, user_id: str, limit: int) -> list[str]:
        """Return up to `limit` candidate item_ids for `user_id`."""


class PopularityStrategy(CandidateStrategy):
    """Most-interacted-with items overall. Always safe to call, even for
    users with no history — this is the universal cold-start fallback."""

    name = "popularity"

    def __init__(self, repo: Repository):
        self.repo = repo

    def generate(self, user_id: str, limit: int) -> list[str]:
        history = self.repo.get_all_user_history()
        if not history:
            return []
        counts: Counter[str] = Counter()
        for items in history.values():
            counts.update(items)
        return [item_id for item_id, _ in counts.most_common(limit)]


class CollaborativeStrategy(CandidateStrategy):
    """Items liked by users whose history overlaps this user's (Jaccard
    similarity over interaction sets) — classic user-user collaborative
    filtering. Falls back to popularity for cold-start users."""

    name = "collaborative"

    def __init__(self, repo: Repository, similarity: SimilarityCalculator | None = None):
        self.repo = repo
        self.similarity = similarity or SimilarityCalculator()
        self._fallback = PopularityStrategy(repo)

    def generate(self, user_id: str, limit: int) -> list[str]:
        target = self.repo.get_user_history(user_id)
        if not target:
            return self._fallback.generate(user_id, limit)

        all_history = self.repo.get_all_user_history()
        ranked_users = sorted(
            ((uid, self.similarity.jaccard_similarity(target, items))
             for uid, items in all_history.items() if uid != user_id),
            key=lambda pair: pair[1], reverse=True,
        )

        candidates, seen = [], set()
        for uid, sim in ranked_users:
            if sim <= 0:
                continue
            for item in all_history[uid]:
                if item in target or item in seen:
                    continue
                seen.add(item)
                candidates.append(item)
                if len(candidates) >= limit:
                    return candidates
        return candidates or self._fallback.generate(user_id, limit)


class ContentBasedStrategy(CandidateStrategy):
    """Items whose tags overlap with the union of tags across the user's
    own history (Jaccard) — classic item-based content filtering.
    Falls back to popularity when there's no history or no feature data."""

    name = "content_based"

    def __init__(self, repo: Repository, similarity: SimilarityCalculator | None = None):
        self.repo = repo
        self.similarity = similarity or SimilarityCalculator()
        self._fallback = PopularityStrategy(repo)

    def generate(self, user_id: str, limit: int) -> list[str]:
        target = self.repo.get_user_history(user_id)
        all_features = self.repo.get_all_item_features()
        if not target or not all_features:
            return self._fallback.generate(user_id, limit)

        profile: set[str] = set()
        for item in target:
            profile |= self.repo.get_item_features(item)
        if not profile:
            return self._fallback.generate(user_id, limit)

        scored = sorted(
            ((iid, self.similarity.jaccard_similarity(profile, tags))
             for iid, tags in all_features.items() if iid not in target),
            key=lambda pair: pair[1], reverse=True,
        )
        candidates = [iid for iid, sim in scored if sim > 0][:limit]
        return candidates or self._fallback.generate(user_id, limit)


class CandidateGenerator:
    """Facade over the registered strategies. `hybrid_candidates` blends
    all of them via round-robin interleaving so no single strategy
    dominates the pool, and always excludes items the user has already
    seen (a final safety net — individual strategies should already
    exclude these, but popularity in particular is un-personalized by
    nature, so this is enforced here too).

    Caching: `popularity_candidates` doesn't depend on `user_id` at all
    (it's a global ranking over every user's history), so it's cached
    after the first call and reused across every user in the process —
    it would otherwise be recomputed via a full history scan on *every*
    `hybrid_candidates` call, for every user, even though the answer is
    the same until someone's history changes. `invalidate_cache` is
    called by `RecommendationService.record_interaction` whenever that
    happens, so the cache can never serve a stale ranking.
    """

    def __init__(self, repo: Repository, max_candidates: int = 20):
        if max_candidates <= 0:
            raise InvalidInputError("max_candidates must be positive")
        self.repo = repo
        self.max_candidates = max_candidates
        similarity = SimilarityCalculator()
        self.popularity_strategy = PopularityStrategy(repo)
        self.collaborative_strategy = CollaborativeStrategy(repo, similarity)
        self.content_strategy = ContentBasedStrategy(repo, similarity)

        self._popularity_cache: list[str] | None = None

    def invalidate_cache(self) -> None:
        """Drop any cached, potentially now-stale candidate data.
        Safe to call unconditionally after any write to the repository."""
        self._popularity_cache = None

    def collaborative_candidates(self, user_id: str) -> list[str]:
        return self.collaborative_strategy.generate(user_id, self.max_candidates)

    def content_based_candidates(self, user_id: str) -> list[str]:
        return self.content_strategy.generate(user_id, self.max_candidates)

    def popularity_candidates(self) -> list[str]:
        if self._popularity_cache is None:
            self._popularity_cache = self.popularity_strategy.generate("", self.max_candidates)
        return self._popularity_cache

    def hybrid_candidates(self, user_id: str) -> list[str]:
        target = self.repo.get_user_history(user_id)
        strategies = [
            self.collaborative_candidates(user_id),
            self.content_based_candidates(user_id),
            self.popularity_candidates(),
        ]

        merged, seen = [], set()
        max_len = max((len(s) for s in strategies), default=0)
        for i in range(max_len):
            for strategy_list in strategies:
                if i < len(strategy_list):
                    item = strategy_list[i]
                    if item in target or item in seen:
                        continue
                    seen.add(item)
                    merged.append(item)
                    if len(merged) >= self.max_candidates:
                        logger.debug("hybrid_candidates user=%s -> %d items", user_id, len(merged))
                        return merged
        logger.debug("hybrid_candidates user=%s -> %d items", user_id, len(merged))
        return merged
