"""Component 3: Scorer & Ranker.

Combines weighted, pluggable scoring functions into a final ranking
with a short explanation for each recommended item.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from .exceptions import InvalidInputError
from .logging_config import get_logger

logger = get_logger(__name__)

ScoreFn = Callable[[str, str, Dict[str, Any]], float]


@dataclass
class ScoredItem:
    item_id: str
    score: float
    explanation: str


class RecommendationScorer:
    """Registers scoring functions and produces weighted, ranked results."""

    def __init__(self):
        self._scorers: List[Tuple[str, ScoreFn, float]] = []

    def add_scorer(self, name: str, function: ScoreFn, weight: float) -> None:
        """Register a scoring function: fn(user_id, item_id, context) -> [0,1]."""
        if weight < 0:
            raise InvalidInputError(f"Scorer weight must be non-negative, got {weight}")
        if not callable(function):
            raise InvalidInputError(f"Scorer '{name}' function must be callable")
        self._scorers.append((name, function, weight))
        logger.debug("Registered scorer '%s' (weight=%.2f)", name, weight)

    def calculate_score(
        self, user_id: str, item_id: str, context: Optional[Dict[str, Any]] = None
    ) -> Tuple[float, str]:
        """Compute the weighted combined score for one item.

        Returns (score, explanation). A scorer that raises is logged and
        skipped rather than failing the whole ranking pass — one bad
        signal shouldn't take down recommendations for everyone.
        """
        context = context or {}
        contributions = []

        for name, fn, weight in self._scorers:
            try:
                raw = float(fn(user_id, item_id, context))
            except Exception:
                logger.warning("Scorer '%s' failed for item=%s", name, item_id, exc_info=True)
                continue
            raw = max(0.0, min(1.0, raw))
            contributions.append((name, raw, weight))

        total_weight = sum(w for _, _, w in contributions)
        if total_weight == 0:
            return 0.0, "No applicable scoring signals"

        score = sum(raw * w for _, raw, w in contributions) / total_weight
        top = sorted(contributions, key=lambda c: c[1] * c[2], reverse=True)[:2]
        explanation = "Recommended due to: " + ", ".join(
            f"{name} ({raw:.2f})" for name, raw, _ in top
        )
        return max(0.0, min(1.0, score)), explanation

    def rank_candidates(
        self,
        user_id: str,
        candidates: List[str],
        context: Optional[Dict[str, Any]] = None,
        limit: int = 10,
    ) -> List[ScoredItem]:
        """Score every candidate (deduplicated) and return the top `limit`."""
        if limit <= 0:
            raise InvalidInputError(f"limit must be positive, got {limit}")

        seen, results = set(), []
        for item_id in candidates:
            if item_id in seen:
                continue
            seen.add(item_id)
            score, explanation = self.calculate_score(user_id, item_id, context)
            results.append(ScoredItem(item_id, score, explanation))

        results.sort(key=lambda r: r.score, reverse=True)
        logger.info("Ranked %d candidates for user=%s, returning top %d",
                    len(results), user_id, min(limit, len(results)))
        return results[:limit]
