"""Component 3: Scorer & Ranker.

Combines weighted, pluggable scoring functions into a final ranking
with a short, honest explanation for each recommended item — built
only from the real signals that actually contributed to that item's
score (no invented or generic-sounding reasons).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .exceptions import InvalidInputError
from .logging_config import get_logger

logger = get_logger(__name__)

ScoreFn = Callable[[str, str, dict[str, Any]], float]

# Only show a signal in the explanation if its own contribution to the
# final score is at least this large — filters out noise (e.g. a
# recency score of 0.02) that's technically nonzero but not something a
# person would recognize as "why" the item was recommended.
_EXPLANATION_THRESHOLD = 0.05


@dataclass
class ScoredItem:
    item_id: str
    score: float
    explanation: str


class RecommendationScorer:
    """Registers scoring functions and produces weighted, ranked results."""

    def __init__(self) -> None:
        self._scorers: list[tuple[str, ScoreFn, float]] = []

    def add_scorer(self, name: str, function: ScoreFn, weight: float) -> None:
        """Register a scoring function: fn(user_id, item_id, context) -> [0,1]."""
        if weight < 0:
            raise InvalidInputError(f"Scorer weight must be non-negative, got {weight}")
        if not callable(function):
            raise InvalidInputError(f"Scorer '{name}' function must be callable")
        self._scorers.append((name, function, weight))
        logger.debug("Registered scorer '%s' (weight=%.2f)", name, weight)

    def calculate_score(
        self, user_id: str, item_id: str, context: dict[str, Any] | None = None
    ) -> tuple[float, str]:
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
        explanation = self._explain(contributions)
        return max(0.0, min(1.0, score)), explanation

    @staticmethod
    def _explain(contributions: list[tuple[str, float, float]]) -> str:
        """Build a plain-language explanation from only the signals that
        actually mattered for this item — every number in it is a real
        (weight * raw-score) contribution, nothing inferred or templated.
        """
        meaningful = [
            (name, raw, weight) for name, raw, weight in contributions
            if raw * weight >= _EXPLANATION_THRESHOLD
        ]
        if not meaningful:
            return "No strong personalization signal for this item; shown as a general suggestion."

        meaningful.sort(key=lambda c: c[1] * c[2], reverse=True)
        parts = [f"{name} match {raw:.0%}" for name, raw, _ in meaningful]
        return "Recommended based on " + ", ".join(parts) + "."

    def rank_candidates(
        self,
        user_id: str,
        candidates: list[str],
        context: dict[str, Any] | None = None,
        limit: int = 10,
    ) -> list[ScoredItem]:
        """Score every candidate (deduplicated) and return the top `limit`."""
        if limit <= 0:
            raise InvalidInputError(f"limit must be positive, got {limit}")

        seen: set[str] = set()
        results: list[ScoredItem] = []
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
