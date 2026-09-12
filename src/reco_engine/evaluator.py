"""Component 4: Evaluator.

Offline metrics for measuring recommendation quality against ground
truth (items the user actually engaged with).
"""

from __future__ import annotations

import math

from .exceptions import InvalidInputError
from .logging_config import get_logger

logger = get_logger(__name__)


class RecommendationEvaluator:
    """precision@k, recall@k, NDCG@k, and a multi-user averaging helper."""

    @staticmethod
    def _validate_k(k: int) -> None:
        if not isinstance(k, int) or k <= 0:
            raise InvalidInputError(f"k must be a positive integer, got {k!r}")

    @staticmethod
    def precision_at_k(recommendations: list[str], relevant_items: list[str], k: int) -> float:
        """Fraction of the top-k recommendations that are relevant."""
        RecommendationEvaluator._validate_k(k)
        if not recommendations:
            return 0.0
        top_k = recommendations[:k]
        relevant = set(relevant_items)
        return sum(1 for item in top_k if item in relevant) / len(top_k)

    @staticmethod
    def recall_at_k(recommendations: list[str], relevant_items: list[str], k: int) -> float:
        """Fraction of all relevant items captured in the top-k."""
        RecommendationEvaluator._validate_k(k)
        if not relevant_items:
            return 0.0
        top_k = set(recommendations[:k])
        relevant = set(relevant_items)
        return len(top_k & relevant) / len(relevant)

    @staticmethod
    def ndcg_at_k(recommendations: list[str], relevant_items: list[str], k: int) -> float:
        """Position-aware ranking quality: relevant items ranked higher score more."""
        RecommendationEvaluator._validate_k(k)
        if not relevant_items or not recommendations:
            return 0.0

        relevant = set(relevant_items)
        top_k = recommendations[:k]

        dcg = sum(
            (1.0 if item in relevant else 0.0) / math.log2(i + 2)
            for i, item in enumerate(top_k)
        )
        ideal_hits = min(k, len(relevant))
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

        return dcg / idcg if idcg else 0.0

    def evaluate_all(
        self,
        recommendations_dict: dict[str, list[str]],
        ground_truth_dict: dict[str, list[str]],
        k: int = 10,
    ) -> dict[str, float]:
        """Average precision/recall/NDCG across users who have ground truth.
        Users missing ground truth are skipped, not counted as zero, so
        incomplete evaluation data doesn't silently deflate the score."""
        precisions, recalls, ndcgs = [], [], []
        skipped = 0

        for user_id, recs in recommendations_dict.items():
            relevant = ground_truth_dict.get(user_id)
            if not relevant:
                skipped += 1
                continue
            precisions.append(self.precision_at_k(recs, relevant, k))
            recalls.append(self.recall_at_k(recs, relevant, k))
            ndcgs.append(self.ndcg_at_k(recs, relevant, k))

        n = len(precisions)
        if skipped:
            logger.info("evaluate_all: skipped %d user(s) with no ground truth", skipped)
        if n == 0:
            return {"precision_at_k": 0.0, "recall_at_k": 0.0,
                    "ndcg_at_k": 0.0, "num_users_evaluated": 0}

        return {
            "precision_at_k": sum(precisions) / n,
            "recall_at_k": sum(recalls) / n,
            "ndcg_at_k": sum(ndcgs) / n,
            "num_users_evaluated": n,
        }
