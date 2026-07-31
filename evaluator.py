"""
evaluator.py — Component 4: Evaluator

Standard top-k ranking metrics for judging recommendation quality:

- precision_at_k : of the top-k items shown, what fraction were relevant?
- recall_at_k    : of all relevant items, what fraction appeared in top-k?
- ndcg_at_k      : like precision, but rewards relevant items ranked higher

`recommendations` is expected to be an ordered list of item IDs
(best first). `relevant_items` is a set/list of item IDs the user
actually engaged with (ground truth).
"""

import math


class RecommendationEvaluator:
    """Computes standard top-k evaluation metrics for one user's
    recommendation list, plus an aggregate evaluator across many users.
    """

    @staticmethod
    def _top_k(recommendations, k):
        return list(recommendations)[:k] if recommendations else []

    @staticmethod
    def precision_at_k(recommendations, relevant_items, k):
        """Fraction of the top-k recommended items that are relevant.

        Returns 0.0 if there are no recommendations or k <= 0.
        """
        if not recommendations or k <= 0:
            return 0.0
        relevant_set = set(relevant_items or [])
        top_k = RecommendationEvaluator._top_k(recommendations, k)
        if not top_k:
            return 0.0
        hits = sum(1 for item in top_k if item in relevant_set)
        return hits / len(top_k)

    @staticmethod
    def recall_at_k(recommendations, relevant_items, k):
        """Fraction of all relevant items that were captured in the top-k.

        Returns 0.0 if there is no ground truth to compare against
        (nothing was "missed" because nothing was known to be relevant,
        but there's also nothing to credit — 0.0 keeps the metric
        well-defined rather than dividing by zero).
        """
        relevant_set = set(relevant_items or [])
        if not relevant_set or not recommendations:
            return 0.0
        top_k = RecommendationEvaluator._top_k(recommendations, k)
        hits = sum(1 for item in top_k if item in relevant_set)
        return hits / len(relevant_set)

    @staticmethod
    def ndcg_at_k(recommendations, relevant_items, k):
        """Normalized Discounted Cumulative Gain at k.

        Uses binary relevance (1 if in relevant_items, else 0).
        Position matters: a relevant item at rank 1 contributes more
        than the same item at rank 10. Normalized against the ideal
        ordering (all relevant items first) so the result is in [0, 1].
        """
        relevant_set = set(relevant_items or [])
        if not recommendations or not relevant_set or k <= 0:
            return 0.0

        top_k = RecommendationEvaluator._top_k(recommendations, k)

        dcg = 0.0
        for i, item in enumerate(top_k):
            rel = 1.0 if item in relevant_set else 0.0
            if rel:
                dcg += rel / math.log2(i + 2)  # i is 0-indexed, rank = i+1

        ideal_hits = min(len(relevant_set), k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

        if idcg == 0:
            return 0.0
        return dcg / idcg

    @classmethod
    def evaluate_all(cls, recommendations_dict, ground_truth_dict, k=10):
        """Average precision/recall/ndcg at k across many users.

        recommendations_dict: {user_id: [item_id, ...]}
        ground_truth_dict:    {user_id: [item_id, ...]}  (relevant items)

        Users missing from ground_truth_dict are skipped from the
        average (there's nothing to evaluate them against) rather
        than silently counted as zero, which would understate quality
        for reasons unrelated to the recommender itself.

        Returns: {"precision_at_k": ..., "recall_at_k": ..., "ndcg_at_k": ...,
                   "k": k, "users_evaluated": int}
        """
        precisions, recalls, ndcgs = [], [], []

        for user_id, recs in (recommendations_dict or {}).items():
            if user_id not in (ground_truth_dict or {}):
                continue
            relevant = ground_truth_dict[user_id]
            if not relevant:
                continue

            precisions.append(cls.precision_at_k(recs, relevant, k))
            recalls.append(cls.recall_at_k(recs, relevant, k))
            ndcgs.append(cls.ndcg_at_k(recs, relevant, k))

        users_evaluated = len(precisions)

        def avg(values):
            return round(sum(values) / len(values), 4) if values else 0.0

        return {
            "precision_at_k": avg(precisions),
            "recall_at_k": avg(recalls),
            "ndcg_at_k": avg(ndcgs),
            "k": k,
            "users_evaluated": users_evaluated,
        }


# ---------------------------------------------------------------------------
# Simple standalone test cases
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ev = RecommendationEvaluator()

    recs = ["i1", "i2", "i3", "i4", "i5"]
    relevant = ["i2", "i4", "i9"]  # i9 never recommended

    p = ev.precision_at_k(recs, relevant, k=5)
    assert round(p, 4) == round(2 / 5, 4)

    r = ev.recall_at_k(recs, relevant, k=5)
    assert round(r, 4) == round(2 / 3, 4)

    n = ev.ndcg_at_k(recs, relevant, k=5)
    assert 0.0 < n < 1.0  # relevant items not in ideal (top) positions

    # perfect ranking -> ndcg of 1.0
    perfect_recs = ["i2", "i4", "i1", "i3", "i5"]
    perfect_relevant = ["i2", "i4"]
    assert round(ev.ndcg_at_k(perfect_recs, perfect_relevant, k=5), 4) == 1.0

    # edge cases
    assert ev.precision_at_k([], relevant, 5) == 0.0
    assert ev.recall_at_k(recs, [], 5) == 0.0
    assert ev.ndcg_at_k(recs, [], 5) == 0.0

    # aggregate
    recommendations_dict = {
        "u1": ["i1", "i2", "i3"],
        "u2": ["i4", "i5", "i6"],
        "u3": ["i7", "i8", "i9"],  # no ground truth for u3
    }
    ground_truth_dict = {
        "u1": ["i2"],
        "u2": ["i4", "i5"],
    }
    summary = ev.evaluate_all(recommendations_dict, ground_truth_dict, k=3)
    assert summary["users_evaluated"] == 2
    assert 0.0 <= summary["precision_at_k"] <= 1.0
    assert "ndcg_at_k" in summary

    print("evaluator.py: all inline tests passed")
