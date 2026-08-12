"""Component 4: Evaluator.

Offline metrics for measuring recommendation quality against ground
truth (items the user actually engaged with).
"""

import math


class RecommendationEvaluator:
    """precision@k, recall@k, NDCG@k, and a multi-user averaging helper."""

    @staticmethod
    def precision_at_k(recommendations, relevant_items, k):
        """Fraction of the top-k recommendations that are relevant."""
        if not recommendations or k <= 0:
            return 0.0
        top_k = recommendations[:k]
        relevant = set(relevant_items)
        return sum(1 for item in top_k if item in relevant) / len(top_k)

    @staticmethod
    def recall_at_k(recommendations, relevant_items, k):
        """Fraction of all relevant items captured in the top-k."""
        if not relevant_items or k <= 0:
            return 0.0
        top_k = set(recommendations[:k])
        relevant = set(relevant_items)
        return len(top_k & relevant) / len(relevant)

    @staticmethod
    def ndcg_at_k(recommendations, relevant_items, k):
        """Position-aware ranking quality: relevant items ranked higher score more."""
        if not relevant_items or k <= 0 or not recommendations:
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

    def evaluate_all(self, recommendations_dict, ground_truth_dict, k=10):
        """Average precision/recall/NDCG across users who have ground truth.
        Users missing ground truth are skipped, not counted as zero."""
        precisions, recalls, ndcgs = [], [], []

        for user_id, recs in recommendations_dict.items():
            relevant = ground_truth_dict.get(user_id)
            if not relevant:
                continue
            precisions.append(self.precision_at_k(recs, relevant, k))
            recalls.append(self.recall_at_k(recs, relevant, k))
            ndcgs.append(self.ndcg_at_k(recs, relevant, k))

        n = len(precisions)
        if n == 0:
            return {"precision_at_k": 0.0, "recall_at_k": 0.0,
                    "ndcg_at_k": 0.0, "num_users_evaluated": 0}

        return {
            "precision_at_k": sum(precisions) / n,
            "recall_at_k": sum(recalls) / n,
            "ndcg_at_k": sum(ndcgs) / n,
            "num_users_evaluated": n,
        }
