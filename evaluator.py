"""
evaluator.py
------------
Component 4: Recommendation Evaluator
=========================================
Measures how good the recommendations are using standard IR metrics.

Why do we need evaluation?
    Without evaluation, we have no idea if our recommendations are actually
    helping users. These metrics tell us how many relevant items we found
    and how well we ranked them.

Three metrics are implemented:
    1. Precision@K  → Of the top-K items shown, how many were actually relevant?
    2. Recall@K     → Of all relevant items, how many did we find in top-K?
    3. NDCG@K       → Like precision, but rewards putting relevant items higher up

Real-world use:
    Netflix, Spotify, and Amazon use these exact metrics to A/B test and
    improve their recommendation systems every day.

Author  : Day 29 - HiDevs Internship
Project : Recommendation Engine Core Components
"""

import math


class RecommendationEvaluator:
    """
    Evaluates recommendation quality using standard information retrieval metrics.
    All methods handle missing or empty data gracefully.
    """

    def precision_at_k(self, recommendations, relevant_items, k):
        """
        Precision@K — "Of the top-K recommendations, what fraction is relevant?"

        This answers: "Are we recommending the right things?"
        High precision = most of what we show is actually useful.

        Formula:
            Precision@K = (# relevant items in top-K) / K

        Args:
            recommendations (list): Ordered list of recommended item IDs
            relevant_items  (list): Ground truth — items the user actually liked
            k               (int) : How many top recommendations to evaluate

        Returns:
            float: Score between 0.0 (nothing relevant) and 1.0 (all relevant)

        Edge Cases Handled:
            - Empty recommendations → returns 0.0
            - Empty relevant items  → returns 0.0
            - k <= 0               → returns 0.0

        Example:
            recs     = ["i1", "i2", "i3", "i4", "i5"]
            relevant = ["i1", "i3", "i5"]
            precision_at_k(recs, relevant, k=3) → 2/3 = 0.667
            (top-3 are i1, i2, i3 → 2 are relevant → 2/3)
        """
        # Edge cases: nothing to evaluate
        if not recommendations or not relevant_items or k <= 0:
            return 0.0

        # Only look at the top-K recommendations
        top_k = recommendations[:k]
        relevant_set = set(relevant_items)  # use set for O(1) lookups

        # Count how many of the top-K items are actually relevant
        hits = sum(1 for item in top_k if item in relevant_set)

        # Divide by K (not by hits) — that's what makes it "precision"
        return hits / k

    def recall_at_k(self, recommendations, relevant_items, k):
        """
        Recall@K — "Of all relevant items, how many did we find in top-K?"

        This answers: "Are we missing relevant items?"
        High recall = we're surfacing most of the things the user would like.

        Formula:
            Recall@K = (# relevant items in top-K) / (total # relevant items)

        Args:
            recommendations (list): Ordered list of recommended item IDs
            relevant_items  (list): Ground truth — items the user actually liked
            k               (int) : How many top recommendations to evaluate

        Returns:
            float: Score between 0.0 (found nothing) and 1.0 (found everything)

        Edge Cases Handled:
            - Empty recommendations → returns 0.0
            - Empty relevant items  → returns 0.0
            - k <= 0               → returns 0.0

        Example:
            recs     = ["i1", "i2", "i3", "i4", "i5"]
            relevant = ["i1", "i3", "i5"]
            recall_at_k(recs, relevant, k=3) → 2/3 = 0.667
            (top-3 are i1, i2, i3 → found 2 of the 3 relevant items)
        """
        # Edge cases: nothing to evaluate
        if not recommendations or not relevant_items or k <= 0:
            return 0.0

        # Only look at the top-K recommendations
        top_k = recommendations[:k]
        relevant_set = set(relevant_items)

        # Count how many of the top-K items are actually relevant
        hits = sum(1 for item in top_k if item in relevant_set)

        # Divide by TOTAL relevant items (not K) — that's what makes it "recall"
        return hits / len(relevant_set)

    def ndcg_at_k(self, recommendations, relevant_items, k):
        """
        NDCG@K — Normalized Discounted Cumulative Gain.

        This answers: "Are we ranking the relevant items near the top?"
        Unlike precision/recall, NDCG cares about ORDER — a relevant item
        at position 1 is worth more than a relevant item at position 5.

        Formula:
            DCG@K  = sum[ 1 / log2(rank + 1) ]  for each relevant item in top-K
            IDCG@K = DCG of a perfect ranking (all relevant items at top)
            NDCG@K = DCG@K / IDCG@K

        Args:
            recommendations (list): Ordered list of recommended item IDs
            relevant_items  (list): Ground truth — items the user actually liked
            k               (int) : How many top recommendations to evaluate

        Returns:
            float: Score between 0.0 (poor ranking) and 1.0 (perfect ranking)

        Edge Cases Handled:
            - Empty recommendations → returns 0.0
            - Empty relevant items  → returns 0.0
            - k <= 0               → returns 0.0
            - IDCG = 0             → returns 0.0 (avoids division by zero)

        Example:
            recs     = ["i1", "i2", "i3"]
            relevant = ["i1", "i3"]
            i1 at rank 1 → 1/log2(2) = 1.0
            i3 at rank 3 → 1/log2(4) = 0.5
            DCG = 1.5
        """
        # Edge cases: nothing to evaluate
        if not recommendations or not relevant_items or k <= 0:
            return 0.0

        top_k = recommendations[:k]
        relevant_set = set(relevant_items)

        # Step 1: Compute actual DCG — reward relevant items, penalize lower ranks
        dcg = 0.0
        for rank, item in enumerate(top_k, start=1):
            if item in relevant_set:
                # Logarithmic discount — rank 1 gets full credit, rank 5 gets less
                dcg += 1.0 / math.log2(rank + 1)

        # Step 2: Compute ideal DCG — what if we had ranked perfectly?
        ideal_hits = min(len(relevant_set), k)
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

        # Edge case: no ideal gain possible
        if idcg == 0:
            return 0.0

        # Step 3: Normalize — DCG relative to the best possible DCG
        return round(dcg / idcg, 4)

    def evaluate_all(self, recommendations_dict, ground_truth_dict, k=10):
        """
        Evaluate recommendations for all users and return averaged metrics.

        This is the main evaluation method for comparing different
        recommendation strategies against each other.

        Args:
            recommendations_dict (dict): {user_id: [recommended_item_ids]}
            ground_truth_dict    (dict): {user_id: [actually_relevant_item_ids]}
            k                    (int) : Cutoff position for all metrics

        Returns:
            dict: {
                "users_evaluated": number of users evaluated,
                "precision@k"    : average precision across all users,
                "recall@k"       : average recall across all users,
                "ndcg@k"         : average NDCG across all users,
                "k"              : the cutoff value used
            }

        Edge Cases Handled:
            - Users with no ground truth are skipped
            - If no users could be evaluated, returns zeros
        """
        precisions = []
        recalls    = []
        ndcgs      = []
        users_evaluated = 0

        for user_id, recs in recommendations_dict.items():
            relevant = ground_truth_dict.get(user_id)

            # Skip users who have no ground truth data to compare against
            if not relevant:
                continue

            # Compute all three metrics for this user
            precisions.append(self.precision_at_k(recs, relevant, k))
            recalls.append(self.recall_at_k(recs, relevant, k))
            ndcgs.append(self.ndcg_at_k(recs, relevant, k))
            users_evaluated += 1

        # Edge case: no users had ground truth — return zero metrics
        if users_evaluated == 0:
            return {
                "users_evaluated": 0,
                "precision@k"    : 0.0,
                "recall@k"       : 0.0,
                "ndcg@k"         : 0.0,
                "k"              : k,
            }

        # Average all metrics across evaluated users
        return {
            "users_evaluated": users_evaluated,
            "precision@k"    : round(sum(precisions) / users_evaluated, 4),
            "recall@k"       : round(sum(recalls)    / users_evaluated, 4),
            "ndcg@k"         : round(sum(ndcgs)      / users_evaluated, 4),
            "k"              : k,
        }