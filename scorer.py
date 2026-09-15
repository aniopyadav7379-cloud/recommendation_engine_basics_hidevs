"""
scorer.py
---------
Component 3: Scorer & Ranker
================================
Scores each candidate item and returns the top N ranked recommendations.

Why do we need a scorer?
    Candidate generation gives us a pool of 20-50 possible items.
    Not all of them are equally good for the user.
    The Scorer applies multiple weighted factors to find the BEST ones.

Three built-in scoring factors:
    1. Relevance   (weight 50%) → how well item tags match user's preferences
    2. Popularity  (weight 30%) → how popular the item is overall
    3. Recency     (weight 20%) → how recently the item was added/updated

You can also add custom scorers using add_scorer().

Author  : Day 29 - HiDevs Internship
Project : Recommendation Engine Core Components
"""

from candidate_gen import ITEM_POPULARITY, ITEM_FEATURES, USER_HISTORY
from similarity import SimilarityCalculator


# Simulated recency scores for each item (1.0 = newest, 0.0 = oldest)
# In a real system this would be calculated from timestamps
ITEM_RECENCY = {
    "i1": 0.4,
    "i2": 0.9,
    "i3": 0.7,
    "i4": 0.5,
    "i5": 0.8,
    "i6": 0.3,
    "i7": 1.0,  # newest item
    "i8": 0.6,
}


class RecommendationScorer:
    """
    Scores candidate items using multiple weighted factors.
    Produces a ranked list of recommendations with explanations.

    Usage:
        scorer = RecommendationScorer()
        ranked = scorer.rank_candidates("u1", ["i4", "i5", "i7"], limit=3)
    """

    def __init__(self):
        self.scorers = []           # list of registered scoring functions
        self.sim = SimilarityCalculator()
        self._register_default_scorers()  # load built-in scorers on startup

    def _register_default_scorers(self):
        """Register the three built-in scoring functions with their weights."""
        self.add_scorer("relevance",  self._relevance_score,  weight=0.5)
        self.add_scorer("popularity", self._popularity_score, weight=0.3)
        self.add_scorer("recency",    self._recency_score,    weight=0.2)

    def add_scorer(self, name, function, weight=1.0):
        """
        Register a custom scoring function.

        This allows you to plug in any business logic as a scoring factor.
        For example: discount score, trending score, diversity score, etc.

        Args:
            name     (str)     : Label for this scorer (shown in explanations)
            function (callable): fn(user_id, item_id, context) → float [0.0, 1.0]
            weight   (float)   : Relative importance of this scorer (e.g., 0.5 = 50%)

        Example:
            scorer.add_scorer("discount", lambda u, i, c: 0.8, weight=0.1)
        """
        self.scorers.append({"name": name, "fn": function, "weight": weight})

    def _relevance_score(self, user_id, item_id, context=None):
        """
        Relevance Score — how well does this item match the user's interests?

        Method:
            Build user's tag profile from their history, then measure
            how much the item's tags overlap using Jaccard similarity.

        Returns:
            float: 0.0 (no match) to 1.0 (perfect tag match)
                   0.5 for cold start users (neutral score)
        """
        user_items = USER_HISTORY.get(user_id, [])

        # Cold start: no history to build a preference profile from
        if not user_items:
            return 0.5  # neutral relevance score

        # Build user's tag profile from all items they've interacted with
        user_tags = set()
        for uid in user_items:
            user_tags.update(ITEM_FEATURES.get(uid, []))

        # Compare item's tags with user's tag profile
        item_tags = set(ITEM_FEATURES.get(item_id, []))
        return self.sim.jaccard_similarity(user_tags, item_tags)

    def _popularity_score(self, user_id, item_id, context=None):
        """
        Popularity Score — how popular is this item among all users?

        Method:
            Normalize raw interaction count to a 0-1 scale
            by dividing by the maximum popularity in the catalog.

        Returns:
            float: 0.0 (no interactions) to 1.0 (most popular item)
        """
        max_pop = max(ITEM_POPULARITY.values()) if ITEM_POPULARITY else 1
        raw_count = ITEM_POPULARITY.get(item_id, 0)
        return raw_count / max_pop  # normalized to [0, 1]

    def _recency_score(self, user_id, item_id, context=None):
        """
        Recency Score — how new or recently updated is this item?

        Method:
            Use a pre-computed recency value (0.0 = old, 1.0 = newest).
            In production, this would be calculated from item timestamps.

        Returns:
            float: 0.0 (oldest) to 1.0 (newest)
        """
        return ITEM_RECENCY.get(item_id, 0.5)  # default to neutral if unknown

    def calculate_score(self, user_id, item_id, context=None):
        """
        Calculate the final weighted score for one item for a given user.

        Method:
            For each registered scorer:
                1. Run the scoring function → raw score [0, 1]
                2. Multiply by its weight
                3. Sum all weighted scores
                4. Divide by total weight to normalize

        Args:
            user_id (str) : The user receiving recommendations
            item_id (str) : The item being scored
            context (dict): Optional extra data (e.g., time, device, location)

        Returns:
            dict: {
                "item_id"   : item_id,
                "score"     : final weighted score [0.0, 1.0],
                "breakdown" : {scorer_name: raw_score, ...},
                "reason"    : human-readable explanation
            }
        """
        total_weight = sum(s["weight"] for s in self.scorers)
        weighted_sum = 0.0
        breakdown = {}
        top_reason = None
        top_contribution = -1

        for scorer in self.scorers:
            # Get raw score from this scorer, clamped to valid range
            raw_score = scorer["fn"](user_id, item_id, context)
            raw_score = max(0.0, min(1.0, raw_score))  # clamp to [0, 1]

            # Weighted contribution of this scorer
            contribution = raw_score * scorer["weight"]
            weighted_sum += contribution

            # Record score breakdown for transparency
            breakdown[scorer["name"]] = round(raw_score, 3)

            # Track which scorer contributed the most (for explanation)
            if contribution > top_contribution:
                top_contribution = contribution
                top_reason = scorer["name"]

        # Normalize by total weight to keep final score in [0, 1]
        final_score = weighted_sum / total_weight if total_weight > 0 else 0.0

        return {
            "item_id"  : item_id,
            "score"    : round(final_score, 4),
            "breakdown": breakdown,
            "reason"   : f"Recommended mainly because of high {top_reason} score",
        }

    def rank_candidates(self, user_id, candidates, limit=10):
        """
        Score all candidate items and return the top N ranked results.

        This is the main method called by the recommendation pipeline.
        It scores every candidate, sorts by score descending, and returns
        the top items with their scores and explanations.

        Args:
            user_id    (str)  : The user to generate recommendations for
            candidates (list) : List of item IDs to score and rank
            limit      (int)  : Number of top items to return (default 10)

        Returns:
            list: Top N result dicts sorted by score (highest first)
                  Each dict has: item_id, score, breakdown, reason

        Edge Cases:
            - Empty candidates list → returns empty list
        """
        if not candidates:
            return []

        # Score every candidate item
        scored = [self.calculate_score(user_id, item_id) for item_id in candidates]

        # Sort by final score in descending order (best recommendations first)
        scored.sort(key=lambda x: x["score"], reverse=True)

        # Return only the top N results
        return scored[:limit]