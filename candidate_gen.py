"""
candidate_gen.py
----------------
Component 2: Candidate Generator
===================================
Generates a pool of items to potentially recommend to a user.

Why do we need candidate generation?
    You cannot score ALL items in a large database — it would be too slow.
    Candidate generation narrows the pool to a manageable set (20-50 items)
    that the Scorer (Component 3) will then rank.

Four strategies are implemented:
    1. collaborative_candidates  → items liked by similar users
    2. content_based_candidates  → items matching user's tag preferences
    3. popularity_candidates     → most popular items overall
    4. hybrid_candidates         → smart combination of all three above

Cold Start Problem:
    New users have no history. Instead of failing, we fall back to
    popularity-based candidates so they still get recommendations.

Author  : Day 29 - HiDevs Internship
Project : Recommendation Engine Core Components
"""

from similarity import SimilarityCalculator


# ─── Mock Database (simulates a real database with dictionaries) ───────────────

# Stores each user's interaction history (items they have seen/liked)
USER_HISTORY = {
    "u1": ["i1", "i2", "i3"],   # user u1 liked items i1, i2, i3
    "u2": ["i2", "i3", "i4"],
    "u3": ["i1", "i4", "i5"],
    "u4": ["i3", "i5", "i6"],
    "u5": [],                    # cold start user — no history yet
}

# Stores each item's tags/features (describes what the item is about)
ITEM_FEATURES = {
    "i1": ["python", "ml", "beginner"],
    "i2": ["python", "data", "beginner"],
    "i3": ["ml", "deep-learning", "advanced"],
    "i4": ["data", "sql", "intermediate"],
    "i5": ["deep-learning", "nlp", "advanced"],
    "i6": ["sql", "analytics", "intermediate"],
    "i7": ["python", "nlp", "intermediate"],
    "i8": ["ml", "analytics", "beginner"],
}

# Stores total interaction count per item (used for popularity ranking)
ITEM_POPULARITY = {
    "i1": 150,
    "i2": 200,  # most popular
    "i3": 180,
    "i4": 90,
    "i5": 120,
    "i6": 60,
    "i7": 75,
    "i8": 110,
}


class CandidateGenerator:
    """
    Generates candidate items for recommendation using multiple strategies.
    Each method returns a list of item IDs to be passed to the Scorer.
    """

    def __init__(self):
        self.sim = SimilarityCalculator()      # used to compare user histories
        self.all_items = list(ITEM_FEATURES.keys())  # full catalog of items

    def collaborative_candidates(self, user_id, limit=20):
        """
        Collaborative Filtering — "users like you also liked..."

        Strategy:
            1. Find users who have similar history to the given user
            2. Collect items those similar users interacted with
            3. Exclude items the user has already seen
            4. Return items ranked by how often similar users liked them

        Cold Start Handling:
            If the user has no history, fall back to popularity_candidates()
            so the user still receives meaningful recommendations.

        Args:
            user_id (str): The user to generate candidates for
            limit   (int): Maximum number of candidates to return (default 20)

        Returns:
            list: Item IDs sorted by collaborative relevance score
        """
        user_items = set(USER_HISTORY.get(user_id, []))

        # Cold start: user has no history — fall back to popular items
        if not user_items:
            return self.popularity_candidates(limit=limit)

        candidates = {}  # item_id → accumulated similarity score

        # Compare this user with every other user
        for other_id, other_items in USER_HISTORY.items():
            if other_id == user_id:
                continue  # don't compare user with themselves

            # Calculate how similar the other user is
            similarity = self.sim.jaccard_similarity(user_items, set(other_items))

            if similarity > 0:
                # Add their items to candidates, weighted by similarity
                for item in other_items:
                    if item not in user_items:  # only suggest unseen items
                        candidates[item] = candidates.get(item, 0) + similarity

        # Sort candidates by accumulated similarity score (highest first)
        sorted_candidates = sorted(candidates, key=candidates.get, reverse=True)
        return sorted_candidates[:limit]

    def content_based_candidates(self, user_id, limit=20):
        """
        Content-Based Filtering — "items similar to what you liked..."

        Strategy:
            1. Extract all tags from items the user has interacted with
            2. Find items in the catalog whose tags overlap with user's preferences
            3. Exclude already seen items
            4. Return items ranked by tag overlap score

        Cold Start Handling:
            If the user has no history, return the first N items from the catalog.

        Args:
            user_id (str): The user to generate candidates for
            limit   (int): Maximum number of candidates to return (default 20)

        Returns:
            list: Item IDs sorted by content similarity score
        """
        user_items = USER_HISTORY.get(user_id, [])

        # Cold start: no history — return items from catalog
        if not user_items:
            return self.all_items[:limit]

        # Step 1: Build user's tag profile from their interaction history
        user_tags = set()
        for item_id in user_items:
            tags = ITEM_FEATURES.get(item_id, [])
            user_tags.update(tags)

        candidates = {}  # item_id → content similarity score

        # Step 2: Compare user's tag profile with every unseen item
        for item_id, item_tags in ITEM_FEATURES.items():
            if item_id in user_items:
                continue  # skip already seen items

            # Measure how much this item's tags match the user's preferences
            score = self.sim.jaccard_similarity(user_tags, set(item_tags))
            if score > 0:
                candidates[item_id] = score

        # Sort by content match score (highest first)
        sorted_candidates = sorted(candidates, key=candidates.get, reverse=True)
        return sorted_candidates[:limit]

    def popularity_candidates(self, limit=20):
        """
        Popularity-Based Filtering — "trending / most popular items..."

        Strategy:
            Simply rank all items by their total interaction count.
            This is a great fallback for cold start users and new items.

        Args:
            limit (int): Maximum number of candidates to return (default 20)

        Returns:
            list: Item IDs sorted by popularity (most popular first)
        """
        # Sort all items by interaction count in descending order
        sorted_items = sorted(ITEM_POPULARITY, key=ITEM_POPULARITY.get, reverse=True)
        return sorted_items[:limit]

    def hybrid_candidates(self, user_id, limit=30):
        """
        Hybrid Filtering — combines all three strategies for best coverage.

        Strategy:
            1. Collect candidates from collaborative, content-based, and popularity
            2. Merge them with priority: collaborative > content-based > popularity
            3. Deduplicate to avoid showing the same item twice
            4. Return the merged pool for scoring

        Why hybrid?
            Each strategy has weaknesses:
            - Collaborative fails for new users (cold start)
            - Content-based misses serendipitous discoveries
            - Popularity ignores personal taste
            Combining them gives the best of all worlds.

        Args:
            user_id (str): The user to generate candidates for
            limit   (int): Maximum total candidates to return (default 30)

        Returns:
            list: Deduplicated item IDs from all three strategies
        """
        # Gather candidates from each strategy
        collab  = set(self.collaborative_candidates(user_id, limit=limit))
        content = set(self.content_based_candidates(user_id, limit=limit))
        popular = set(self.popularity_candidates(limit=limit))

        # Merge with priority order: collaborative first, then content, then popular
        combined = list(collab)
        for item in content:
            if item not in combined:
                combined.append(item)
        for item in popular:
            if item not in combined:
                combined.append(item)

        return combined[:limit]