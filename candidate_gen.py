"""
candidate_gen.py — Component 2: Candidate Generator

Generates pools of candidate item IDs using several strategies.
Today this runs on plain in-memory dictionaries (no DB yet) — the
same interface will sit on top of real storage tomorrow.

Expected data shape (all dictionaries, injected via the constructor):

    user_item_history: {user_id: [item_id, ...]}       # items a user liked/interacted with
    item_tags:         {item_id: {tag, ...}}            # tags/categories per item
    item_popularity:   {item_id: int}                   # e.g. total likes/purchases
    user_similarity_fn: optional callable(user_id) -> [(other_user_id, score), ...]
                         sorted by score descending. Defaults to a simple
                         co-occurrence heuristic if not provided.
"""

from similarity import SimilarityCalculator


class CandidateGenerator:
    """Produces candidate item pools for a given user using several
    independent strategies, plus a hybrid combiner.
    """

    DEFAULT_LIMIT = 20
    MAX_LIMIT = 50

    def __init__(self, user_item_history=None, item_tags=None, item_popularity=None):
        self.user_item_history = user_item_history or {}
        self.item_tags = item_tags or {}
        self.item_popularity = item_popularity or {}
        self.similarity = SimilarityCalculator()

    # -- helpers ------------------------------------------------------

    def _all_items(self):
        items = set(self.item_popularity.keys())
        for hist in self.user_item_history.values():
            items.update(hist)
        return items

    def _similar_users(self, user_id):
        """Rank other users by Jaccard similarity of their item history.

        Simple, dependency-free stand-in for a real collaborative
        filtering model — good enough to generate candidates today.
        """
        target_items = set(self.user_item_history.get(user_id, []))
        scored = []
        for other_id, other_items in self.user_item_history.items():
            if other_id == user_id:
                continue
            score = self.similarity.jaccard_similarity(target_items, set(other_items))
            if score > 0:
                scored.append((other_id, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored

    def _limit(self, items, limit):
        limit = max(1, min(limit, self.MAX_LIMIT))
        return items[:limit]

    # -- strategies -----------------------------------------------------

    def collaborative_candidates(self, user_id, limit=DEFAULT_LIMIT):
        """Items liked by users similar to `user_id` (that this user
        hasn't already interacted with). Falls back to popularity for
        cold-start users with no history or no similar peers.
        """
        seen = set(self.user_item_history.get(user_id, []))
        similar_users = self._similar_users(user_id)

        if not similar_users:
            return self.popularity_candidates(limit)

        candidates = []
        seen_candidates = set()
        for other_id, _score in similar_users:
            for item in self.user_item_history.get(other_id, []):
                if item not in seen and item not in seen_candidates:
                    candidates.append(item)
                    seen_candidates.add(item)

        if not candidates:
            return self.popularity_candidates(limit)

        return self._limit(candidates, limit)

    def content_based_candidates(self, user_id, limit=DEFAULT_LIMIT):
        """Items whose tags overlap with tags of items the user has
        already liked. Falls back to popularity for cold-start users.
        """
        history = self.user_item_history.get(user_id, [])
        if not history:
            return self.popularity_candidates(limit)

        user_tags = set()
        for item in history:
            user_tags.update(self.item_tags.get(item, set()))

        if not user_tags:
            return self.popularity_candidates(limit)

        seen = set(history)
        scored = []
        for item in self._all_items():
            if item in seen:
                continue
            tags = self.item_tags.get(item, set())
            score = self.similarity.jaccard_similarity(user_tags, tags)
            if score > 0:
                scored.append((item, score))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        candidates = [item for item, _score in scored]

        if not candidates:
            return self.popularity_candidates(limit)

        return self._limit(candidates, limit)

    def popularity_candidates(self, limit=DEFAULT_LIMIT):
        """Most popular items overall, regardless of user. Always
        available — this is the ultimate cold-start fallback.
        """
        ranked = sorted(self.item_popularity.items(), key=lambda pair: pair[1], reverse=True)
        candidates = [item for item, _count in ranked]
        return self._limit(candidates, limit)

    def hybrid_candidates(self, user_id, limit=DEFAULT_LIMIT):
        """Combine collaborative, content-based, and popularity candidates,
        interleaved so no single strategy dominates the pool, then
        de-duplicated while preserving order of first appearance.
        """
        collab = self.collaborative_candidates(user_id, limit)
        content = self.content_based_candidates(user_id, limit)
        popular = self.popularity_candidates(limit)

        combined = []
        seen = set()
        for triple in zip_longest_manual(collab, content, popular):
            for item in triple:
                if item is not None and item not in seen:
                    combined.append(item)
                    seen.add(item)

        return self._limit(combined, limit)


def zip_longest_manual(*lists):
    """Small local zip_longest so we don't need itertools for one call site."""
    max_len = max((len(lst) for lst in lists), default=0)
    for i in range(max_len):
        yield tuple(lst[i] if i < len(lst) else None for lst in lists)


# ---------------------------------------------------------------------------
# Simple standalone test cases
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    history = {
        "u1": ["i1", "i2", "i3"],
        "u2": ["i1", "i2", "i4"],
        "u3": ["i5", "i6"],
    }
    tags = {
        "i1": {"python", "backend"},
        "i2": {"python", "ml"},
        "i3": {"frontend"},
        "i4": {"python", "backend"},
        "i5": {"design"},
        "i6": {"design", "frontend"},
    }
    popularity = {"i1": 50, "i2": 40, "i3": 10, "i4": 30, "i5": 5, "i6": 5}

    gen = CandidateGenerator(history, tags, popularity)

    collab = gen.collaborative_candidates("u1")
    assert "i4" in collab  # u2 is similar to u1 and liked i4

    content = gen.content_based_candidates("u1")
    assert "i4" in content  # shares python/backend tags with i1

    popular = gen.popularity_candidates(3)
    assert popular == ["i1", "i2", "i4"]

    cold_start = gen.collaborative_candidates("new_user")
    assert cold_start == gen.popularity_candidates()

    hybrid = gen.hybrid_candidates("u1")
    assert len(hybrid) > 0

    print("candidate_gen.py: all inline tests passed")
