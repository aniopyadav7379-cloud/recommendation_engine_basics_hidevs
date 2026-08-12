"""Component 2: Candidate Generator.

Generates pools of candidate item IDs using several strategies. Uses
plain dicts as a stand-in "database" for now.
"""

from collections import Counter

from similarity import SimilarityCalculator


class CandidateGenerator:
    """Generates candidate items for a user via multiple strategies."""

    def __init__(self, user_history=None, item_features=None, max_candidates=20):
        self.user_history = user_history or {}
        self.item_features = item_features or {}
        self.max_candidates = max_candidates
        self.similarity = SimilarityCalculator()

    def collaborative_candidates(self, user_id):
        """Items liked by users similar to this user (Jaccard on history)."""
        target = self.user_history.get(user_id, set())
        if not target:
            return self.popularity_candidates()

        similar_users = sorted(
            ((uid, self.similarity.jaccard_similarity(target, items))
             for uid, items in self.user_history.items() if uid != user_id),
            key=lambda pair: pair[1], reverse=True,
        )

        candidates, seen = [], set()
        for uid, sim in similar_users:
            if sim <= 0:
                continue
            for item in self.user_history[uid]:
                if item in target or item in seen:
                    continue
                seen.add(item)
                candidates.append(item)
                if len(candidates) >= self.max_candidates:
                    return candidates
        return candidates or self.popularity_candidates()

    def content_based_candidates(self, user_id):
        """Items similar to the user's history, by tag/feature overlap."""
        target = self.user_history.get(user_id, set())
        if not target or not self.item_features:
            return self.popularity_candidates()

        profile = set()
        for item in target:
            profile |= self.item_features.get(item, set())
        if not profile:
            return self.popularity_candidates()

        scored = sorted(
            ((iid, self.similarity.jaccard_similarity(profile, tags))
             for iid, tags in self.item_features.items() if iid not in target),
            key=lambda pair: pair[1], reverse=True,
        )
        candidates = [iid for iid, sim in scored if sim > 0][: self.max_candidates]
        return candidates or self.popularity_candidates()

    def popularity_candidates(self):
        """Most popular items overall, ranked by interaction count."""
        if not self.user_history:
            return []
        counts = Counter()
        for items in self.user_history.values():
            counts.update(items)
        return [iid for iid, _ in counts.most_common(self.max_candidates)]

    def hybrid_candidates(self, user_id):
        """Interleave collaborative, content-based, and popularity results,
        excluding items the user has already seen."""
        target = self.user_history.get(user_id, set())
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
                        return merged
        return merged
