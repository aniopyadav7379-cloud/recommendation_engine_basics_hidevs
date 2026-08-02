"""
candidate_gen.py — Candidate Generator

Given a catalog of items (each described by a set of tags, e.g.
genres or categories) and a user's preferences (a set of tags they
like), finds candidate items worth recommending.

This is the "narrow down the whole catalog to a shortlist" step —
the Scorer decides which of these candidates are actually the best
picks.
"""

from similarity import SimilarityCalculator


class CandidateGenerator:
    """Finds candidate items based on how well their tags match a
    user's stated preferences.
    """

    def __init__(self, item_catalog):
        """
        item_catalog: dict of {item_id: set_of_tags}
            e.g. {"movie1": {"action", "sci-fi"}, "movie2": {"romance"}}
        """
        self.item_catalog = item_catalog or {}
        self.similarity = SimilarityCalculator()

    def find_candidates(self, user_preferences, limit=10):
        """
        Return a list of item IDs whose tags overlap with the user's
        preferences, ranked by how much they overlap (best match first).

        user_preferences: a set (or list) of tags the user likes,
            e.g. {"action", "sci-fi"}
        limit: maximum number of candidates to return

        Edge cases handled:
        - empty catalog -> returns []
        - empty/no preferences -> returns [] (nothing to match against,
          so we can't meaningfully guess what they'd like)
        - no items overlap at all -> returns []
        """
        if not self.item_catalog or not user_preferences:
            return []

        preferences = set(user_preferences)
        scored_items = []

        for item_id, tags in self.item_catalog.items():
            score = self.similarity.jaccard_similarity(preferences, tags)
            if score > 0:
                scored_items.append((item_id, score))

        # best matches first
        scored_items.sort(key=lambda pair: pair[1], reverse=True)

        limit = max(1, limit)
        return [item_id for item_id, _score in scored_items[:limit]]


# ---------------------------------------------------------------------------
# Simple test cases you can run directly: python3 candidate_gen.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    catalog = {
        "movie1": {"action", "sci-fi"},
        "movie2": {"romance", "comedy"},
        "movie3": {"sci-fi", "thriller"},
        "movie4": {"comedy"},
    }

    gen = CandidateGenerator(catalog)

    # user who likes sci-fi and action should get movie1 and movie3
    candidates = gen.find_candidates({"action", "sci-fi"})
    print("Candidates for sci-fi/action fan:", candidates)
    assert "movie1" in candidates
    assert "movie3" in candidates
    assert "movie2" not in candidates

    # empty preferences -> no candidates
    assert gen.find_candidates(set()) == []

    # empty catalog -> no candidates
    empty_gen = CandidateGenerator({})
    assert empty_gen.find_candidates({"action"}) == []

    # preferences with no matching items at all
    assert gen.find_candidates({"horror"}) == []

    # limit is respected
    limited = gen.find_candidates({"action", "sci-fi", "romance", "comedy", "thriller"}, limit=2)
    assert len(limited) == 2

    print("\nAll candidate_gen.py tests passed!")
