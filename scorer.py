"""
scorer.py — Scorer

Takes a shortlist of candidate items (from the Candidate Generator)
and ranks them, combining two signals:

- relevance: how well the item's tags match the user's preferences
  (using Jaccard similarity)
- rating: the item's average rating, if available (so a highly-rated
  item edges out an equally-relevant but poorly-rated one)

Returns the top N picks, best first.
"""

from similarity import SimilarityCalculator


class Scorer:
    """Scores and ranks candidate items for a user."""

    # how much weight relevance gets vs. rating when combining scores
    RELEVANCE_WEIGHT = 0.7
    RATING_WEIGHT = 0.3
    MAX_RATING = 5.0  # assumes a 5-star rating scale

    def __init__(self, item_catalog, item_ratings=None):
        """
        item_catalog: dict of {item_id: set_of_tags}
        item_ratings: optional dict of {item_id: average_rating (0-5)}
        """
        self.item_catalog = item_catalog or {}
        self.item_ratings = item_ratings or {}
        self.similarity = SimilarityCalculator()

    def score_item(self, item_id, user_preferences):
        """
        Score a single item for a user. Returns a float in [0, 1].

        Edge cases handled:
        - item not in catalog -> treated as having no tags, so
          relevance is 0.0
        - item has no rating on file -> treated as rating 0.0 (doesn't
          crash, just doesn't get a rating boost)
        """
        tags = self.item_catalog.get(item_id, set())
        relevance = self.similarity.jaccard_similarity(user_preferences, tags)

        raw_rating = self.item_ratings.get(item_id, 0.0)
        normalized_rating = raw_rating / self.MAX_RATING if self.MAX_RATING else 0.0

        combined = (self.RELEVANCE_WEIGHT * relevance) + (self.RATING_WEIGHT * normalized_rating)
        return round(combined, 4)

    def rank_candidates(self, candidates, user_preferences, top_n=5):
        """
        Score every candidate and return the top_n highest-scored,
        as a list of (item_id, score) tuples, best first.

        Edge cases handled:
        - empty candidate list -> returns []
        """
        if not candidates:
            return []

        scored = [
            (item_id, self.score_item(item_id, user_preferences))
            for item_id in candidates
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)

        top_n = max(1, top_n)
        return scored[:top_n]


# ---------------------------------------------------------------------------
# Simple test cases you can run directly: python3 scorer.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    catalog = {
        "movie1": {"action", "sci-fi"},
        "movie2": {"sci-fi"},
        "movie3": {"comedy"},
    }
    ratings = {"movie1": 4.5, "movie2": 3.0, "movie3": 4.8}

    scorer = Scorer(catalog, ratings)
    preferences = {"action", "sci-fi"}

    # movie1 matches both preference tags AND has a high rating,
    # so it should outrank movie2 (matches one tag, lower rating)
    ranked = scorer.rank_candidates(["movie1", "movie2", "movie3"], preferences, top_n=2)
    print("Ranked recommendations:", ranked)

    assert ranked[0][0] == "movie1"
    assert len(ranked) == 2

    # empty candidates -> empty result
    assert scorer.rank_candidates([], preferences) == []

    # item not in catalog doesn't crash, just scores low
    score = scorer.score_item("unknown_movie", preferences)
    assert score == 0.0

    print("\nAll scorer.py tests passed!")
