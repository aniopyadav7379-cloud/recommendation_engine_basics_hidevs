"""
evaluator.py — Evaluator

Checks how good a set of recommendations actually is, using
precision: of the items we recommended, what fraction did the user
actually want (based on some known "ground truth" of relevant items)?

This is how you'd know, after the fact, whether the Scorer's rankings
were any good — not just that the code ran, but that it recommended
the right things.
"""


class Evaluator:
    """Basic evaluation metrics for a list of recommendations."""

    @staticmethod
    def precision(recommended_items, relevant_items):
        """
        Precision = (number of recommended items that are relevant)
                    / (total number of recommended items)

        recommended_items: list of item IDs the system recommended
        relevant_items: list/set of item IDs the user actually liked
                         (the "ground truth")

        Returns a float in [0, 1].

        Edge cases handled:
        - no recommendations made -> 0.0 (can't have precision with
          nothing recommended)
        - no relevant items known -> 0.0 (nothing to be "correct" about)
        """
        if not recommended_items:
            return 0.0
        if not relevant_items:
            return 0.0

        relevant_set = set(relevant_items)
        hits = sum(1 for item in recommended_items if item in relevant_set)

        return hits / len(recommended_items)

    @staticmethod
    def precision_at_k(recommended_items, relevant_items, k):
        """
        Precision, but only looking at the top k recommendations.
        Useful because most systems only show a user the top few
        picks — precision over the whole list can be misleadingly low
        if the good stuff is all near the top.

        Edge cases handled:
        - k <= 0 -> 0.0
        - fewer than k recommendations -> just evaluates however many
          there are, rather than crashing
        """
        if k <= 0 or not recommended_items:
            return 0.0

        top_k = recommended_items[:k]
        return Evaluator.precision(top_k, relevant_items)


# ---------------------------------------------------------------------------
# Simple test cases you can run directly: python3 evaluator.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    evaluator = Evaluator()

    recommended = ["movie1", "movie2", "movie3", "movie4"]
    relevant = ["movie1", "movie3", "movie9"]  # movie9 was never recommended

    p = evaluator.precision(recommended, relevant)
    print("Precision:", p)
    assert round(p, 4) == round(2 / 4, 4)  # movie1 and movie3 are hits

    p_at_2 = evaluator.precision_at_k(recommended, relevant, k=2)
    print("Precision@2:", p_at_2)
    # top 2 recommendations are movie1, movie2 -> only movie1 is relevant
    assert round(p_at_2, 4) == 0.5

    # edge cases
    assert evaluator.precision([], relevant) == 0.0
    assert evaluator.precision(recommended, []) == 0.0
    assert evaluator.precision_at_k(recommended, relevant, k=0) == 0.0

    print("\nAll evaluator.py tests passed!")
