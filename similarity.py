"""
similarity.py — Similarity Calculator

Measures how similar two things are, using two different metrics:

- cosine_similarity: compares two numeric vectors (e.g. rating
  patterns, feature scores). Good when magnitude and direction both
  matter.
- jaccard_similarity: compares two sets (e.g. genres, tags, skills).
  Good for "how much do these two collections overlap".
"""

import math


class SimilarityCalculator:
    """Two similarity metrics, both safe to call on empty/edge-case input."""

    @staticmethod
    def cosine_similarity(vec1, vec2):
        """
        Cosine similarity between two equal-length numeric vectors.
        Returns a float, typically in [0, 1] for non-negative vectors
        like ratings, or [-1, 1] in general.

        Edge cases handled:
        - empty vector(s) -> 0.0 (nothing to compare)
        - a zero vector (all zeros) -> 0.0 (no direction to compare)
        """
        if not vec1 or not vec2:
            return 0.0
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must be the same length")

        # dot product of the two vectors
        dot_product = sum(a * b for a, b in zip(vec1, vec2))

        # magnitude (length) of each vector
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    @staticmethod
    def jaccard_similarity(set1, set2):
        """
        Jaccard similarity between two sets: size of the overlap
        divided by size of the combined set. Returns a float in [0, 1].

        Edge cases handled:
        - both sets empty -> 1.0 (two "nothing"s are identical)
        - only one set empty -> 0.0 (no overlap possible)
        """
        s1, s2 = set(set1), set(set2)

        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0

        intersection = s1 & s2
        union = s1 | s2
        return len(intersection) / len(union)


# ---------------------------------------------------------------------------
# Simple test cases you can run directly: python3 similarity.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sim = SimilarityCalculator()

    # cosine similarity tests
    print("Cosine similarity, identical vectors:", sim.cosine_similarity([1, 2, 3], [1, 2, 3]))
    print("Cosine similarity, opposite direction:", sim.cosine_similarity([1, 0], [0, 1]))
    print("Cosine similarity, empty vector:", sim.cosine_similarity([], [1, 2]))
    print("Cosine similarity, zero vector:", sim.cosine_similarity([0, 0], [1, 2]))

    assert round(sim.cosine_similarity([1, 2, 3], [1, 2, 3]), 4) == 1.0
    assert sim.cosine_similarity([1, 0], [0, 1]) == 0.0
    assert sim.cosine_similarity([], [1, 2]) == 0.0
    assert sim.cosine_similarity([0, 0], [1, 2]) == 0.0

    # jaccard similarity tests
    print("Jaccard similarity, identical sets:", sim.jaccard_similarity({"a", "b"}, {"a", "b"}))
    print("Jaccard similarity, no overlap:", sim.jaccard_similarity({"a"}, {"b"}))
    print("Jaccard similarity, both empty:", sim.jaccard_similarity(set(), set()))
    print("Jaccard similarity, partial overlap:", sim.jaccard_similarity({"a", "b", "c"}, {"b", "c", "d"}))

    assert sim.jaccard_similarity({"a", "b"}, {"a", "b"}) == 1.0
    assert sim.jaccard_similarity({"a"}, {"b"}) == 0.0
    assert sim.jaccard_similarity(set(), set()) == 1.0
    assert round(sim.jaccard_similarity({"a", "b", "c"}, {"b", "c", "d"}), 4) == round(2 / 4, 4)

    print("\nAll similarity.py tests passed!")
