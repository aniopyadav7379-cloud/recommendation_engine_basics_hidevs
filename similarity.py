"""
similarity.py — Component 1: Similarity Calculator

Provides three similarity/correlation metrics used throughout a
recommendation engine:

- cosine_similarity   : angle-based similarity between two vectors
                        (e.g. user embedding vs item embedding)
- jaccard_similarity  : overlap between two sets (e.g. skill tags)
- pearson_correlation : linear correlation between two rating series
                        (e.g. how similarly two users rate the same items)
"""

import math
from typing import Sequence, Set, FrozenSet, Union

Number = Union[int, float]
NumericSet = Union[Set, FrozenSet]


class SimilarityCalculator:
    """Stateless collection of similarity metrics.

    All methods are safe to call with malformed/empty input — they
    return a sensible default (usually 0.0) instead of raising,
    since a recommendation pipeline should degrade gracefully rather
    than crash on one bad pair.
    """

    @staticmethod
    def cosine_similarity(vec1: Sequence[Number], vec2: Sequence[Number]) -> float:
        """Cosine similarity between two equal-length numeric vectors.

        Returns a value in [-1, 1] (in practice [0, 1] for non-negative
        feature vectors like ratings or embeddings). Returns 0.0 if
        either vector is empty, mismatched in length, or has zero
        magnitude (a zero vector has no direction to compare).
        """
        if not vec1 or not vec2:
            return 0.0
        if len(vec1) != len(vec2):
            raise ValueError("cosine_similarity: vectors must be the same length")

        dot = sum(a * b for a, b in zip(vec1, vec2))
        mag1 = math.sqrt(sum(a * a for a in vec1))
        mag2 = math.sqrt(sum(b * b for b in vec2))

        if mag1 == 0 or mag2 == 0:
            return 0.0

        return dot / (mag1 * mag2)

    @staticmethod
    def jaccard_similarity(set1: NumericSet, set2: NumericSet) -> float:
        """Jaccard similarity between two sets: |intersection| / |union|.

        Useful for comparing skill sets, tag lists, or category lists.
        Returns 1.0 when both sets are empty (identical — vacuously
        similar), and 0.0 when exactly one is empty.
        """
        s1, s2 = set(set1), set(set2)

        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0

        intersection = s1 & s2
        union = s1 | s2
        return len(intersection) / len(union)

    @staticmethod
    def pearson_correlation(ratings1: Sequence[Number], ratings2: Sequence[Number]) -> float:
        """Pearson correlation coefficient between two rating series.

        Measures whether two users rate items in a similarly-trending
        way (both above/below their own average), independent of
        rating scale offsets. Returns a value in [-1, 1].

        Returns 0.0 for: empty input, mismatched lengths, fewer than
        2 points, or zero variance in either series (a constant
        series has no correlation to measure).
        """
        if not ratings1 or not ratings2:
            return 0.0
        if len(ratings1) != len(ratings2):
            raise ValueError("pearson_correlation: series must be the same length")
        n = len(ratings1)
        if n < 2:
            return 0.0

        mean1 = sum(ratings1) / n
        mean2 = sum(ratings2) / n

        cov = sum((a - mean1) * (b - mean2) for a, b in zip(ratings1, ratings2))
        var1 = sum((a - mean1) ** 2 for a in ratings1)
        var2 = sum((b - mean2) ** 2 for b in ratings2)

        denom = math.sqrt(var1 * var2)
        if denom == 0:
            return 0.0

        return cov / denom


# ---------------------------------------------------------------------------
# Simple standalone test cases (also exercised from test.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sc = SimilarityCalculator()

    # cosine
    assert round(sc.cosine_similarity([1, 0], [1, 0]), 4) == 1.0
    assert round(sc.cosine_similarity([1, 0], [0, 1]), 4) == 0.0
    assert sc.cosine_similarity([], [1, 2]) == 0.0
    assert sc.cosine_similarity([0, 0], [1, 2]) == 0.0

    # jaccard
    assert sc.jaccard_similarity({"python", "sql"}, {"python", "sql"}) == 1.0
    assert sc.jaccard_similarity({"python"}, {"java"}) == 0.0
    assert sc.jaccard_similarity(set(), set()) == 1.0
    assert round(sc.jaccard_similarity({"a", "b", "c"}, {"b", "c", "d"}), 4) == round(2 / 4, 4)

    # pearson
    assert round(sc.pearson_correlation([1, 2, 3], [1, 2, 3]), 4) == 1.0
    assert round(sc.pearson_correlation([1, 2, 3], [3, 2, 1]), 4) == -1.0
    assert sc.pearson_correlation([1, 1, 1], [1, 2, 3]) == 0.0

    print("similarity.py: all inline tests passed")
