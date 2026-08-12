"""Component 1: Similarity Calculator.

Metrics for comparing users, items, or skill sets.
"""

import math


class SimilarityCalculator:
    """Similarity metrics used across the recommendation pipeline."""

    @staticmethod
    def cosine_similarity(vec1, vec2):
        """Cosine similarity between two numeric vectors, clipped to [0, 1].
        Handles empty vectors, zero vectors, and mismatched lengths."""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0

        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0  # zero vector has no direction

        sim = dot / (norm1 * norm2)
        return max(0.0, min(1.0, sim))

    @staticmethod
    def jaccard_similarity(set1, set2):
        """Jaccard similarity (|A∩B| / |A∪B|) between two collections.
        Two empty sets are treated as identical; one empty is fully dissimilar."""
        s1, s2 = set(set1), set(set2)

        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0

        return len(s1 & s2) / len(s1 | s2)

    @staticmethod
    def pearson_correlation(ratings1, ratings2):
        """Pearson correlation coefficient in [-1, 1] between two paired
        rating series. Returns 0.0 for empty/mismatched/constant input."""
        if not ratings1 or not ratings2 or len(ratings1) != len(ratings2):
            return 0.0

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
            return 0.0  # constant series -> no meaningful correlation

        return max(-1.0, min(1.0, cov / denom))
