"""Component 1: Similarity Calculator.

Metrics for comparing users, items, or skill sets. Pure functions with
no external state, so they're trivially unit-testable and safe to call
from concurrent contexts.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from functools import lru_cache

from .exceptions import InvalidInputError


class SimilarityCalculator:
    """Similarity metrics used across the recommendation pipeline."""

    # ------------------------------------------------------------------
    @staticmethod
    def cosine_similarity(vec1: Sequence[float], vec2: Sequence[float]) -> float:
        """Cosine similarity between two numeric vectors, clipped to [0, 1].

        Raises InvalidInputError on mismatched (non-empty) lengths, since
        that signals a caller bug rather than a legitimate edge case.
        Empty or all-zero vectors return 0.0 (similarity is undefined,
        not an error — cold-start data is common in this domain).
        """
        if vec1 and vec2 and len(vec1) != len(vec2):
            raise InvalidInputError(
                f"cosine_similarity: length mismatch ({len(vec1)} vs {len(vec2)})"
            )
        if not vec1 or not vec2:
            return 0.0

        dot = sum(a * b for a, b in zip(vec1, vec2, strict=True))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        if norm1 == 0 or norm2 == 0:
            return 0.0

        return max(0.0, min(1.0, dot / (norm1 * norm2)))

    # ------------------------------------------------------------------
    @staticmethod
    @lru_cache(maxsize=4096)
    def _jaccard_cached(s1: tuple[str, ...], s2: tuple[str, ...]) -> float:
        """Cached core computation; keyed on sorted tuples so set order
        never causes cache misses. Item/skill catalogs are typically
        small and re-compared often (e.g. every candidate against a
        user profile), so this cache meaningfully cuts repeat work."""
        a, b = set(s1), set(s2)
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    @classmethod
    def jaccard_similarity(cls, set1: Iterable, set2: Iterable) -> float:
        """Jaccard similarity (|A∩B| / |A∪B|) between two collections.
        Two empty sets are identical (1.0); one empty is fully dissimilar (0.0)."""
        key1 = tuple(sorted(str(x) for x in set1))
        key2 = tuple(sorted(str(x) for x in set2))
        return cls._jaccard_cached(key1, key2)

    # ------------------------------------------------------------------
    @staticmethod
    def pearson_correlation(ratings1: Sequence[float], ratings2: Sequence[float]) -> float:
        """Pearson correlation coefficient in [-1, 1] between two paired
        rating series. Raises InvalidInputError on length mismatch;
        returns 0.0 for empty input or zero-variance (constant) series,
        since those have no meaningful correlation but aren't caller errors."""
        if ratings1 and ratings2 and len(ratings1) != len(ratings2):
            raise InvalidInputError(
                f"pearson_correlation: length mismatch ({len(ratings1)} vs {len(ratings2)})"
            )
        if not ratings1 or not ratings2 or len(ratings1) < 2:
            return 0.0

        n = len(ratings1)
        mean1 = sum(ratings1) / n
        mean2 = sum(ratings2) / n

        cov = sum((a - mean1) * (b - mean2) for a, b in zip(ratings1, ratings2, strict=True))
        var1 = sum((a - mean1) ** 2 for a in ratings1)
        var2 = sum((b - mean2) ** 2 for b in ratings2)

        denom = math.sqrt(var1 * var2)
        if denom == 0:
            return 0.0

        return max(-1.0, min(1.0, cov / denom))

    # ------------------------------------------------------------------
    @staticmethod
    def adjusted_cosine_similarity(
        ratings1: Sequence[float], ratings2: Sequence[float], item_means: Sequence[float]
    ) -> float:
        """Adjusted cosine similarity: cosine similarity computed after
        subtracting each item's mean rating from both series.

        Standard cosine similarity treats a "generous" rater (who rates
        everything 4-5) and a "harsh" rater (who rates everything 1-2)
        as dissimilar even when their *relative* preferences agree.
        Adjusted cosine removes each item's average rating first, which
        is the standard fix used in item-based collaborative filtering
        (Sarwar et al., 2001) and gives a more honest similarity signal
        for skewed raters. Falls back to 0.0 on mismatched/empty input.
        """
        if not (len(ratings1) == len(ratings2) == len(item_means)) or not ratings1:
            return 0.0

        centered1 = [r - m for r, m in zip(ratings1, item_means, strict=True)]
        centered2 = [r - m for r, m in zip(ratings2, item_means, strict=True)]

        dot = sum(a * b for a, b in zip(centered1, centered2, strict=True))
        norm1 = math.sqrt(sum(a * a for a in centered1))
        norm2 = math.sqrt(sum(b * b for b in centered2))
        if norm1 == 0 or norm2 == 0:
            return 0.0

        return max(-1.0, min(1.0, dot / (norm1 * norm2)))
