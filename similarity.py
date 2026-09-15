"""
similarity.py
-------------
Component 1: Similarity Calculator
====================================
This module measures how similar two users, items, or skills are.
Similarity is the foundation of any recommendation engine.

"Users like you also liked..." → needs similarity between users
"Items similar to what you liked..." → needs similarity between items

Three methods are provided:
    1. cosine_similarity   → for comparing user/item feature vectors
    2. jaccard_similarity  → for comparing skill sets or tags
    3. pearson_correlation → for comparing rating patterns

Author  : Day 29 - HiDevs Internship
Project : Recommendation Engine Core Components
"""

import math


class SimilarityCalculator:
    """
    Calculates similarity scores between vectors, sets, and rating lists.
    All methods handle edge cases gracefully and return safe default values.
    """

    def cosine_similarity(self, vec1, vec2):
        """
        Cosine Similarity — measures the angle between two vectors.
        The closer the angle to 0 degrees, the more similar the vectors are.

        Formula:
            cosine_sim = (vec1 . vec2) / (|vec1| x |vec2|)

        Args:
            vec1 (list): First numeric vector (e.g., user feature vector)
            vec2 (list): Second numeric vector (e.g., item feature vector)

        Returns:
            float: Similarity score between 0.0 (no similarity) and 1.0 (identical)

        Edge Cases Handled:
            - Empty vectors       → returns 0.0
            - Mismatched lengths  → returns 0.0
            - Zero vectors [0,0]  → returns 0.0 (avoids division by zero)

        Example:
            sim.cosine_similarity([1, 0, 1], [1, 0, 1]) → 1.0  (identical)
            sim.cosine_similarity([1, 0], [0, 1])       → 0.0  (orthogonal)
        """
        # Edge case: empty input
        if not vec1 or not vec2:
            return 0.0

        # Edge case: vectors must be same length to compare
        if len(vec1) != len(vec2):
            return 0.0

        # Step 1: Calculate dot product (numerator)
        dot_product = sum(a * b for a, b in zip(vec1, vec2))

        # Step 2: Calculate magnitudes of each vector
        magnitude1 = math.sqrt(sum(a ** 2 for a in vec1))
        magnitude2 = math.sqrt(sum(b ** 2 for b in vec2))

        # Edge case: zero vector has no direction, similarity is undefined
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        # Step 3: Divide dot product by product of magnitudes
        return dot_product / (magnitude1 * magnitude2)

    def jaccard_similarity(self, set1, set2):
        """
        Jaccard Similarity — measures overlap between two sets.
        Used when comparing tags, categories, skill sets, genres etc.

        Formula:
            jaccard_sim = |Intersection| / |Union|

        Args:
            set1 (set or list): First set of items (e.g., user's skills)
            set2 (set or list): Second set of items (e.g., item's tags)

        Returns:
            float: Similarity score between 0.0 (no overlap) and 1.0 (identical sets)

        Edge Cases Handled:
            - Both empty → returns 1.0 (two empty sets are considered identical)
            - One empty  → returns 0.0 (nothing in common)
            - No overlap → returns 0.0

        Example:
            sim.jaccard_similarity({"python","ml"}, {"python","ml"}) → 1.0
            sim.jaccard_similarity({"a","b","c"}, {"b","c","d"})     → 0.5
        """
        # Edge case: both empty sets are identical
        if not set1 and not set2:
            return 1.0

        # Edge case: one empty set means zero overlap
        if not set1 or not set2:
            return 0.0

        # Convert to sets to remove duplicates and enable set operations
        set1 = set(set1)
        set2 = set(set2)

        # Step 1: Count shared elements (intersection)
        intersection = len(set1 & set2)

        # Step 2: Count all unique elements (union)
        union = len(set1 | set2)

        # Edge case: avoid division by zero
        if union == 0:
            return 0.0

        # Step 3: Ratio of shared to total
        return intersection / union

    def pearson_correlation(self, ratings1, ratings2):
        """
        Pearson Correlation — measures how similarly two users rate items.
        Unlike cosine similarity, this accounts for rating bias
        (e.g., a user who always rates high vs. one who rates low).

        Formula:
            pearson = sum[(r1-mean1)(r2-mean2)] / (std1 x std2)

        Args:
            ratings1 (list): First user's ratings  (e.g., [4, 3, 5, 2])
            ratings2 (list): Second user's ratings (e.g., [5, 3, 4, 1])

        Returns:
            float: Correlation between -1.0 (opposite) and 1.0 (identical pattern)
                   0.0 means no correlation

        Edge Cases Handled:
            - Empty lists        → returns 0.0
            - Mismatched lengths → returns 0.0
            - Less than 2 items  → returns 0.0 (cannot compute correlation)
            - Constant ratings   → returns 0.0 (no variance, avoids division by zero)

        Example:
            sim.pearson_correlation([1,2,3], [1,2,3]) →  1.0  (perfect match)
            sim.pearson_correlation([1,2,3], [3,2,1]) → -1.0  (exact opposite)
        """
        # Edge case: empty lists
        if not ratings1 or not ratings2:
            return 0.0

        # Edge case: must have same number of ratings to compare
        if len(ratings1) != len(ratings2):
            return 0.0

        n = len(ratings1)

        # Edge case: need at least 2 points to compute correlation
        if n < 2:
            return 0.0

        # Step 1: Compute means to center the data
        mean1 = sum(ratings1) / n
        mean2 = sum(ratings2) / n

        # Step 2: Compute numerator — covariance between the two rating lists
        numerator = sum((r1 - mean1) * (r2 - mean2) for r1, r2 in zip(ratings1, ratings2))

        # Step 3: Compute denominators — standard deviations
        denom1 = math.sqrt(sum((r - mean1) ** 2 for r in ratings1))
        denom2 = math.sqrt(sum((r - mean2) ** 2 for r in ratings2))

        # Edge case: constant ratings have zero std deviation — avoid division by zero
        if denom1 == 0 or denom2 == 0:
            return 0.0

        # Step 4: Normalize the covariance by standard deviations
        return numerator / (denom1 * denom2)