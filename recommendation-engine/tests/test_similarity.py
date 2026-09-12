import pytest

from reco_engine.exceptions import InvalidInputError
from reco_engine.similarity import SimilarityCalculator

sc = SimilarityCalculator()


class TestCosineSimilarity:
    def test_identical_vectors(self):
        assert sc.cosine_similarity([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert sc.cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_zero_vector(self):
        assert sc.cosine_similarity([0, 0, 0], [1, 2, 3]) == 0.0

    def test_empty_vectors(self):
        assert sc.cosine_similarity([], []) == 0.0

    def test_mismatched_length_raises(self):
        with pytest.raises(InvalidInputError):
            sc.cosine_similarity([1, 2], [1, 2, 3])

    def test_scaled_vectors_are_maximally_similar(self):
        assert sc.cosine_similarity([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)


class TestJaccardSimilarity:
    def test_partial_overlap(self):
        result = sc.jaccard_similarity({"python", "sql", "aws"}, {"python", "sql", "gcp"})
        assert result == pytest.approx(0.5)

    def test_identical_sets(self):
        assert sc.jaccard_similarity({"a", "b"}, {"a", "b"}) == 1.0

    def test_disjoint_sets(self):
        assert sc.jaccard_similarity({"a"}, {"b"}) == 0.0

    def test_both_empty(self):
        assert sc.jaccard_similarity(set(), set()) == 1.0

    def test_one_empty(self):
        assert sc.jaccard_similarity({"a"}, set()) == 0.0


class TestPearsonCorrelation:
    def test_perfect_positive_correlation(self):
        assert sc.pearson_correlation([1, 2, 3, 4], [2, 4, 6, 8]) == pytest.approx(1.0)

    def test_perfect_negative_correlation(self):
        assert sc.pearson_correlation([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)

    def test_zero_variance_returns_zero(self):
        assert sc.pearson_correlation([3, 3, 3], [1, 2, 3]) == 0.0

    def test_empty_input(self):
        assert sc.pearson_correlation([], []) == 0.0

    def test_single_point(self):
        assert sc.pearson_correlation([1], [2]) == 0.0

    def test_mismatched_length_raises(self):
        with pytest.raises(InvalidInputError):
            sc.pearson_correlation([1, 2, 3], [1, 2])


class TestAdjustedCosineSimilarity:
    def test_removes_rater_bias(self):
        # Both users prefer item1 most and item3 least relative to the
        # community average (item_means) -- same relative taste -- even
        # though user2's ratings are compressed toward a lower range.
        ratings1 = [5, 4, 2]     # deviations from item_means: +2, +1, -1
        ratings2 = [4, 3.5, 2.5]  # deviations from item_means: +1, +0.5, -0.5
        item_means = [3.0, 3.0, 3.0]
        result = sc.adjusted_cosine_similarity(ratings1, ratings2, item_means)
        assert result > 0.5

    def test_empty_input(self):
        assert sc.adjusted_cosine_similarity([], [], []) == 0.0

    def test_mismatched_length_returns_zero(self):
        assert sc.adjusted_cosine_similarity([1, 2], [1], [1, 1]) == 0.0
