import pytest

from similarity import SimilarityCalculator


@pytest.fixture
def sc():
    return SimilarityCalculator()


class TestCosineSimilarity:
    def test_identical_vectors(self, sc):
        assert sc.cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self, sc):
        assert sc.cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_opposite_vectors(self, sc):
        assert sc.cosine_similarity([1, 2], [-1, -2]) == pytest.approx(-1.0)

    def test_empty_vector_returns_zero(self, sc):
        assert sc.cosine_similarity([], [1, 2]) == 0.0
        assert sc.cosine_similarity([1, 2], []) == 0.0

    def test_zero_vector_returns_zero(self, sc):
        assert sc.cosine_similarity([0, 0], [1, 2]) == 0.0

    def test_mismatched_length_raises(self, sc):
        with pytest.raises(ValueError):
            sc.cosine_similarity([1, 2], [1, 2, 3])


class TestJaccardSimilarity:
    def test_identical_sets(self, sc):
        assert sc.jaccard_similarity({"python", "sql"}, {"python", "sql"}) == 1.0

    def test_disjoint_sets(self, sc):
        assert sc.jaccard_similarity({"python"}, {"java"}) == 0.0

    def test_both_empty_returns_one(self, sc):
        assert sc.jaccard_similarity(set(), set()) == 1.0

    def test_one_empty_returns_zero(self, sc):
        assert sc.jaccard_similarity({"a"}, set()) == 0.0

    def test_partial_overlap(self, sc):
        result = sc.jaccard_similarity({"a", "b", "c"}, {"b", "c", "d"})
        assert result == pytest.approx(2 / 4)

    def test_accepts_lists_not_just_sets(self, sc):
        assert sc.jaccard_similarity(["a", "b"], ["b", "a"]) == 1.0


class TestPearsonCorrelation:
    def test_perfect_positive_correlation(self, sc):
        assert sc.pearson_correlation([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)

    def test_perfect_negative_correlation(self, sc):
        assert sc.pearson_correlation([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)

    def test_constant_series_returns_zero(self, sc):
        assert sc.pearson_correlation([1, 1, 1], [1, 2, 3]) == 0.0

    def test_empty_series_returns_zero(self, sc):
        assert sc.pearson_correlation([], []) == 0.0

    def test_single_point_returns_zero(self, sc):
        assert sc.pearson_correlation([1], [2]) == 0.0

    def test_mismatched_length_raises(self, sc):
        with pytest.raises(ValueError):
            sc.pearson_correlation([1, 2], [1, 2, 3])
