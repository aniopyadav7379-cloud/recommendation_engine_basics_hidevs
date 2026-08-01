import pytest

from evaluator import RecommendationEvaluator


@pytest.fixture
def ev():
    return RecommendationEvaluator()


@pytest.fixture
def recs():
    return ["i1", "i2", "i3", "i4", "i5"]


@pytest.fixture
def relevant():
    return ["i2", "i4", "i9"]  # i9 is never recommended


class TestPrecisionAtK:
    def test_computes_correct_fraction(self, ev, recs, relevant):
        assert ev.precision_at_k(recs, relevant, k=5) == pytest.approx(2 / 5)

    def test_empty_recommendations_returns_zero(self, ev, relevant):
        assert ev.precision_at_k([], relevant, k=5) == 0.0

    def test_zero_k_returns_zero(self, ev, recs, relevant):
        assert ev.precision_at_k(recs, relevant, k=0) == 0.0


class TestRecallAtK:
    def test_computes_correct_fraction(self, ev, recs, relevant):
        assert ev.recall_at_k(recs, relevant, k=5) == pytest.approx(2 / 3)

    def test_no_ground_truth_returns_zero(self, ev, recs):
        assert ev.recall_at_k(recs, [], k=5) == 0.0

    def test_perfect_recall(self, ev):
        assert ev.recall_at_k(["a", "b", "c"], ["a", "b"], k=3) == pytest.approx(1.0)


class TestNdcgAtK:
    def test_within_bounds(self, ev, recs, relevant):
        score = ev.ndcg_at_k(recs, relevant, k=5)
        assert 0.0 < score < 1.0

    def test_perfect_ranking_scores_one(self, ev):
        perfect_recs = ["i2", "i4", "i1", "i3", "i5"]
        perfect_relevant = ["i2", "i4"]
        assert ev.ndcg_at_k(perfect_recs, perfect_relevant, k=5) == pytest.approx(1.0)

    def test_worse_ranking_scores_lower_than_perfect(self, ev):
        perfect = ev.ndcg_at_k(["a", "b"], ["a", "b"], k=2)
        worse = ev.ndcg_at_k(["b", "a"], ["a", "b"], k=2)
        # order doesn't matter here since both are relevant either way,
        # so instead compare a case where only the lower rank is relevant
        worse_case = ev.ndcg_at_k(["x", "a"], ["a"], k=2)
        best_case = ev.ndcg_at_k(["a", "x"], ["a"], k=2)
        assert worse_case < best_case

    def test_no_ground_truth_returns_zero(self, ev, recs):
        assert ev.ndcg_at_k(recs, [], k=5) == 0.0


class TestEvaluateAll:
    def test_averages_across_users(self, ev):
        recommendations_dict = {
            "u1": ["i1", "i2", "i3"],
            "u2": ["i4", "i5", "i6"],
        }
        ground_truth_dict = {
            "u1": ["i2"],
            "u2": ["i4", "i5"],
        }
        summary = ev.evaluate_all(recommendations_dict, ground_truth_dict, k=3)
        assert summary["users_evaluated"] == 2
        assert 0.0 <= summary["precision_at_k"] <= 1.0
        assert 0.0 <= summary["recall_at_k"] <= 1.0
        assert 0.0 <= summary["ndcg_at_k"] <= 1.0

    def test_skips_users_missing_ground_truth(self, ev):
        recommendations_dict = {"u1": ["i1"], "u2": ["i2"]}
        ground_truth_dict = {"u1": ["i1"]}
        summary = ev.evaluate_all(recommendations_dict, ground_truth_dict, k=3)
        assert summary["users_evaluated"] == 1

    def test_empty_inputs_return_zeroed_summary(self, ev):
        summary = ev.evaluate_all({}, {}, k=5)
        assert summary["users_evaluated"] == 0
        assert summary["precision_at_k"] == 0.0
