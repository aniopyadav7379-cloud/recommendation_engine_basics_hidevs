import pytest

from reco_engine.evaluator import RecommendationEvaluator
from reco_engine.exceptions import InvalidInputError

ev = RecommendationEvaluator()

RECS = ["item_a", "item_b", "item_c", "item_d", "item_e"]
RELEVANT = ["item_b", "item_d", "item_z"]  # item_z never recommended


class TestPrecisionAtK:
    def test_basic(self):
        assert ev.precision_at_k(RECS, RELEVANT, 5) == pytest.approx(2 / 5)

    def test_empty_recommendations(self):
        assert ev.precision_at_k([], RELEVANT, 5) == 0.0

    def test_invalid_k_raises(self):
        with pytest.raises(InvalidInputError):
            ev.precision_at_k(RECS, RELEVANT, 0)


class TestRecallAtK:
    def test_basic(self):
        assert ev.recall_at_k(RECS, RELEVANT, 5) == pytest.approx(2 / 3)

    def test_no_ground_truth(self):
        assert ev.recall_at_k(RECS, [], 5) == 0.0


class TestNdcgAtK:
    def test_perfect_ranking_scores_one(self):
        assert ev.ndcg_at_k(["item_b", "item_d"], ["item_b", "item_d"], 5) == pytest.approx(1.0)

    def test_no_relevant_items_found(self):
        assert ev.ndcg_at_k(["item_x", "item_y"], ["item_b"], 5) == 0.0

    def test_position_matters(self):
        # relevant item first should score higher than relevant item last
        high = ev.ndcg_at_k(["item_b", "item_x"], ["item_b"], 2)
        low = ev.ndcg_at_k(["item_x", "item_b"], ["item_b"], 2)
        assert high > low


class TestEvaluateAll:
    def test_skips_users_without_ground_truth(self):
        recommendations_dict = {
            "u1": ["item_a", "item_b", "item_c"],
            "u2": ["item_x", "item_y", "item_z"],
            "u3": ["item_p", "item_q"],  # no ground truth
        }
        ground_truth_dict = {
            "u1": ["item_b"],
            "u2": ["item_x", "item_z"],
        }
        metrics = ev.evaluate_all(recommendations_dict, ground_truth_dict, k=3)
        assert metrics["num_users_evaluated"] == 2

    def test_no_ground_truth_at_all_returns_zeroed_metrics(self):
        metrics = ev.evaluate_all({"u1": ["item_a"]}, {}, k=3)
        assert metrics == {
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "ndcg_at_k": 0.0,
            "num_users_evaluated": 0,
        }
