import json

import pytest
from fastapi.testclient import TestClient

from reco_engine.config import get_settings
from reco_engine.persistence import InMemoryRepository, JSONFileRepository
from reco_engine.service import RecommendationService


class TestServiceEndToEnd:
    def test_recommendations_are_personalized_per_user(self, sample_repo):
        service = RecommendationService(sample_repo)
        alice_like = service.get_recommendations("u1", k=5)
        stranger_like = service.get_recommendations("u3", k=5)
        # u1 and u3 have disjoint history/interests -> different top picks
        assert [r.item_id for r in alice_like] != [r.item_id for r in stranger_like]

    def test_cold_start_user_gets_results_not_error(self, sample_repo):
        service = RecommendationService(sample_repo)
        results = service.get_recommendations("cold_user", k=5)
        assert isinstance(results, list)

    def test_record_interaction_affects_future_recommendations(self):
        repo = InMemoryRepository(
            user_history={"u1": {"item_a"}, "u2": {"item_a", "item_b"}},
            item_features={"item_a": {"python"}, "item_b": {"python", "sql"}},
        )
        service = RecommendationService(repo)
        before = repo.get_user_history("u1")
        service.record_interaction("u1", "item_b")
        after = repo.get_user_history("u1")
        assert "item_b" not in before
        assert "item_b" in after

    def test_evaluate_runs_full_pipeline_and_scores_it(self, sample_repo):
        service = RecommendationService(sample_repo)
        metrics = service.evaluate(
            user_ids=["u1", "u2"],
            ground_truth={"u1": ["item_d"]},
            k=5,
        )
        assert metrics["num_users_evaluated"] == 1
        assert 0.0 <= metrics["precision_at_k"] <= 1.0


class TestJSONFileRepositoryPersistence:
    def test_write_then_reload_roundtrip(self, tmp_path):
        repo1 = JSONFileRepository(str(tmp_path), "history.json", "features.json")
        repo1.record_interaction("newuser", "item_1")

        # Simulate a fresh process reading the same files back
        repo2 = JSONFileRepository(str(tmp_path), "history.json", "features.json")
        assert repo2.get_user_history("newuser") == {"item_1"}

    def test_write_is_valid_json_on_disk(self, tmp_path):
        repo = JSONFileRepository(str(tmp_path), "history.json", "features.json")
        repo.record_interaction("u1", "item_a")
        with open(tmp_path / "history.json") as f:
            data = json.load(f)
        assert data["u1"] == ["item_a"]

    def test_missing_files_start_empty_without_error(self, tmp_path):
        repo = JSONFileRepository(str(tmp_path / "fresh"), "history.json", "features.json")
        assert repo.get_all_user_history() == {}
        assert repo.get_all_item_features() == {}


class TestAPI:
    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RECO_DATA_DIR", str(tmp_path))
        get_settings.cache_clear()
        from reco_engine import api as api_module
        import importlib
        importlib.reload(api_module)
        return TestClient(api_module.app)

    def test_health(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_record_then_get_recommendations(self, client):
        record_response = client.post(
            "/interactions", json={"user_id": "u1", "item_id": "job_1"}
        )
        assert record_response.status_code == 204

        rec_response = client.get("/recommendations/u1")
        assert rec_response.status_code == 200
        assert isinstance(rec_response.json(), list)

    def test_invalid_k_returns_400(self, client):
        response = client.get("/recommendations/u1?k=0")
        assert response.status_code == 400
