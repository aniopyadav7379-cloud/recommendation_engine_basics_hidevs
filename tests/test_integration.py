import importlib
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

    def test_cold_start_user_gets_honest_explanation(self, sample_repo):
        """A user with no history has no relevance signal at all, so the
        explanation must say so rather than inventing a personalized reason."""
        service = RecommendationService(sample_repo)
        results = service.get_recommendations("cold_user", k=5)
        assert results, "popularity fallback should still produce candidates"
        for item in results:
            assert "relevance" not in item.explanation

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


class TestProfileCacheInvalidation:
    """The service caches each user's tag profile to avoid rebuilding it
    for every candidate scored in a request. That cache must never serve
    a stale profile once a new interaction has actually changed the
    user's history."""

    def test_profile_cache_reflects_new_interaction(self):
        repo = InMemoryRepository(
            item_features={"item_a": {"python"}, "item_b": {"design"}},
        )
        service = RecommendationService(repo)

        # u1 starts with no history -> empty (cached) profile.
        assert service._get_user_profile("u1") == frozenset()

        service.record_interaction("u1", "item_a")
        # Cache must have been invalidated: profile now reflects item_a's tags.
        assert service._get_user_profile("u1") == frozenset({"python"})

    def test_cache_is_populated_on_first_access(self, sample_repo):
        service = RecommendationService(sample_repo)
        assert "u1" not in service._profile_cache
        service._get_user_profile("u1")
        assert "u1" in service._profile_cache

    def test_other_users_cache_is_untouched_by_unrelated_interaction(self, sample_repo):
        service = RecommendationService(sample_repo)
        service._get_user_profile("u1")
        service._get_user_profile("u2")
        service.record_interaction("u3", "item_x")
        # u1/u2 profiles weren't invalidated by an interaction for u3.
        assert "u1" in service._profile_cache
        assert "u2" in service._profile_cache

    def test_popularity_candidate_cache_is_invalidated_on_interaction(self, sample_repo):
        generator = RecommendationService(sample_repo).generator
        first = generator.popularity_candidates()
        assert generator._popularity_cache is not None
        generator.invalidate_cache()
        assert generator._popularity_cache is None
        # Recomputing after invalidation still produces a valid (possibly
        # identical) ranking rather than erroring.
        second = generator.popularity_candidates()
        assert isinstance(second, list)
        assert first  # sanity: fixture data does produce a ranking


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


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("RECO_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    from reco_engine import api as api_module
    importlib.reload(api_module)
    return TestClient(api_module.app)


class TestHealthEndpoints:
    def test_health_back_compat(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_health_live(self, client):
        response = client.get("/health/live")
        assert response.status_code == 200
        assert response.json() == {"status": "alive"}

    def test_health_ready_when_storage_is_reachable(self, client):
        response = client.get("/health/ready")
        assert response.status_code == 200
        assert response.json() == {"status": "ready"}

    def test_health_ready_reports_503_when_storage_fails(self, client, monkeypatch):
        from reco_engine import api as api_module

        def broken_history():
            raise OSError("disk unavailable")

        monkeypatch.setattr(api_module._repo, "get_all_user_history", broken_history)
        response = client.get("/health/ready")
        assert response.status_code == 503
        body = response.json()
        assert body["error"]["type"] == "not_ready"


class TestRecommendationsEndpoint:
    def test_record_then_get_recommendations(self, client):
        record_response = client.post(
            "/interactions", json={"user_id": "u1", "item_id": "job_1"}
        )
        assert record_response.status_code == 204

        rec_response = client.get("/recommendations/u1")
        assert rec_response.status_code == 200
        assert isinstance(rec_response.json(), list)

    def test_invalid_k_zero_returns_422(self, client):
        response = client.get("/recommendations/u1?k=0")
        assert response.status_code == 422
        body = response.json()
        assert body["error"]["type"] == "validation_error"

    def test_invalid_k_negative_returns_422(self, client):
        response = client.get("/recommendations/u1?k=-5")
        assert response.status_code == 422

    def test_k_above_configured_maximum_returns_422(self, client):
        max_k = get_settings().max_k
        response = client.get(f"/recommendations/u1?k={max_k + 1}")
        assert response.status_code == 422

    def test_k_at_configured_maximum_is_accepted(self, client):
        max_k = get_settings().max_k
        response = client.get(f"/recommendations/u1?k={max_k}")
        assert response.status_code == 200

    def test_invalid_user_id_with_bad_characters_returns_422(self, client):
        response = client.get("/recommendations/../etc")
        # Path traversal-shaped ids are rejected by the id pattern.
        assert response.status_code in (404, 422)

    def test_invalid_user_id_with_whitespace_returns_422(self, client):
        response = client.get("/recommendations/has space")
        assert response.status_code == 422

    def test_valid_user_id_with_dots_and_dashes_is_accepted(self, client):
        response = client.get("/recommendations/user.name-1")
        assert response.status_code == 200


class TestInteractionsEndpointValidation:
    def test_invalid_user_id_in_body_returns_422(self, client):
        response = client.post("/interactions", json={"user_id": "bad user!", "item_id": "job_1"})
        assert response.status_code == 422
        assert response.json()["error"]["type"] == "validation_error"

    def test_invalid_item_id_in_body_returns_422(self, client):
        response = client.post("/interactions", json={"user_id": "u1", "item_id": "bad item!"})
        assert response.status_code == 422

    def test_missing_field_returns_422(self, client):
        response = client.post("/interactions", json={"user_id": "u1"})
        assert response.status_code == 422

    def test_oversized_id_returns_422(self, client):
        response = client.post("/interactions", json={"user_id": "u" * 100, "item_id": "job_1"})
        assert response.status_code == 422

    def test_valid_interaction_updates_future_recommendations(self, client):
        before = client.get("/recommendations/u1").json()

        record_response = client.post(
            "/interactions", json={"user_id": "u1", "item_id": "job_2"}
        )
        assert record_response.status_code == 204

        after = client.get("/recommendations/u1").json()
        # Not a strict inequality requirement (candidate pools can
        # legitimately coincide), just confirms the round trip works and
        # the newly-interacted item is now excluded from candidates.
        assert "job_2" not in [item["item_id"] for item in after]
        assert isinstance(before, list)
        assert isinstance(after, list)


class TestErrorEnvelope:
    def test_error_responses_never_include_a_traceback(self, client, monkeypatch):
        from reco_engine import api as api_module

        def boom(*args, **kwargs):
            raise RuntimeError("something exploded internally")

        monkeypatch.setattr(api_module._service, "get_recommendations", boom)
        # raise_server_exceptions=False: exercise the actual HTTP response
        # our exception handler sends, rather than TestClient's default
        # debug behavior of re-raising the exception into the test.
        no_raise_client = TestClient(api_module.app, raise_server_exceptions=False)
        response = no_raise_client.get("/recommendations/u1")
        assert response.status_code == 500
        body = response.json()
        assert body == {"error": {"type": "internal_error", "message": "An internal error occurred."}}
        assert "Traceback" not in response.text
        assert "RuntimeError" not in response.text

    def test_validation_error_envelope_shape(self, client):
        response = client.get("/recommendations/u1?k=0")
        body = response.json()
        assert set(body.keys()) == {"error"}
        assert set(body["error"].keys()) == {"type", "message"}
