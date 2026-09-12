import pytest

from reco_engine.candidate_gen import CandidateGenerator
from reco_engine.exceptions import InvalidInputError


class TestCandidateGenerator:
    def test_invalid_max_candidates_raises(self, sample_repo):
        with pytest.raises(InvalidInputError):
            CandidateGenerator(sample_repo, max_candidates=0)

    def test_collaborative_excludes_own_history(self, sample_repo):
        gen = CandidateGenerator(sample_repo, max_candidates=10)
        result = gen.collaborative_candidates("u1")
        assert not (set(result) & sample_repo.get_user_history("u1"))

    def test_collaborative_finds_similar_users_items(self, sample_repo):
        gen = CandidateGenerator(sample_repo, max_candidates=10)
        result = gen.collaborative_candidates("u1")
        # u2 shares item_a/item_b with u1 and additionally has item_d
        assert "item_d" in result

    def test_collaborative_cold_start_falls_back_to_popularity(self, sample_repo):
        gen = CandidateGenerator(sample_repo, max_candidates=10)
        result = gen.collaborative_candidates("cold_user")
        assert result == gen.popularity_candidates()

    def test_content_based_excludes_own_history(self, sample_repo):
        gen = CandidateGenerator(sample_repo, max_candidates=10)
        result = gen.content_based_candidates("u1")
        assert not (set(result) & sample_repo.get_user_history("u1"))

    def test_popularity_returns_ranked_items(self, sample_repo):
        gen = CandidateGenerator(sample_repo, max_candidates=10)
        result = gen.popularity_candidates()
        assert len(result) > 0
        assert result[0] in {"item_a", "item_b"}  # both interacted with twice

    def test_popularity_empty_history_returns_empty(self):
        from reco_engine.persistence import InMemoryRepository
        gen = CandidateGenerator(InMemoryRepository(), max_candidates=10)
        assert gen.popularity_candidates() == []

    def test_hybrid_excludes_already_seen_items(self, sample_repo):
        gen = CandidateGenerator(sample_repo, max_candidates=10)
        result = gen.hybrid_candidates("u1")
        assert not (set(result) & sample_repo.get_user_history("u1"))

    def test_hybrid_respects_max_candidates_cap(self, sample_repo):
        gen = CandidateGenerator(sample_repo, max_candidates=2)
        result = gen.hybrid_candidates("u1")
        assert len(result) <= 2

    def test_hybrid_cold_start_does_not_error(self, sample_repo):
        gen = CandidateGenerator(sample_repo, max_candidates=10)
        result = gen.hybrid_candidates("cold_user")
        assert isinstance(result, list)

    def test_hybrid_unknown_user_does_not_error(self, sample_repo):
        gen = CandidateGenerator(sample_repo, max_candidates=10)
        result = gen.hybrid_candidates("does_not_exist")
        assert isinstance(result, list)
