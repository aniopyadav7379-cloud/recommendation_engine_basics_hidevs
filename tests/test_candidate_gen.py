import pytest

from candidate_gen import CandidateGenerator


@pytest.fixture
def history():
    return {
        "alice": ["book1", "book2"],
        "bob": ["book1", "book2", "book3"],
        "carol": ["book4"],
    }


@pytest.fixture
def tags():
    return {
        "book1": {"scifi"},
        "book2": {"scifi", "adventure"},
        "book3": {"scifi", "mystery"},
        "book4": {"romance"},
    }


@pytest.fixture
def popularity():
    return {"book1": 100, "book2": 80, "book3": 40, "book4": 10}


@pytest.fixture
def gen(history, tags, popularity):
    return CandidateGenerator(history, tags, popularity)


class TestCollaborativeCandidates:
    def test_finds_items_from_similar_users(self, gen):
        # bob shares alice's whole history plus book3 -> book3 should surface
        assert "book3" in gen.collaborative_candidates("alice")

    def test_excludes_items_user_already_has(self, gen):
        assert "book1" not in gen.collaborative_candidates("alice")

    def test_cold_start_falls_back_to_popularity(self, gen):
        assert gen.collaborative_candidates("unknown_user") == gen.popularity_candidates()


class TestContentBasedCandidates:
    def test_finds_items_with_overlapping_tags(self, gen):
        assert "book3" in gen.content_based_candidates("alice")

    def test_cold_start_falls_back_to_popularity(self, gen):
        assert gen.content_based_candidates("unknown_user") == gen.popularity_candidates()

    def test_no_history_falls_back(self, gen):
        assert gen.content_based_candidates("dave") == gen.popularity_candidates()


class TestPopularityCandidates:
    def test_ranks_by_popularity_descending(self, gen):
        assert gen.popularity_candidates(3) == ["book1", "book2", "book3"]

    def test_respects_limit(self, gen):
        assert len(gen.popularity_candidates(2)) == 2

    def test_limit_capped_at_max(self, gen):
        assert len(gen.popularity_candidates(9999)) <= CandidateGenerator.MAX_LIMIT


class TestHybridCandidates:
    def test_returns_nonempty_for_active_user(self, gen):
        assert len(gen.hybrid_candidates("alice")) > 0

    def test_no_duplicate_items(self, gen):
        candidates = gen.hybrid_candidates("alice")
        assert len(candidates) == len(set(candidates))

    def test_cold_start_still_returns_results(self, gen):
        assert len(gen.hybrid_candidates("unknown_user")) > 0


class TestEmptyGenerator:
    def test_empty_data_returns_empty_lists(self):
        empty_gen = CandidateGenerator()
        assert empty_gen.popularity_candidates() == []
        assert empty_gen.collaborative_candidates("anyone") == []
        assert empty_gen.content_based_candidates("anyone") == []
