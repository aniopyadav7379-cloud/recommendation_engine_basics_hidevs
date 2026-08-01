import pytest

from scorer import RecommendationScorer


@pytest.fixture
def scorer():
    return RecommendationScorer()


@pytest.fixture
def context():
    return {
        "user_tags": {"python", "backend"},
        "item_tags": {
            "i1": {"python", "backend"},
            "i2": {"design"},
            "i3": {"python", "ml"},
        },
        "item_popularity": {"i1": 50, "i2": 10, "i3": 30},
        "new_items": {"i3"},
    }


def relevance_scorer(user_id, item_id, context):
    user_tags = context.get("user_tags", set())
    item_tags = context.get("item_tags", {}).get(item_id, set())
    if not user_tags or not item_tags:
        return 0.0
    overlap = user_tags & item_tags
    return len(overlap) / len(user_tags | item_tags)


def popularity_scorer(user_id, item_id, context):
    pop = context.get("item_popularity", {})
    if not pop:
        return 0.0
    max_pop = max(pop.values()) or 1
    return pop.get(item_id, 0) / max_pop


def broken_scorer(user_id, item_id, context):
    raise RuntimeError("intentionally broken scorer")


class TestAddScorer:
    def test_rejects_non_callable(self, scorer):
        with pytest.raises(TypeError):
            scorer.add_scorer("bad", "not a function")

    def test_rejects_negative_weight(self, scorer):
        with pytest.raises(ValueError):
            scorer.add_scorer("bad", relevance_scorer, weight=-1)


class TestCalculateScore:
    def test_no_scorers_registered_returns_zero(self, scorer):
        result = scorer.calculate_score("u1", "i1")
        assert result["score"] == 0.0
        assert result["breakdown"] == {}

    def test_score_within_bounds(self, scorer, context):
        scorer.add_scorer("relevance", relevance_scorer, weight=2.0)
        scorer.add_scorer("popularity", popularity_scorer, weight=1.0)
        result = scorer.calculate_score("u1", "i1", context)
        assert 0.0 <= result["score"] <= 1.0
        assert "relevance" in result["breakdown"]
        assert "popularity" in result["breakdown"]

    def test_broken_scorer_is_skipped_not_fatal(self, scorer, context):
        scorer.add_scorer("relevance", relevance_scorer, weight=1.0)
        scorer.add_scorer("broken", broken_scorer, weight=1.0)
        result = scorer.calculate_score("u1", "i1", context)
        assert "broken" not in result["breakdown"]
        assert "relevance" in result["breakdown"]

    def test_explanation_names_top_factor(self, scorer, context):
        scorer.add_scorer("relevance", relevance_scorer, weight=2.0)
        scorer.add_scorer("popularity", popularity_scorer, weight=1.0)
        result = scorer.calculate_score("u1", "i1", context)
        assert "relevance" in result["explanation"] or "popularity" in result["explanation"]


class TestRankCandidates:
    def test_ranks_highest_score_first(self, scorer, context):
        scorer.add_scorer("relevance", relevance_scorer, weight=2.0)
        scorer.add_scorer("popularity", popularity_scorer, weight=1.0)
        ranked = scorer.rank_candidates("u1", ["i1", "i2", "i3"], context, limit=3)
        assert ranked[0]["item_id"] == "i1"

    def test_respects_limit(self, scorer, context):
        scorer.add_scorer("relevance", relevance_scorer, weight=1.0)
        ranked = scorer.rank_candidates("u1", ["i1", "i2", "i3"], context, limit=2)
        assert len(ranked) == 2

    def test_empty_candidates_returns_empty(self, scorer):
        assert scorer.rank_candidates("u1", []) == []
