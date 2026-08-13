import pytest

from reco_engine.exceptions import InvalidInputError
from reco_engine.scorer import RecommendationScorer


@pytest.fixture
def scorer_with_signals():
    relevance = {"item_a": 0.9, "item_b": 0.4}
    popularity = {"item_a": 0.6, "item_b": 0.6}

    scorer = RecommendationScorer()
    scorer.add_scorer("relevance", lambda u, i, ctx: relevance.get(i, 0.0), weight=0.7)
    scorer.add_scorer("popularity", lambda u, i, ctx: popularity.get(i, 0.0), weight=0.3)
    return scorer


class TestRecommendationScorer:
    def test_negative_weight_raises(self):
        scorer = RecommendationScorer()
        with pytest.raises(InvalidInputError):
            scorer.add_scorer("bad", lambda u, i, ctx: 1.0, weight=-1)

    def test_non_callable_raises(self):
        scorer = RecommendationScorer()
        with pytest.raises(InvalidInputError):
            scorer.add_scorer("bad", "not callable", weight=1.0)

    def test_score_in_bounds(self, scorer_with_signals):
        score, explanation = scorer_with_signals.calculate_score("u1", "item_a")
        assert 0.0 <= score <= 1.0
        assert explanation

    def test_no_scorers_returns_zero(self):
        scorer = RecommendationScorer()
        score, explanation = scorer.calculate_score("u1", "item_a")
        assert score == 0.0

    def test_failing_scorer_is_skipped_not_fatal(self):
        def broken(u, i, ctx):
            raise RuntimeError("boom")

        scorer = RecommendationScorer()
        scorer.add_scorer("broken", broken, weight=1.0)
        scorer.add_scorer("ok", lambda u, i, ctx: 0.5, weight=1.0)
        score, _ = scorer.calculate_score("u1", "item_a")
        assert score == pytest.approx(0.5)

    def test_rank_candidates_deduplicates(self, scorer_with_signals):
        ranked = scorer_with_signals.rank_candidates("u1", ["item_a", "item_b", "item_a"], limit=5)
        assert len(ranked) == 2

    def test_rank_candidates_sorts_descending(self, scorer_with_signals):
        ranked = scorer_with_signals.rank_candidates("u1", ["item_a", "item_b"], limit=5)
        assert ranked[0].score >= ranked[1].score
        assert ranked[0].item_id == "item_a"  # higher relevance + popularity

    def test_rank_candidates_respects_limit(self, scorer_with_signals):
        ranked = scorer_with_signals.rank_candidates("u1", ["item_a", "item_b"], limit=1)
        assert len(ranked) == 1

    def test_rank_candidates_invalid_limit_raises(self, scorer_with_signals):
        with pytest.raises(InvalidInputError):
            scorer_with_signals.rank_candidates("u1", ["item_a"], limit=0)
