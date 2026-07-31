"""
test.py — Day 29 test suite

Runs the inline tests inside each module, then a small integration
test that chains all four components together the way the full
system will tomorrow:

    candidates -> scored/ranked -> evaluated against ground truth
"""

from similarity import SimilarityCalculator
from candidate_gen import CandidateGenerator
from scorer import RecommendationScorer
from evaluator import RecommendationEvaluator


def test_similarity():
    sc = SimilarityCalculator()
    assert round(sc.cosine_similarity([1, 2, 3], [1, 2, 3]), 4) == 1.0
    assert sc.cosine_similarity([1, 0], [0, 1]) == 0.0
    assert sc.jaccard_similarity({"a", "b"}, {"a", "b"}) == 1.0
    assert round(sc.pearson_correlation([1, 2, 3, 4], [2, 4, 6, 8]), 4) == 1.0
    print("[PASS] similarity")


def test_candidate_gen():
    history = {
        "alice": ["book1", "book2"],
        "bob": ["book1", "book2", "book3"],
        "carol": ["book4"],
    }
    tags = {
        "book1": {"scifi"},
        "book2": {"scifi", "adventure"},
        "book3": {"scifi", "mystery"},
        "book4": {"romance"},
    }
    popularity = {"book1": 100, "book2": 80, "book3": 40, "book4": 10}

    gen = CandidateGenerator(history, tags, popularity)

    assert "book3" in gen.collaborative_candidates("alice")
    assert "book3" in gen.content_based_candidates("alice")
    assert gen.popularity_candidates(2) == ["book1", "book2"]
    assert len(gen.hybrid_candidates("alice")) > 0
    # cold start: unknown user falls back to popularity
    assert gen.collaborative_candidates("dave") == gen.popularity_candidates()
    print("[PASS] candidate_gen")


def test_scorer():
    scorer = RecommendationScorer()
    scorer.add_scorer("popularity", lambda u, i, ctx: ctx["pop"].get(i, 0) / 100, weight=1.0)
    scorer.add_scorer("match", lambda u, i, ctx: 1.0 if i in ctx["preferred"] else 0.0, weight=2.0)

    context = {"pop": {"book1": 100, "book2": 50}, "preferred": {"book2"}}
    ranked = scorer.rank_candidates("alice", ["book1", "book2"], context, limit=2)

    assert ranked[0]["item_id"] == "book2"  # wins on weighted "match"
    assert "explanation" in ranked[0]
    print("[PASS] scorer")


def test_evaluator():
    ev = RecommendationEvaluator()
    recs = ["a", "b", "c", "d"]
    relevant = ["b", "d"]

    assert round(ev.precision_at_k(recs, relevant, 4), 4) == 0.5
    assert round(ev.recall_at_k(recs, relevant, 4), 4) == 1.0
    assert 0.0 < ev.ndcg_at_k(recs, relevant, 4) <= 1.0

    summary = ev.evaluate_all({"u1": recs}, {"u1": relevant}, k=4)
    assert summary["users_evaluated"] == 1
    print("[PASS] evaluator")


def test_integration():
    """Chain all four components together end to end, the way the
    full recommendation engine will do tomorrow.
    """
    history = {
        "alice": ["book1", "book2"],
        "bob": ["book1", "book2", "book3"],
    }
    tags = {
        "book1": {"scifi"},
        "book2": {"scifi", "adventure"},
        "book3": {"scifi", "mystery"},
    }
    popularity = {"book1": 100, "book2": 80, "book3": 40}

    # 1. Generate candidates
    gen = CandidateGenerator(history, tags, popularity)
    candidates = gen.hybrid_candidates("alice", limit=10)
    assert len(candidates) > 0

    # 2. Score and rank them
    scorer = RecommendationScorer()
    scorer.add_scorer(
        "popularity",
        lambda u, i, ctx: ctx["item_popularity"].get(i, 0) / max(ctx["item_popularity"].values()),
        weight=1.0,
    )
    context = {"item_popularity": popularity}
    ranked = scorer.rank_candidates("alice", candidates, context, limit=5)
    ranked_items = [entry["item_id"] for entry in ranked]
    assert len(ranked_items) > 0

    # 3. Evaluate against a known "ground truth" of what alice actually liked next
    ground_truth = {"alice": ["book3"]}
    recommendations_dict = {"alice": ranked_items}
    metrics = RecommendationEvaluator.evaluate_all(recommendations_dict, ground_truth, k=5)
    assert metrics["users_evaluated"] == 1
    assert 0.0 <= metrics["precision_at_k"] <= 1.0

    print("[PASS] integration (candidates -> scoring -> evaluation)")


if __name__ == "__main__":
    test_similarity()
    test_candidate_gen()
    test_scorer()
    test_evaluator()
    test_integration()
    print("\nAll Day 29 tests passed.")
