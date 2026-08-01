"""
End-to-end test chaining all four components the way the full
recommendation engine will do tomorrow:

    candidates -> scored/ranked -> evaluated against ground truth
"""

from candidate_gen import CandidateGenerator
from scorer import RecommendationScorer
from evaluator import RecommendationEvaluator


def test_full_pipeline():
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

    # 3. Evaluate against a known "ground truth" of what alice liked next
    ground_truth = {"alice": ["book3"]}
    recommendations_dict = {"alice": ranked_items}
    metrics = RecommendationEvaluator.evaluate_all(recommendations_dict, ground_truth, k=5)

    assert metrics["users_evaluated"] == 1
    assert 0.0 <= metrics["precision_at_k"] <= 1.0
    assert 0.0 <= metrics["recall_at_k"] <= 1.0
    assert 0.0 <= metrics["ndcg_at_k"] <= 1.0
