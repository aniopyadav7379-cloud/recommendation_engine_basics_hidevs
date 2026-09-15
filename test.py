"""
test.py
-------
Test cases for all four Recommendation Engine components.
Run this file directly: python test.py
"""

from similarity import SimilarityCalculator
from candidate_gen import CandidateGenerator
from scorer import RecommendationScorer
from evaluator import RecommendationEvaluator

PASS = "✅ PASS"
FAIL = "❌ FAIL"


def check(label, condition):
    print(f"  {PASS if condition else FAIL}  {label}")


# ──────────────────────────────────────────────
# 1. SIMILARITY CALCULATOR TESTS
# ──────────────────────────────────────────────
def test_similarity():
    print("\n📐 SimilarityCalculator Tests")
    sim = SimilarityCalculator()

    # Cosine similarity
    score = sim.cosine_similarity([1, 0, 1], [1, 0, 1])
    check("Identical vectors → cosine = 1.0", abs(score - 1.0) < 0.001)

    score = sim.cosine_similarity([1, 0], [0, 1])
    check("Orthogonal vectors → cosine = 0.0", abs(score - 0.0) < 0.001)

    score = sim.cosine_similarity([], [])
    check("Empty vectors → cosine = 0.0", score == 0.0)

    score = sim.cosine_similarity([0, 0], [0, 0])
    check("Zero vectors → cosine = 0.0", score == 0.0)

    # Jaccard similarity
    score = sim.jaccard_similarity({"a", "b"}, {"a", "b"})
    check("Identical sets → jaccard = 1.0", abs(score - 1.0) < 0.001)

    score = sim.jaccard_similarity({"a"}, {"b"})
    check("Disjoint sets → jaccard = 0.0", abs(score - 0.0) < 0.001)

    score = sim.jaccard_similarity({"a", "b", "c"}, {"b", "c", "d"})
    check("Partial overlap → jaccard = 0.5", abs(score - 0.5) < 0.001)

    score = sim.jaccard_similarity(set(), set())
    check("Both empty → jaccard = 1.0", score == 1.0)

    score = sim.jaccard_similarity(set(), {"a"})
    check("One empty → jaccard = 0.0", score == 0.0)

    # Pearson correlation
    score = sim.pearson_correlation([1, 2, 3], [1, 2, 3])
    check("Same ratings → pearson = 1.0", abs(score - 1.0) < 0.001)

    score = sim.pearson_correlation([1, 2, 3], [3, 2, 1])
    check("Opposite ratings → pearson = -1.0", abs(score - (-1.0)) < 0.001)

    score = sim.pearson_correlation([], [])
    check("Empty ratings → pearson = 0.0", score == 0.0)


# ──────────────────────────────────────────────
# 2. CANDIDATE GENERATOR TESTS
# ──────────────────────────────────────────────
def test_candidate_generator():
    print("\n🎯 CandidateGenerator Tests")
    gen = CandidateGenerator()

    # Regular user
    candidates = gen.collaborative_candidates("u1")
    check("Collaborative candidates returns a list", isinstance(candidates, list))
    check("Collaborative respects limit (≤ 20)", len(candidates) <= 20)
    check("Collaborative excludes already-seen items", "i1" not in candidates)

    candidates = gen.content_based_candidates("u1")
    check("Content-based returns a list", isinstance(candidates, list))
    check("Content-based respects limit (≤ 20)", len(candidates) <= 20)

    # Cold start user (u5 has empty history)
    cold_collab = gen.collaborative_candidates("u5")
    check("Cold start collaborative returns items", len(cold_collab) > 0)

    cold_content = gen.content_based_candidates("u5")
    check("Cold start content-based returns items", len(cold_content) > 0)

    # Popularity
    popular = gen.popularity_candidates()
    check("Popularity returns a list", isinstance(popular, list))
    check("Most popular item is first", popular[0] == "i2")

    # Hybrid
    hybrid = gen.hybrid_candidates("u1")
    check("Hybrid returns a list", isinstance(hybrid, list))
    check("Hybrid respects limit (≤ 30)", len(hybrid) <= 30)


# ──────────────────────────────────────────────
# 3. SCORER TESTS
# ──────────────────────────────────────────────
def test_scorer():
    print("\n⭐ RecommendationScorer Tests")
    scorer = RecommendationScorer()

    # Single item score
    result = scorer.calculate_score("u1", "i5")
    check("Score result has 'score' key", "score" in result)
    check("Score is between 0 and 1", 0.0 <= result["score"] <= 1.0)
    check("Score has 'breakdown' key", "breakdown" in result)
    check("Score has 'reason' key", "reason" in result)

    # Ranking
    candidates = ["i4", "i5", "i7", "i8"]
    ranked = scorer.rank_candidates("u1", candidates, limit=3)
    check("Ranking returns list", isinstance(ranked, list))
    check("Ranking respects limit", len(ranked) <= 3)
    check("Ranking is sorted descending", ranked[0]["score"] >= ranked[-1]["score"])

    # Edge cases
    empty_ranked = scorer.rank_candidates("u1", [], limit=5)
    check("Empty candidates returns empty list", empty_ranked == [])

    cold_result = scorer.calculate_score("u5", "i2")  # cold start user
    check("Cold start user gets a score", 0.0 <= cold_result["score"] <= 1.0)

    # Custom scorer
    scorer.add_scorer("test_scorer", lambda u, i, c: 0.9, weight=0.1)
    result2 = scorer.calculate_score("u1", "i5")
    check("Custom scorer registered and used", "test_scorer" in result2["breakdown"])


# ──────────────────────────────────────────────
# 4. EVALUATOR TESTS
# ──────────────────────────────────────────────
def test_evaluator():
    print("\n📊 RecommendationEvaluator Tests")
    ev = RecommendationEvaluator()

    recs = ["i1", "i2", "i3", "i4", "i5"]
    relevant = ["i1", "i3", "i5"]

    # Precision@3: top-3 = [i1, i2, i3], hits = 2 → 2/3
    p = ev.precision_at_k(recs, relevant, k=3)
    check("Precision@3 = 0.667", abs(p - 0.667) < 0.01)

    # Recall@3: hits=2, total relevant=3 → 2/3
    r = ev.recall_at_k(recs, relevant, k=3)
    check("Recall@3 = 0.667", abs(r - 0.667) < 0.01)

    # NDCG@5
    n = ev.ndcg_at_k(recs, relevant, k=5)
    check("NDCG@5 is between 0 and 1", 0.0 <= n <= 1.0)

    # Perfect recommendations
    p_perfect = ev.precision_at_k(["i1", "i3", "i5"], ["i1", "i3", "i5"], k=3)
    check("Perfect recs → precision = 1.0", abs(p_perfect - 1.0) < 0.001)

    n_perfect = ev.ndcg_at_k(["i1", "i3", "i5"], ["i1", "i3", "i5"], k=3)
    check("Perfect recs → NDCG = 1.0", abs(n_perfect - 1.0) < 0.001)

    # Edge cases
    check("Empty recs → precision = 0.0", ev.precision_at_k([], relevant, k=3) == 0.0)
    check("Empty relevant → recall = 0.0", ev.recall_at_k(recs, [], k=3) == 0.0)
    check("k=0 → ndcg = 0.0", ev.ndcg_at_k(recs, relevant, k=0) == 0.0)

    # evaluate_all
    recs_dict = {"u1": ["i4", "i5", "i7"], "u2": ["i1", "i3", "i6"]}
    gt_dict = {"u1": ["i5", "i7"], "u2": ["i1", "i6"]}
    metrics = ev.evaluate_all(recs_dict, gt_dict, k=3)
    check("evaluate_all returns dict", isinstance(metrics, dict))
    check("evaluate_all has precision@k", "precision@k" in metrics)
    check("evaluate_all has recall@k", "recall@k" in metrics)
    check("evaluate_all has ndcg@k", "ndcg@k" in metrics)
    check("evaluate_all evaluated 2 users", metrics["users_evaluated"] == 2)

    # No ground truth
    empty_metrics = ev.evaluate_all({"u1": ["i1"]}, {}, k=3)
    check("No ground truth → users_evaluated = 0", empty_metrics["users_evaluated"] == 0)


# ──────────────────────────────────────────────
# 5. INTEGRATION TEST
# ──────────────────────────────────────────────
def test_integration():
    print("\n🔗 Integration Test (Full Pipeline)")

    gen = CandidateGenerator()
    scorer = RecommendationScorer()
    ev = RecommendationEvaluator()

    user_id = "u1"
    candidates = gen.hybrid_candidates(user_id, limit=30)
    ranked = scorer.rank_candidates(user_id, candidates, limit=5)
    top_items = [r["item_id"] for r in ranked]

    ground_truth = {"u1": ["i5", "i7", "i8"]}
    recs_dict = {"u1": top_items}
    metrics = ev.evaluate_all(recs_dict, ground_truth, k=5)

    check("Pipeline produces recommendations", len(top_items) > 0)
    check("Metrics computed successfully", metrics["users_evaluated"] == 1)
    print(f"\n  📈 Pipeline Metrics for user '{user_id}':")
    print(f"     Top recommendations : {top_items}")
    print(f"     Precision@5         : {metrics['precision@k']}")
    print(f"     Recall@5            : {metrics['recall@k']}")
    print(f"     NDCG@5              : {metrics['ndcg@k']}")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("   DAY 29 — Recommendation Engine: All Tests")
    print("=" * 55)

    test_similarity()
    test_candidate_generator()
    test_scorer()
    test_evaluator()
    test_integration()

    print("\n" + "=" * 55)
    print("   All tests complete!")
    print("=" * 55)