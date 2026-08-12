"""Simple tests for similarity.py, candidate_gen.py, scorer.py, evaluator.py.

Run: python test.py
"""

from similarity import SimilarityCalculator
from candidate_gen import CandidateGenerator
from scorer import RecommendationScorer
from evaluator import RecommendationEvaluator

passed = 0
failed = 0


def check(label, actual, expected, tol=1e-6):
    global passed, failed
    ok = abs(actual - expected) <= tol if isinstance(expected, (int, float)) else actual == expected
    status = "PASS" if ok else "FAIL"
    if ok:
        passed += 1
    else:
        failed += 1
    print(f"[{status}] {label} -> got {actual}, expected {expected}")


# ---------------------------------------------------------------------
# Component 1: SimilarityCalculator
# ---------------------------------------------------------------------
print("\n=== SimilarityCalculator ===")
sc = SimilarityCalculator()
check("cosine identical vectors", sc.cosine_similarity([1, 2, 3], [1, 2, 3]), 1.0)
check("cosine orthogonal vectors", sc.cosine_similarity([1, 0], [0, 1]), 0.0)
check("cosine zero vector", sc.cosine_similarity([0, 0, 0], [1, 2, 3]), 0.0)
check("cosine empty vectors", sc.cosine_similarity([], []), 0.0)

check("jaccard partial overlap",
      sc.jaccard_similarity({"python", "sql", "aws"}, {"python", "sql", "gcp"}), 0.5)
check("jaccard identical sets", sc.jaccard_similarity({"a", "b"}, {"a", "b"}), 1.0)
check("jaccard disjoint sets", sc.jaccard_similarity({"a"}, {"b"}), 0.0)
check("jaccard both empty", sc.jaccard_similarity(set(), set()), 1.0)

check("pearson perfectly correlated", sc.pearson_correlation([1, 2, 3, 4], [2, 4, 6, 8]), 1.0)
check("pearson perfectly anti-correlated", sc.pearson_correlation([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
check("pearson zero variance", sc.pearson_correlation([3, 3, 3], [1, 2, 3]), 0.0)
check("pearson empty input", sc.pearson_correlation([], []), 0.0)

# ---------------------------------------------------------------------
# Component 2: CandidateGenerator
# ---------------------------------------------------------------------
print("\n=== CandidateGenerator ===")
user_history = {
    "u1": {"item_a", "item_b", "item_c"},
    "u2": {"item_a", "item_b", "item_d"},  # similar to u1
    "u3": {"item_x", "item_y"},            # dissimilar
    "cold_user": set(),                    # no history -> cold start
}
item_features = {
    "item_a": {"python", "backend"},
    "item_b": {"python", "ml"},
    "item_c": {"frontend", "react"},
    "item_d": {"python", "backend", "sql"},
    "item_x": {"design"},
    "item_y": {"design", "figma"},
}
gen = CandidateGenerator(user_history, item_features, max_candidates=10)

collab = gen.collaborative_candidates("u1")
check("collaborative excludes own history", any(i in user_history["u1"] for i in collab), False)
check("collaborative finds item_d from similar user u2", "item_d" in collab, True)

content = gen.content_based_candidates("u1")
check("content-based excludes own history", any(i in user_history["u1"] for i in content), False)

pop = gen.popularity_candidates()
check("popularity returns non-empty list for populated history", len(pop) > 0, True)

hybrid_cold = gen.hybrid_candidates("cold_user")
check("hybrid handles cold-start user without error", isinstance(hybrid_cold, list), True)

hybrid_u1 = gen.hybrid_candidates("u1")
check("hybrid excludes already-seen items", any(i in user_history["u1"] for i in hybrid_u1), False)
check("hybrid respects max_candidates cap", len(hybrid_u1) <= 10, True)

# ---------------------------------------------------------------------
# Component 3: RecommendationScorer
# ---------------------------------------------------------------------
print("\n=== RecommendationScorer ===")
relevance = {"item_a": 0.9, "item_b": 0.4}
popularity = {"item_a": 0.6, "item_b": 0.6}

scorer = RecommendationScorer()
scorer.add_scorer("relevance", lambda u, i, ctx: relevance.get(i, 0.0), weight=0.7)
scorer.add_scorer("popularity", lambda u, i, ctx: popularity.get(i, 0.0), weight=0.3)

score_a, explanation_a = scorer.calculate_score("u1", "item_a")
check("score is within [0, 1]", 0.0 <= score_a <= 1.0, True)
check("explanation is non-empty", len(explanation_a) > 0, True)

ranked = scorer.rank_candidates("u1", ["item_a", "item_b", "item_a"], limit=5)
check("rank_candidates deduplicates input", len(ranked), 2)
check("rank_candidates sorts descending by score", ranked[0].score >= ranked[1].score, True)

empty_scorer = RecommendationScorer()
empty_score, _ = empty_scorer.calculate_score("u1", "item_a")
check("no scorers registered -> score 0.0", empty_score, 0.0)

# ---------------------------------------------------------------------
# Component 4: RecommendationEvaluator
# ---------------------------------------------------------------------
print("\n=== RecommendationEvaluator ===")
ev = RecommendationEvaluator()
recs = ["item_a", "item_b", "item_c", "item_d", "item_e"]
relevant_items = ["item_b", "item_d", "item_z"]  # item_z never recommended

check("precision@5", ev.precision_at_k(recs, relevant_items, 5), 2 / 5)
check("recall@5", ev.recall_at_k(recs, relevant_items, 5), 2 / 3)
check("ndcg perfect ranking", ev.ndcg_at_k(["item_b", "item_d"], ["item_b", "item_d"], 5), 1.0)
check("precision empty recommendations", ev.precision_at_k([], relevant_items, 5), 0.0)
check("recall no ground truth", ev.recall_at_k(recs, [], 5), 0.0)

recommendations_dict = {
    "u1": ["item_a", "item_b", "item_c"],
    "u2": ["item_x", "item_y", "item_z"],
    "u3": ["item_p", "item_q"],  # no ground truth -> should be skipped
}
ground_truth_dict = {
    "u1": ["item_b"],
    "u2": ["item_x", "item_z"],
}
metrics = ev.evaluate_all(recommendations_dict, ground_truth_dict, k=3)
check("evaluate_all skips users with no ground truth", metrics["num_users_evaluated"], 2)

# ---------------------------------------------------------------------
print(f"\n{passed} passed, {failed} failed")
if failed:
    raise SystemExit(1)
