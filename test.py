"""
test.py — runs all tests for the recommendation engine

Run with: python3 test.py
"""

from similarity import SimilarityCalculator
from candidate_gen import CandidateGenerator
from scorer import Scorer
from evaluator import Evaluator


def test_similarity_calculator():
    sim = SimilarityCalculator()

    # cosine similarity
    assert round(sim.cosine_similarity([1, 2, 3], [1, 2, 3]), 4) == 1.0
    assert sim.cosine_similarity([1, 0], [0, 1]) == 0.0
    assert sim.cosine_similarity([], [1, 2]) == 0.0       # empty vector
    assert sim.cosine_similarity([0, 0], [1, 2]) == 0.0   # zero vector

    # jaccard similarity
    assert sim.jaccard_similarity({"a", "b"}, {"a", "b"}) == 1.0
    assert sim.jaccard_similarity({"a"}, {"b"}) == 0.0
    assert sim.jaccard_similarity(set(), set()) == 1.0    # both empty
    assert round(sim.jaccard_similarity({"a", "b", "c"}, {"b", "c", "d"}), 4) == round(2 / 4, 4)

    print("[PASS] similarity_calculator")


def test_candidate_generator():
    catalog = {
        "movie1": {"action", "sci-fi"},
        "movie2": {"romance"},
        "movie3": {"sci-fi", "thriller"},
    }
    gen = CandidateGenerator(catalog)

    candidates = gen.find_candidates({"action", "sci-fi"})
    assert "movie1" in candidates
    assert "movie3" in candidates
    assert "movie2" not in candidates

    # edge cases
    assert gen.find_candidates(set()) == []                    # no preferences
    assert CandidateGenerator({}).find_candidates({"action"}) == []  # empty catalog
    assert gen.find_candidates({"horror"}) == []                # no matches

    print("[PASS] candidate_generator")


def test_scorer():
    catalog = {
        "movie1": {"action", "sci-fi"},
        "movie2": {"sci-fi"},
    }
    ratings = {"movie1": 4.5, "movie2": 3.0}
    scorer = Scorer(catalog, ratings)

    ranked = scorer.rank_candidates(["movie1", "movie2"], {"action", "sci-fi"}, top_n=2)
    assert ranked[0][0] == "movie1"  # better tag match + higher rating
    assert len(ranked) == 2

    # edge cases
    assert scorer.rank_candidates([], {"action"}) == []          # no candidates
    assert scorer.score_item("unknown", {"action"}) == 0.0       # item not in catalog

    print("[PASS] scorer")


def test_evaluator():
    evaluator = Evaluator()

    recommended = ["movie1", "movie2", "movie3"]
    relevant = ["movie1", "movie3"]

    assert round(evaluator.precision(recommended, relevant), 4) == round(2 / 3, 4)
    assert evaluator.precision_at_k(recommended, relevant, k=1) == 1.0  # movie1 is a hit

    # edge cases
    assert evaluator.precision([], relevant) == 0.0     # no recommendations
    assert evaluator.precision(recommended, []) == 0.0  # no ground truth
    assert evaluator.precision_at_k(recommended, relevant, k=0) == 0.0

    print("[PASS] evaluator")


def test_full_pipeline():
    """Chains all four components together: candidates -> scoring ->
    ranking -> evaluation, using the same sample data as demo.py.
    """
    catalog = {
        "inception": {"sci-fi", "action"},
        "the_notebook": {"romance"},
        "interstellar": {"sci-fi", "drama"},
    }
    ratings = {"inception": 4.8, "the_notebook": 4.2, "interstellar": 4.9}
    user_preferences = {"sci-fi", "action"}
    user_actually_liked = ["inception", "interstellar"]

    # 1. Generate candidates
    generator = CandidateGenerator(catalog)
    candidates = generator.find_candidates(user_preferences)
    assert len(candidates) > 0

    # 2. Score and rank
    scorer = Scorer(catalog, ratings)
    top_picks = scorer.rank_candidates(candidates, user_preferences, top_n=2)
    assert len(top_picks) > 0
    recommended_ids = [item_id for item_id, _score in top_picks]

    # 3. Evaluate
    evaluator = Evaluator()
    precision = evaluator.precision(recommended_ids, user_actually_liked)
    assert 0.0 <= precision <= 1.0

    print("[PASS] full_pipeline (candidates -> scoring -> evaluation)")


if __name__ == "__main__":
    test_similarity_calculator()
    test_candidate_generator()
    test_scorer()
    test_evaluator()
    test_full_pipeline()
    print("\nAll tests passed!")
