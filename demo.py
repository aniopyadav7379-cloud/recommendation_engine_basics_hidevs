"""
demo.py — puts all four components together

Simulates a tiny movie recommendation flow end to end:

    user preferences -> Candidate Generator -> Scorer -> ranked picks
                                                              |
                                                              v
                                          Evaluator checks the picks
                                          against what the user
                                          actually went on to watch
"""

from candidate_gen import CandidateGenerator
from scorer import Scorer
from evaluator import Evaluator

# --- sample data ------------------------------------------------------

movie_catalog = {
    "inception": {"sci-fi", "thriller", "action"},
    "the_notebook": {"romance", "drama"},
    "interstellar": {"sci-fi", "drama"},
    "the_hangover": {"comedy"},
    "mad_max": {"action", "sci-fi"},
    "titanic": {"romance", "drama"},
}

movie_ratings = {
    "inception": 4.8,
    "the_notebook": 4.2,
    "interstellar": 4.9,
    "the_hangover": 4.0,
    "mad_max": 4.5,
    "titanic": 4.3,
}

# a user who likes sci-fi and action movies
user_preferences = {"sci-fi", "action"}

# what this user actually went on to watch and enjoy (ground truth,
# for evaluating how good our recommendations were)
user_actually_liked = ["inception", "interstellar"]


def main():
    # 1. Generate candidates: narrow the catalog down to relevant items
    generator = CandidateGenerator(movie_catalog)
    candidates = generator.find_candidates(user_preferences, limit=10)
    print("Candidates:", candidates)

    # 2. Score and rank the candidates, get the top picks
    scorer = Scorer(movie_catalog, movie_ratings)
    top_picks = scorer.rank_candidates(candidates, user_preferences, top_n=3)
    print("\nTop recommendations:")
    for item_id, score in top_picks:
        print(f"  {item_id}: score={score}")

    # 3. Evaluate: how many of our top picks did the user actually like?
    recommended_ids = [item_id for item_id, _score in top_picks]
    evaluator = Evaluator()
    precision = evaluator.precision(recommended_ids, user_actually_liked)
    print(f"\nPrecision of top {len(recommended_ids)} recommendations: {precision}")


if __name__ == "__main__":
    main()
