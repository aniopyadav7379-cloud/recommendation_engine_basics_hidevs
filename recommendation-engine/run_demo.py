"""End-to-end demo: run the full pipeline against the seed data in
data/ and print personalized recommendations, scores, explanations,
evaluation metrics, and cold-start behavior.

Run: python run_demo.py
"""

from __future__ import annotations

from reco_engine.config import get_settings
from reco_engine.persistence import JSONFileRepository
from reco_engine.scorer import ScoredItem
from reco_engine.service import RecommendationService

# Item-level signals (recency/popularity describe the item itself, not
# a specific user's affinity for it -- in a real system these would be
# computed/cached by a batch job and read from a feature store).
RECENCY_SIGNAL = {"job_1": 0.4, "job_2": 0.9, "job_3": 0.7, "job_5": 0.6,
                   "job_6": 0.5, "job_9": 0.2, "job_10": 0.2}
POPULARITY_SIGNAL = {"job_1": 0.8, "job_2": 0.6, "job_3": 0.5, "job_5": 0.5,
                      "job_6": 0.3, "job_9": 0.4, "job_10": 0.4}

GROUND_TRUTH = {
    "alice": ["job_3", "job_6"],
    "bob": ["job_5"],
}

# A brand-new user id that appears nowhere in data/user_history.json,
# used below to demonstrate cold-start behavior explicitly.
COLD_START_USER = "grace_newuser"


def _print_section(title: str) -> None:
    print(f"\n=== {title} ===")


def _print_results(user_id: str, results: list[ScoredItem]) -> None:
    print(f"--- Recommendations for {user_id} ---")
    if not results:
        print("  (no candidates generated)")
        return
    for rank, item in enumerate(results, start=1):
        print(f"  {rank}. {item.item_id:<10} score={item.score:.3f} ({item.score:.0%})  {item.explanation}")


def main() -> None:
    settings = get_settings()
    repo = JSONFileRepository(settings.data_dir, settings.history_file, settings.features_file)
    service = RecommendationService(repo, recency=RECENCY_SIGNAL, popularity=POPULARITY_SIGNAL)

    _print_section("Personalized recommendations (existing users)")
    for user_id in repo.get_all_user_history():
        results = service.get_recommendations(user_id, k=5)
        _print_results(user_id, results)
        print()

    _print_section("Cold-start behavior")
    print(
        f"'{COLD_START_USER}' has no interaction history at all -- this is exactly\n"
        "the scenario collaborative filtering and content-based filtering can't\n"
        "handle on their own. The candidate generator falls back to a\n"
        "popularity ranking instead of returning nothing:\n"
    )
    cold_start_results = service.get_recommendations(COLD_START_USER, k=5)
    _print_results(COLD_START_USER, cold_start_results)

    _print_section("Evaluation (users with known ground truth)")
    metrics = service.evaluate(user_ids=list(GROUND_TRUTH), ground_truth=GROUND_TRUTH, k=5)
    for name, value in metrics.items():
        if name == "num_users_evaluated":
            print(f"  {name}: {value}")
        else:
            print(f"  {name}: {value:.3f}")


if __name__ == "__main__":
    main()
