"""End-to-end demo: run the full pipeline against the seed data in
data/ and print recommendations + evaluation metrics.

Run: python run_demo.py
"""

from reco_engine.config import get_settings
from reco_engine.persistence import JSONFileRepository
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


def main() -> None:
    settings = get_settings()
    repo = JSONFileRepository(settings.data_dir, settings.history_file, settings.features_file)
    service = RecommendationService(repo, recency=RECENCY_SIGNAL, popularity=POPULARITY_SIGNAL)

    for user_id in repo.get_all_user_history():
        results = service.get_recommendations(user_id, k=5)
        print(f"--- Recommendations for {user_id} ---")
        if not results:
            print("  (no candidates generated)")
        for item in results:
            print(f"  {item.item_id}: {item.score:.3f} | {item.explanation}")
        print()

    metrics = service.evaluate(user_ids=list(GROUND_TRUTH), ground_truth=GROUND_TRUTH, k=5)
    print("=== Evaluation (users with ground truth only) ===")
    print(metrics)


if __name__ == "__main__":
    main()
