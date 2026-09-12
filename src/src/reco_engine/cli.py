"""Command-line interface.

Usage:
    python -m reco_engine.cli recommend <user_id> [--k N]
    python -m reco_engine.cli record <user_id> <item_id>
"""

from __future__ import annotations

import argparse
import sys

from .config import get_settings
from .logging_config import get_logger
from .persistence import JSONFileRepository
from .service import RecommendationService

logger = get_logger(__name__)


def build_service() -> RecommendationService:
    settings = get_settings()
    repo = JSONFileRepository(settings.data_dir, settings.history_file, settings.features_file)
    return RecommendationService(repo)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="reco_engine", description="Recommendation engine CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    recommend_parser = subparsers.add_parser("recommend", help="Get recommendations for a user")
    recommend_parser.add_argument("user_id")
    recommend_parser.add_argument("--k", type=int, default=None)

    record_parser = subparsers.add_parser("record", help="Record a user-item interaction")
    record_parser.add_argument("user_id")
    record_parser.add_argument("item_id")

    args = parser.parse_args(argv)
    service = build_service()

    if args.command == "recommend":
        results = service.get_recommendations(args.user_id, args.k)
        if not results:
            print(f"No recommendations available for '{args.user_id}'")
            return 0
        for item in results:
            print(f"{item.item_id}\t{item.score:.3f}\t{item.explanation}")
        return 0

    if args.command == "record":
        service.record_interaction(args.user_id, args.item_id)
        print(f"Recorded: {args.user_id} -> {args.item_id}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
