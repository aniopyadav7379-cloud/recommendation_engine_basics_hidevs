"""Configuration management.

All tunables live here instead of being hard-coded through the
codebase, and every value can be overridden via environment variable
so the same code runs unchanged across dev/test/prod:

    RECO_MAX_CANDIDATES=50 RECO_LOG_LEVEL=DEBUG python -m reco_engine.cli ...

`get_settings()` is cached (loaded once per process) via lru_cache,
so repeated calls are free and every component sees a consistent view.
"""

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw is not None else default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw is not None else default


@dataclass(frozen=True)
class Settings:
    # Candidate generation
    max_candidates: int = 20

    # Default scorer weights (used by the service when the caller
    # doesn't register custom scorers)
    weight_relevance: float = 0.5
    weight_recency: float = 0.2
    weight_popularity: float = 0.3

    # Ranking
    default_top_k: int = 10

    # Persistence
    data_dir: str = "data"
    history_file: str = "user_history.json"
    features_file: str = "item_features.json"

    # Logging
    log_level: str = "INFO"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    @property
    def default_weights(self) -> Dict[str, float]:
        return {
            "relevance": self.weight_relevance,
            "recency": self.weight_recency,
            "popularity": self.weight_popularity,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once per process, from environment variables
    (prefixed RECO_) with defaults falling back to the dataclass above."""
    return Settings(
        max_candidates=_env_int("RECO_MAX_CANDIDATES", 20),
        weight_relevance=_env_float("RECO_WEIGHT_RELEVANCE", 0.5),
        weight_recency=_env_float("RECO_WEIGHT_RECENCY", 0.2),
        weight_popularity=_env_float("RECO_WEIGHT_POPULARITY", 0.3),
        default_top_k=_env_int("RECO_DEFAULT_TOP_K", 10),
        data_dir=os.getenv("RECO_DATA_DIR", "data"),
        history_file=os.getenv("RECO_HISTORY_FILE", "user_history.json"),
        features_file=os.getenv("RECO_FEATURES_FILE", "item_features.json"),
        log_level=os.getenv("RECO_LOG_LEVEL", "INFO"),
        api_host=os.getenv("RECO_API_HOST", "0.0.0.0"),
        api_port=_env_int("RECO_API_PORT", 8000),
    )
