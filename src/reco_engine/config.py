"""Configuration management.

All tunables live here instead of being hard-coded through the
codebase, and every value can be overridden via environment variable
so the same code runs unchanged across dev/test/prod:

    RECO_MAX_CANDIDATES=50 RECO_LOG_LEVEL=DEBUG python -m reco_engine.cli ...

Every existing ``RECO_*`` variable keeps its name and default; one new
variable (``RECO_MAX_K``) was added for the API's bounded ``k`` limit.

Settings are validated eagerly in ``__post_init__`` so a bad value
(negative weight, out-of-range port, all-zero weights) fails loudly at
startup with a clear message, instead of surfacing later as a subtle
runtime bug.

``get_settings()`` is cached (loaded once per process) via lru_cache,
so repeated calls are free and every component sees a consistent view.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from .exceptions import ConfigurationError

_VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


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

    # API-level ceiling on the caller-supplied `k` (recommendations per
    # request), so a request can't force the service to rank/return an
    # unbounded number of items.
    max_k: int = 100

    # Persistence
    data_dir: str = "data"
    history_file: str = "user_history.json"
    features_file: str = "item_features.json"

    # Logging
    log_level: str = "INFO"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    def __post_init__(self) -> None:
        """Validate every value eagerly so misconfiguration fails at
        startup with a clear message, not later as a confusing bug."""
        if self.max_candidates <= 0:
            raise ConfigurationError(
                f"RECO_MAX_CANDIDATES must be a positive integer, got {self.max_candidates}"
            )
        if self.default_top_k <= 0:
            raise ConfigurationError(
                f"RECO_DEFAULT_TOP_K must be a positive integer, got {self.default_top_k}"
            )
        if self.max_k <= 0:
            raise ConfigurationError(f"RECO_MAX_K must be a positive integer, got {self.max_k}")
        if self.default_top_k > self.max_k:
            raise ConfigurationError(
                f"RECO_DEFAULT_TOP_K ({self.default_top_k}) cannot exceed RECO_MAX_K ({self.max_k})"
            )

        for name, value in (
            ("RECO_WEIGHT_RELEVANCE", self.weight_relevance),
            ("RECO_WEIGHT_RECENCY", self.weight_recency),
            ("RECO_WEIGHT_POPULARITY", self.weight_popularity),
        ):
            if value < 0:
                raise ConfigurationError(f"{name} must be non-negative, got {value}")
        if self.weight_relevance == 0 and self.weight_recency == 0 and self.weight_popularity == 0:
            raise ConfigurationError(
                "At least one of RECO_WEIGHT_RELEVANCE, RECO_WEIGHT_RECENCY, "
                "RECO_WEIGHT_POPULARITY must be positive, or no scorer can ever contribute"
            )

        if not 1 <= self.api_port <= 65535:
            raise ConfigurationError(f"RECO_API_PORT must be between 1 and 65535, got {self.api_port}")

        normalized_level = self.log_level.strip().upper()
        if normalized_level not in _VALID_LOG_LEVELS:
            raise ConfigurationError(
                f"RECO_LOG_LEVEL must be one of {sorted(_VALID_LOG_LEVELS)}, got {self.log_level!r}"
            )
        # Settings is frozen; use object.__setattr__ to store the
        # normalized (uppercase) value without relaxing immutability.
        object.__setattr__(self, "log_level", normalized_level)

    @property
    def default_weights(self) -> dict[str, float]:
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
        max_k=_env_int("RECO_MAX_K", 100),
        data_dir=os.getenv("RECO_DATA_DIR", "data"),
        history_file=os.getenv("RECO_HISTORY_FILE", "user_history.json"),
        features_file=os.getenv("RECO_FEATURES_FILE", "item_features.json"),
        log_level=os.getenv("RECO_LOG_LEVEL", "INFO"),
        api_host=os.getenv("RECO_API_HOST", "0.0.0.0"),
        api_port=_env_int("RECO_API_PORT", 8000),
    )
