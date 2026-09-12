"""Shared pytest fixtures."""

import pytest

from reco_engine.persistence import InMemoryRepository


@pytest.fixture
def sample_repo() -> InMemoryRepository:
    """A small, deterministic dataset used across multiple test modules."""
    user_history = {
        "u1": {"item_a", "item_b", "item_c"},
        "u2": {"item_a", "item_b", "item_d"},  # similar to u1
        "u3": {"item_x", "item_y"},            # dissimilar taste
        "cold_user": set(),                    # no history
    }
    item_features = {
        "item_a": {"python", "backend"},
        "item_b": {"python", "ml"},
        "item_c": {"frontend", "react"},
        "item_d": {"python", "backend", "sql"},
        "item_x": {"design"},
        "item_y": {"design", "figma"},
    }
    return InMemoryRepository(user_history, item_features)
