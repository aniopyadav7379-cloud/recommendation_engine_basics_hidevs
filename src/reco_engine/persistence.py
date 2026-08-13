"""Persistence layer.

Defines a `Repository` interface so the rest of the engine never talks
to storage directly — it asks for user history / item features / to
record an interaction, and doesn't care whether that's backed by a
Python dict, a JSON file, or (later) a real database.

Two implementations are provided:
    InMemoryRepository  - ephemeral, for tests and quick experiments
    JSONFileRepository   - durable, atomic writes to disk

Swapping in a SQL/NoSQL-backed repository later means implementing
this same interface — nothing upstream (candidate generation, scoring,
the service layer) needs to change.
"""

import json
import os
import tempfile
from abc import ABC, abstractmethod
from typing import Dict, Set

from .exceptions import PersistenceError
from .logging_config import get_logger

logger = get_logger(__name__)


class Repository(ABC):
    """Abstract data-access interface for user history and item features."""

    @abstractmethod
    def get_user_history(self, user_id: str) -> Set[str]:
        """Return the set of item_ids this user has interacted with.
        Returns an empty set for unknown/cold-start users (not an error)."""

    @abstractmethod
    def get_all_user_history(self) -> Dict[str, Set[str]]:
        """Return the full user_id -> item_ids mapping."""

    @abstractmethod
    def get_item_features(self, item_id: str) -> Set[str]:
        """Return the tag/feature set for an item. Empty set if unknown."""

    @abstractmethod
    def get_all_item_features(self) -> Dict[str, Set[str]]:
        """Return the full item_id -> tags mapping."""

    @abstractmethod
    def record_interaction(self, user_id: str, item_id: str) -> None:
        """Persist that `user_id` interacted with `item_id`."""


class InMemoryRepository(Repository):
    """Non-durable repository backed by plain dicts. Good for tests and
    the algorithmic-prototype stage; state is lost on process exit."""

    def __init__(self, user_history: Dict[str, Set[str]] = None,
                 item_features: Dict[str, Set[str]] = None):
        self._history: Dict[str, Set[str]] = {
            uid: set(items) for uid, items in (user_history or {}).items()
        }
        self._features: Dict[str, Set[str]] = {
            iid: set(tags) for iid, tags in (item_features or {}).items()
        }

    def get_user_history(self, user_id: str) -> Set[str]:
        return set(self._history.get(user_id, set()))

    def get_all_user_history(self) -> Dict[str, Set[str]]:
        return {uid: set(items) for uid, items in self._history.items()}

    def get_item_features(self, item_id: str) -> Set[str]:
        return set(self._features.get(item_id, set()))

    def get_all_item_features(self) -> Dict[str, Set[str]]:
        return {iid: set(tags) for iid, tags in self._features.items()}

    def record_interaction(self, user_id: str, item_id: str) -> None:
        self._history.setdefault(user_id, set()).add(item_id)
        logger.debug("Recorded interaction user=%s item=%s", user_id, item_id)


class JSONFileRepository(Repository):
    """Durable repository that persists to two JSON files on disk.

    Writes are atomic (write-to-temp-then-rename) so a crash mid-write
    can't corrupt the data file. Item features are treated as static
    reference data (loaded once); user history is read fresh from
    memory but flushed to disk on every `record_interaction` call.
    """

    def __init__(self, data_dir: str, history_file: str, features_file: str):
        self._history_path = os.path.join(data_dir, history_file)
        self._features_path = os.path.join(data_dir, features_file)
        os.makedirs(data_dir, exist_ok=True)

        self._history = self._load(self._history_path)
        self._features = self._load(self._features_path)

    @staticmethod
    def _load(path: str) -> Dict[str, Set[str]]:
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            return {key: set(values) for key, values in raw.items()}
        except (json.JSONDecodeError, OSError) as exc:
            raise PersistenceError(f"Failed to load {path}: {exc}") from exc

    @staticmethod
    def _atomic_write(path: str, data: Dict[str, Set[str]]) -> None:
        serializable = {key: sorted(values) for key, values in data.items()}
        directory = os.path.dirname(path) or "."
        try:
            fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2)
            os.replace(tmp_path, path)  # atomic on POSIX and Windows
        except OSError as exc:
            raise PersistenceError(f"Failed to write {path}: {exc}") from exc

    def get_user_history(self, user_id: str) -> Set[str]:
        return set(self._history.get(user_id, set()))

    def get_all_user_history(self) -> Dict[str, Set[str]]:
        return {uid: set(items) for uid, items in self._history.items()}

    def get_item_features(self, item_id: str) -> Set[str]:
        return set(self._features.get(item_id, set()))

    def get_all_item_features(self) -> Dict[str, Set[str]]:
        return {iid: set(tags) for iid, tags in self._features.items()}

    def record_interaction(self, user_id: str, item_id: str) -> None:
        self._history.setdefault(user_id, set()).add(item_id)
        self._atomic_write(self._history_path, self._history)
        logger.info("Persisted interaction user=%s item=%s", user_id, item_id)
