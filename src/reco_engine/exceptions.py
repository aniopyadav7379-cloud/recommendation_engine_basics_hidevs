"""Custom exceptions for the recommendation engine.

Using a dedicated hierarchy (instead of bare ValueError/KeyError
everywhere) lets callers — a REST layer, a CLI, a batch job — catch
exactly what they care about without swallowing unrelated bugs.
"""


class RecoEngineError(Exception):
    """Base class for all errors raised by this package."""


class InvalidInputError(RecoEngineError):
    """Raised when caller-supplied data fails validation (bad shapes,
    mismatched lengths, negative weights, etc.)."""


class UserNotFoundError(RecoEngineError):
    """Raised when an operation requires a known user and none exists.

    Note: cold-start users (known but with empty history) are NOT an
    error — only a user_id absent from the repository entirely raises
    this.
    """

    def __init__(self, user_id: str):
        super().__init__(f"Unknown user_id: {user_id!r}")
        self.user_id = user_id


class ItemNotFoundError(RecoEngineError):
    """Raised when an operation references an item_id the repository
    has no features/metadata for."""

    def __init__(self, item_id: str):
        super().__init__(f"Unknown item_id: {item_id!r}")
        self.item_id = item_id


class PersistenceError(RecoEngineError):
    """Raised when the storage backend fails to read or write data."""
