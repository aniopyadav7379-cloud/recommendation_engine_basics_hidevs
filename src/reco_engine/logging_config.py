"""Shared logging setup.

Uses Python's stdlib `logging` (not `print`) so log level, format, and
destination are controllable in one place and callers can redirect
output (file, log aggregator, stdout) without touching library code.
"""

import logging

from .config import get_settings

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger, configuring the root handler once."""
    global _CONFIGURED
    if not _CONFIGURED:
        settings = get_settings()
        logging.basicConfig(
            level=getattr(logging, settings.log_level.upper(), logging.INFO),
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        )
        _CONFIGURED = True
    return logging.getLogger(name)
