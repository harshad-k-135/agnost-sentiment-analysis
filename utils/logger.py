"""Logging helpers for the sentiment analytics engine."""

from __future__ import annotations

import logging
from typing import Final


LOG_FORMAT: Final[str] = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module name.

    Args:
        name: Logger namespace.

    Returns:
        A logger that emits structured, human-readable messages.
    """

    logger = logging.getLogger(name)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    return logger
