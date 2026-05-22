"""Schema bootstrap command used by Docker Compose and local startup."""

from __future__ import annotations

from db import create_schema
from utils.logger import get_logger


logger = get_logger(__name__)


def run_migrations() -> None:
    """Create all database tables if they do not already exist."""

    logger.info("Running database migrations")
    create_schema()


def main() -> None:
    """Entry point for ``python -m db.migrations``."""

    run_migrations()


if __name__ == "__main__":
    main()
