"""Database engine and session helpers."""

from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from config import SETTINGS
from utils.logger import get_logger


logger = get_logger(__name__)
SQLITE_PREFIX = "sqlite"


@lru_cache(maxsize=4)
def _build_engine(url: str):
    """Create and cache a SQLAlchemy engine for a normalized database URL."""

    engine_kwargs: dict[str, object] = {"future": True, "pool_pre_ping": True}
    if url.startswith(SQLITE_PREFIX):
        engine_kwargs["connect_args"] = {"check_same_thread": False}
        if ":memory:" in url:
            engine_kwargs["poolclass"] = StaticPool
    logger.info("Creating database engine for %s", url)
    return create_engine(url, **engine_kwargs)


def get_engine(database_url: str | None = None):
    """Create a SQLAlchemy engine for the active database URL."""

    url = database_url or SETTINGS.database_url
    return _build_engine(url)


@lru_cache(maxsize=4)
def _build_session_factory(url: str):
    """Create and cache a SQLAlchemy session factory for a normalized URL."""

    return sessionmaker(bind=get_engine(url), autoflush=False, autocommit=False, expire_on_commit=False, future=True)


def get_session_factory(database_url: str | None = None):
    """Return a reusable SQLAlchemy session factory."""

    url = database_url or SETTINGS.database_url
    return _build_session_factory(url)


@contextmanager
def session_scope(database_url: str | None = None) -> Generator[Session, None, None]:
    """Provide a transactional session boundary."""

    session_factory = get_session_factory(database_url)
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def ping_database(database_url: str | None = None) -> bool:
    """Check whether the configured database accepts a trivial query."""

    try:
        engine = get_engine(database_url)
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception as exc:  # pragma: no cover - defensive logging path
        logger.error("Database ping failed: %s", exc)
        return False
