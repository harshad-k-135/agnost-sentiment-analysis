"""Database configuration, schema creation, and persistence helpers.

The app defaults to SQLite so the repository runs locally with no extra setup,
but the same code path works with PostgreSQL by setting ``DATABASE_URL``.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime
from typing import Generator, Sequence

from sqlalchemy import DateTime, Float, Integer, Text, create_engine, func
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


DEFAULT_SQLITE_URL = "sqlite:///./agnost_sentiment.db"
DATABASE_URL_ENV = "DATABASE_URL"
SQLITE_CONNECT_ARGS = {"check_same_thread": False}


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""

    pass


class Conversation(Base):
    """Persisted raw conversation text and its assigned cluster ID."""

    __tablename__ = "conversations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    cluster_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class Insight(Base):
    """Persisted insight generated from a cluster."""

    __tablename__ = "insights"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    cluster_id: Mapped[int] = mapped_column(Integer, nullable=False)
    insight_text: Mapped[str] = mapped_column(Text, nullable=False)
    percentage: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


def get_database_url() -> str:
    """Return the active database URL, falling back to local SQLite."""

    return os.getenv(DATABASE_URL_ENV, DEFAULT_SQLITE_URL)


def get_engine():
    """Create the SQLAlchemy engine for the configured database."""

    url = get_database_url()
    engine_kwargs = {"future": True}
    if url.startswith("sqlite"):
        engine_kwargs["connect_args"] = SQLITE_CONNECT_ARGS
    return create_engine(url, **engine_kwargs)


def get_session_factory():
    """Create a session factory bound to the active engine."""

    return sessionmaker(bind=get_engine(), autoflush=False, autocommit=False, expire_on_commit=False, future=True)


def create_schema() -> None:
    """Create database tables if they do not already exist."""

    Base.metadata.create_all(bind=get_engine())


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """Yield a transactional SQLAlchemy session."""

    SessionLocal = get_session_factory()
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def save_conversations(texts: Sequence[str], cluster_ids: Sequence[int] | None = None) -> list[Conversation]:
    """Persist a batch of conversations, optionally with cluster IDs."""

    if cluster_ids is not None and len(cluster_ids) != len(texts):
        raise ValueError("Conversation and cluster counts must match.")

    saved: list[Conversation] = []
    with session_scope() as session:
        for i, text in enumerate(texts):
            cid = None if cluster_ids is None else int(cluster_ids[i])
            row = Conversation(text=text, cluster_id=cid)
            session.add(row)
            session.flush()
            saved.append(row)
    return saved


def save_insights(cluster_results: Sequence[object]) -> list[Insight]:
    """Persist a batch of generated cluster insights."""

    saved: list[Insight] = []
    with session_scope() as session:
        for res in cluster_results:
            cid = int(getattr(res, "cluster_id"))
            text = str(getattr(res, "insight_text"))
            pct = float(getattr(res, "percentage"))
            row = Insight(cluster_id=cid, insight_text=text, percentage=pct)
            session.add(row)
            session.flush()
            saved.append(row)
    return saved


def fetch_all_insights() -> list[Insight]:
    """Return all stored insights ordered from newest to oldest."""

    with session_scope() as session:
        rows = session.query(Insight).order_by(Insight.created_at.desc(), Insight.id.desc()).all()
        return list(rows)
