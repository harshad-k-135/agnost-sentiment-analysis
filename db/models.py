"""SQLAlchemy ORM models for conversations, clusters, insights, and analysis metadata."""

from __future__ import annotations

from datetime import datetime
from typing import Any, List

from sqlalchemy import DateTime, Float, Index, Integer, JSON, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base for all database tables."""


class Conversation(Base):
    """Immutable conversation log plus analysis annotations."""

    __tablename__ = "conversations"
    __table_args__ = (
        Index("idx_conversations_cluster_id", "cluster_id"),
        Index("idx_conversations_created_at", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[str | None] = mapped_column(String(50), nullable=True)
    batch_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    cluster_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sentiment_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


class Cluster(Base):
    """Cluster snapshot created from one analysis batch."""

    __tablename__ = "clusters"
    __table_args__ = (
        Index("idx_clusters_batch_id", "batch_id"),
        Index("idx_clusters_created_at", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    batch_id: Mapped[str] = mapped_column(String(36), nullable=False)
    topic: Mapped[str] = mapped_column(String(255), nullable=False)
    topic_keywords: Mapped[List[str]] = mapped_column(JSON, nullable=False, default=list)
    size: Mapped[int] = mapped_column(Integer, nullable=False)
    percentage: Mapped[float] = mapped_column(Float, nullable=False)
    dominant_sentiment: Mapped[str] = mapped_column(String(20), nullable=False)
    sample_conversations: Mapped[List[str]] = mapped_column(JSON, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class Insight(Base):
    """Quantified insight derived from a cluster."""

    __tablename__ = "insights"
    __table_args__ = (
        Index("idx_insights_cluster_id", "cluster_id"),
        Index("idx_insights_created_at", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    cluster_id: Mapped[int] = mapped_column(Integer, nullable=False)
    insight_text: Mapped[str] = mapped_column(Text, nullable=False)
    metric_type: Mapped[str] = mapped_column(String(50), nullable=False)
    metric_value: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    sample_conversation_ids: Mapped[List[int]] = mapped_column(JSON, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class AnalysisMetadata(Base):
    """Metadata describing one completed analysis batch."""

    __tablename__ = "analysis_metadata"
    __table_args__ = (
        Index("idx_analysis_metadata_batch_id", "batch_id", unique=True),
        Index("idx_analysis_metadata_created_at", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    batch_id: Mapped[str] = mapped_column(String(36), nullable=False)
    total_conversations: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(100), nullable=False)
    clustering_algo: Mapped[str] = mapped_column(String(50), nullable=False)
    optimal_k: Mapped[int] = mapped_column(Integer, nullable=False)
    silhouette_score: Mapped[float] = mapped_column(Float, nullable=False)
    processing_time_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
