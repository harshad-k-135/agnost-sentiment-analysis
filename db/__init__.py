"""Database package exports and persistence helpers."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import func

from config import SETTINGS
from clustering.insight_extractor import ClusterSummary, cluster_summary_to_dict
from utils.errors import DatabaseOperationError
from utils.logger import get_logger

from .connection import get_engine, get_session_factory, ping_database, session_scope
from .models import AnalysisMetadata, Base, Cluster, Conversation, Insight


logger = get_logger(__name__)


def create_schema() -> None:
    """Create all database tables if they do not already exist."""

    try:
        Base.metadata.create_all(bind=get_engine())
        logger.info("Database schema ensured")
    except Exception as exc:  # pragma: no cover - defensive logging path
        logger.error("Failed to create schema: %s", exc)
        raise DatabaseOperationError(str(exc)) from exc


def save_conversations(
    texts: Sequence[str],
    cluster_ids: Sequence[int] | None = None,
    *,
    source: str | None = None,
    batch_id: str | None = None,
    sentiment_scores: Sequence[float] | None = None,
) -> list[Conversation]:
    """Persist a batch of conversations and optional analysis annotations."""

    create_schema()
    if cluster_ids is not None and len(cluster_ids) != len(texts):
        raise ValueError("Conversation and cluster counts must match.")
    if sentiment_scores is not None and len(sentiment_scores) != len(texts):
        raise ValueError("Conversation and sentiment counts must match.")

    saved: list[Conversation] = []
    with session_scope() as session:
        for index, text in enumerate(texts):
            row = Conversation(
                text=text,
                source=source,
                batch_id=batch_id,
                cluster_id=None if cluster_ids is None else int(cluster_ids[index]),
                sentiment_score=None if sentiment_scores is None else float(sentiment_scores[index]),
            )
            session.add(row)
            session.flush()
            saved.append(row)
    return saved


def save_clusters(batch_id: str, summaries: Sequence[ClusterSummary]) -> list[Cluster]:
    """Persist cluster summaries for a completed analysis batch."""

    create_schema()
    saved: list[Cluster] = []
    with session_scope() as session:
        for summary in summaries:
            row = Cluster(
                batch_id=batch_id,
                topic=summary.topic,
                topic_keywords=list(summary.key_phrases),
                size=summary.size,
                percentage=summary.percentage,
                dominant_sentiment=summary.sentiment,
                sample_conversations=list(summary.sample_conversations),
            )
            session.add(row)
            session.flush()
            saved.append(row)
    return saved


def save_insights(cluster_rows: Sequence[Cluster], summaries: Sequence[ClusterSummary]) -> list[Insight]:
    """Persist quantified insights associated with cluster rows."""

    create_schema()
    if len(cluster_rows) != len(summaries):
        raise ValueError("Cluster and summary counts must match.")

    saved: list[Insight] = []
    with session_scope() as session:
        for cluster_row, summary in zip(cluster_rows, summaries):
            sample_ids = list(range(1, len(summary.sample_conversations) + 1))
            row = Insight(
                cluster_id=cluster_row.id,
                insight_text=summary.insight_text,
                metric_type=summary.metric_type,
                metric_value=summary.metric_value,
                confidence=summary.confidence,
                sample_conversation_ids=sample_ids,
            )
            session.add(row)
            session.flush()
            saved.append(row)
    return saved


def save_analysis_metadata(
    batch_id: str,
    total_conversations: int,
    embedding_model: str,
    clustering_algo: str,
    optimal_k: int,
    silhouette_score: float,
    processing_time_ms: int,
) -> AnalysisMetadata:
    """Persist analysis metadata for cache and auditing purposes."""

    create_schema()
    with session_scope() as session:
        row = AnalysisMetadata(
            batch_id=batch_id,
            total_conversations=total_conversations,
            embedding_model=embedding_model,
            clustering_algo=clustering_algo,
            optimal_k=optimal_k,
            silhouette_score=silhouette_score,
            processing_time_ms=processing_time_ms,
        )
        session.add(row)
        session.flush()
        return row


def fetch_latest_metadata() -> AnalysisMetadata | None:
    """Return the latest analysis metadata row, if one exists."""

    create_schema()
    with session_scope() as session:
        return session.query(AnalysisMetadata).order_by(AnalysisMetadata.created_at.desc(), AnalysisMetadata.id.desc()).first()


def fetch_insights(limit: int, offset: int) -> tuple[list[dict[str, Any]], int, AnalysisMetadata | None]:
    """Return paginated insights for the latest analysis batch."""

    create_schema()
    with session_scope() as session:
        latest_metadata = session.query(AnalysisMetadata).order_by(AnalysisMetadata.created_at.desc(), AnalysisMetadata.id.desc()).first()
        if latest_metadata is None:
            return [], 0, None

        base_query = (
            session.query(Insight, Cluster)
            .join(Cluster, Insight.cluster_id == Cluster.id)
            .filter(Cluster.batch_id == latest_metadata.batch_id)
            .order_by(Insight.created_at.desc(), Insight.id.desc())
        )
        total = base_query.count()
        rows = base_query.offset(offset).limit(limit).all()

        insights: list[dict[str, Any]] = []
        for insight, cluster in rows:
            insights.append(
                {
                    "id": insight.id,
                    "cluster_id": insight.cluster_id,
                    "batch_id": cluster.batch_id,
                    "topic": cluster.topic,
                    "dominant_sentiment": cluster.dominant_sentiment,
                    "insight_text": insight.insight_text,
                    "metric_type": insight.metric_type,
                    "metric_value": insight.metric_value,
                    "confidence": insight.confidence,
                    "sample_conversation_ids": list(insight.sample_conversation_ids or []),
                    "created_at": insight.created_at,
                }
            )
        return insights, total, latest_metadata
