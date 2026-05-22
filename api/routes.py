"""FastAPI routes for conversation storage, analysis, and cached insights."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from time import perf_counter
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool

from api.schemas import (
    AnalysisDataResponse,
    AnalysisMetadataResponse,
    AnalyzeRequest,
    AnalyzeResponse,
    BatchConversationRequest,
    CachedInsightsResponse,
    ClusterInsight,
    ConversationStoreResponse,
    HealthResponse,
    PaginationResponse,
    InsightRecord,
)
from clustering import ClusterSummary, analyze_conversations, normalize_conversations, score_sentiment
from config import SETTINGS
from db import create_schema, fetch_insights, ping_database, save_analysis_metadata, save_clusters, save_conversations, save_insights
from utils.cache import analysis_cache
from utils.errors import DatabaseOperationError
from utils.logger import get_logger


logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Ensure the database schema exists before the app starts serving."""

    await run_in_threadpool(create_schema)
    yield


app = FastAPI(title=SETTINGS.app_title, description=SETTINGS.app_description, version=SETTINGS.app_version, lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return a lightweight health signal for uptime checks."""

    database_ok = await run_in_threadpool(ping_database)
    if not database_ok:
        raise HTTPException(status_code=503, detail="Database is unavailable.")

    return HealthResponse(status="healthy", database="connected", model_loaded=True)


async def _run_analysis(conversations: list[str], num_clusters: int | None) -> dict[str, Any]:
    """Run the analysis pipeline in a worker thread."""

    return await run_in_threadpool(analyze_conversations, conversations, num_clusters)


@app.post("/api/v1/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """Analyze a batch of conversations and persist the resulting insights."""

    conversations = normalize_conversations(request.conversations)
    if not conversations:
        raise HTTPException(status_code=400, detail="At least one non-empty conversation is required.")
    if len(conversations) > SETTINGS.max_conversations_per_request:
        raise HTTPException(status_code=400, detail=f"A maximum of {SETTINGS.max_conversations_per_request} conversations is allowed.")

    cache_key = analysis_cache.make_key(conversations, request.num_clusters)
    cached_entry = None if request.force_recompute else analysis_cache.get(cache_key)
    if cached_entry is not None:
        logger.info("Serving analysis from in-memory cache for key %s", cache_key)
        return AnalyzeResponse(**cached_entry.payload)

    started_at = perf_counter()
    try:
        analysis = await _run_analysis(conversations, request.num_clusters)
        batch_id = str(uuid4())
        cluster_summaries = analysis["cluster_results"]
        labels = analysis["labels"]
        normalized_scores = [score_in_text for score_in_text in (score_sentiment(text) for text in conversations)]

        saved_conversations = await run_in_threadpool(
            save_conversations,
            conversations,
            labels,
            batch_id=batch_id,
            sentiment_scores=normalized_scores,
        )
        cluster_dataclasses = [ClusterSummary(**summary) for summary in cluster_summaries]
        saved_clusters = await run_in_threadpool(
            save_clusters,
            batch_id,
            cluster_dataclasses,
        )
        await run_in_threadpool(
            save_insights,
            saved_clusters,
            cluster_dataclasses,
        )
        metadata_row = await run_in_threadpool(
            save_analysis_metadata,
            batch_id,
            len(conversations),
            SETTINGS.embedding_model,
            "kmeans",
            analysis["optimal_k"],
            analysis["silhouette_score"],
            int((perf_counter() - started_at) * 1000),
        )

        response_payload = AnalyzeResponse(
            data=AnalysisDataResponse(
                total_conversations=len(conversations),
                clusters=[ClusterInsight(**summary) for summary in cluster_summaries],
                metadata=AnalysisMetadataResponse(
                    embedding_model=metadata_row.embedding_model,
                    clustering_algo=metadata_row.clustering_algo,
                    optimal_k=metadata_row.optimal_k,
                    silhouette_score=metadata_row.silhouette_score,
                    processing_time_ms=metadata_row.processing_time_ms,
                    processed_at=metadata_row.created_at,
                ),
            )
        )
        serialized_response = response_payload.model_dump() if hasattr(response_payload, "model_dump") else response_payload.dict()
        analysis_cache.set(cache_key, serialized_response)
        return response_payload
    except DatabaseOperationError as exc:
        logger.error("Database error while analyzing conversations: %s", exc)
        raise HTTPException(status_code=500, detail="Database error while processing the batch.") from exc
    except Exception as exc:
        logger.exception("Unexpected analysis failure")
        raise HTTPException(status_code=500, detail="Processing error while analyzing conversations.") from exc


@app.get("/api/v1/insights", response_model=CachedInsightsResponse)
async def get_cached_insights(limit: int = 10, offset: int = 0) -> CachedInsightsResponse:
    """Return the most recent stored insights with pagination metadata."""

    if limit < 1 or offset < 0:
        raise HTTPException(status_code=400, detail="Invalid pagination parameters.")

    insights, total, metadata = await run_in_threadpool(fetch_insights, limit, offset)
    cache_age_seconds = 0
    if metadata is not None:
        created_at = metadata.created_at
        if created_at.tzinfo is None:
            cache_age_seconds = max(0, int((datetime.now() - created_at).total_seconds()))
        else:
            cache_age_seconds = max(0, int((datetime.now(timezone.utc) - created_at).total_seconds()))

    return CachedInsightsResponse(
        insights=[InsightRecord(**row) for row in insights],
        pagination=PaginationResponse(limit=limit, offset=offset, total=total),
        cached=True,
        cache_age_seconds=cache_age_seconds,
    )


@app.post("/api/v1/conversations/batch", status_code=201, response_model=ConversationStoreResponse)
async def store_conversations(request: BatchConversationRequest) -> ConversationStoreResponse:
    """Persist raw conversations without running clustering."""

    conversations = normalize_conversations(request.conversations)
    if not conversations:
        raise HTTPException(status_code=400, detail="At least one non-empty conversation is required.")

    saved_rows = await run_in_threadpool(save_conversations, conversations, source=request.source)
    return ConversationStoreResponse(stored_count=len(saved_rows), conversation_ids=[row.id for row in saved_rows])


@app.post("/analyze", response_model=AnalyzeResponse)
async def legacy_analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """Backward-compatible alias for the original endpoint path."""

    return await analyze(request)


@app.post("/conversations", response_model=ConversationStoreResponse, status_code=201)
async def legacy_store_conversations(request: BatchConversationRequest) -> ConversationStoreResponse:
    """Backward-compatible alias for the original storage endpoint."""

    return await store_conversations(request)


@app.get("/insights", response_model=CachedInsightsResponse)
async def legacy_get_insights(limit: int = 10, offset: int = 0) -> CachedInsightsResponse:
    """Backward-compatible alias for the original insights path."""

    return await get_cached_insights(limit=limit, offset=offset)
