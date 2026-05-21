"""FastAPI application for sentiment clustering and insight generation.

Run with:
    python main.py

By default this uses SQLite for local runs. Set `DATABASE_URL` to use PostgreSQL.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from clustering import analyze_conversations, ClusterResult
from db import Conversation, Insight, create_schema, fetch_all_insights, save_conversations, save_insights


APP_TITLE = "Agnost AI Sentiment Analytics Engine"
APP_DESCRIPTION = "Cluster user conversations and surface PM-ready insights."
APP_VERSION = "1.0.0"
MAX_CONVERSATIONS_PER_REQUEST = 500


class AnalyzeRequest(BaseModel):
    """Request body for `/analyze`.

    The endpoint accepts a batch of raw conversation strings and returns
    clustered insights for product and research teams.
    """

    conversations: List[str] = Field(..., min_length=1)


class ConversationStoreRequest(BaseModel):
    """Request body for `/conversations`.

    This endpoint only persists conversation text without running the
    clustering pipeline.
    """

    conversations: List[str] = Field(..., min_length=1)


class ClusterInsight(BaseModel):
    """Cluster summary returned by the analysis endpoint."""

    cluster_id: int
    size: int
    percentage: float
    top_keywords: List[str]
    insight_text: str
    sample_conversations: List[str]


class AnalyzeResponse(BaseModel):
    """Response schema for `/analyze`."""

    total_conversations: int
    cluster_count: int
    clusters: List[ClusterInsight]


class InsightItem(BaseModel):
    """Persisted insight record returned by `/insights`."""

    id: int
    cluster_id: int
    insight_text: str
    percentage: float
    created_at: datetime


class InsightsResponse(BaseModel):
    """Response schema for `/insights`."""

    total_insights: int
    insights: List[InsightItem]


class ConversationResponse(BaseModel):
    """Persisted conversation record returned by `/conversations`."""

    id: int
    text: str
    cluster_id: int | None
    created_at: datetime


class ConversationStoreResponse(BaseModel):
    """Response schema for `/conversations`."""

    total_conversations: int
    conversations: List[ConversationResponse]


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Create the database schema before the app starts serving requests."""

    create_schema()
    yield


app = FastAPI(title=APP_TITLE, description=APP_DESCRIPTION, version=APP_VERSION, lifespan=lifespan)


@app.get("/health")
def health_check() -> dict:
    """Return a lightweight health signal for uptime and smoke checks."""

    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """Cluster a batch of conversations and persist the generated insights."""

    conversations = [c.strip() for c in request.conversations if c and c.strip()]
    if not conversations:
        raise HTTPException(status_code=400, detail="At least one non-empty conversation is required.")
    if len(conversations) > MAX_CONVERSATIONS_PER_REQUEST:
        raise HTTPException(status_code=400, detail=f"A maximum of {MAX_CONVERSATIONS_PER_REQUEST} conversations is allowed.")

    analysis = analyze_conversations(conversations)
    labels = analysis.get("labels", [])
    cluster_results: List[ClusterResult] = analysis.get("cluster_results", [])

    # Persist conversations with their assigned cluster ids
    save_conversations(conversations, labels)

    # Persist insights
    save_insights(cluster_results)

    clusters_payload = [
        ClusterInsight(
            cluster_id=cr.cluster_id,
            size=cr.size,
            percentage=cr.percentage,
            top_keywords=cr.top_keywords,
            insight_text=cr.insight_text,
            sample_conversations=cr.sample_conversations,
        )
        for cr in cluster_results
    ]

    return AnalyzeResponse(total_conversations=len(conversations), cluster_count=len(clusters_payload), clusters=clusters_payload)


@app.post("/conversations", response_model=ConversationStoreResponse)
def store_conversations(request: ConversationStoreRequest) -> ConversationStoreResponse:
    """Persist raw conversations without clustering them."""

    conversations = [c.strip() for c in request.conversations if c and c.strip()]
    if not conversations:
        raise HTTPException(status_code=400, detail="At least one non-empty conversation is required.")
    if len(conversations) > MAX_CONVERSATIONS_PER_REQUEST:
        raise HTTPException(status_code=400, detail=f"A maximum of {MAX_CONVERSATIONS_PER_REQUEST} conversations is allowed.")

    saved = save_conversations(conversations)
    response_rows = [ConversationResponse(id=row.id, text=row.text, cluster_id=row.cluster_id, created_at=row.created_at) for row in saved]
    return ConversationStoreResponse(total_conversations=len(response_rows), conversations=response_rows)


@app.get("/insights", response_model=InsightsResponse)
def get_insights() -> InsightsResponse:
    """Return all stored insights in reverse chronological order."""

    rows = fetch_all_insights()
    insights = [
        InsightItem(id=row.id, cluster_id=row.cluster_id, insight_text=row.insight_text, percentage=row.percentage, created_at=row.created_at)
        for row in rows
    ]
    return InsightsResponse(total_insights=len(insights), insights=insights)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000)
