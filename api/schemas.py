"""Pydantic request and response schemas for the public API."""

from __future__ import annotations

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, validator


class AnalyzeRequest(BaseModel):
    """Request payload for conversation analysis."""

    conversations: List[str] = Field(...)
    force_recompute: bool = False
    num_clusters: Optional[int] = None

    @validator("conversations")
    def validate_conversations(cls, value: List[str]) -> List[str]:
        """Strip empty rows and ensure at least one conversation exists."""

        cleaned = [" ".join(str(item).split()).strip() for item in value if str(item).strip()]
        if not cleaned:
            raise ValueError("At least one non-empty conversation is required.")
        return cleaned

    @validator("num_clusters")
    def validate_num_clusters(cls, value: Optional[int]) -> Optional[int]:
        """Ensure manual cluster counts are positive when provided."""

        if value is not None and value < 1:
            raise ValueError("num_clusters must be positive.")
        return value


class BatchConversationRequest(BaseModel):
    """Request payload for storing a batch of raw conversations."""

    conversations: List[str] = Field(...)
    source: Optional[str] = None

    @validator("conversations")
    def validate_conversations(cls, value: List[str]) -> List[str]:
        """Strip empty rows and ensure at least one conversation exists."""

        cleaned = [" ".join(str(item).split()).strip() for item in value if str(item).strip()]
        if not cleaned:
            raise ValueError("At least one non-empty conversation is required.")
        return cleaned


class ClusterInsight(BaseModel):
    """Cluster summary returned by the analysis endpoint."""

    cluster_id: int
    topic: str
    size: int
    percentage: float
    sentiment: str
    key_phrases: List[str]
    sample_conversations: List[str]
    insight_text: str
    confidence: float


class AnalysisMetadataResponse(BaseModel):
    """Metadata about the analysis batch."""

    embedding_model: str
    clustering_algo: str
    optimal_k: int
    silhouette_score: float
    processing_time_ms: int
    processed_at: datetime


class AnalysisDataResponse(BaseModel):
    """Envelope returned inside the analysis response."""

    total_conversations: int
    clusters: List[ClusterInsight]
    metadata: AnalysisMetadataResponse


class AnalyzeResponse(BaseModel):
    """Top-level response model for POST /api/v1/analyze."""

    status: Literal["success"] = "success"
    data: AnalysisDataResponse


class PaginationResponse(BaseModel):
    """Pagination metadata for cached insight retrieval."""

    limit: int
    offset: int
    total: int


class InsightRecord(BaseModel):
    """Stored insight row returned by GET /api/v1/insights."""

    id: int
    cluster_id: int
    batch_id: str
    topic: str
    dominant_sentiment: str
    insight_text: str
    metric_type: str
    metric_value: float
    confidence: float
    sample_conversation_ids: List[int]
    created_at: datetime


class CachedInsightsResponse(BaseModel):
    """Response model for cached insight retrieval."""

    insights: List[InsightRecord]
    pagination: PaginationResponse
    cached: bool
    cache_age_seconds: int


class ConversationStoreResponse(BaseModel):
    """Response model for batch conversation storage."""

    stored_count: int
    conversation_ids: List[int]


class HealthResponse(BaseModel):
    """Response model for the health endpoint."""

    status: Literal["healthy"]
    database: Literal["connected"]
    model_loaded: bool
