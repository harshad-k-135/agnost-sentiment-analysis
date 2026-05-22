"""Application configuration and shared constants for the sentiment analytics service."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class Settings:
    """Runtime configuration loaded from environment variables.

    Attributes:
        app_title: Human-readable FastAPI title.
        app_description: Short description of the service.
        app_version: Semantic version string.
        embedding_model: Sentence-transformer used for embeddings.
        database_url: SQLAlchemy database URL.
        max_conversations_per_request: Hard limit for batch endpoints.
        default_max_clusters: Upper bound for automatic k selection.
        sample_conversation_limit: Number of example conversations per cluster.
        cache_ttl_seconds: Retention window for in-memory analysis cache entries.
    """

    app_title: str = os.getenv("APP_TITLE", "Agnost AI Sentiment Analytics Engine")
    app_description: str = os.getenv(
        "APP_DESCRIPTION",
        "Cluster conversations, surface sentiment patterns, and generate quantified product insights.",
    )
    app_version: str = os.getenv("APP_VERSION", "2.0.0")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./agnost_sentiment.db")
    max_conversations_per_request: int = int(os.getenv("MAX_CONVERSATIONS_PER_REQUEST", "500"))
    default_max_clusters: int = int(os.getenv("DEFAULT_MAX_CLUSTERS", "10"))
    sample_conversation_limit: int = int(os.getenv("SAMPLE_CONVERSATION_LIMIT", "3"))
    cache_ttl_seconds: int = int(os.getenv("CACHE_TTL_SECONDS", "3600"))


SETTINGS = Settings()
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
