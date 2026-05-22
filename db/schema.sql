-- Reference schema for the sentiment analytics database.
-- The ORM mirrors this layout so PostgreSQL and SQLite can both be used in development.

CREATE TABLE IF NOT EXISTS conversations (
    id SERIAL PRIMARY KEY,
    text TEXT NOT NULL,
    source VARCHAR(50),
    batch_id VARCHAR(36),
    cluster_id INTEGER,
    sentiment_score DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_conversations_cluster_id ON conversations (cluster_id);
CREATE INDEX IF NOT EXISTS idx_conversations_created_at ON conversations (created_at DESC);

CREATE TABLE IF NOT EXISTS clusters (
    id SERIAL PRIMARY KEY,
    batch_id VARCHAR(36) NOT NULL,
    topic VARCHAR(255) NOT NULL,
    topic_keywords JSON NOT NULL,
    size INTEGER NOT NULL,
    percentage DOUBLE PRECISION NOT NULL,
    dominant_sentiment VARCHAR(20) NOT NULL,
    sample_conversations JSON NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_clusters_batch_id ON clusters (batch_id);
CREATE INDEX IF NOT EXISTS idx_clusters_created_at ON clusters (created_at DESC);

CREATE TABLE IF NOT EXISTS insights (
    id SERIAL PRIMARY KEY,
    cluster_id INTEGER NOT NULL REFERENCES clusters(id),
    insight_text TEXT NOT NULL,
    metric_type VARCHAR(50) NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    sample_conversation_ids JSON NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_insights_cluster_id ON insights (cluster_id);
CREATE INDEX IF NOT EXISTS idx_insights_created_at ON insights (created_at DESC);

CREATE TABLE IF NOT EXISTS analysis_metadata (
    id SERIAL PRIMARY KEY,
    batch_id VARCHAR(36) UNIQUE NOT NULL,
    total_conversations INTEGER NOT NULL,
    embedding_model VARCHAR(100) NOT NULL,
    clustering_algo VARCHAR(50) NOT NULL,
    optimal_k INTEGER NOT NULL,
    silhouette_score DOUBLE PRECISION NOT NULL,
    processing_time_ms INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_analysis_metadata_created_at ON analysis_metadata (created_at DESC);
