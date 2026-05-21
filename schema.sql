CREATE TABLE IF NOT EXISTS conversations (
    id SERIAL PRIMARY KEY,
    text TEXT NOT NULL,
    cluster_id INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS insights (
    id SERIAL PRIMARY KEY,
    cluster_id INTEGER NOT NULL,
    insight_text TEXT NOT NULL,
    percentage DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
