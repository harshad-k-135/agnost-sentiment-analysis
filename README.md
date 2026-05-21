# Agnost AI Sentiment Analytics Engine

Production-ready MVP that clusters user conversations and surfaces PM-ready insights.

## Quick start

1. Create a virtualenv and activate it:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. (Optional) Configure PostgreSQL:

```bash
export DATABASE_URL=postgresql+psycopg2://user:password@localhost:5432/agnost_sentiment
```

4. Run the API:

```bash
python main.py
```

The API will be available at `http://127.0.0.1:8000` and interactive docs at `/docs`.

## Endpoints

- `POST /analyze` — Accepts JSON `{ "conversations": [ ... ] }` and returns clusters and insights.
- `POST /conversations` — Persist raw conversations.
- `GET /insights` — Retrieve stored insights.

See `sample_data.json` for example input.

## Notes

- The app defaults to SQLite for quick local runs. Set `DATABASE_URL` to a PostgreSQL DSN to use Postgres in production.
- The clustering uses `sentence-transformers/all-MiniLM-L6-v2` for embeddings and K-means for clustering.
