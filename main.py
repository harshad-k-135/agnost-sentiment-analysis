"""Application entry point for the sentiment analytics engine."""

from __future__ import annotations

from api.routes import app

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000)

