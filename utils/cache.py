"""In-memory cache for repeated analysis requests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from threading import RLock
from typing import Any, Dict, Sequence


@dataclass(slots=True)
class CacheEntry:
    """Stored analysis payload plus the timestamp it was cached."""

    payload: Dict[str, Any]
    created_at: datetime


class AnalysisCache:
    """Small thread-safe cache keyed by a deterministic batch hash."""

    def __init__(self) -> None:
        self._entries: Dict[str, CacheEntry] = {}
        self._lock = RLock()

    def make_key(self, conversations: Sequence[str], num_clusters: int | None) -> str:
        """Build a stable cache key for a batch of conversations.

        Args:
            conversations: Cleaned conversation strings.
            num_clusters: Optional manual cluster override.

        Returns:
            A SHA-256 digest that uniquely identifies the request payload.
        """

        payload = json.dumps({"conversations": list(conversations), "num_clusters": num_clusters}, ensure_ascii=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> CacheEntry | None:
        """Return a cached entry if present."""

        with self._lock:
            return self._entries.get(key)

    def set(self, key: str, payload: Dict[str, Any]) -> None:
        """Store a payload under the given key."""

        with self._lock:
            self._entries[key] = CacheEntry(payload=payload, created_at=datetime.now(timezone.utc))

    def age_seconds(self, key: str) -> int | None:
        """Return the cached entry age in whole seconds."""

        entry = self.get(key)
        if entry is None:
            return None
        delta = datetime.now(timezone.utc) - entry.created_at
        return max(0, int(delta.total_seconds()))


analysis_cache = AnalysisCache()
