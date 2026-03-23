from __future__ import annotations

import threading
import time

from app.models import DashboardStats
from app.pipeline.qdrant_store import QdrantStore
from app.storage.state import StateStore

_CACHE_TTL_SECONDS = 60


class DashboardService:
    def __init__(self, qdrant: QdrantStore, state_store: StateStore) -> None:
        self.qdrant = qdrant
        self.state_store = state_store
        self._lock = threading.Lock()
        self._cache: DashboardStats | None = None
        self._cache_ts: float = 0.0

    def stats(self) -> DashboardStats:
        now = time.monotonic()
        with self._lock:
            if self._cache is not None and (now - self._cache_ts) < _CACHE_TTL_SECONDS:
                return self._cache

        raw = self.qdrant.dashboard_stats(limit_recent=20)
        state_recent = self.state_store.recent(limit=20)
        if state_recent:
            raw["recent_uploads"] = state_recent
        result = DashboardStats(**raw)

        with self._lock:
            self._cache = result
            self._cache_ts = time.monotonic()
        return result

    def invalidate(self) -> None:
        with self._lock:
            self._cache = None
            self._cache_ts = 0.0
