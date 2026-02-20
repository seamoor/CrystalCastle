from __future__ import annotations

from app.models import DashboardStats
from app.pipeline.qdrant_store import QdrantStore
from app.storage.state import StateStore


class DashboardService:
    def __init__(self, qdrant: QdrantStore, state_store: StateStore) -> None:
        self.qdrant = qdrant
        self.state_store = state_store

    def stats(self) -> DashboardStats:
        stats = self.qdrant.dashboard_stats(limit_recent=20)
        state_recent = self.state_store.recent(limit=20)
        if state_recent:
            stats["recent_uploads"] = state_recent
        return DashboardStats(**stats)
