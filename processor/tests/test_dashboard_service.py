from app.services.dashboard_service import DashboardService


class FakeQdrant:
    def dashboard_stats(self, limit_recent: int = 20):
        assert limit_recent == 20
        return {
            "document_count": 2,
            "total_indexed_duration_seconds": 120.0,
            "tag_distribution": {"ml": 3},
            "recent_uploads": [{"doc_id": "fallback", "filename": "fallback.pdf", "indexed_at": "x", "duration_seconds": 0.0}],
        }


class FakeState:
    def __init__(self, recent_items):
        self._recent_items = recent_items

    def recent(self, limit: int = 20):
        assert limit == 20
        return self._recent_items


def test_dashboard_service_prefers_state_recent_uploads() -> None:
    service = DashboardService(
        qdrant=FakeQdrant(),
        state_store=FakeState([{"doc_id": "doc-9", "filename": "latest.pdf", "indexed_at": "2026-02-20", "duration_seconds": 10.0, "status": "indexed"}]),
    )

    stats = service.stats()

    assert stats.document_count == 2
    assert stats.total_indexed_duration_seconds == 120.0
    assert stats.tag_distribution == {"ml": 3}
    assert stats.recent_uploads[0]["doc_id"] == "doc-9"


def test_dashboard_service_uses_qdrant_recent_when_state_empty() -> None:
    service = DashboardService(qdrant=FakeQdrant(), state_store=FakeState([]))
    stats = service.stats()
    assert stats.recent_uploads[0]["doc_id"] == "fallback"
