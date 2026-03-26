from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from app.models import IndexedDocument
from app.storage.state import StateStore


def test_state_store_upsert_and_seen_success(tmp_path: Path) -> None:
    store = StateStore(tmp_path)

    doc = IndexedDocument(
        doc_id="doc-1",
        filename="deck.pdf",
        path="/watch/deck.pdf",
        indexed_at=datetime.now(timezone.utc),
        status="indexed",
        tags=["sales"],
        speakers=["Speaker 1"],
    )
    store.upsert(doc)

    assert store.seen_success(Path("/watch/deck.pdf")) is True


def test_state_store_recent_sorted(tmp_path: Path) -> None:
    store = StateStore(tmp_path)

    older = IndexedDocument(
        doc_id="doc-old",
        filename="old.pdf",
        path="/watch/old.pdf",
        indexed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        status="indexed",
    )
    newer = IndexedDocument(
        doc_id="doc-new",
        filename="new.pdf",
        path="/watch/new.pdf",
        indexed_at=datetime(2026, 2, 1, tzinfo=timezone.utc),
        status="processing",
    )
    store.upsert(older)
    store.upsert(newer)

    recent = store.recent(limit=2)
    assert len(recent) == 2
    assert recent[0]["doc_id"] == "doc-new"
    assert recent[1]["doc_id"] == "doc-old"
