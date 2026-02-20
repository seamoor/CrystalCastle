from __future__ import annotations

import queue
from pathlib import Path

from app.watcher import WatchHandler, WatchService


class Event:
    def __init__(self, src_path: str, is_directory: bool = False) -> None:
        self.src_path = src_path
        self.is_directory = is_directory


class FakeStateStore:
    def __init__(self, seen: set[str] | None = None) -> None:
        self.seen = seen or set()

    def seen_success(self, path: Path) -> bool:
        return str(path) in self.seen


class FakeOrchestrator:
    def __init__(self) -> None:
        self.processed: list[Path] = []

    def process_file(self, path: Path) -> None:
        self.processed.append(path)


def test_watch_handler_enqueues_only_supported_files(tmp_path: Path) -> None:
    q: queue.Queue[Path] = queue.Queue()
    handler = WatchHandler(q, supported_ext={".pdf", ".mp4"})

    handler.on_created(Event(str(tmp_path / "a.pdf")))
    handler.on_created(Event(str(tmp_path / "a.txt")))
    handler.on_created(Event(str(tmp_path / "folder"), is_directory=True))

    queued = [q.get_nowait()]
    assert queued[0].suffix == ".pdf"
    assert q.empty()


def test_enqueue_existing_files_skips_already_indexed(tmp_path: Path) -> None:
    pdf_new = tmp_path / "new.pdf"
    pdf_old = tmp_path / "old.pdf"
    txt = tmp_path / "skip.txt"
    pdf_new.write_text("x", encoding="utf-8")
    pdf_old.write_text("x", encoding="utf-8")
    txt.write_text("x", encoding="utf-8")

    state = FakeStateStore(seen={str(pdf_old)})
    svc = WatchService(
        watch_dir=tmp_path,
        orchestrator=FakeOrchestrator(),
        state_store=state,
        supported_extensions=[".pdf"],
    )

    svc._enqueue_existing_files()

    queued = [svc.ingest_queue.get_nowait()]
    assert queued == [pdf_new]
    assert svc.ingest_queue.empty()
