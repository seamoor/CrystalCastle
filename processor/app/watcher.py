from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from app.pipeline.orchestrator import PipelineOrchestrator
from app.storage.state import StateStore

logger = logging.getLogger(__name__)


class WatchHandler(FileSystemEventHandler):
    def __init__(self, ingest_queue: queue.Queue[Path], supported_ext: set[str]) -> None:
        self.ingest_queue = ingest_queue
        self.supported_ext = supported_ext

    def on_created(self, event) -> None:  # type: ignore[override]
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix.lower() in self.supported_ext:
            self.ingest_queue.put(path)


class WatchService:
    def __init__(
        self,
        watch_dir: Path,
        orchestrator: PipelineOrchestrator,
        state_store: StateStore,
        supported_extensions: list[str],
        poll_interval_seconds: float = 2.0,
    ) -> None:
        self.watch_dir = watch_dir
        self.orchestrator = orchestrator
        self.state_store = state_store
        self.supported_extensions = {e.lower() for e in supported_extensions}
        self.poll_interval_seconds = poll_interval_seconds
        self.ingest_queue: queue.Queue[Path] = queue.Queue()
        self.observer: Observer | None = None
        self.worker_thread: threading.Thread | None = None
        self.stop_event = threading.Event()

    def start(self) -> None:
        self.watch_dir.mkdir(parents=True, exist_ok=True)
        self._enqueue_existing_files()

        handler = WatchHandler(self.ingest_queue, self.supported_extensions)
        self.observer = Observer()
        self.observer.schedule(handler, str(self.watch_dir), recursive=True)
        self.observer.start()

        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        logger.info("Watcher started at %s", self.watch_dir)

    def stop(self) -> None:
        self.stop_event.set()
        if self.observer:
            self.observer.stop()
            self.observer.join(timeout=5)
        if self.worker_thread:
            self.worker_thread.join(timeout=5)

    def enqueue_path(self, path: Path) -> None:
        self.ingest_queue.put(path)

    def _enqueue_existing_files(self) -> None:
        for path in self.watch_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in self.supported_extensions:
                if not self.state_store.seen_success(path):
                    self.ingest_queue.put(path)

    def _worker_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                path = self.ingest_queue.get(timeout=self.poll_interval_seconds)
            except queue.Empty:
                continue

            if not path.exists() or path.suffix.lower() not in self.supported_extensions:
                continue

            if self.state_store.seen_success(path):
                continue

            try:
                self._wait_until_stable(path)
                self.orchestrator.process_file(path)
            except Exception:  # noqa: BLE001
                logger.exception("Queue worker failed for %s", path)
            finally:
                self.ingest_queue.task_done()

    @staticmethod
    def _wait_until_stable(path: Path, checks: int = 3, wait_seconds: float = 1.0) -> None:
        last_size = -1
        stable = 0
        while stable < checks:
            size = path.stat().st_size
            if size == last_size:
                stable += 1
            else:
                stable = 0
                last_size = size
            time.sleep(wait_seconds)
