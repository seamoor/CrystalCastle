from __future__ import annotations

from pathlib import Path

from app.models import IngestResponse
from app.watcher import WatchService


class IngestService:
    def __init__(self, watcher: WatchService) -> None:
        self.watcher = watcher

    def enqueue(self, path: str, force: bool = False) -> IngestResponse:
        p = Path(path)
        self.watcher.enqueue_path(p, force=force)
        return IngestResponse(accepted=True, path=str(p))
