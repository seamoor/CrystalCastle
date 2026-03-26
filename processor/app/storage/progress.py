from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any


class ProgressStore:
    STAGE_WEIGHTS: dict[str, float] = {
        "extract_audio": 0.05,
        "transcription": 0.40,
        "ocr_sampling": 0.10,
        "ocr_inference": 0.15,
        "vision_inference": 0.10,
        "chunking": 0.05,
        "embedding": 0.05,
        "summary_tagging": 0.05,
        "upsert": 0.05,
    }

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}

    def start(self, doc_id: str, filename: str, path: str, file_type: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._jobs[doc_id] = {
                "doc_id": doc_id,
                "filename": filename,
                "path": path,
                "file_type": file_type,
                "status": "processing",
                "started_at": now,
                "updated_at": now,
                "stage": "queued",
                "stage_progress": 0.0,
                "overall_progress": 0.0,
                "details": {},
                "error": None,
            }

    def update(
        self,
        doc_id: str,
        stage: str,
        stage_progress: float,
        details: dict[str, Any] | None = None,
    ) -> float | None:
        with self._lock:
            job = self._jobs.get(doc_id)
            if not job:
                return None
            stage_progress = max(0.0, min(1.0, stage_progress))
            job["stage"] = stage
            job["stage_progress"] = stage_progress
            job["details"] = details or {}
            job["overall_progress"] = self._overall_from_stage(stage, stage_progress, job.get("file_type", "media"))
            job["updated_at"] = datetime.now(timezone.utc).isoformat()
            return float(job["overall_progress"])

    def complete(self, doc_id: str, status: str, error: str | None = None) -> None:
        with self._lock:
            job = self._jobs.get(doc_id)
            if not job:
                return
            job["status"] = status
            job["error"] = error
            job["overall_progress"] = 1.0 if status == "indexed" else job.get("overall_progress", 0.0)
            job["updated_at"] = datetime.now(timezone.utc).isoformat()

    def all(self) -> list[dict[str, Any]]:
        with self._lock:
            return sorted(
                (v.copy() for v in self._jobs.values()),
                key=lambda x: x.get("updated_at", ""),
                reverse=True,
            )

    def by_filename(self, filename: str) -> list[dict[str, Any]]:
        target = _normalize(filename)
        with self._lock:
            matches = [v.copy() for v in self._jobs.values() if _normalize(str(v.get("filename", ""))) == target]
        return sorted(matches, key=lambda x: x.get("updated_at", ""), reverse=True)

    def _overall_from_stage(self, stage: str, stage_progress: float, file_type: str) -> float:
        if file_type in {"pdf", "pptx"}:
            mapping = {
                "extract": (0.0, 0.60),
                "chunking": (0.60, 0.72),
                "embedding": (0.72, 0.84),
                "summary_tagging": (0.84, 0.94),
                "upsert": (0.94, 1.00),
            }
            start, end = mapping.get(stage, (0.0, 0.05))
            return start + (end - start) * stage_progress

        order = [
            "extract_audio",
            "transcription",
            "ocr_sampling",
            "ocr_inference",
            "vision_inference",
            "chunking",
            "embedding",
            "summary_tagging",
            "upsert",
        ]
        if stage not in self.STAGE_WEIGHTS:
            return 0.0
        done = 0.0
        for s in order:
            w = self.STAGE_WEIGHTS[s]
            if s == stage:
                done += w * stage_progress
                break
            done += w
        return min(1.0, done)


def _normalize(name: str) -> str:
    return " ".join(name.strip().lower().split())
