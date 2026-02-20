from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path

from app.config import AppConfig
from app.models import IndexedDocument
from app.pipeline.chunking import chunk_text
from app.pipeline.diarization import DiarizationService
from app.pipeline.embeddings import EmbeddingService
from app.pipeline.file_types import classify_file
from app.pipeline.llm import LLMService
from app.pipeline.loaders import extract_pdf_text, extract_pptx_text
from app.pipeline.media import MediaProcessor
from app.pipeline.qdrant_store import QdrantStore
from app.storage.progress import ProgressStore
from app.storage.state import StateStore

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    def __init__(self, cfg: AppConfig, state_store: StateStore, progress_store: ProgressStore) -> None:
        self.cfg = cfg
        self.state_store = state_store
        self.progress_store = progress_store

        self.embedding = EmbeddingService(
            model_name=cfg.embedding.model_name,
            normalize=cfg.embedding.normalize_embeddings,
        )
        self.qdrant = QdrantStore(
            host=cfg.qdrant.host,
            port=cfg.qdrant.port,
            collection_name=cfg.qdrant.collection_name,
            vector_size=self.embedding.vector_size(),
        )
        self.llm = LLMService(
            base_url=cfg.llm.base_url,
            model=cfg.llm.model,
            enabled=cfg.llm.enabled,
            timeout_seconds=cfg.llm.timeout_seconds,
        )
        self.diarization = DiarizationService(
            model_path=cfg.diarization.model_path,
            enabled=cfg.diarization.enabled,
            gpu_enabled=cfg.processor.gpu_enabled,
        )
        self.media = MediaProcessor(
            data_dir=Path(cfg.processor.data_dir),
            whisper_model_size=cfg.whisper.model_size,
            whisper_device=cfg.whisper.device,
            whisper_compute_type=cfg.whisper.compute_type,
            ocr_enabled=cfg.ocr.enabled,
            ocr_fps=cfg.ocr.fps,
            slide_change_threshold=cfg.ocr.slide_change_threshold,
            ocr_languages=cfg.ocr.languages,
            diarization_service=self.diarization,
        )

    def process_file(self, file_path: Path, force: bool = False) -> None:
        ext_type = classify_file(file_path)
        if ext_type == "unsupported":
            logger.info("Skipping unsupported file: %s", file_path)
            return

        if force:
            logger.info("Force reprocess requested: path=%s. Cleaning previous index entries.", file_path)
            self.qdrant.delete_by_path(str(file_path))
            self.state_store.delete_by_path(file_path)

        doc_id = str(uuid.uuid4())
        now = datetime.now(UTC)
        self.progress_store.start(doc_id, file_path.name, str(file_path), ext_type)
        logger.info(
            "Indexing started: doc_id=%s path=%s type=%s",
            doc_id,
            file_path,
            ext_type,
        )
        self.state_store.upsert(
            IndexedDocument(
                doc_id=doc_id,
                filename=file_path.name,
                path=str(file_path),
                indexed_at=now,
                status="processing",
            )
        )

        try:
            logger.info("Extract phase started: doc_id=%s path=%s", doc_id, file_path)
            self.progress_store.update(doc_id, "extract", 0.0, {"path": str(file_path)})
            result = self._extract(file_path, ext_type, doc_id)
            self.progress_store.update(doc_id, "extract", 1.0, {"path": str(file_path)})
            logger.info(
                "Extract phase finished: doc_id=%s language=%s duration_seconds=%.2f",
                doc_id,
                result.get("language"),
                float(result.get("duration_seconds", 0.0) or 0.0),
            )
            text = result.get("text", "")
            if not text.strip():
                raise ValueError("No text extracted from file")

            chunks = chunk_text(
                text,
                chunk_size=self.cfg.chunking.chunk_size,
                chunk_overlap=self.cfg.chunking.chunk_overlap,
            )
            if not chunks:
                raise ValueError("Chunking returned no chunks")
            self.progress_store.update(doc_id, "chunking", 1.0, {"chunks": len(chunks)})
            logger.info("Chunking finished: doc_id=%s chunks=%d", doc_id, len(chunks))

            chunk_texts = [c.text for c in chunks]
            logger.info("Embedding phase started: doc_id=%s", doc_id)
            vectors = self.embedding.embed(chunk_texts)
            self.progress_store.update(doc_id, "embedding", 1.0, {"vectors": len(vectors)})
            logger.info("Embedding phase finished: doc_id=%s vectors=%d", doc_id, len(vectors))

            logger.info("Summary/tag phase started: doc_id=%s", doc_id)
            summary, tags, llm_language = self.llm.summarize_and_tag(text)
            logger.info(
                "Summary/tag phase finished: doc_id=%s tags=%d llm_language=%s",
                doc_id,
                len(tags),
                llm_language,
            )
            language = result.get("language") or llm_language
            speakers = result.get("speakers", [])
            duration = float(result.get("duration_seconds", 0.0) or 0.0)
            segments = result.get("segments", [])

            payloads = []
            for c in chunks:
                ts_start, ts_end = _chunk_timestamps(c.text, segments)
                payloads.append(
                    {
                        "doc_id": doc_id,
                        "filename": file_path.name,
                        "path": str(file_path),
                        "chunk_id": f"{doc_id}:{c.index}",
                        "chunk_index": c.index,
                        "text": c.text,
                        "summary": summary,
                        "date_indexed": now.isoformat(),
                        "language": language,
                        "speakers": speakers,
                        "tags": tags,
                        "duration_seconds": duration,
                        "timestamp_start": ts_start,
                        "timestamp_end": ts_end,
                    }
                )

            logger.info("Qdrant upsert started: doc_id=%s points=%d", doc_id, len(payloads))
            self.qdrant.upsert_chunks(vectors=vectors, payloads=payloads)
            self.progress_store.update(doc_id, "upsert", 1.0, {"points": len(payloads)})
            logger.info("Qdrant upsert finished: doc_id=%s", doc_id)
            self.state_store.upsert(
                IndexedDocument(
                    doc_id=doc_id,
                    filename=file_path.name,
                    path=str(file_path),
                    indexed_at=now,
                    language=language,
                    duration_seconds=duration,
                    tags=tags,
                    speakers=speakers,
                    status="indexed",
                )
            )
            logger.info(
                "Indexing finished: doc_id=%s path=%s status=indexed chunks=%d tags=%d speakers=%d",
                doc_id,
                file_path,
                len(chunks),
                len(tags),
                len(speakers),
            )
            self.progress_store.complete(doc_id, status="indexed")
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Indexing failed: doc_id=%s path=%s error=%s",
                doc_id,
                file_path,
                exc,
            )
            self.state_store.upsert(
                IndexedDocument(
                    doc_id=doc_id,
                    filename=file_path.name,
                    path=str(file_path),
                    indexed_at=now,
                    status="failed",
                    error=str(exc),
                )
            )
            self.progress_store.complete(doc_id, status="failed", error=str(exc))

    def _extract(self, file_path: Path, ext_type: str, doc_id: str) -> dict:
        if ext_type == "media":
            def media_progress(stage: str, progress: float, details: dict | None = None) -> None:
                overall = self.progress_store.update(doc_id, stage, progress, details or {})
                if overall is not None:
                    logger.info(
                        "Progress update: doc_id=%s stage=%s stage_progress=%.1f%% overall_progress=%.1f%%",
                        doc_id,
                        stage,
                        progress * 100.0,
                        overall * 100.0,
                    )

            return self.media.process(file_path, progress_cb=media_progress)
        if ext_type == "pdf":
            return {"text": extract_pdf_text(file_path), "language": None, "duration_seconds": 0.0, "segments": []}
        if ext_type == "pptx":
            return {"text": extract_pptx_text(file_path), "language": None, "duration_seconds": 0.0, "segments": []}
        return {"text": "", "segments": []}


def _chunk_timestamps(chunk_text_value: str, segments: list[dict]) -> tuple[float | None, float | None]:
    if not segments:
        return None, None

    first = None
    last = None
    low = chunk_text_value.lower()
    for seg in segments:
        seg_text = str(seg.get("text", "")).lower()
        if seg_text and seg_text in low:
            if first is None:
                first = float(seg.get("start", 0.0))
            last = float(seg.get("end", 0.0))
    return first, last
