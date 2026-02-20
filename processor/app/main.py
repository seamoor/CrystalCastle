from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import json

from app.config import AppConfig, load_config
from app.logging_config import setup_logging
from app.models import DashboardStats, IngestResponse, QueryRequest, QueryResponse
from app.pipeline.orchestrator import PipelineOrchestrator
from app.services.dashboard_service import DashboardService
from app.services.ingest_service import IngestService
from app.services.query_service import QueryService
from app.storage.state import StateStore
from app.watcher import WatchService

setup_logging()
logger = logging.getLogger(__name__)

cfg: AppConfig | None = None
state_store: StateStore | None = None
orchestrator: PipelineOrchestrator | None = None
watcher: WatchService | None = None
ingest_service: IngestService | None = None
query_service: QueryService | None = None
dashboard_service: DashboardService | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global cfg, state_store, orchestrator, watcher, ingest_service, query_service, dashboard_service

    cfg = load_config()
    state_store = StateStore(Path(cfg.processor.data_dir))
    orchestrator = PipelineOrchestrator(cfg, state_store)
    watcher = WatchService(
        watch_dir=Path(cfg.processor.watch_dir),
        orchestrator=orchestrator,
        state_store=state_store,
        supported_extensions=cfg.processor.supported_extensions,
        poll_interval_seconds=cfg.processor.poll_interval_seconds,
    )
    watcher.start()

    ingest_service = IngestService(watcher)
    query_service = QueryService(orchestrator.embedding, orchestrator.qdrant, orchestrator.llm)
    dashboard_service = DashboardService(orchestrator.qdrant, state_store)

    logger.info("Processor service started")
    try:
        yield
    finally:
        if watcher:
            watcher.stop()
        logger.info("Processor service stopped")


app = FastAPI(title="Local Knowledge Processor", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
def ingest(payload: dict[str, str]) -> IngestResponse:
    if not ingest_service:
        raise HTTPException(status_code=503, detail="Service not ready")
    path = payload.get("path")
    if not path:
        raise HTTPException(status_code=400, detail="Missing path")
    return ingest_service.enqueue(path)


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    if not query_service:
        raise HTTPException(status_code=503, detail="Service not ready")
    return query_service.query(query_text=req.query, top_k=req.top_k, filters=req.filters)


@app.get("/dashboard/stats", response_model=DashboardStats)
def dashboard_stats() -> DashboardStats:
    if not dashboard_service:
        raise HTTPException(status_code=503, detail="Service not ready")
    return dashboard_service.stats()


@app.get("/v1/models")
def openai_models() -> dict[str, Any]:
    if not cfg:
        raise HTTPException(status_code=503, detail="Service not ready")
    return {
        "object": "list",
        "data": [
            {
                "id": cfg.llm.model,
                "object": "model",
                "created": 0,
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/chat/completions")
def openai_chat_completions(payload: dict[str, Any]) -> dict[str, Any]:
    if not query_service or not cfg:
        raise HTTPException(status_code=503, detail="Service not ready")

    messages = payload.get("messages", [])
    if not messages:
        raise HTTPException(status_code=400, detail="messages are required")

    user_messages = [m for m in messages if m.get("role") == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="user message is required")

    query_text = str(user_messages[-1].get("content", "")).strip()
    filters = payload.get("filters") or payload.get("metadata", {}).get("filters")
    top_k = int(payload.get("top_k", 8))

    result = query_service.query(query_text=query_text, top_k=top_k, filters=filters)
    stream = bool(payload.get("stream", False))

    if stream:
        def event_stream():
            first_chunk = {
                "id": "chatcmpl-local",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": cfg.llm.model,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(first_chunk)}\n\n"

            content_chunk = {
                "id": "chatcmpl-local",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": cfg.llm.model,
                "choices": [{"index": 0, "delta": {"content": result.answer}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(content_chunk)}\n\n"

            final_chunk = {
                "id": "chatcmpl-local",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": cfg.llm.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")  # type: ignore[return-value]

    return {
        "id": "chatcmpl-local",
        "object": "chat.completion",
        "created": 0,
        "model": cfg.llm.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.answer},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
