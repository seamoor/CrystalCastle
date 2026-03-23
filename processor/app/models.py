from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class SourceRef(BaseModel):
    filename: str
    doc_id: str
    chunk_id: str
    score: float
    timestamp_start: float | None = None
    timestamp_end: float | None = None
    page_start: int | None = None
    page_end: int | None = None
    slide_start: int | None = None
    slide_end: int | None = None
    text_preview: str


class QueryRequest(BaseModel):
    query: str
    top_k: int = 8
    filters: dict[str, Any] | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceRef]


class IngestRequest(BaseModel):
    path: str
    force: bool = False


class IngestResponse(BaseModel):
    accepted: bool
    path: str


class DashboardStats(BaseModel):
    document_count: int
    total_indexed_duration_seconds: float
    tag_distribution: dict[str, int]
    recent_uploads: list[dict[str, Any]]


class IndexedDocument(BaseModel):
    doc_id: str
    filename: str
    path: str
    indexed_at: datetime
    language: str | None = None
    duration_seconds: float | None = None
    tags: list[str] = Field(default_factory=list)
    speakers: list[str] = Field(default_factory=list)
    status: str = "indexed"
    error: str | None = None
