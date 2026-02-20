from __future__ import annotations

import re

from app.models import QueryResponse, SourceRef
from app.pipeline.embeddings import EmbeddingService
from app.pipeline.llm import LLMService
from app.pipeline.qdrant_store import QdrantStore


class QueryService:
    def __init__(
        self,
        embedding: EmbeddingService,
        qdrant: QdrantStore,
        llm: LLMService,
        strict_grounding: bool = False,
        extractive_max_snippets: int = 8,
    ) -> None:
        self.embedding = embedding
        self.qdrant = qdrant
        self.llm = llm
        self.strict_grounding = strict_grounding
        self.extractive_max_snippets = extractive_max_snippets

    def query(self, query_text: str, top_k: int = 8, filters: dict | None = None) -> QueryResponse:
        merged_filters = self._merge_filters_with_filename_hint(query_text, filters)
        filename_filter = str(merged_filters.get("filename", "")).strip() if merged_filters else ""
        if filename_filter:
            by_file = self._query_by_filename(query_text=query_text, filename=filename_filter, top_k=top_k)
            if by_file is not None:
                return by_file

        query_vector = self.embedding.embed([query_text])[0]
        results = self.qdrant.search(query_vector=query_vector, top_k=top_k, filters=merged_filters)

        sources: list[SourceRef] = []
        contexts: list[str] = []

        for hit in results:
            payload = hit.payload or {}
            text = str(payload.get("text", ""))
            sources.append(
                SourceRef(
                    filename=str(payload.get("filename", "unknown")),
                    doc_id=str(payload.get("doc_id", "unknown")),
                    chunk_id=str(payload.get("chunk_id", "unknown")),
                    score=float(hit.score),
                    timestamp_start=payload.get("timestamp_start"),
                    timestamp_end=payload.get("timestamp_end"),
                    page_start=payload.get("page_start"),
                    page_end=payload.get("page_end"),
                    slide_start=payload.get("slide_start"),
                    slide_end=payload.get("slide_end"),
                    text_preview=text[:260],
                )
            )
            contexts.append(text)

        context = "\n\n".join(contexts)
        answer = self._build_answer(query_text=query_text, context=context, sources=sources)

        if sources:
            src_block = ["\n\nSources:"]
            for s in sources:
                if s.timestamp_start is not None:
                    src_block.append(
                        f"- {s.filename} [{s.timestamp_start:.2f}s-{(s.timestamp_end or 0.0):.2f}s]"
                    )
                elif s.slide_start is not None:
                    src_block.append(f"- {s.filename} [{_format_range('slide', s.slide_start, s.slide_end)}]")
                elif s.page_start is not None:
                    src_block.append(f"- {s.filename} [{_format_range('page', s.page_start, s.page_end)}]")
                else:
                    src_block.append(f"- {s.filename}")
            answer += "\n" + "\n".join(src_block)

        return QueryResponse(answer=answer, sources=sources)

    def _query_by_filename(self, query_text: str, filename: str, top_k: int) -> QueryResponse | None:
        payloads = self.qdrant.get_chunks_by_filename(filename=filename, limit=2000)
        if not payloads:
            return None

        contexts: list[str] = []
        sources: list[SourceRef] = []

        for payload in payloads:
            text = str(payload.get("text", ""))
            if text:
                contexts.append(text)

        for payload in payloads[:top_k]:
            text = str(payload.get("text", ""))
            sources.append(
                SourceRef(
                    filename=str(payload.get("filename", "unknown")),
                    doc_id=str(payload.get("doc_id", "unknown")),
                    chunk_id=str(payload.get("chunk_id", "unknown")),
                    score=1.0,
                    timestamp_start=payload.get("timestamp_start"),
                    timestamp_end=payload.get("timestamp_end"),
                    page_start=payload.get("page_start"),
                    page_end=payload.get("page_end"),
                    slide_start=payload.get("slide_start"),
                    slide_end=payload.get("slide_end"),
                    text_preview=text[:260],
                )
            )

        context = "\n\n".join(contexts)
        answer = self._build_answer(query_text=query_text, context=context, sources=sources)
        src_block = ["\n\nSources:"]
        for s in sources:
            if s.timestamp_start is not None:
                src_block.append(f"- {s.filename} [{s.timestamp_start:.2f}s-{(s.timestamp_end or 0.0):.2f}s]")
            elif s.slide_start is not None:
                src_block.append(f"- {s.filename} [{_format_range('slide', s.slide_start, s.slide_end)}]")
            elif s.page_start is not None:
                src_block.append(f"- {s.filename} [{_format_range('page', s.page_start, s.page_end)}]")
            else:
                src_block.append(f"- {s.filename}")
        answer += "\n" + "\n".join(src_block)

        return QueryResponse(answer=answer, sources=sources)

    def _build_answer(self, query_text: str, context: str, sources: list[SourceRef]) -> str:
        if not sources or not context.strip():
            return "NO_INDEXED_CONTEXT"

        if self.strict_grounding:
            return self._extractive_answer(query_text=query_text, sources=sources)

        response = self.llm.answer(query_text, context)
        return response if response else "NO_INDEXED_CONTEXT"

    def _extractive_answer(self, query_text: str, sources: list[SourceRef]) -> str:
        lines = [
            "Grounded answer (extractive mode, no free-form generation):",
            f"Question: {query_text}",
            "Evidence snippets:",
        ]
        for i, src in enumerate(sources[: self.extractive_max_snippets], start=1):
            ts = ""
            if src.timestamp_start is not None:
                ts = f" [{src.timestamp_start:.2f}s-{(src.timestamp_end or 0.0):.2f}s]"
            lines.append(f"{i}. {src.filename}{ts} :: {src.text_preview}")
        return "\n".join(lines)

    @staticmethod
    def _merge_filters_with_filename_hint(query_text: str, filters: dict | None) -> dict | None:
        merged = dict(filters or {})
        if "filename" not in merged:
            match = re.search(r"<([^>]+\.(?:mp4|mkv|mov|mp3|wav|m4a|pdf|pptx))>", query_text, flags=re.IGNORECASE)
            if not match:
                match = re.search(
                    r"([A-Za-z0-9 _\-.]+\.(?:mp4|mkv|mov|mp3|wav|m4a|pdf|pptx))",
                    query_text,
                    flags=re.IGNORECASE,
                )
            if match:
                merged["filename"] = match.group(1).strip()
        return merged or None


def _format_range(label: str, start: int | None, end: int | None) -> str:
    if start is None:
        return f"{label}:?"
    if end is None or end == start:
        return f"{label}:{start}"
    return f"{label}s:{start}-{end}"
