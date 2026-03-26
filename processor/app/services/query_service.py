from __future__ import annotations

import re
from collections.abc import Iterator
from collections import Counter

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
        min_score: float = 0.25,
        max_context_chars: int = 12000,
        rerank_enabled: bool = True,
        rerank_top_n: int = 12,
        max_chunks_per_doc: int = 4,
    ) -> None:
        self.embedding = embedding
        self.qdrant = qdrant
        self.llm = llm
        self.strict_grounding = strict_grounding
        self.extractive_max_snippets = extractive_max_snippets
        self.min_score = min_score
        self.max_context_chars = max_context_chars
        self.rerank_enabled = rerank_enabled
        self.rerank_top_n = rerank_top_n
        self.max_chunks_per_doc = max_chunks_per_doc

    def query(self, query_text: str, top_k: int = 8, filters: dict | None = None) -> QueryResponse:
        merged_filters = self._merge_filters_with_filename_hint(query_text, filters)
        query_vector = self.embedding.embed([query_text])[0]
        results = self.qdrant.search(query_vector=query_vector, top_k=top_k, filters=merged_filters)
        results = self._rerank_hits(query_text, results)

        sources, contexts = self._build_sources_and_contexts(results)
        context = "\n\n".join(contexts)
        answer = self._build_answer(query_text=query_text, context=context, sources=sources)
        answer += _sources_block(sources)
        return QueryResponse(answer=answer, sources=sources)

    def query_stream(self, query_text: str, top_k: int = 8, filters: dict | None = None) -> Iterator[str]:
        """Yield answer tokens progressively, then a final sources block."""
        merged_filters = self._merge_filters_with_filename_hint(query_text, filters)
        query_vector = self.embedding.embed([query_text])[0]
        results = self.qdrant.search(query_vector=query_vector, top_k=top_k, filters=merged_filters)
        results = self._rerank_hits(query_text, results)

        sources, contexts = self._build_sources_and_contexts(results)
        context = "\n\n".join(contexts)

        if not sources or not context.strip():
            yield "NO_INDEXED_CONTEXT"
            return

        if self.strict_grounding:
            yield self._extractive_answer(query_text=query_text, sources=sources)
        else:
            for token in self.llm.answer_stream(query_text, context):
                yield token

        block = _sources_block(sources)
        if block:
            yield block

    def _build_sources_and_contexts(self, results) -> tuple[list[SourceRef], list[str]]:
        sources: list[SourceRef] = []
        contexts: list[str] = []
        seen_chunk_ids: set[str] = set()
        per_doc_counts: Counter[str] = Counter()
        used_chars = 0

        for hit in results:
            payload = hit.payload or {}
            text = str(payload.get("text", ""))
            score = float(hit.score)
            if score < self.min_score or not text.strip():
                continue

            chunk_id = str(payload.get("chunk_id", "unknown"))
            if chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)
            doc_id = str(payload.get("doc_id", "unknown"))
            if per_doc_counts[doc_id] >= self.max_chunks_per_doc:
                continue
            per_doc_counts[doc_id] += 1

            src = SourceRef(
                filename=str(payload.get("filename", "unknown")),
                doc_id=doc_id,
                chunk_id=chunk_id,
                score=score,
                timestamp_start=payload.get("timestamp_start"),
                timestamp_end=payload.get("timestamp_end"),
                page_start=payload.get("page_start"),
                page_end=payload.get("page_end"),
                slide_start=payload.get("slide_start"),
                slide_end=payload.get("slide_end"),
                text_preview=text[:260],
            )

            block = (
                f"[source: {src.filename}; chunk={src.chunk_id}; score={src.score:.3f}]\n"
                f"{text}"
            )
            if used_chars + len(block) > self.max_context_chars:
                remaining = self.max_context_chars - used_chars
                if remaining <= 0:
                    break
                block = block[:remaining]

            sources.append(src)
            contexts.append(block)
            used_chars += len(block)
            if used_chars >= self.max_context_chars:
                break
        return sources, contexts

    def _rerank_hits(self, query_text: str, results: list) -> list:
        if not self.rerank_enabled or not results:
            return results

        query_tokens = _tokens(query_text)
        reranked: list[tuple[float, object]] = []
        for hit in results:
            payload = hit.payload or {}
            text = str(payload.get("text", ""))
            lexical = _overlap_ratio(query_tokens, _tokens(text))
            base = float(hit.score)
            combined = (0.75 * base) + (0.25 * lexical)
            reranked.append((combined, hit))

        reranked.sort(key=lambda item: item[0], reverse=True)
        if self.rerank_top_n <= 0:
            return [hit for _, hit in reranked]
        return [hit for _, hit in reranked[: self.rerank_top_n]]

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


def _sources_block(sources: list[SourceRef]) -> str:
    if not sources:
        return ""
    lines = ["\n\nSources:"]
    for s in sources:
        if s.timestamp_start is not None:
            lines.append(f"- {s.filename} [{s.timestamp_start:.2f}s-{(s.timestamp_end or 0.0):.2f}s]")
        elif s.slide_start is not None:
            lines.append(f"- {s.filename} [{_format_range('slide', s.slide_start, s.slide_end)}]")
        elif s.page_start is not None:
            lines.append(f"- {s.filename} [{_format_range('page', s.page_start, s.page_end)}]")
        else:
            lines.append(f"- {s.filename}")
    return "\n" + "\n".join(lines)


def _format_range(label: str, start: int | None, end: int | None) -> str:
    if start is None:
        return f"{label}:?"
    if end is None or end == start:
        return f"{label}:{start}"
    return f"{label}s:{start}-{end}"


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"\w+", text.lower())
        if len(token) > 2
    }


def _overlap_ratio(query_tokens: set[str], context_tokens: set[str]) -> float:
    if not query_tokens or not context_tokens:
        return 0.0
    common = query_tokens & context_tokens
    return len(common) / len(query_tokens)
