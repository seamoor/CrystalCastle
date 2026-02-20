from __future__ import annotations

from app.models import QueryResponse, SourceRef
from app.pipeline.embeddings import EmbeddingService
from app.pipeline.llm import LLMService
from app.pipeline.qdrant_store import QdrantStore


class QueryService:
    def __init__(self, embedding: EmbeddingService, qdrant: QdrantStore, llm: LLMService) -> None:
        self.embedding = embedding
        self.qdrant = qdrant
        self.llm = llm

    def query(self, query_text: str, top_k: int = 8, filters: dict | None = None) -> QueryResponse:
        query_vector = self.embedding.embed([query_text])[0]
        results = self.qdrant.search(query_vector=query_vector, top_k=top_k, filters=filters)

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
                    text_preview=text[:260],
                )
            )
            contexts.append(text)

        context = "\n\n".join(contexts)
        answer = self.llm.answer(query_text, context)

        if sources:
            src_block = ["\n\nSources:"]
            for s in sources:
                if s.timestamp_start is not None:
                    src_block.append(
                        f"- {s.filename} [{s.timestamp_start:.2f}s-{(s.timestamp_end or 0.0):.2f}s]"
                    )
                else:
                    src_block.append(f"- {s.filename}")
            answer += "\n" + "\n".join(src_block)

        return QueryResponse(answer=answer, sources=sources)
