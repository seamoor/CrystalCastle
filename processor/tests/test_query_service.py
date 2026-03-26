from __future__ import annotations

from dataclasses import dataclass

from app.services.query_service import QueryService


class FakeEmbedding:
    def embed(self, texts: list[str]) -> list[list[float]]:
        assert len(texts) == 1
        return [[0.1, 0.2, 0.3]]


@dataclass
class FakeHit:
    payload: dict
    score: float


class FakeQdrant:
    def __init__(self) -> None:
        self.last_filters = None

    def search(self, query_vector: list[float], top_k: int, filters: dict | None = None):
        self.last_filters = filters
        assert query_vector == [0.1, 0.2, 0.3]
        assert top_k == 2
        return [
            FakeHit(
                payload={
                    "filename": "session.mp4",
                    "doc_id": "doc-1",
                    "chunk_id": "doc-1:0",
                    "timestamp_start": 12.0,
                    "timestamp_end": 18.0,
                    "text": "Important architecture decision",
                },
                score=0.88,
            ),
            FakeHit(
                payload={
                    "filename": "session.mp4",
                    "doc_id": "doc-1",
                    "chunk_id": "doc-1:0",
                    "timestamp_start": 12.0,
                    "timestamp_end": 18.0,
                    "text": "Important architecture decision duplicate",
                },
                score=0.87,
            ),
            FakeHit(
                payload={
                    "filename": "session.mp4",
                    "doc_id": "doc-1",
                    "chunk_id": "doc-1:2",
                    "timestamp_start": 22.0,
                    "timestamp_end": 25.0,
                    "text": "Low confidence snippet should be filtered out",
                },
                score=0.12,
            ),
        ]


class FakeLLM:
    def answer(self, query: str, context: str) -> str:
        assert "Important architecture decision" in context
        assert "Low confidence snippet should be filtered out" not in context
        assert "[source: session.mp4; chunk=doc-1:0; score=0.880]" in context
        return f"Answer for: {query}"

    def answer_stream(self, query: str, context: str):
        yield self.answer(query, context)


class SimpleLLM:
    def answer(self, query: str, context: str) -> str:
        return f"Answer for: {query}"

    def answer_stream(self, query: str, context: str):
        yield self.answer(query, context)


def test_query_service_returns_sources_and_answer_suffix() -> None:
    qdrant = FakeQdrant()
    service = QueryService(embedding=FakeEmbedding(), qdrant=qdrant, llm=FakeLLM())

    result = service.query(
        query_text="What was decided?",
        top_k=2,
        filters={"tags": ["architecture"]},
    )

    assert "Answer for: What was decided?" in result.answer
    assert "Sources:" in result.answer
    assert "session.mp4 [12.00s-18.00s]" in result.answer
    assert len(result.sources) == 1
    assert result.sources[0].filename == "session.mp4"
    assert qdrant.last_filters == {"tags": ["architecture"]}


def test_query_service_reranks_by_lexical_overlap() -> None:
    class RerankQdrant:
        def search(self, query_vector: list[float], top_k: int, filters: dict | None = None):
            return [
                FakeHit(
                    payload={
                        "filename": "a.pdf",
                        "doc_id": "doc-a",
                        "chunk_id": "doc-a:1",
                        "text": "Budget numbers and finance planning for next quarter",
                    },
                    score=0.91,
                ),
                FakeHit(
                    payload={
                        "filename": "b.pdf",
                        "doc_id": "doc-b",
                        "chunk_id": "doc-b:1",
                        "text": "Kubernetes deployment rollback strategy and canary release",
                    },
                    score=0.84,
                ),
            ]

    service = QueryService(
        embedding=FakeEmbedding(),
        qdrant=RerankQdrant(),
        llm=SimpleLLM(),
        min_score=0.0,
        rerank_enabled=True,
    )
    result = service.query("Jak działa rollback deployment strategy?", top_k=2)
    assert result.sources[0].filename == "b.pdf"


def test_query_service_limits_chunks_per_document() -> None:
    class MultiDocQdrant:
        def search(self, query_vector: list[float], top_k: int, filters: dict | None = None):
            return [
                FakeHit(payload={"filename": "x.pdf", "doc_id": "doc-x", "chunk_id": "doc-x:1", "text": "alpha one"}, score=0.9),
                FakeHit(payload={"filename": "x.pdf", "doc_id": "doc-x", "chunk_id": "doc-x:2", "text": "alpha two"}, score=0.89),
                FakeHit(payload={"filename": "y.pdf", "doc_id": "doc-y", "chunk_id": "doc-y:1", "text": "beta one"}, score=0.88),
            ]

    service = QueryService(
        embedding=FakeEmbedding(),
        qdrant=MultiDocQdrant(),
        llm=SimpleLLM(),
        min_score=0.0,
        max_chunks_per_doc=1,
    )
    result = service.query("alpha beta", top_k=3)
    doc_ids = [s.doc_id for s in result.sources]
    assert doc_ids.count("doc-x") == 1
    assert doc_ids.count("doc-y") == 1
