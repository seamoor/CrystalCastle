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
            )
        ]


class FakeLLM:
    def answer(self, query: str, context: str) -> str:
        assert "Important architecture decision" in context
        return f"Answer for: {query}"


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
