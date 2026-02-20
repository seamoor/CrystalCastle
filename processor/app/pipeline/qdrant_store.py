from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models


class QdrantStore:
    def __init__(self, host: str, port: int, collection_name: str, vector_size: int) -> None:
        self.collection_name = collection_name
        self.client = QdrantClient(host=host, port=port)
        self._ensure_collection(vector_size)

    def _ensure_collection(self, vector_size: int) -> None:
        collections = self.client.get_collections().collections
        if any(c.name == self.collection_name for c in collections):
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )

    def upsert_chunks(self, vectors: list[list[float]], payloads: list[dict[str, Any]]) -> None:
        points = [
            models.PointStruct(id=str(uuid.uuid4()), vector=vectors[i], payload=payloads[i])
            for i in range(len(vectors))
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_vector: list[float], top_k: int, filters: dict[str, Any] | None = None):
        query_filter = self._build_filter(filters)
        return self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=query_filter,
        )

    def delete_by_path(self, path: str) -> None:
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[models.FieldCondition(key="path", match=models.MatchValue(value=path))]
                )
            ),
            wait=True,
        )

    def dashboard_stats(self, limit_recent: int = 20) -> dict[str, Any]:
        docs: dict[str, dict[str, Any]] = {}
        tag_distribution: dict[str, int] = {}
        offset = None

        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            if not points:
                break
            for point in points:
                payload = point.payload or {}
                doc_id = str(payload.get("doc_id", "unknown"))
                doc = docs.setdefault(
                    doc_id,
                    {
                        "filename": payload.get("filename", "unknown"),
                        "indexed_at": payload.get("date_indexed", datetime.now(UTC).isoformat()),
                        "duration_seconds": float(payload.get("duration_seconds", 0.0) or 0.0),
                    },
                )
                tags = payload.get("tags", []) or []
                for tag in tags:
                    if isinstance(tag, str):
                        tag_distribution[tag] = tag_distribution.get(tag, 0) + 1

            if offset is None:
                break

        recent_uploads = sorted(
            (
                {
                    "doc_id": k,
                    "filename": v["filename"],
                    "indexed_at": v["indexed_at"],
                    "duration_seconds": v["duration_seconds"],
                }
                for k, v in docs.items()
            ),
            key=lambda x: x["indexed_at"],
            reverse=True,
        )[:limit_recent]

        total_duration = sum(v["duration_seconds"] for v in docs.values())
        return {
            "document_count": len(docs),
            "total_indexed_duration_seconds": total_duration,
            "tag_distribution": dict(sorted(tag_distribution.items(), key=lambda x: x[1], reverse=True)),
            "recent_uploads": recent_uploads,
        }

    @staticmethod
    def _build_filter(filters: dict[str, Any] | None) -> models.Filter | None:
        if not filters:
            return None

        conditions: list[models.FieldCondition] = []
        if filename := filters.get("filename"):
            conditions.append(
                models.FieldCondition(key="filename", match=models.MatchValue(value=str(filename)))
            )
        if tags := filters.get("tags"):
            for tag in tags if isinstance(tags, list) else [tags]:
                conditions.append(
                    models.FieldCondition(key="tags", match=models.MatchValue(value=str(tag)))
                )
        if date_from := filters.get("date_from"):
            conditions.append(
                models.FieldCondition(
                    key="date_indexed",
                    range=models.Range(gte=str(date_from)),
                )
            )
        if date_to := filters.get("date_to"):
            conditions.append(
                models.FieldCondition(
                    key="date_indexed",
                    range=models.Range(lte=str(date_to)),
                )
            )
        return models.Filter(must=conditions) if conditions else None
