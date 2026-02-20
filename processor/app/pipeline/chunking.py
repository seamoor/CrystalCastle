from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    index: int


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return []

    chunks: list[Chunk] = []
    start = 0
    idx = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + chunk_size)
        chunks.append(Chunk(text=cleaned[start:end], index=idx))
        if end >= len(cleaned):
            break
        start = max(0, end - chunk_overlap)
        idx += 1
    return chunks
