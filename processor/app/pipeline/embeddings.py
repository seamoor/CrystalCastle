from __future__ import annotations

from sentence_transformers import SentenceTransformer


class EmbeddingService:
    def __init__(self, model_name: str, normalize: bool = True) -> None:
        self.model = SentenceTransformer(model_name)
        self.normalize = normalize

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return [vec.tolist() for vec in vectors]

    def vector_size(self) -> int:
        return self.model.get_sentence_embedding_dimension()
