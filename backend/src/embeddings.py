from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingModel:
    """Wraps sentence-transformers for local, free embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    def _load(self):
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)

    def embed(self, text: str) -> list[float]:
        self._load()
        vector = self._model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    def embed_batch(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        self._load()
        vectors = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
        )
        return vectors.tolist()

    @property
    def dimension(self) -> int:
        self._load()
        return self._model.get_sentence_embedding_dimension()
