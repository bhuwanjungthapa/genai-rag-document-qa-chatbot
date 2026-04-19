"""
Sentence-Transformers embedder with L2-normalization.

Normalizing the vectors lets us use FAISS inner-product search
as a stand-in for cosine similarity.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


class Embedder:
    """Thin wrapper around a sentence-transformers model."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        # Lazy import so the rest of the codebase can be imported without
        # pulling in torch at module load time (useful for light scripts).
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    @property
    def dim(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: Iterable[str], batch_size: int = 64) -> np.ndarray:
        """Return a (n, dim) float32 array of L2-normalized embeddings."""
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")

        vectors = self._model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")
        return vectors

    def embed_one(self, text: str) -> np.ndarray:
        """Return a (dim,) float32 vector for a single string."""
        return self.embed([text])[0]
