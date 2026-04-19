"""
FAISS-based vector store with on-disk persistence.

We store:
- `faiss.index`    : the FAISS IndexFlatIP over normalized vectors
- `chunks.parquet` : the chunk metadata (aligned by row to the FAISS ids)
- `manifest.json`  : ingestion parameters used to build the index
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import pandas as pd

from .chunker import Chunk


class FaissVectorStore:
    """A flat inner-product FAISS index + aligned pandas metadata table."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.index: faiss.Index = faiss.IndexFlatIP(dim)
        self.metadata: pd.DataFrame = pd.DataFrame(
            columns=[
                "chunk_id",
                "doc_name",
                "page_start",
                "page_end",
                "section_title",
                "raw_text",
            ]
        )

    # --- Build ------------------------------------------------------------

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) == 0:
            return
        assert vectors.shape[0] == len(chunks), "vectors/chunks length mismatch"
        assert vectors.shape[1] == self.dim, "embedding dim mismatch"

        self.index.add(vectors.astype("float32"))

        rows = [asdict(c) for c in chunks]
        new_df = pd.DataFrame(rows)
        self.metadata = pd.concat([self.metadata, new_df], ignore_index=True)

    def __len__(self) -> int:
        return int(self.index.ntotal)

    # --- Search -----------------------------------------------------------

    def search(
        self, query_vector: np.ndarray, top_k: int = 4
    ) -> list[tuple[float, dict]]:
        """Return `top_k` (score, metadata_dict) tuples for a single query vector."""
        if len(self) == 0:
            return []

        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        query_vector = query_vector.astype("float32")

        scores, idxs = self.index.search(query_vector, min(top_k, len(self)))
        results: list[tuple[float, dict]] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            row = self.metadata.iloc[int(idx)].to_dict()
            results.append((float(score), row))
        return results

    # --- Persistence ------------------------------------------------------

    def save(
        self,
        index_file: Path,
        metadata_file: Path,
        manifest_file: Optional[Path] = None,
        manifest: Optional[dict] = None,
    ) -> None:
        index_file.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_file))
        self.metadata.to_parquet(metadata_file, index=False)
        if manifest_file is not None and manifest is not None:
            manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    @classmethod
    def load(
        cls, index_file: Path, metadata_file: Path, dim: Optional[int] = None
    ) -> "FaissVectorStore":
        idx = faiss.read_index(str(index_file))
        store = cls(dim=dim or idx.d)
        store.index = idx
        store.metadata = pd.read_parquet(metadata_file)
        return store

    @staticmethod
    def exists(index_file: Path, metadata_file: Path) -> bool:
        return index_file.exists() and metadata_file.exists()
