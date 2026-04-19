"""Retriever wires an embedder + vector store together for top-k search."""

from __future__ import annotations

from dataclasses import dataclass

from .embedder import Embedder
from .vector_store import FaissVectorStore


@dataclass
class RetrievedChunk:
    score: float
    chunk_id: str
    doc_name: str
    page_start: int
    page_end: int
    section_title: str
    raw_text: str

    @property
    def citation(self) -> str:
        if self.page_start == self.page_end:
            return f"[{self.doc_name} p.{self.page_start}]"
        return f"[{self.doc_name} p.{self.page_start}-{self.page_end}]"


class Retriever:
    def __init__(self, embedder: Embedder, store: FaissVectorStore) -> None:
        self.embedder = embedder
        self.store = store

    def retrieve(self, query: str, top_k: int = 4) -> list[RetrievedChunk]:
        if not query.strip() or len(self.store) == 0:
            return []

        q_vec = self.embedder.embed_one(query)
        raw_results = self.store.search(q_vec, top_k=top_k)

        return [
            RetrievedChunk(
                score=score,
                chunk_id=meta["chunk_id"],
                doc_name=meta["doc_name"],
                page_start=int(meta["page_start"]),
                page_end=int(meta["page_end"]),
                section_title=meta.get("section_title", ""),
                raw_text=meta["raw_text"],
            )
            for score, meta in raw_results
        ]
