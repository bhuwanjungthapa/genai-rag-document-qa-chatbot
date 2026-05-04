"""Retriever wires dense FAISS and sparse BM25 search together."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from .bm25_retriever import BM25Retriever
from .embedder import Embedder
from .vector_store import FaissVectorStore

RetrievalMode = Literal["dense", "bm25", "hybrid"]


@dataclass
class RetrievedChunk:
    score: float
    chunk_id: str
    doc_name: str
    page_start: int
    page_end: int
    section_title: str
    raw_text: str
    retrieval_source: str = "dense"
    dense_score: Optional[float] = None
    bm25_score: Optional[float] = None

    @property
    def citation(self) -> str:
        if self.page_start == self.page_end:
            return f"[{self.doc_name} p.{self.page_start}]"
        return f"[{self.doc_name} p.{self.page_start}-{self.page_end}]"


class Retriever:
    def __init__(self, embedder: Embedder, store: FaissVectorStore) -> None:
        self.embedder = embedder
        self.store = store

    @staticmethod
    def _chunk_from_meta(
        *,
        score: float,
        meta: dict,
        retrieval_source: str,
        dense_score: Optional[float] = None,
        bm25_score: Optional[float] = None,
    ) -> RetrievedChunk:
        return RetrievedChunk(
            score=score,
            chunk_id=meta["chunk_id"],
            doc_name=meta["doc_name"],
            page_start=int(meta["page_start"]),
            page_end=int(meta["page_end"]),
            section_title=meta.get("section_title", ""),
            raw_text=meta["raw_text"],
            retrieval_source=retrieval_source,
            dense_score=dense_score,
            bm25_score=bm25_score,
        )

    def _dense_results(self, query: str, top_k: int) -> list[tuple[float, dict]]:
        q_vec = self.embedder.embed_one(query)
        return self.store.search(q_vec, top_k=top_k)

    def _bm25_results(self, query: str, top_k: int) -> list[tuple[float, dict]]:
        bm25 = BM25Retriever(self.store.metadata)
        hits = bm25.search(query, top_k=top_k)
        if not hits:
            return []
        max_score = max(hit.score for hit in hits) or 1.0
        return [(hit.score / max_score, hit.metadata) for hit in hits]

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        *,
        mode: RetrievalMode = "dense",
        hybrid_alpha: float = 0.65,
    ) -> list[RetrievedChunk]:
        if not query.strip() or len(self.store) == 0:
            return []

        mode = mode if mode in {"dense", "bm25", "hybrid"} else "dense"
        hybrid_alpha = max(0.0, min(1.0, float(hybrid_alpha)))

        if mode == "dense":
            return [
                self._chunk_from_meta(
                    score=score,
                    meta=meta,
                    retrieval_source="dense",
                    dense_score=score,
                )
                for score, meta in self._dense_results(query, top_k=top_k)
            ]

        if mode == "bm25":
            return [
                self._chunk_from_meta(
                    score=score,
                    meta=meta,
                    retrieval_source="bm25",
                    bm25_score=score,
                )
                for score, meta in self._bm25_results(query, top_k=top_k)
            ]

        pool_k = min(len(self.store), max(top_k * 4, top_k))
        dense_raw = self._dense_results(query, top_k=pool_k)
        bm25_raw = self._bm25_results(query, top_k=pool_k)

        fused: dict[str, dict] = {}
        for score, meta in dense_raw:
            cid = meta["chunk_id"]
            item = fused.setdefault(cid, {"meta": meta, "dense": 0.0, "bm25": 0.0})
            # Dense vectors are L2-normalized, so inner product is cosine.
            item["dense"] = max(0.0, min(1.0, float(score)))
        for score, meta in bm25_raw:
            cid = meta["chunk_id"]
            item = fused.setdefault(cid, {"meta": meta, "dense": 0.0, "bm25": 0.0})
            item["bm25"] = max(0.0, min(1.0, float(score)))

        ranked = sorted(
            fused.values(),
            key=lambda item: hybrid_alpha * item["dense"] + (1 - hybrid_alpha) * item["bm25"],
            reverse=True,
        )

        return [
            self._chunk_from_meta(
                score=hybrid_alpha * item["dense"] + (1 - hybrid_alpha) * item["bm25"],
                meta=item["meta"],
                retrieval_source="hybrid",
                dense_score=item["dense"],
                bm25_score=item["bm25"],
            )
            for item in ranked[:top_k]
        ]
