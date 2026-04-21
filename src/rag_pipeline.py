"""
End-to-end RAG pipeline: ingest -> index -> retrieve -> generate.

This module is the high-level API used by `app.py`. It hides the details
of embeddings, FAISS, chunking, and LLM calls behind a small class.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from config import (
    ANSWER_SYSTEM_PROMPT,
    ANSWER_USER_PROMPT_TEMPLATE,
    AppConfig,
    FAISS_INDEX_FILE,
    METADATA_FILE,
    MANIFEST_FILE,
    QA_LOG_FILE,
)

from .chunker import chunk_pages
from .embedder import Embedder
from .llm_client import LLMClient, NullClient, build_llm_client
from .pdf_loader import PageRecord, load_pdfs
from .retriever import RetrievedChunk, Retriever
from .utils import append_jsonl, now_iso
from .vector_store import FaissVectorStore


# ---------------------------------------------------------------------------
# Answer container
# ---------------------------------------------------------------------------


@dataclass
class RAGAnswer:
    question: str
    answer: str
    citations: list[str]
    retrieved: list[RetrievedChunk]
    grounded: bool
    provider: str

    def to_log_record(self) -> dict:
        return {
            "timestamp": now_iso(),
            "question": self.question,
            "answer": self.answer,
            "citations": self.citations,
            "grounded": self.grounded,
            "provider": self.provider,
            "retrieved": [
                {
                    "score": r.score,
                    "doc_name": r.doc_name,
                    "page_start": r.page_start,
                    "page_end": r.page_end,
                    "section_title": r.section_title,
                    "chunk_id": r.chunk_id,
                }
                for r in self.retrieved
            ],
        }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


INSUFFICIENT_EVIDENCE_MSG = (
    "I could not find a supported answer in the uploaded documents."
)


class RAGPipeline:
    """High-level wrapper for the whole retrieval + generation flow."""

    def __init__(
        self,
        config: AppConfig,
        embedder: Optional[Embedder] = None,
        store: Optional[FaissVectorStore] = None,
        llm: Optional[LLMClient] = None,
    ) -> None:
        self.config = config
        self.embedder = embedder or Embedder(config.embedding_model)
        self.store = store or FaissVectorStore(dim=self.embedder.dim)
        self.retriever = Retriever(self.embedder, self.store)
        self.llm: LLMClient = llm or NullClient()

    # -------- Index lifecycle --------------------------------------------

    def build_index(self, pdf_paths: Iterable[str | Path]) -> dict:
        """Ingest PDFs, chunk them, embed them, and (re)build the index.

        If `pdf_paths` is empty or no text can be extracted, the in-memory
        store AND the on-disk index are both reset, so the app never shows
        stale chunks from a previous build.
        """
        pdf_paths = list(pdf_paths)
        pages: list[PageRecord] = load_pdfs(pdf_paths) if pdf_paths else []
        if not pages:
            self.reset_index()
            return {"num_documents": 0, "num_pages": 0, "num_chunks": 0}

        chunks = chunk_pages(
            pages,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            min_chunk_size=self.config.min_chunk_size,
        )
        if not chunks:
            self.reset_index()
            return {
                "num_documents": len({p.doc_name for p in pages}),
                "num_pages": len(pages),
                "num_chunks": 0,
            }

        vectors = self.embedder.embed([c.raw_text for c in chunks])

        # Fresh index — don't concatenate onto old vectors.
        self.store = FaissVectorStore(dim=self.embedder.dim)
        self.store.add(chunks, vectors)
        self.retriever = Retriever(self.embedder, self.store)

        manifest = {
            "embedding_model": self.config.embedding_model,
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
            "min_chunk_size": self.config.min_chunk_size,
            "num_documents": len({p.doc_name for p in pages}),
            "num_pages": len(pages),
            "num_chunks": len(chunks),
            "documents": sorted({p.doc_name for p in pages}),
            "built_at": now_iso(),
        }
        self.store.save(
            index_file=FAISS_INDEX_FILE,
            metadata_file=METADATA_FILE,
            manifest_file=MANIFEST_FILE,
            manifest=manifest,
        )
        return manifest

    def reset_index(self) -> None:
        """Drop the in-memory store and delete persisted index files on disk."""
        self.store = FaissVectorStore(dim=self.embedder.dim)
        self.retriever = Retriever(self.embedder, self.store)
        for p in (FAISS_INDEX_FILE, METADATA_FILE, MANIFEST_FILE):
            try:
                if p.exists():
                    p.unlink()
            except OSError:
                pass

    def load_index_if_exists(self) -> Optional[dict]:
        if not FaissVectorStore.exists(FAISS_INDEX_FILE, METADATA_FILE):
            return None
        self.store = FaissVectorStore.load(
            FAISS_INDEX_FILE, METADATA_FILE, dim=self.embedder.dim
        )
        self.retriever = Retriever(self.embedder, self.store)
        if MANIFEST_FILE.exists():
            try:
                return json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                return None
        return None

    # -------- Inspection helpers -----------------------------------------

    def is_ready(self) -> bool:
        return len(self.store) > 0

    def document_summary(self) -> pd.DataFrame:
        """Aggregate per-document stats for the Documents tab."""
        if self.store.metadata.empty:
            return pd.DataFrame(
                columns=["doc_name", "num_chunks", "page_start", "page_end"]
            )
        df = self.store.metadata
        agg = (
            df.groupby("doc_name")
            .agg(
                num_chunks=("chunk_id", "count"),
                page_start=("page_start", "min"),
                page_end=("page_end", "max"),
            )
            .reset_index()
            .sort_values("doc_name")
        )
        return agg

    def example_chunks(self, n_per_doc: int = 2) -> pd.DataFrame:
        """Return a small preview of chunks per document for the UI."""
        if self.store.metadata.empty:
            return self.store.metadata
        return (
            self.store.metadata.groupby("doc_name", group_keys=False)
            .head(n_per_doc)
            .reset_index(drop=True)
        )

    # -------- LLM configuration -------------------------------------------

    def set_llm_from_config(self, provider: Optional[str] = None) -> str:
        provider = provider or self.config.default_provider
        self.llm = build_llm_client(
            provider,
            gemini_api_key=self.config.gemini_api_key,
            gemini_model=self.config.gemini_model,
            openai_api_key=self.config.openai_api_key,
            openai_model=self.config.openai_model,
        )
        return self.llm.name

    # -------- Retrieval + Generation -------------------------------------

    @staticmethod
    def _build_context_block(retrieved: list[RetrievedChunk]) -> str:
        blocks = []
        for i, r in enumerate(retrieved, start=1):
            header = (
                f"[{i}] {r.doc_name} p.{r.page_start}"
                + (f"-{r.page_end}" if r.page_end != r.page_start else "")
                + (f" | section: {r.section_title}" if r.section_title else "")
            )
            blocks.append(f"{header}\n{r.raw_text}")
        return "\n\n---\n\n".join(blocks)

    def answer(
        self,
        question: str,
        top_k: Optional[int] = None,
        log: bool = True,
    ) -> RAGAnswer:
        """Run one retrieval + generation cycle. Always safe; never raises."""
        top_k = top_k or self.config.top_k
        question = (question or "").strip()
        if not question:
            return RAGAnswer(
                question="",
                answer="Please enter a question.",
                citations=[],
                retrieved=[],
                grounded=False,
                provider=self.llm.name,
            )

        retrieved = self.retriever.retrieve(question, top_k=top_k)
        top_score = retrieved[0].score if retrieved else 0.0

        # Guardrail: weak retrieval -> refuse rather than hallucinate.
        if not retrieved or top_score < self.config.min_score:
            ans = INSUFFICIENT_EVIDENCE_MSG
            result = RAGAnswer(
                question=question,
                answer=ans,
                citations=[],
                retrieved=retrieved,
                grounded=False,
                provider=self.llm.name,
            )
            if log:
                append_jsonl(QA_LOG_FILE, result.to_log_record())
            return result

        context = self._build_context_block(retrieved)
        user_prompt = ANSWER_USER_PROMPT_TEMPLATE.format(
            question=question, context=context
        )
        raw_answer = self.llm.generate(ANSWER_SYSTEM_PROMPT, user_prompt)
        answer_text = (raw_answer or "").strip() or INSUFFICIENT_EVIDENCE_MSG

        grounded = INSUFFICIENT_EVIDENCE_MSG.lower() not in answer_text.lower()

        # Collect de-duplicated citations from the retrieved chunks.
        seen = set()
        citations: list[str] = []
        for r in retrieved:
            cite = r.citation
            if cite not in seen:
                seen.add(cite)
                citations.append(cite)

        result = RAGAnswer(
            question=question,
            answer=answer_text,
            citations=citations if grounded else [],
            retrieved=retrieved,
            grounded=grounded,
            provider=self.llm.name,
        )
        if log:
            append_jsonl(QA_LOG_FILE, result.to_log_record())
        return result
