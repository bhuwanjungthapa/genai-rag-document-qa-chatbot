"""Small dependency-free BM25 retriever for sparse keyword search.

The dense FAISS retriever is good at semantic similarity. BM25 complements it
by rewarding exact course/document terms such as assignment names, dates, or
policy labels that embeddings may blur together.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

import pandas as pd


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Tokenize text for sparse retrieval."""
    return [t for t in _TOKEN_RE.findall(str(text).lower()) if len(t) > 1]


@dataclass
class BM25Hit:
    score: float
    row_index: int
    metadata: dict


class BM25Retriever:
    """In-memory BM25 index built from the current chunk metadata table."""

    def __init__(self, metadata: pd.DataFrame, *, k1: float = 1.5, b: float = 0.75) -> None:
        self.metadata = metadata.reset_index(drop=True).copy()
        self.k1 = k1
        self.b = b
        self.doc_tokens: list[list[str]] = []
        self.term_freqs: list[Counter[str]] = []
        self.doc_freqs: Counter[str] = Counter()
        self.avg_doc_len = 0.0
        self._build()

    def _build(self) -> None:
        for _, row in self.metadata.iterrows():
            tokens = tokenize(row.get("raw_text", ""))
            self.doc_tokens.append(tokens)
            freqs = Counter(tokens)
            self.term_freqs.append(freqs)
            self.doc_freqs.update(freqs.keys())

        if self.doc_tokens:
            self.avg_doc_len = sum(len(toks) for toks in self.doc_tokens) / len(self.doc_tokens)
        else:
            self.avg_doc_len = 0.0

    def _idf(self, term: str) -> float:
        n_docs = len(self.doc_tokens)
        if n_docs == 0:
            return 0.0
        df = self.doc_freqs.get(term, 0)
        # Robertson-Sparck Jones idf with +1 to keep scores positive.
        return math.log(1 + (n_docs - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 4) -> list[BM25Hit]:
        query_terms = tokenize(query)
        if not query_terms or self.metadata.empty:
            return []

        scores: list[tuple[float, int]] = []
        avgdl = self.avg_doc_len or 1.0
        for idx, freqs in enumerate(self.term_freqs):
            doc_len = len(self.doc_tokens[idx]) or 1
            score = 0.0
            for term in query_terms:
                freq = freqs.get(term, 0)
                if freq == 0:
                    continue
                denom = freq + self.k1 * (1 - self.b + self.b * doc_len / avgdl)
                score += self._idf(term) * (freq * (self.k1 + 1)) / denom
            if score > 0:
                scores.append((score, idx))

        scores.sort(key=lambda item: item[0], reverse=True)
        return [
            BM25Hit(
                score=float(score),
                row_index=idx,
                metadata=self.metadata.iloc[idx].to_dict(),
            )
            for score, idx in scores[:top_k]
        ]
