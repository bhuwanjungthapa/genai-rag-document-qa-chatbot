"""
Evaluation utilities.

Input CSV columns (see `eval/sample_eval_questions.csv`):
- question         : required
- gold_answer      : required (short reference answer)
- gold_doc         : optional (expected supporting document filename)
- gold_page        : optional (expected supporting page number)

Output columns produced by `run_evaluation`:
- question, gold_answer, predicted_answer,
  retrieved_doc_names, retrieved_pages,
  hit_at_k, grounded_or_not, top_score, notes
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from .rag_pipeline import RAGAnswer, RAGPipeline
from .retriever import RetrievedChunk


# ---------------------------------------------------------------------------
# Forgiving document-name matching
# ---------------------------------------------------------------------------
#
# Eval CSVs usually say something short like `Syllabus.pdf`, but the actual
# uploaded file might be named `CSE_434___CSE_534_Syllabus.pdf`. A strict
# equality check would mark every row a miss even though retrieval is working
# correctly. We normalize both sides and accept a substring match in either
# direction (with a 4-char minimum to avoid silly "a" matches).
#

_NONALNUM = re.compile(r"[^a-z0-9]+")


def _normalize_doc_name(name: str | None) -> str:
    if not name:
        return ""
    n = str(name).lower().strip()
    if n.endswith(".pdf"):
        n = n[:-4]
    return _NONALNUM.sub("_", n).strip("_")


def docs_match(indexed_name: str, gold_name: str | None) -> bool:
    """Return True if an indexed document plausibly corresponds to a CSV
    `gold_doc` value. Normalized exact match OR substring match (min 4 chars)
    in either direction."""
    a = _normalize_doc_name(indexed_name)
    b = _normalize_doc_name(gold_name)
    if not a or not b:
        return False
    if a == b:
        return True
    if len(b) >= 4 and b in a:
        return True
    if len(a) >= 4 and a in b:
        return True
    return False


# ---------------------------------------------------------------------------
# Retrieval metric
# ---------------------------------------------------------------------------


def hit_at_k(
    retrieved: list[RetrievedChunk],
    gold_doc: Optional[str],
    gold_page: Optional[int],
) -> Optional[bool]:
    """
    Return True if the gold doc/page appears in the retrieved chunks.
    Returns None if no gold supporting source was provided (so it can be
    excluded from the retrieval hit rate).
    """
    if not gold_doc:
        return None
    for r in retrieved:
        if not docs_match(r.doc_name, gold_doc):
            continue
        if gold_page is None:
            return True
        try:
            gp = int(gold_page)
        except (TypeError, ValueError):
            return True
        if r.page_start <= gp <= r.page_end:
            return True
    return False


# ---------------------------------------------------------------------------
# Rough answer-comparison helper
# ---------------------------------------------------------------------------


def compare_answer_to_context(
    predicted_answer: str,
    gold_answer: str,
    retrieved: list[RetrievedChunk],
) -> dict:
    """
    Compare the predicted answer against the retrieved context and the gold
    answer using simple token overlap (Jaccard). This is a *rough* signal, not
    a replacement for manual labels - it's meant to help a grader prioritize
    which rows to review.
    """

    def tokenize(s: str) -> set[str]:
        return {t for t in "".join(ch.lower() if ch.isalnum() else " " for ch in s).split() if len(t) > 2}

    def jaccard(a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    pred_tokens = tokenize(predicted_answer)
    gold_tokens = tokenize(gold_answer)
    context_tokens: set[str] = set()
    for r in retrieved:
        context_tokens |= tokenize(r.raw_text)

    return {
        "overlap_pred_vs_gold": round(jaccard(pred_tokens, gold_tokens), 3),
        "overlap_pred_vs_context": round(jaccard(pred_tokens, context_tokens), 3),
        "overlap_gold_vs_context": round(jaccard(gold_tokens, context_tokens), 3),
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


@dataclass
class EvalRow:
    question: str
    gold_answer: str
    predicted_answer: str
    retrieved_doc_names: str
    retrieved_pages: str
    hit_at_k: Optional[bool]
    grounded_or_not: bool
    top_score: float
    overlap_pred_vs_gold: float
    overlap_pred_vs_context: float
    notes: str = ""
    label: str = ""   # Correct / Partially Correct / Unsupported / Hallucinated


VALID_LABELS = ["", "Correct", "Partially Correct", "Unsupported", "Hallucinated"]


def run_evaluation(
    pipeline: RAGPipeline,
    questions_df: pd.DataFrame,
    top_k: Optional[int] = None,
) -> pd.DataFrame:
    """Run each question through the pipeline and return a results DataFrame."""
    required = {"question", "gold_answer"}
    missing = required - set(questions_df.columns)
    if missing:
        raise ValueError(f"Evaluation CSV is missing columns: {missing}")

    rows: list[EvalRow] = []
    for _, row in questions_df.iterrows():
        question = str(row["question"]).strip()
        gold_answer = str(row.get("gold_answer", "")).strip()
        gold_doc = (
            str(row["gold_doc"]).strip()
            if "gold_doc" in questions_df.columns and pd.notna(row.get("gold_doc"))
            else None
        )
        gold_page = None
        if "gold_page" in questions_df.columns and pd.notna(row.get("gold_page")):
            try:
                gold_page = int(row["gold_page"])
            except (TypeError, ValueError):
                gold_page = None

        ans: RAGAnswer = pipeline.answer(question, top_k=top_k, log=False)

        doc_names = ", ".join(r.doc_name for r in ans.retrieved)
        pages = ", ".join(
            f"{r.page_start}"
            + (f"-{r.page_end}" if r.page_end != r.page_start else "")
            for r in ans.retrieved
        )
        hit = hit_at_k(ans.retrieved, gold_doc, gold_page)
        overlap = compare_answer_to_context(ans.answer, gold_answer, ans.retrieved)
        top_score = ans.retrieved[0].score if ans.retrieved else 0.0

        rows.append(
            EvalRow(
                question=question,
                gold_answer=gold_answer,
                predicted_answer=ans.answer,
                retrieved_doc_names=doc_names,
                retrieved_pages=pages,
                hit_at_k=hit,
                grounded_or_not=ans.grounded,
                top_score=round(top_score, 4),
                overlap_pred_vs_gold=overlap["overlap_pred_vs_gold"],
                overlap_pred_vs_context=overlap["overlap_pred_vs_context"],
                notes="",
                label="",
            )
        )

    return pd.DataFrame([r.__dict__ for r in rows])


def summarize_evaluation(results: pd.DataFrame) -> dict:
    """Compute headline metrics for the Evaluation tab."""
    if results.empty:
        return {
            "n": 0,
            "retrieval_hit_rate": None,
            "correctness_rate": None,
            "hallucination_rate": None,
            "grounded_rate": None,
        }

    hits = results["hit_at_k"].dropna()
    retrieval_hit_rate = float(hits.mean()) if not hits.empty else None

    labels = results["label"].fillna("").astype(str)
    labeled = labels[labels != ""]
    n_labeled = len(labeled)

    correctness_rate = (
        float((labeled == "Correct").mean()) if n_labeled else None
    )
    hallucination_rate = (
        float((labeled == "Hallucinated").mean()) if n_labeled else None
    )

    grounded_rate = float(results["grounded_or_not"].mean())

    return {
        "n": int(len(results)),
        "n_labeled": n_labeled,
        "retrieval_hit_rate": retrieval_hit_rate,
        "correctness_rate": correctness_rate,
        "hallucination_rate": hallucination_rate,
        "grounded_rate": grounded_rate,
    }


def load_questions_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Retrieval-only pass used by report figures (Hit@k at multiple k, score hist)
# ---------------------------------------------------------------------------


def compute_retrieval_ranks(
    pipeline,
    questions_df: pd.DataFrame,
    max_k: int = 10,
) -> pd.DataFrame:
    """Retrieve top `max_k` chunks per question and record the rank at which
    the gold supporting chunk first appears.

    Unlike `run_evaluation`, this does NOT call the LLM - it's a cheap
    retrieval-only pass used to compute Hit@k for multiple k values and to
    plot the top-1 similarity-score distribution.

    Returns a DataFrame with columns: question, gold_doc, gold_page,
    top_score, gold_rank (1-indexed; NaN if the gold chunk is not in the
    top `max_k`).
    """
    records: list[dict] = []
    for _, row in questions_df.iterrows():
        q = str(row["question"]).strip()

        gold_doc: Optional[str] = None
        if "gold_doc" in questions_df.columns and pd.notna(row.get("gold_doc")):
            gold_doc = str(row["gold_doc"]).strip()

        gold_page: Optional[int] = None
        if "gold_page" in questions_df.columns and pd.notna(row.get("gold_page")):
            try:
                gold_page = int(row["gold_page"])
            except (TypeError, ValueError):
                gold_page = None

        retrieved = pipeline.retriever.retrieve(q, top_k=max_k)
        top_score = float(retrieved[0].score) if retrieved else 0.0

        gold_rank: Optional[int] = None
        if gold_doc:
            for rank, r in enumerate(retrieved, start=1):
                if not docs_match(r.doc_name, gold_doc):
                    continue
                if gold_page is None:
                    gold_rank = rank
                    break
                if r.page_start <= gold_page <= r.page_end:
                    gold_rank = rank
                    break

        records.append(
            {
                "question": q,
                "gold_doc": gold_doc,
                "gold_page": gold_page,
                "top_score": top_score,
                "gold_rank": gold_rank,
            }
        )
    return pd.DataFrame(records)
