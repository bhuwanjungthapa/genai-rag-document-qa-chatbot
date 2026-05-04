"""Create document-grounded QA examples from the current RAG chunks.

This script intentionally does not call an LLM. It creates reproducible,
template-based QA examples from `indexes/chunks.parquet` so the LoRA experiment
can run without API keys. The goal is to fine-tune answer format and grounding
behavior, not to replace RAG as the dynamic knowledge mechanism.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHUNKS = PROJECT_ROOT / "indexes" / "chunks.parquet"
DEFAULT_TRAIN = PROJECT_ROOT / "experiments" / "lora_qlora" / "data" / "train.jsonl"
DEFAULT_EVAL = PROJECT_ROOT / "experiments" / "lora_qlora" / "data" / "eval.jsonl"


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def citation(row: pd.Series) -> str:
    doc = row.get("doc_name", "document.pdf")
    start = int(row.get("page_start", 1))
    end = int(row.get("page_end", start))
    page = f"p.{start}" if start == end else f"p.{start}-{end}"
    return f"[{doc} {page}]"


def answer_from_text(text: str, cite: str, max_chars: int) -> str:
    text = clean_text(text)
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0].rstrip(".,;:")
    return f"{text} {cite}"


def make_examples(row: pd.Series, examples_per_chunk: int, max_context_chars: int, max_answer_chars: int) -> list[dict]:
    raw_text = clean_text(row.get("raw_text", ""))
    if not raw_text:
        return []

    doc = str(row.get("doc_name", "document.pdf"))
    section = clean_text(row.get("section_title", "")) or "this section"
    cite = citation(row)
    context = raw_text[:max_context_chars].rsplit(" ", 1)[0] if len(raw_text) > max_context_chars else raw_text
    answer = answer_from_text(raw_text, cite, max_answer_chars)

    templates = [
        f"What does {doc} say about {section}?",
        f"Summarize the key information from {section}.",
        f"According to {doc}, what is the relevant information on this page?",
        f"What answer is supported by the provided context from {doc}?",
        f"Using only the context, explain {section}.",
    ]

    examples = []
    for question in templates[:examples_per_chunk]:
        examples.append(
            {
                "instruction": "Answer using only the provided context. If the answer is unsupported, say you could not find a supported answer. Cite the source.",
                "context": context,
                "question": question,
                "answer": answer,
                "doc_name": doc,
                "page_start": int(row.get("page_start", 1)),
                "page_end": int(row.get("page_end", row.get("page_start", 1))),
                "chunk_id": str(row.get("chunk_id", "")),
            }
        )
    return examples


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", default=str(DEFAULT_CHUNKS))
    parser.add_argument("--train-out", default=str(DEFAULT_TRAIN))
    parser.add_argument("--eval-out", default=str(DEFAULT_EVAL))
    parser.add_argument("--examples-per-chunk", type=int, default=3)
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--max-context-chars", type=int, default=1400)
    parser.add_argument("--max-answer-chars", type=int, default=420)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    chunks_path = Path(args.chunks)
    if not chunks_path.exists():
        raise SystemExit(f"Chunk metadata not found: {chunks_path}. Build the index first.")

    df = pd.read_parquet(chunks_path)
    if df.empty:
        raise SystemExit("Chunk metadata is empty. Upload PDFs and rebuild the index first.")

    examples: list[dict] = []
    for _, row in df.iterrows():
        examples.extend(
            make_examples(
                row,
                examples_per_chunk=max(1, min(5, args.examples_per_chunk)),
                max_context_chars=args.max_context_chars,
                max_answer_chars=args.max_answer_chars,
            )
        )

    if len(examples) < 2:
        raise SystemExit("Not enough examples generated for a train/eval split.")

    rng = random.Random(args.seed)
    rng.shuffle(examples)
    eval_n = max(1, int(round(len(examples) * max(0.05, min(0.5, args.eval_fraction)))))
    eval_rows = examples[:eval_n]
    train_rows = examples[eval_n:]

    write_jsonl(Path(args.train_out), train_rows)
    write_jsonl(Path(args.eval_out), eval_rows)

    summary = {
        "chunks": int(len(df)),
        "examples_total": len(examples),
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "train_out": str(Path(args.train_out).resolve()),
        "eval_out": str(Path(args.eval_out).resolve()),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
