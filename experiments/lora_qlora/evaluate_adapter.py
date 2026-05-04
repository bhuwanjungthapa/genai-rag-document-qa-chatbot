"""Evaluate a saved LoRA adapter on generated document-grounded QA examples."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL = PROJECT_ROOT / "experiments" / "lora_qlora" / "data" / "eval.jsonl"
DEFAULT_ADAPTER = PROJECT_ROOT / "experiments" / "lora_qlora" / "adapters" / "flan-t5-small-lora"
DEFAULT_OUT = PROJECT_ROOT / "experiments" / "lora_qlora" / "results" / "adapter_eval.jsonl"


def format_source(example: dict) -> str:
    return (
        f"Instruction: {example['instruction']}\n"
        f"Context: {example['context']}\n"
        f"Question: {example['question']}\n"
        "Answer:"
    )


def tokenize(text: str) -> set[str]:
    return {t for t in re.sub(r"[^a-z0-9]+", " ", str(text).lower()).split() if len(t) > 2}


def jaccard(a: str, b: str) -> float:
    left = tokenize(a)
    right = tokenize(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/flan-t5-small")
    parser.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    parser.add_argument("--eval-jsonl", default=str(DEFAULT_EVAL))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as e:
        raise SystemExit("Missing evaluation dependencies. Install transformers and peft.") from e

    eval_path = Path(args.eval_jsonl)
    adapter_path = Path(args.adapter)
    if not eval_path.exists():
        raise SystemExit("Evaluation dataset missing. Generate the QA dataset first.")
    if not adapter_path.exists():
        raise SystemExit("Adapter directory missing. Train the LoRA adapter first.")

    rows = read_jsonl(eval_path)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(adapter_path if (adapter_path / "tokenizer_config.json").exists() else args.model)
    base = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()

    outputs = []
    for row in rows:
        prompt = format_source(row)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        pred = tokenizer.decode(generated[0], skip_special_tokens=True)
        outputs.append(
            {
                "question": row["question"],
                "gold_answer": row["answer"],
                "predicted_answer": pred,
                "jaccard_pred_vs_gold": round(jaccard(pred, row["answer"]), 4),
                "doc_name": row.get("doc_name"),
                "chunk_id": row.get("chunk_id"),
            }
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in outputs:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    mean_jaccard = sum(row["jaccard_pred_vs_gold"] for row in outputs) / len(outputs) if outputs else 0.0
    summary = {
        "examples": len(outputs),
        "mean_jaccard_pred_vs_gold": round(mean_jaccard, 4),
        "out": str(out_path.resolve()),
    }
    (out_path.parent / "adapter_eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
