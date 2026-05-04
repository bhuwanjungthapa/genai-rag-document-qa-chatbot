"""
One-shot evaluation runner used to populate every report figure.

What it does:
1. Loads `eval/sample_eval_questions.csv`.
2. Runs the full pipeline with BERTScore enabled and saves the labeled-ready
   results CSV to `reports/evaluation_results.csv`.
3. Auto-applies a *heuristic* manual label per row so that figures 3, 5, 6
   (which require labels) render. Heuristic:
      - grounded_or_not == False     -> "Unsupported"
      - semantic similarity >= 0.60  -> "Correct"
      - semantic similarity >= 0.35  -> "Partially Correct"
      - hit_at_k True but low sim    -> "Hallucinated"
      - otherwise                     -> "Partially Correct"
   These are starting-point labels; you should review them in the Streamlit
   Evaluation tab and override anything that looks wrong.
4. Calls `reports/make_figures.py` so every PNG ends up under
   `reports/figures/`.

Run from the project root:
    python reports/run_full_evaluation.py
"""

from __future__ import annotations

# Defuse macOS OpenMP conflicts (faiss + torch + bert-score) BEFORE any heavy
# import. Without these three env vars the eval process segfaults (exit 139)
# the moment BERTScore tries to spawn worker threads.
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import AppConfig  # noqa: E402
from src.evaluation import (  # noqa: E402
    load_questions_csv,
    run_evaluation,
    summarize_evaluation,
)
from src.rag_pipeline import RAGPipeline  # noqa: E402


EVAL_CSV = PROJECT_ROOT / "eval" / "sample_eval_questions.csv"
OUT_CSV = PROJECT_ROOT / "reports" / "evaluation_results.csv"
SUMMARY_JSON = PROJECT_ROOT / "reports" / "evaluation_summary.json"


def auto_label(row: pd.Series) -> str:
    grounded = bool(row.get("grounded_or_not", False))
    if not grounded:
        return "Unsupported"

    sim = row.get("semantic_similarity_pred_vs_gold")
    try:
        sim_val = float(sim) if sim is not None else None
    except (TypeError, ValueError):
        sim_val = None

    hit = row.get("hit_at_k")
    hit_bool = bool(hit) if hit is not None and not (isinstance(hit, float) and pd.isna(hit)) else None

    if sim_val is None:
        return "Partially Correct"
    if sim_val >= 0.60:
        return "Correct"
    if sim_val >= 0.35:
        return "Partially Correct"
    if hit_bool is True:
        return "Hallucinated"
    return "Partially Correct"


def main() -> None:
    cfg = AppConfig()
    pipeline = RAGPipeline(cfg)
    pipeline.load_index_if_exists()
    if not pipeline.is_ready():
        raise SystemExit(
            "No FAISS index found. Build the index in the Streamlit app first "
            "(or run the LoRA tab's step-1 dataset prep after building it)."
        )

    provider_name = pipeline.set_llm_from_config(cfg.default_provider)
    if provider_name == "none":
        print(
            "[warn] No LLM API key configured (GEMINI_API_KEY or OPENAI_API_KEY). "
            "Predicted answers will be the no-LLM fallback message and the report "
            "numbers will reflect a retrieval-only run.",
        )
    else:
        print(f"[info] Using LLM provider: {provider_name}")

    if not EVAL_CSV.exists():
        raise SystemExit(f"Evaluation CSV not found: {EVAL_CSV}")

    print(f"[1/3] Loading evaluation questions from {EVAL_CSV.name}...")
    questions_df = load_questions_csv(EVAL_CSV)
    print(f"      {len(questions_df)} questions queued.")

    print("[2/3] Running full pipeline with BERTScore enabled (this is the slow step)...")
    results = run_evaluation(pipeline, questions_df, compute_bertscore=True)

    print("[3/3] Auto-labeling rows with a starter heuristic...")
    results["label"] = results.apply(auto_label, axis=1)
    results["notes"] = "auto-labeled by reports/run_full_evaluation.py - review in Streamlit"

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT_CSV, index=False)
    print(f"      wrote {OUT_CSV.relative_to(PROJECT_ROOT)}")

    summary = summarize_evaluation(results)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"      wrote {SUMMARY_JSON.relative_to(PROJECT_ROOT)}")

    label_counts = results["label"].value_counts().to_dict()
    print("\n--- Headline metrics ---")
    print(json.dumps(summary, indent=2, default=str))
    print("\n--- Auto-label distribution ---")
    print(json.dumps(label_counts, indent=2))

    print("\nGenerating the six report figures...")
    fig_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "reports" / "make_figures.py"),
        "--eval-csv",
        str(EVAL_CSV),
        "--results-csv",
        str(OUT_CSV),
    ]
    subprocess.run(fig_cmd, check=True, cwd=PROJECT_ROOT)
    print("\nDone.")


if __name__ == "__main__":
    main()
