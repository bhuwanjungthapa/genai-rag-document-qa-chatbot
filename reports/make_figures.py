"""
Generate the 5 report figures from the command line.

This is a thin CLI wrapper around `reports/figures.py` (pure builders) and
`src/evaluation.py::compute_retrieval_ranks` (retrieval-only pass). The
Streamlit app in `app.py` uses the same functions so figures look identical
whether you generate them here or render them in the UI.

Usage (run from project root, after building the index in the Streamlit app):

    python reports/make_figures.py \
        --eval-csv eval/sample_eval_questions.csv \
        --results-csv path/to/evaluation_results.csv \   # optional (figs 3 + 5)
        --out reports/figures

Figures 2 and 4 need a built FAISS index in `indexes/`.
Figures 3 and 5 need a labeled evaluation_results.csv from the Evaluation tab.
Figure 1 only needs the persisted chunk metadata.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Make the project root importable when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import AppConfig, METADATA_FILE  # noqa: E402
from src.evaluation import compute_retrieval_ranks, load_questions_csv  # noqa: E402
from src.rag_pipeline import RAGPipeline  # noqa: E402

from reports.figures import (  # noqa: E402
    fig_chunks_per_doc,
    fig_heatmap,
    fig_hit_at_k,
    fig_label_distribution,
    fig_score_hist,
)


def _save(fig, out: Path) -> None:
    fig.savefig(out, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    print(f"[ok] wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-csv", default="eval/sample_eval_questions.csv")
    parser.add_argument(
        "--results-csv",
        default=None,
        help="Optional labeled evaluation_results.csv from the Evaluation tab "
        "(enables figs 3 and 5).",
    )
    parser.add_argument("--out", default="reports/figures")
    parser.add_argument("--max-k", type=int, default=10)
    parser.add_argument("--heatmap-k", type=int, default=5)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Figure 1: chunks per document ---------------------------------
    if METADATA_FILE.exists():
        chunks_df = pd.read_parquet(METADATA_FILE)
        if not chunks_df.empty:
            _save(fig_chunks_per_doc(chunks_df), out_dir / "fig1_chunks_per_doc.png")
        else:
            print("[skip] chunk metadata is empty.")
    else:
        print(f"[skip] {METADATA_FILE} not found - build the index first.")

    # --- Retrieval ranks pass (needed by figs 2, 4, 5) -----------------
    ranks_df = pd.DataFrame()
    eval_csv = Path(args.eval_csv)
    if not eval_csv.exists():
        print(f"[warn] eval CSV not found: {eval_csv}. Skipping figs 2, 4, 5.")
    else:
        cfg = AppConfig()
        pipeline = RAGPipeline(cfg)
        pipeline.load_index_if_exists()
        if not pipeline.is_ready():
            print(
                "[warn] no persisted FAISS index in `indexes/`. "
                "Build it in the Streamlit app first. Skipping figs 2, 4, 5."
            )
        else:
            questions_df = load_questions_csv(eval_csv)
            ranks_df = compute_retrieval_ranks(pipeline, questions_df, max_k=args.max_k)
            ranks_df.to_csv(out_dir / "ranks_debug.csv", index=False)
            _save(fig_hit_at_k(ranks_df), out_dir / "fig2_hit_at_k.png")
            _save(fig_score_hist(ranks_df), out_dir / "fig4_score_hist.png")

    # --- Figs 3, 5: require labeled results CSV ------------------------
    results_df: pd.DataFrame | None = None
    if args.results_csv:
        p = Path(args.results_csv)
        if p.exists():
            results_df = pd.read_csv(p)
        else:
            print(f"[warn] results CSV not found: {p}")

    fig3 = fig_label_distribution(results_df)
    if fig3 is not None:
        _save(fig3, out_dir / "fig3_label_distribution.png")
    else:
        print("[skip] figure 3 (need labeled results CSV).")

    fig5 = fig_heatmap(results_df, ranks_df, k=args.heatmap_k)
    if fig5 is not None:
        _save(fig5, out_dir / "fig5_heatmap.png")
    else:
        print("[skip] figure 5 (need labeled results CSV + built index).")

    print(f"\nDone. Figures written under: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
