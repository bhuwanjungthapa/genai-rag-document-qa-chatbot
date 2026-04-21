"""
Pure figure builders for the Generative AI RAG Document Q&A Chatbot report.

Each function takes pandas DataFrames and returns a matplotlib Figure.
No disk I/O happens here - callers decide whether to save (CLI) or render
(Streamlit). Keeping this module I/O-free makes it trivial to unit test
and to call from both the batch script and the web UI.

Figures:
    fig_chunks_per_doc(chunks_df)                -> Figure
    fig_hit_at_k(ranks_df, ks=(1, 3, 5, 10))     -> Figure
    fig_label_distribution(results_df)           -> Figure | None
    fig_score_hist(ranks_df)                     -> Figure
    fig_heatmap(results_df, ranks_df, k=5)       -> Figure | None
"""

from __future__ import annotations

from typing import Iterable

import matplotlib

# Safe default for headless environments. Streamlit overrides this anyway
# when it grabs the figure object.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LABEL_ORDER: list[str] = [
    "Correct",
    "Partially Correct",
    "Unsupported",
    "Hallucinated",
]
LABEL_COLORS: dict[str, str] = {
    "Correct": "#55A868",
    "Partially Correct": "#DD8452",
    "Unsupported": "#8172B3",
    "Hallucinated": "#C44E52",
}


# ---------------------------------------------------------------------------
# Figure 1 - chunks per document
# ---------------------------------------------------------------------------


def fig_chunks_per_doc(chunks_df: pd.DataFrame) -> plt.Figure:
    counts = chunks_df.groupby("doc_name").size().sort_values(ascending=True)

    height = max(3.0, 0.5 * len(counts) + 1.5)
    fig, ax = plt.subplots(figsize=(8, height))
    ax.barh(counts.index, counts.values, color="#4C72B0")
    ax.set_xlabel("Number of chunks")
    ax.set_ylabel("Document")
    ax.set_title(
        f"Chunks per document  (n_docs={counts.shape[0]}, n_chunks={int(counts.sum())})"
    )
    max_v = float(counts.values.max()) if len(counts.values) else 1.0
    for i, v in enumerate(counts.values):
        ax.text(v + max_v * 0.01, i, str(int(v)), va="center", fontsize=9)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 2 - Hit@k at k = 1, 3, 5, 10
# ---------------------------------------------------------------------------


def fig_hit_at_k(
    ranks_df: pd.DataFrame, ks: Iterable[int] = (1, 3, 5, 10)
) -> plt.Figure:
    elig = ranks_df[ranks_df["gold_doc"].notna()].copy()
    n = len(elig)

    ks = list(ks)
    if n == 0:
        rates = [0.0 for _ in ks]
    else:
        rates = [
            float(
                elig["gold_rank"]
                .apply(lambda r, k=k: pd.notna(r) and r <= k)
                .mean()
            )
            for k in ks
        ]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar([str(k) for k in ks], rates, color="#55A868")
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("k")
    ax.set_ylabel("Hit@k")
    ax.set_title(f"Retrieval Hit@k  (n={n})")
    for bar, r in zip(bars, rates):
        hits = int(round(r * n))
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{hits}/{n}\n{r:.0%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Helpful hint when every bar is empty.
    if n > 0 and all(r == 0.0 for r in rates):
        ax.text(
            0.5,
            0.5,
            "No retrieved docs matched any gold_doc.\n"
            "Check the Retrieval diagnostics panel\n"
            "or edit the gold_doc / gold_page values.",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="#555",
            bbox=dict(boxstyle="round,pad=0.5", fc="#FFF3CD", ec="#F0AD4E"),
        )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 3 - label distribution
# ---------------------------------------------------------------------------


def fig_label_distribution(results_df: pd.DataFrame | None) -> plt.Figure | None:
    if results_df is None or "label" not in results_df.columns:
        return None

    labels = results_df["label"].fillna("").astype(str)
    labeled = labels[labels != ""]
    if labeled.empty:
        return None

    counts = labeled.value_counts().reindex(LABEL_ORDER).fillna(0).astype(int)

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(
        counts.index,
        counts.values,
        color=[LABEL_COLORS[c] for c in counts.index],
    )
    ax.set_ylabel("Number of questions")
    ax.set_title(f"Answer label distribution  (n_labeled={int(counts.sum())})")
    max_v = max(counts.values) if len(counts.values) else 1
    for b, v in zip(bars, counts.values):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + max(max_v * 0.02, 0.1),
            str(int(v)),
            ha="center",
            fontsize=10,
        )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 4 - Top-1 similarity histogram, split by Hit@1
# ---------------------------------------------------------------------------


def fig_score_hist(ranks_df: pd.DataFrame) -> plt.Figure:
    elig = ranks_df[ranks_df["gold_doc"].notna()].copy()
    elig["hit1"] = elig["gold_rank"].apply(lambda r: pd.notna(r) and r <= 1)

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1, 21)

    hit_scores = elig.loc[elig["hit1"], "top_score"]
    miss_scores = elig.loc[~elig["hit1"], "top_score"]

    if not hit_scores.empty:
        ax.hist(
            hit_scores,
            bins=bins,
            alpha=0.7,
            label=f"Hit@1 = True  (n={len(hit_scores)})",
            color="#55A868",
        )
    if not miss_scores.empty:
        ax.hist(
            miss_scores,
            bins=bins,
            alpha=0.7,
            label=f"Hit@1 = False (n={len(miss_scores)})",
            color="#C44E52",
        )
    ax.set_xlabel("Top-1 retrieval similarity (cosine)")
    ax.set_ylabel("Count")
    ax.set_title(f"Top-1 retrieval score by Hit@1  (n={len(elig)})")
    if not elig.empty:
        ax.legend()

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 5 - Hit@k x answer label heatmap
# ---------------------------------------------------------------------------


def fig_heatmap(
    results_df: pd.DataFrame | None,
    ranks_df: pd.DataFrame,
    k: int = 5,
) -> plt.Figure | None:
    if results_df is None or "label" not in results_df.columns or ranks_df.empty:
        return None

    merged = results_df.merge(
        ranks_df[["question", "gold_rank"]], on="question", how="left"
    )
    merged["hit"] = merged["gold_rank"].apply(lambda r: pd.notna(r) and r <= k)
    labels = merged["label"].fillna("").astype(str)
    merged = merged[labels != ""]
    if merged.empty:
        return None

    merged["hit_label"] = merged["hit"].map(
        {True: f"Hit@{k}=True", False: f"Hit@{k}=False"}
    )
    matrix = pd.crosstab(merged["hit_label"], merged["label"])
    for col in LABEL_ORDER:
        if col not in matrix.columns:
            matrix[col] = 0
    matrix = matrix[LABEL_ORDER]
    matrix = (
        matrix.reindex(index=[f"Hit@{k}=True", f"Hit@{k}=False"])
        .fillna(0)
        .astype(int)
    )

    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    im = ax.imshow(matrix.values, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_title(
        f"Retrieval success (Hit@{k}) x answer label  "
        f"(n_labeled={int(matrix.values.sum())})"
    )

    vmax = float(matrix.values.max()) if matrix.values.size else 1.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = int(matrix.values[i, j])
            text_color = "white" if v > vmax / 2 else "black"
            ax.text(
                j, i, str(v), ha="center", va="center",
                color=text_color, fontsize=11,
            )

    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig
