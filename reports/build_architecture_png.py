"""
Standalone PNG version of the slide-5 architecture diagram.

This is a portable backup of the same two-row layout that
`reports/build_presentation.py` draws inside the .pptx using native shapes.
Use this when you want to drop the diagram into Word, Google Slides, the
README, or any other tool that doesn't speak python-pptx.

Run from the project root:
    .venv/bin/python3.11 reports/build_architecture_png.py

Outputs:
    reports/figures/architecture_diagram.png  (high-DPI, 16:9)
"""

from __future__ import annotations

import os
from pathlib import Path

# Defuse macOS OpenMP conflicts before any heavy imports (matplotlib pulls
# numpy which sometimes triggers the same libomp clash we hit elsewhere).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = PROJECT_ROOT / "reports" / "figures" / "architecture_diagram.png"


# Palette -- matches the .pptx version
PRIMARY = "#103A5C"
ACCENT = "#E86A33"
INGEST_FILL = "#DCE6F0"
QUERY_FILL = "#F7E0CE"
HUB_FILL = "#FCCCA0"
ARROW_COLOR = "#4A4A4A"
TEXT_DARK = "#222222"
SUBTLE = "#666666"


def _draw_node(ax, x, y, w, h, title, subtitle, fill, edge):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        facecolor=fill, edgecolor=edge, linewidth=1.4,
    )
    ax.add_patch(box)
    # With invert_yaxis(), small y is visually higher → title goes near y, subtitle below it.
    ax.text(
        x + w / 2, y + 0.18, title,
        ha="center", va="top",
        fontsize=11, fontweight="bold", color=TEXT_DARK,
    )
    ax.text(
        x + w / 2, y + 0.42, subtitle,
        ha="center", va="top",
        fontsize=8, style="italic", color="#555555",
    )


def _draw_h_arrow(ax, x_start, x_end, y):
    ax.annotate(
        "", xy=(x_end, y), xytext=(x_start, y),
        arrowprops=dict(arrowstyle="-|>", color=ARROW_COLOR, lw=1.4, mutation_scale=14),
    )


def main() -> None:
    fig, ax = plt.subplots(figsize=(13.33, 7.5), dpi=160)
    ax.set_xlim(0, 13.33)
    ax.set_ylim(0, 7.5)
    ax.invert_yaxis()  # match the .pptx coordinate system (y grows downward)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title bar
    title_bar = patches.Rectangle((0, 0), 13.33, 0.9, color=PRIMARY)
    ax.add_patch(title_bar)
    ax.text(
        0.4, 0.45, "Methods — System Architecture",
        ha="left", va="center", fontsize=20, fontweight="bold", color="white",
    )

    # Layout (mirrors _draw_architecture_diagram in build_presentation.py)
    BOX_W, BOX_H = 1.95, 0.95
    GAP = 0.35
    LABEL_H = 0.30
    ROW_GAP = 0.85
    top_y = 1.6

    total_w = 5 * BOX_W + 4 * GAP
    left_margin = (13.33 - total_w) / 2
    box_x = [left_margin + i * (BOX_W + GAP) for i in range(5)]

    label1_y = top_y + 0.05
    row1_y = label1_y + LABEL_H
    label2_y = row1_y + BOX_H + ROW_GAP - 0.10
    row2_y = label2_y + LABEL_H

    # Section labels
    ax.text(0.4, label1_y + 0.18, "INGEST  (build the index)",
            fontsize=12, fontweight="bold", color=PRIMARY)
    ax.text(0.4, label2_y + 0.18, "QUERY  (answer a question)",
            fontsize=12, fontweight="bold", color=ACCENT)

    # INGEST row
    ingest = [
        ("PDFs", "uploaded course\ndocuments"),
        ("Pages", "pypdf\nper-page text"),
        ("Chunks", "heading-aware\n+ recursive split\n(800c, overlap 150)"),
        ("Vectors", "MiniLM-L6-v2\n384-d embeddings\nL2-normalized"),
        ("FAISS Index", "IndexFlatIP\npersisted on disk"),
    ]
    for i, (title, sub) in enumerate(ingest):
        fill = HUB_FILL if i == 4 else INGEST_FILL
        _draw_node(ax, box_x[i], row1_y, BOX_W, BOX_H, title, sub, fill, PRIMARY)
    for i in range(4):
        _draw_h_arrow(ax, box_x[i] + BOX_W, box_x[i + 1], row1_y + BOX_H / 2)

    # QUERY row
    query = [
        ("User question", "free-form\ntext input"),
        ("Top-k retrieval", "Dense / BM25 /\nhybrid (α = 0.65)\ntop-k = 4"),
        ("Confidence guardrail", "if top score\n< min_score (0.25)\n→ refuse"),
        ("LLM", "Gemini-1.5-flash\nor GPT-4o-mini\n(strict prompt)"),
        ("Answer + citations", "[file p.PAGE]\ngrounded or refused"),
    ]
    for i, (title, sub) in enumerate(query):
        _draw_node(ax, box_x[i], row2_y, BOX_W, BOX_H, title, sub, QUERY_FILL, ACCENT)
    for i in range(4):
        _draw_h_arrow(ax, box_x[i] + BOX_W, box_x[i + 1], row2_y + BOX_H / 2)

    # Cross-arrow: FAISS Index (top last) -> Top-k retrieval (bottom 2nd)
    faiss_x = box_x[4] + BOX_W / 2
    topk_x = box_x[1] + BOX_W / 2
    bridge_y = (row1_y + BOX_H + row2_y) / 2

    ax.plot([faiss_x, faiss_x], [row1_y + BOX_H, bridge_y], color=ARROW_COLOR, lw=1.7)
    ax.plot([faiss_x, topk_x], [bridge_y, bridge_y], color=ARROW_COLOR, lw=1.7)
    ax.annotate(
        "", xy=(topk_x, row2_y), xytext=(topk_x, bridge_y),
        arrowprops=dict(arrowstyle="-|>", color=ARROW_COLOR, lw=1.7, mutation_scale=18),
    )
    ax.text(
        (faiss_x + topk_x) / 2, bridge_y - 0.10,
        "retrieve  (same vector space ⇒ same embedding model)",
        ha="center", va="bottom", fontsize=10, style="italic", color=ARROW_COLOR,
    )

    # Caption
    cap_y = row2_y + BOX_H + 0.30
    ax.text(
        13.33 / 2, cap_y,
        "QUERY reads from the FAISS index built by INGEST.  The confidence "
        "guardrail short-circuits the LLM call when retrieval is weak — the "
        "primary anti-hallucination defense.",
        ha="center", va="top", fontsize=10, style="italic", color=SUBTLE,
        wrap=True,
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Wrote {OUT_PATH.relative_to(PROJECT_ROOT)} ({OUT_PATH.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
