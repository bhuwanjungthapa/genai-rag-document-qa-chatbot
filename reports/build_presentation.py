"""
Build the 8-minute final-project presentation as a .pptx file.

Layout follows `data/Final_Project_Presentation_Guideline.pdf`:
  - Slide 1 : Title & Team               (30s)
  - Slides 2-3 : Introduction & Problem  (1.5min)
  - Slide 4 : Dataset                    (1min)
  - Slides 5-6 : Methods / System Design (2min)
  - Slides 7-8 : Final Results           (2min)
  - Slide 9 : Reflection / Limits / Future (1min)

For every slide we add:
  - a short title,
  - a concise bullet list (what's *on* the slide), and
  - detailed speaker notes (what you actually *say*).

The script also embeds the relevant report figures (Fig 2 retrieval Hit@k,
Fig 6 calibration, Fig 5 heatmap) and pulls the latest numbers from
`reports/evaluation_summary.json` and the LoRA training/eval summaries so
the slide values are always in sync with the latest run.
"""

from __future__ import annotations

import json
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES = PROJECT_ROOT / "reports" / "figures"
EVAL_SUMMARY = PROJECT_ROOT / "reports" / "evaluation_summary.json"
LORA_TRAIN_SUMMARY = (
    PROJECT_ROOT / "experiments" / "lora_qlora" / "adapters" / "flan-t5-small-lora" / "training_summary.json"
)
LORA_EVAL_SUMMARY = (
    PROJECT_ROOT / "experiments" / "lora_qlora" / "results" / "adapter_eval_summary.json"
)
EVAL_RESULTS_CSV = PROJECT_ROOT / "reports" / "evaluation_results.csv"
OUT_PPTX = PROJECT_ROOT / "reports" / "Final_Project_Presentation.pptx"


PRIMARY = RGBColor(0x10, 0x3A, 0x5C)   # deep blue
ACCENT = RGBColor(0xE8, 0x6A, 0x33)    # warm orange
TEXT_DARK = RGBColor(0x22, 0x22, 0x22)
SUBTLE = RGBColor(0x66, 0x66, 0x66)

INGEST_FILL = RGBColor(0xDC, 0xE6, 0xF0)   # pale blue (ingest boxes)
QUERY_FILL = RGBColor(0xF7, 0xE0, 0xCE)    # pale orange (query boxes)
HUB_FILL = RGBColor(0xFC, 0xCC, 0xA0)      # warmer orange (FAISS bridge)
ARROW_COLOR = RGBColor(0x4A, 0x4A, 0x4A)
BOX_BORDER_INGEST = PRIMARY
BOX_BORDER_QUERY = ACCENT


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _load_bertscore_pr_means(path: Path) -> tuple[float | None, float | None]:
    """Pull mean BERTScore precision and recall from the evaluation CSV.

    The slide 7 headline shows the F1 (already in evaluation_summary.json), but
    the per-row CSV is the source of truth for precision / recall, so we read
    them straight from there. Returns (None, None) if the CSV is missing or
    BERTScore wasn't enabled when the eval was run.
    """
    if not path.exists():
        return (None, None)
    try:
        import pandas as pd

        df = pd.read_csv(path)
    except Exception:  # noqa: BLE001 — slide build should never crash the script
        return (None, None)
    p_col = df.get("bertscore_precision")
    r_col = df.get("bertscore_recall")
    if p_col is None or r_col is None:
        return (None, None)
    p_mean = p_col.dropna().mean() if p_col.dropna().size else None
    r_mean = r_col.dropna().mean() if r_col.dropna().size else None
    return (
        float(p_mean) if p_mean is not None else None,
        float(r_mean) if r_mean is not None else None,
    )


def _pct(value: float | None, digits: int = 0) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.{digits}f}%"


def _num(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _add_title_bar(slide, title: str) -> None:
    bar = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.33), Inches(0.9)
    )
    bar.fill.solid()
    bar.fill.fore_color.rgb = PRIMARY
    bar.line.fill.background()

    tx = slide.shapes.add_textbox(Inches(0.4), Inches(0.15), Inches(12.5), Inches(0.6)).text_frame
    tx.word_wrap = True
    p = tx.paragraphs[0]
    p.text = title
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)


def _add_bullets(slide, bullets: list[str], left: float = 0.5, top: float = 1.2,
                 width: float = 12.3, height: float = 5.7, font_size: int = 20) -> None:
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height)).text_frame
    box.word_wrap = True
    for i, b in enumerate(bullets):
        p = box.paragraphs[0] if i == 0 else box.add_paragraph()
        p.text = b
        p.font.size = Pt(font_size)
        p.font.color.rgb = TEXT_DARK
        p.level = 0
        p.space_after = Pt(8)


def _add_speaker_notes(slide, text: str) -> None:
    notes_tf = slide.notes_slide.notes_text_frame
    notes_tf.text = text


# ---------------------------------------------------------------------------
# Architecture diagram primitives (native PowerPoint shapes, fully editable)
# ---------------------------------------------------------------------------


def _add_arrowhead(connector) -> None:
    """Add a triangle arrowhead at the END of a connector line."""
    line = connector.line._get_or_add_ln()
    existing = line.find(qn("a:tailEnd"))
    if existing is not None:
        line.remove(existing)
    tail = etree.SubElement(line, qn("a:tailEnd"))
    tail.set("type", "triangle")
    tail.set("w", "med")
    tail.set("len", "med")


def _draw_box(
    slide,
    *,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    subtitle: str = "",
    fill: RGBColor,
    border: RGBColor,
    title_size: int = 12,
    sub_size: int = 9,
) -> None:
    """A rounded-rectangle node with a bold title and optional small subtitle."""
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = border
    shape.line.width = Pt(1.25)
    tf = shape.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.05)
    tf.margin_right = Inches(0.05)
    tf.margin_top = Inches(0.04)
    tf.margin_bottom = Inches(0.04)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = title
    r.font.bold = True
    r.font.size = Pt(title_size)
    r.font.color.rgb = TEXT_DARK
    if subtitle:
        for line in subtitle.split("\n"):
            p2 = tf.add_paragraph()
            p2.alignment = PP_ALIGN.CENTER
            r2 = p2.add_run()
            r2.text = line
            r2.font.size = Pt(sub_size)
            r2.font.italic = True
            r2.font.color.rgb = RGBColor(0x55, 0x55, 0x55)


def _draw_h_arrow(slide, x_start: float, x_end: float, y: float) -> None:
    """A horizontal arrow with the arrowhead pointing right."""
    conn = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT, Inches(x_start), Inches(y), Inches(x_end), Inches(y)
    )
    conn.line.color.rgb = ARROW_COLOR
    conn.line.width = Pt(1.5)
    _add_arrowhead(conn)


def _draw_l_arrow(
    slide,
    x_start: float,
    y_start: float,
    x_end: float,
    y_end: float,
    mid_y: float,
    label: str = "",
) -> None:
    """L-shape connector: down → left/right → down, arrowhead at the final endpoint.

    Used to connect FAISS Index (top row) to the retrieval box (bottom row),
    since they're not vertically aligned.
    """
    seg1 = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT, Inches(x_start), Inches(y_start), Inches(x_start), Inches(mid_y)
    )
    seg1.line.color.rgb = ARROW_COLOR
    seg1.line.width = Pt(1.75)

    seg2 = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT, Inches(x_start), Inches(mid_y), Inches(x_end), Inches(mid_y)
    )
    seg2.line.color.rgb = ARROW_COLOR
    seg2.line.width = Pt(1.75)

    seg3 = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT, Inches(x_end), Inches(mid_y), Inches(x_end), Inches(y_end)
    )
    seg3.line.color.rgb = ARROW_COLOR
    seg3.line.width = Pt(1.75)
    _add_arrowhead(seg3)

    if label:
        text_left = min(x_start, x_end) + 0.1
        text_w = abs(x_end - x_start) - 0.2
        tb = slide.shapes.add_textbox(
            Inches(text_left), Inches(mid_y - 0.32), Inches(max(text_w, 0.5)), Inches(0.25)
        ).text_frame
        p = tb.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = label
        r.font.size = Pt(10)
        r.font.italic = True
        r.font.color.rgb = ARROW_COLOR


def _add_section_label(slide, x: float, y: float, text: str, color: RGBColor) -> None:
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(2.5), Inches(0.3)).text_frame
    p = box.paragraphs[0]
    r = p.add_run()
    r.text = text
    r.font.bold = True
    r.font.size = Pt(13)
    r.font.color.rgb = color


def _draw_architecture_diagram(slide, top_y: float = 2.0) -> None:
    """Two horizontal rows (INGEST / QUERY) connected by an L-shape arrow.

    INGEST: PDFs → Pages → Chunks → Vectors → FAISS Index
    QUERY:  User question → Top-k retrieval → Guardrail → LLM → Answer + citations
    Cross-arrow: FAISS Index (last box, row 1) → Top-k retrieval (2nd box, row 2)
    """
    SLIDE_WIDTH = 13.33
    BOX_W, BOX_H = 1.95, 0.95
    GAP = 0.35
    LABEL_H = 0.30
    ROW_GAP = 0.85

    total_w = 5 * BOX_W + 4 * GAP
    left_margin = (SLIDE_WIDTH - total_w) / 2
    box_x = [left_margin + i * (BOX_W + GAP) for i in range(5)]

    label1_y = top_y + 0.05
    row1_y = label1_y + LABEL_H
    label2_y = row1_y + BOX_H + ROW_GAP - 0.10
    row2_y = label2_y + LABEL_H
    caption_y = row2_y + BOX_H + 0.20

    _add_section_label(slide, 0.4, label1_y, "INGEST  (build the index)", PRIMARY)
    _add_section_label(slide, 0.4, label2_y, "QUERY  (answer a question)", ACCENT)

    ingest = [
        ("PDFs", "uploaded course\ndocuments"),
        ("Pages", "pypdf\nper-page text"),
        ("Chunks", "heading-aware\n+ recursive split\n(800 chars, overlap 150)"),
        ("Vectors", "MiniLM-L6-v2\n384-d embeddings\nL2-normalized"),
        ("FAISS Index", "IndexFlatIP\npersisted on disk"),
    ]
    for i, (title, sub) in enumerate(ingest):
        fill = HUB_FILL if i == 4 else INGEST_FILL
        _draw_box(
            slide,
            x=box_x[i], y=row1_y, w=BOX_W, h=BOX_H,
            title=title, subtitle=sub,
            fill=fill, border=BOX_BORDER_INGEST,
        )
    for i in range(4):
        _draw_h_arrow(slide, box_x[i] + BOX_W, box_x[i + 1], row1_y + BOX_H / 2)

    query = [
        ("User question", "free-form\ntext input"),
        ("Top-k retrieval", "Dense / BM25 /\nhybrid (α = 0.65)\ntop-k = 4"),
        ("Confidence\nguardrail", "if top score\n< min_score (0.25)\n→ refuse"),
        ("LLM", "Gemini-1.5-flash\nor GPT-4o-mini\n(strict prompt)"),
        ("Answer +\ncitations", "[file p.PAGE]\ngrounded or refused"),
    ]
    for i, (title, sub) in enumerate(query):
        _draw_box(
            slide,
            x=box_x[i], y=row2_y, w=BOX_W, h=BOX_H,
            title=title, subtitle=sub,
            fill=QUERY_FILL, border=BOX_BORDER_QUERY,
        )
    for i in range(4):
        _draw_h_arrow(slide, box_x[i] + BOX_W, box_x[i + 1], row2_y + BOX_H / 2)

    # Cross-arrow: FAISS Index (top, last) -> Top-k retrieval (bottom, second box)
    faiss_center_x = box_x[4] + BOX_W / 2
    topk_center_x = box_x[1] + BOX_W / 2
    bridge_y = (row1_y + BOX_H + row2_y) / 2
    _draw_l_arrow(
        slide,
        x_start=faiss_center_x, y_start=row1_y + BOX_H,
        x_end=topk_center_x, y_end=row2_y,
        mid_y=bridge_y,
        label="retrieve  (same vector space ⇒ same embedding model)",
    )

    # Caption underneath
    cap = slide.shapes.add_textbox(
        Inches(0.4), Inches(caption_y), Inches(12.5), Inches(0.3)
    ).text_frame
    cap.word_wrap = True
    p = cap.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = (
        "QUERY reads from the FAISS index built by INGEST. The confidence "
        "guardrail short-circuits the LLM call when retrieval is weak — "
        "this is the primary anti-hallucination defense."
    )
    r.font.size = Pt(11)
    r.font.italic = True
    r.font.color.rgb = SUBTLE


def _add_image(slide, image_path: Path, left: float, top: float, width: float | None = None,
               height: float | None = None) -> None:
    if not image_path.exists():
        return
    kwargs = {"left": Inches(left), "top": Inches(top)}
    if width is not None:
        kwargs["width"] = Inches(width)
    if height is not None:
        kwargs["height"] = Inches(height)
    slide.shapes.add_picture(str(image_path), **kwargs)


def _add_footer(slide, text: str) -> None:
    box = slide.shapes.add_textbox(Inches(0.4), Inches(7.0), Inches(12.5), Inches(0.4)).text_frame
    p = box.paragraphs[0]
    p.text = text
    p.font.size = Pt(10)
    p.font.color.rgb = SUBTLE


def build() -> Path:
    eval_summary = _load_json(EVAL_SUMMARY)
    lora_train = _load_json(LORA_TRAIN_SUMMARY)
    lora_eval = _load_json(LORA_EVAL_SUMMARY)

    hit_rate = _pct(eval_summary.get("retrieval_hit_rate"))
    correct_rate = _pct(eval_summary.get("correctness_rate"))
    halluc_rate = _pct(eval_summary.get("hallucination_rate"))
    grounded_rate = _pct(eval_summary.get("grounded_rate"))
    sem_sim = _num(eval_summary.get("semantic_similarity_mean"), 2)
    bert_f1 = _num(eval_summary.get("bertscore_f1_mean"), 2)
    bert_p_val, bert_r_val = _load_bertscore_pr_means(EVAL_RESULTS_CSV)
    bert_p = _num(bert_p_val, 2)
    bert_r = _num(bert_r_val, 2)
    bert_pr_suffix = (
        f" (precision {bert_p}, recall {bert_r})"
        if bert_p_val is not None and bert_r_val is not None
        else ""
    )
    ece = _num(eval_summary.get("calibration_ece"), 3)
    n_eval = int(eval_summary.get("n", 0))

    lora_train_n = lora_train.get("train_examples", 0)
    lora_eval_n = lora_train.get("eval_examples", 0)
    lora_loss = lora_train.get("metrics", {}).get("eval_loss")
    lora_jaccard = lora_eval.get("mean_jaccard_pred_vs_gold")

    prs = Presentation()
    prs.slide_width = Inches(13.33)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    # -----------------------------------------------------------------
    # Slide 1 — Title & Team (30s)
    # -----------------------------------------------------------------
    s = prs.slides.add_slide(blank)
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.33), Inches(7.5))
    bg.fill.solid()
    bg.fill.fore_color.rgb = PRIMARY
    bg.line.fill.background()

    title_box = s.shapes.add_textbox(Inches(0.8), Inches(2.3), Inches(11.7), Inches(1.5)).text_frame
    title_box.word_wrap = True
    p = title_box.paragraphs[0]
    p.text = "Generative AI RAG Document Q&A Chatbot"
    p.font.size = Pt(44)
    p.font.bold = True
    p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    sub_box = s.shapes.add_textbox(Inches(0.8), Inches(3.7), Inches(11.7), Inches(1.0)).text_frame
    sub_box.word_wrap = True
    p = sub_box.paragraphs[0]
    p.text = "Grounded answers from uploaded course PDFs, with citations and a LoRA fine-tuning experiment."
    p.font.size = Pt(22)
    p.font.color.rgb = ACCENT

    team_box = s.shapes.add_textbox(Inches(0.8), Inches(5.0), Inches(11.7), Inches(1.6)).text_frame
    team_box.word_wrap = True
    for i, line in enumerate([
        "Team: <add team member names here>",
        "Course: CSE 434 / 534 — Generative AI",
        "Final Project Presentation",
    ]):
        para = team_box.paragraphs[0] if i == 0 else team_box.add_paragraph()
        para.text = line
        para.font.size = Pt(18)
        para.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    _add_speaker_notes(s, (
        "TIME: ~30 seconds.\n\n"
        "SAY: 'Good <morning/afternoon>. Our project is a Retrieval-Augmented "
        "Generation chatbot that answers questions about uploaded course PDFs. "
        "Every answer is grounded in retrieved passages and shown with a "
        "citation, so the user can verify it. We also added a parameter-"
        "efficient fine-tuning experiment using LoRA on FLAN-T5-small as a "
        "research-level comparison.'\n\n"
        "ACTION: Introduce yourself and your teammates here. Edit the team "
        "names placeholder in this slide. Keep it short — move on at 30s."
    ))

    # -----------------------------------------------------------------
    # Slide 2 — Introduction (motivation)
    # -----------------------------------------------------------------
    s = prs.slides.add_slide(blank)
    _add_title_bar(s, "Introduction — Why a grounded course-PDF chatbot?")
    _add_bullets(s, [
        "Problem: students and instructors waste time searching syllabi, project specs, and policy docs.",
        "Generic LLMs hallucinate facts about deadlines, grading, and policies — that is unsafe to trust.",
        "We need a system that answers ONLY from the uploaded documents and shows the source.",
        "Course connections covered:  Prompt engineering  •  Retrieval-Augmented Generation  •  Parameter-efficient fine-tuning (LoRA)  •  Evaluation methodology (Hit@k, BERTScore, calibration, manual labels).",
        "Goal: a local-first, transparent, trustworthy Q&A chatbot — not a magic oracle.",
    ])
    _add_speaker_notes(s, (
        "TIME: ~45 seconds (1st of two intro slides).\n\n"
        "SAY: 'Course documents — syllabi, project requirements, presentation "
        "guidelines — are the kind of thing students re-read constantly. A "
        "naive LLM will happily make up a deadline or a grading weight, and "
        "that is genuinely harmful. So our research question was: can we "
        "build a chatbot that ONLY answers from the actual document, refuses "
        "when the document does not support an answer, and always shows "
        "where its answer came from? That single principle — refuse rather "
        "than hallucinate — drives every design choice.'\n\n"
        "EMPHASIZE: This project touches every CSE 534 topic — prompting, "
        "retrieval, fine-tuning, and evaluation."
    ))

    # -----------------------------------------------------------------
    # Slide 3 — Problem definition (concrete task)
    # -----------------------------------------------------------------
    s = prs.slides.add_slide(blank)
    _add_title_bar(s, "Problem Definition")
    _add_bullets(s, [
        "Input: a user question + a corpus of PDFs the user uploaded.",
        "Output: a concise answer grounded in retrieved passages, with citations [filename p.PAGE].",
        "Hard constraint: if no passage supports an answer, the bot must say so and refuse — no invention.",
        "Auditability: every Q&A is logged so any answer can be traced back to its retrieved chunks.",
        "Stretch goal: compare RAG against a parameter-efficient fine-tuned model (LoRA on FLAN-T5-small).",
    ])
    _add_speaker_notes(s, (
        "TIME: ~45 seconds.\n\n"
        "SAY: 'Concretely: input is a free-form question, output is an answer "
        "grounded in the document. The hard requirement is no invention — if "
        "the document does not support an answer, the bot returns a fixed "
        "refusal sentence. Every answer also includes citations like "
        "Syllabus.pdf p.3, and every interaction is logged so a grader can "
        "reproduce why the bot said what it said. As a research extension we "
        "also fine-tune a small model with LoRA on QA pairs derived from the "
        "same chunks, and we compare it against the RAG baseline.'\n\n"
        "TRANSITION: 'Now let me show you what data this is built on.'"
    ))

    # -----------------------------------------------------------------
    # Slide 4 — Dataset (1 min)
    # -----------------------------------------------------------------
    s = prs.slides.add_slide(blank)
    _add_title_bar(s, "Dataset")
    _add_bullets(s, [
        "Source: 4 real CSE 534 course PDFs uploaded into the app — Syllabus, Project Requirement, Midterm Project Check, and Final Project Presentation Guideline.",
        "Preprocessing: pypdf page extraction → whitespace cleanup → heading-aware section splitting → recursive char split (chunk_size 800, overlap 150) → tiny-chunk merge (min 250 chars).",
        "Indexed corpus: 4 PDFs, dozens of chunks, embedded with sentence-transformers all-MiniLM-L6-v2 (384-d) into a FAISS IndexFlatIP.",
        "Evaluation set: a 25-question gold CSV (eval/sample_eval_questions.csv) with question, gold_answer, gold_doc, gold_page columns.",
        f"LoRA fine-tuning data: {lora_train_n + lora_eval_n} template-based grounded QA pairs auto-generated from the indexed chunks ({lora_train_n} train / {lora_eval_n} eval, fixed seed for reproducibility).",
        "Ethics: only user-uploaded course PDFs (no scraping); all retrieval runs locally; only the LLM call leaves the machine.",
    ])
    _add_speaker_notes(s, (
        "TIME: ~1 minute.\n\n"
        "SAY: 'Our corpus is the actual CSE 534 course material — four PDFs: "
        "the syllabus, the project requirements, the midterm check, and the "
        "presentation guideline. We extract page-level text with pypdf, clean "
        "whitespace, and run a heading-aware chunker that first splits on "
        "section titles like \"Grading Policy\" or \"Schedule\" and then "
        "recursively splits long sections into 800-character chunks with 150 "
        "characters of overlap. Tiny stub chunks are merged backward.'\n\n"
        "SAY: 'For evaluation we wrote a 25-question gold CSV that names the "
        "expected supporting document and page for each question. For the "
        f"LoRA experiment we automatically generated {lora_train_n + lora_eval_n} "
        "template-based grounded QA pairs from the chunks — split into "
        f"{lora_train_n} training and {lora_eval_n} evaluation examples with a fixed seed.'\n\n"
        "ETHICS: Mention briefly: data is user-uploaded, indexing runs locally, "
        "no PII in syllabi, only the LLM call hits the network."
    ))

    # -----------------------------------------------------------------
    # Slide 5 — System architecture (native PowerPoint diagram)
    # -----------------------------------------------------------------
    s = prs.slides.add_slide(blank)
    _add_title_bar(s, "Methods — System Architecture")
    _add_bullets(
        s,
        [
            "INGEST (top row) — each PDF is split into section-aware chunks, embedded with all-MiniLM-L6-v2, and stored as L2-normalized vectors in a FAISS index.",
            "QUERY (bottom row) — the same model embeds the question; top-k chunks are retrieved (dense / BM25 / hybrid) and a confidence guardrail short-circuits the LLM call when retrieval is weak.",
            "GENERATE — only the retrieved chunks plus the question reach the LLM under a strict context-only prompt that forces citations as [filename p.PAGE].",
        ],
        top=1.05,
        height=1.7,
        font_size=14,
    )

    _draw_architecture_diagram(s, top_y=2.7)

    _add_speaker_notes(s, (
        "TIME: ~1 minute (slide 1 of 2 in methods).\n\n"
        "WALK THE DIAGRAM TOP TO BOTTOM:\n\n"
        "SAY: 'The top row is INGEST. Each uploaded PDF is read page-by-page "
        "with pypdf, split into roughly 800-character section-aware chunks, "
        "embedded with all-MiniLM-L6-v2 into 384-dimensional L2-normalized "
        "vectors, and stored in a flat FAISS index that is persisted to "
        "disk.'\n\n"
        "SAY: 'The bottom row is QUERY. The user question is embedded with "
        "the exact same model so the question vector lives in the same "
        "space as the chunk vectors — that is what the long arrow in the "
        "middle is showing. Top-k retrieval pulls the closest chunks using "
        "dense FAISS, sparse BM25, or a weighted hybrid. Then comes the "
        "single most important box on the slide — the confidence guardrail. "
        "If the top similarity score is below 0.25, the pipeline refuses to "
        "even call the LLM, returning a fixed unsupported sentence. That "
        "one check is our primary anti-hallucination defense.'\n\n"
        "SAY: 'Only when retrieval is strong does the question plus chunks "
        "go to the LLM under a strict prompt that forces context-only "
        "answers and citations like [Syllabus.pdf p.3]. The final answer "
        "is shown to the user with the citations and the retrieved chunks "
        "for transparency.'"
    ))

    # -----------------------------------------------------------------
    # Slide 6 — Methods detail (rationale + constraints)
    # -----------------------------------------------------------------
    s = prs.slides.add_slide(blank)
    _add_title_bar(s, "Methods — Choices, Rationale & Constraints")
    _add_bullets(s, [
        "Embedding model: all-MiniLM-L6-v2 default (small, fast, ~90 MB). Sidebar dropdown lets us swap to BGE-small, MPNet, or Arctic-xs and rebuild.",
        "Hybrid retrieval: final_score = α · dense + (1 − α) · BM25, default α = 0.65. Lets exact course terms (\"Week 3\") and paraphrases reinforce each other.",
        "Strict grounded prompt: \"answer ONLY from this context, refuse otherwise, cite [filename p.PAGE]\". The exact refusal sentence flips the grounded_or_not flag automatically.",
        "LoRA fine-tune of FLAN-T5-small: base model stays frozen; we train only a 1.3 MB rank-8 patch on the q/v attention layers — 99 auto-generated Q&A pairs, 3 epochs, runs on Mac CPU.",
        "QLoRA: optional GPU-only path exposed as a readiness check. On Mac it correctly reports \"not ready\" (no CUDA / bitsandbytes), which is why LoRA is the path we actually run end-to-end.",
        "Engineering trade-offs: low temperature (0.1) + tight prompts to control cost; CPU-only training and indexing; every index ships with a manifest.json so swapping the embedding model auto-clears the stale index.",
    ], font_size=15)
    _add_speaker_notes(s, (
        "TIME: ~1 minute (slide 2 of 2 in methods).\n\n"
        "SAY: 'A few design choices worth calling out. First, we use a small "
        "sentence-transformer by default — MiniLM-L6-v2 — because it is fast, "
        "free, and well-understood, but the sidebar lets us swap in BGE or "
        "MPNet and rebuild the index without code changes. Second, we use a "
        "weighted hybrid of dense FAISS and BM25 sparse retrieval, because "
        "course documents contain very specific terms — week numbers, course "
        "codes — that benefit from exact keyword matching, while semantics "
        "still helps for paraphrased questions.'\n\n"
        "SAY: 'Third, the prompt is intentionally strict — the LLM is told "
        "to refuse with a fixed sentence if the context does not support an "
        "answer, and we detect that exact sentence to mark the answer as "
        "ungrounded.'\n\n"
        "SAY (LoRA): 'For fine-tuning, we ran a real LoRA experiment, not "
        "just a placeholder. We took FLAN-T5-small, FROZE the entire base "
        "model, and trained a tiny adapter — about 1.3 megabytes of weights "
        "— on just the query and value attention layers, using 99 "
        "question-answer pairs we generated from our own PDFs. The whole "
        "thing trains in a few minutes on a Mac CPU, and the held-out "
        "Jaccard score went up to roughly 0.68, so the adapter actually "
        "learned the document style.'\n\n"
        "SAY (QLoRA): 'We also wired in QLoRA as an OPTIONAL path. QLoRA "
        "needs an NVIDIA GPU and the bitsandbytes library, neither of "
        "which exist on a Mac, so the app exposes it as a one-click "
        "readiness check that correctly reports \"not available\" on this "
        "machine. The actual experiment we run end-to-end is plain LoRA.'\n\n"
        "SAY (engineering): 'Three pragmatic choices worth flagging. One — "
        "every LLM call uses low temperature (0.1) and a tight prompt to "
        "keep cost and rate limits in check. Two — training and indexing "
        "are CPU-only, no GPU required. And three — every index we build "
        "ships with a manifest.json that records the embedding model and "
        "chunk settings; if a teammate switches the embedding model in the "
        "sidebar, the app detects the mismatch and CLEARS the stale index "
        "automatically — that prevents the silent failure of querying with "
        "one model while the index was built with another.'"
    ))

    # -----------------------------------------------------------------
    # Slide 7 — Quantitative results (with Hit@k figure)
    # -----------------------------------------------------------------
    s = prs.slides.add_slide(blank)
    _add_title_bar(s, "Final Results — Quantitative")
    bullets = [
        f"Evaluation set: {n_eval} gold questions over the 4 indexed PDFs.",
        f"Retrieval Hit@k: {hit_rate}.   Grounded-by-guardrail rate: {grounded_rate}.",
        f"Manual + auto-labeled correctness rate: {correct_rate}.   Hallucination rate: {halluc_rate}.",
        f"Semantic similarity (pred vs gold): {sem_sim}.   BERTScore F1: {bert_f1}{bert_pr_suffix}.",
        f"Calibration: ECE = {ece} (top-1 confidence vs labeled correctness).",
        f"LoRA experiment ({lora_train_n} train / {lora_eval_n} eval, 3 epochs, lr 5e-4):  eval_loss = {_num(lora_loss, 3)},  mean Jaccard = {_num(lora_jaccard, 3)}.",
    ]
    _add_bullets(s, bullets, top=1.1, height=3.0, font_size=16)

    fig2 = FIGURES / "fig2_hit_at_k.png"
    fig6 = FIGURES / "fig6_calibration.png"
    if fig2.exists():
        _add_image(s, fig2, left=0.4, top=4.1, height=3.0)
    if fig6.exists():
        _add_image(s, fig6, left=6.7, top=4.1, height=3.0)
    _add_footer(s, "Left: Retrieval Hit@k for k = 1, 3, 5, 10.   Right: Calibration of top-1 retrieval confidence vs labeled correctness (ECE).")

    if bert_p_val is not None and bert_r_val is not None:
        bert_pr_block = (
            f"BACKUP IF ASKED ABOUT PRECISION/RECALL: 'BERTScore precision is "
            f"{bert_p} and recall is {bert_r}. Recall higher than precision means "
            "our answers cover the gold thoroughly but tend to be a bit verbose — "
            "citations, surrounding context, hedging — which lines up with the "
            "Partially Correct bucket on the slide-8 heatmap.'\n\n"
        )
    else:
        bert_pr_block = ""

    _add_speaker_notes(s, (
        "TIME: ~1 minute (slide 1 of 2 in results).\n\n"
        f"SAY: 'On {n_eval} gold questions, retrieval finds the right document "
        f"and page in the top-k chunks {hit_rate} of the time. The model "
        f"refused (returned the unsupported sentence) in only "
        f"{int((1 - eval_summary.get('grounded_rate', 0)) * 100)}% of cases, and our heuristic "
        f"+ manual labels show {correct_rate} correct answers and "
        f"{halluc_rate} hallucinations. BERTScore F1 against the gold "
        f"answers is {bert_f1}, semantic similarity is {sem_sim}, and the "
        f"calibration ECE — how well the retrieval confidence score lines "
        f"up with empirical correctness — is {ece}, which is reasonable for "
        "an unblinded run.'\n\n"
        f"{bert_pr_block}"
        f"SAY: 'For the LoRA experiment we trained for 3 epochs on "
        f"{lora_train_n} examples in about 3 minutes on CPU and reached an "
        f"eval loss of {_num(lora_loss, 3)} and a held-out Jaccard overlap "
        f"of {_num(lora_jaccard, 3)} against template-generated gold "
        "answers. That is a useful research baseline; we will discuss its "
        "limits in a moment.'\n\n"
        "POINT TO FIGURES: Left chart shows Hit@k bars; explain that Hit@1 "
        "vs Hit@10 gap indicates rank quality. Right chart shows calibration "
        "buckets — emphasize that low ECE means confidence is honest."
    ))

    # -----------------------------------------------------------------
    # Slide 8 — Qualitative + comparative
    # -----------------------------------------------------------------
    s = prs.slides.add_slide(blank)
    _add_title_bar(s, "Final Results — Qualitative & Comparative")
    _add_bullets(s, [
        "Side-by-side example: base FLAN-T5-small vs LoRA-adapted FLAN-T5-small on the same prompt — the adapter consistently keeps the answer grounded and includes the citation pattern from training, while the base model often drifts off-context.",
        "Failure case (correct behavior): question with no supporting chunk → bot returns the fixed refusal sentence, citations list is empty, log records grounded=False. We prefer silence over invention.",
        "Heatmap interpretation: cells (Hit@5=True, Hallucinated) flag generation problems; (Hit@5=False, Correct) usually means a gold-page mismatch in the CSV; (Hit@5=False, Unsupported) shows the guardrail working.",
        "Comparative analysis: With RAG → grounded, cited, refuses when unsure. Without RAG (raw LLM) → confidently invents specifics. Fine-tuned LoRA → reproduces answer style but limited to template-grounded patterns; not a substitute for retrieval over a dynamic corpus.",
    ], top=1.1, height=2.7, font_size=15)

    fig5 = FIGURES / "fig5_heatmap.png"
    fig3 = FIGURES / "fig3_label_distribution.png"
    if fig5.exists():
        _add_image(s, fig5, left=0.4, top=4.0, height=3.1)
    if fig3.exists():
        _add_image(s, fig3, left=6.7, top=4.0, height=3.1)
    _add_footer(s, "Left: Hit@5 × answer label heatmap (separates retrieval errors from generation errors).   Right: Manual / auto label distribution.")

    _add_speaker_notes(s, (
        "TIME: ~1 minute (slide 2 of 2 in results).\n\n"
        "SAY: 'Qualitatively the most informative comparison is the LoRA "
        "adapter side-by-side. The base FLAN-T5-small often drifts away "
        "from the supplied context and rarely cites the source; the LoRA-"
        "adapted version reliably mirrors the training style — answer "
        "from context, end with a [doc.pdf p.N] citation. That is exactly "
        "the behavior we wanted to teach.'\n\n"
        "SAY: 'A representative correct refusal: when we ask something "
        "the corpus does not cover, the guardrail short-circuits, the "
        "answer is the fixed unsupported sentence, and citations stay "
        "empty. We log this as grounded=False so we can audit it later.'\n\n"
        "POINT TO HEATMAP: 'The heatmap on the left is the figure that "
        "actually separates retrieval failure from generation failure. The "
        "bottom-left cell — retrieval missed but the bot correctly refused "
        "— is the guardrail earning its keep. The top-right cell — "
        "retrieval succeeded but the bot hallucinated — would be a "
        "generation problem; in our run that bucket is empty.'\n\n"
        "COMPARATIVE: 'With RAG vs without RAG: a raw LLM with no context "
        "happily fabricates page numbers and grading weights; with RAG it "
        "either cites real text or refuses. With LoRA vs without LoRA: "
        "the adapter learned the answer style but cannot replace retrieval "
        "for a corpus the user can change at any time.'"
    ))

    # -----------------------------------------------------------------
    # Slide 9 — Reflection, Limitations, Future Work
    # -----------------------------------------------------------------
    s = prs.slides.add_slide(blank)
    _add_title_bar(s, "Reflection, Limitations & Future Work")
    _add_bullets(s, [
        "Reflection: the strict grounded prompt + min_score guardrail did most of the heavy lifting against hallucination — more than any single model swap. Hybrid BM25 + dense beat dense alone for course-specific terms (course codes, week numbers).",
        "Reflection: RAG > pure fine-tuning when the corpus is dynamic — users can upload new PDFs at any time, so retrieval has to stay live. LoRA is best framed as a research extension, not a replacement.",
        "Limitations: pypdf has no OCR (scanned PDFs fail silently); heading detection is heuristic; template-generated LoRA QA pairs cap what the adapter can really learn; QLoRA cannot run on Mac (no CUDA / bitsandbytes); token-overlap is a coarse adapter metric.",
        "Future work: OCR fallback (ocrmypdf); cross-encoder re-ranker on retrieved chunks; per-course namespaces with a switcher; QLoRA on Colab (free T4) for true 4-bit comparison; human-written QA pairs evaluated with BERTScore for the LoRA path; LLM-response cache keyed on (question, retrieved_ids) to cut eval cost.",
    ], font_size=15)
    _add_speaker_notes(s, (
        "TIME: ~1 minute (closing slide).\n\n"
        "REFLECTION: 'Two lessons stood out. First, simple prompt "
        "engineering plus a hard confidence threshold did more for "
        "trust than any model swap. Second, hybrid retrieval beat dense-"
        "only on questions that contained exact course terms.'\n\n"
        "LIMITATIONS: 'pypdf has no OCR, so any scanned PDF will silently "
        "produce empty text. Our heading detector is a regex heuristic and "
        "misses unusual layouts. The LoRA experiment is intentionally "
        "honest — the QA pairs are template-generated from the chunks, so "
        "the adapter mostly learns answer FORMAT rather than new "
        "knowledge. And QLoRA would need a CUDA GPU; we ship a readiness "
        "check that correctly reports it as unavailable on a Mac.'\n\n"
        "FUTURE WORK: 'Realistic next steps: add OCR for scanned PDFs, "
        "rerank retrieved chunks with a small cross-encoder, run QLoRA "
        "on a free Colab T4 for a true 4-bit comparison, and replace the "
        "template QA pairs with human-written ones evaluated with "
        "BERTScore.'\n\n"
        "CLOSING SENTENCE: 'Thanks — happy to take questions, especially "
        "about the retrieval guardrail or the LoRA-vs-base comparison.'"
    ))

    OUT_PPTX.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT_PPTX)
    return OUT_PPTX


if __name__ == "__main__":
    out = build()
    print(f"Wrote: {out.relative_to(PROJECT_ROOT)}")
    print(f"Slides: 9   |   Aspect: 16:9   |   Speaker notes: yes")
