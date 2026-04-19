"""
Course Syllabus QA Chatbot - Streamlit UI.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st

from config import (
    AppConfig,
    DATA_DIR,
    EVAL_DIR,
    QA_LOG_FILE,
    STARTER_QUESTIONS,
)
from src.evaluation import (
    VALID_LABELS,
    load_questions_csv,
    run_evaluation,
    summarize_evaluation,
)
from src.llm_client import describe_available
from src.rag_pipeline import RAGPipeline
from src.utils import safe_filename


st.set_page_config(
    page_title="Course Syllabus QA Chatbot",
    page_icon=":books:",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading embedding model...")
def _bootstrap_pipeline(
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_size: int,
    top_k: int,
    min_score: float,
) -> RAGPipeline:
    """Build the pipeline once per session; cached on config values."""
    cfg = AppConfig()
    cfg.embedding_model = embedding_model
    cfg.chunk_size = chunk_size
    cfg.chunk_overlap = chunk_overlap
    cfg.min_chunk_size = min_chunk_size
    cfg.top_k = top_k
    cfg.min_score = min_score
    pipeline = RAGPipeline(cfg)
    pipeline.load_index_if_exists()
    return pipeline


def _init_state() -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []   # list[dict]
    if "eval_results" not in st.session_state:
        st.session_state.eval_results = None
    if "last_manifest" not in st.session_state:
        st.session_state.last_manifest = None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def render_sidebar(base_cfg: AppConfig) -> dict:
    st.sidebar.header(":gear: Configuration")

    available = describe_available(base_cfg.gemini_api_key, base_cfg.openai_api_key)
    if available:
        st.sidebar.success(f"LLM providers detected: {available}")
    else:
        st.sidebar.warning(
            "No LLM API key detected. Ingestion/retrieval will work, but answer "
            "generation is disabled. Add a key in `.env` and restart."
        )

    # Provider choice
    provider_options = ["gemini", "openai"]
    default_idx = 0 if base_cfg.gemini_api_key or not base_cfg.openai_api_key else 1
    provider = st.sidebar.selectbox(
        "LLM provider",
        options=provider_options,
        index=default_idx,
        help="Pick which LLM answers questions. Requires the matching API key in `.env`.",
    )

    st.sidebar.subheader("Chunking")
    chunk_size = st.sidebar.number_input(
        "Chunk size (chars)", min_value=200, max_value=4000, value=base_cfg.chunk_size, step=50
    )
    chunk_overlap = st.sidebar.number_input(
        "Chunk overlap (chars)", min_value=0, max_value=1000, value=base_cfg.chunk_overlap, step=25
    )
    min_chunk_size = st.sidebar.number_input(
        "Min chunk size (chars)", min_value=50, max_value=1500, value=base_cfg.min_chunk_size, step=25
    )

    st.sidebar.subheader("Retrieval")
    top_k = st.sidebar.slider("Top-k chunks", min_value=1, max_value=10, value=base_cfg.top_k)
    min_score = st.sidebar.slider(
        "Min retrieval score",
        min_value=0.0, max_value=1.0,
        value=float(base_cfg.min_score), step=0.05,
        help="If the top retrieved chunk scores below this, the assistant refuses to answer.",
    )

    st.sidebar.subheader("Documents")
    uploaded = st.sidebar.file_uploader(
        "Upload course PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )
    build_clicked = st.sidebar.button(
        ":hammer_and_wrench: Build / Rebuild Index",
        use_container_width=True,
        type="primary",
    )

    return {
        "provider": provider,
        "chunk_size": int(chunk_size),
        "chunk_overlap": int(chunk_overlap),
        "min_chunk_size": int(min_chunk_size),
        "top_k": int(top_k),
        "min_score": float(min_score),
        "uploaded": uploaded,
        "build_clicked": build_clicked,
    }


def _persist_uploads(uploaded_files) -> list[Path]:
    saved: list[Path] = []
    for f in uploaded_files or []:
        fname = safe_filename(f.name)
        target = DATA_DIR / fname
        with open(target, "wb") as out:
            out.write(f.read())
        saved.append(target)
    return saved


# ---------------------------------------------------------------------------
# Chat tab
# ---------------------------------------------------------------------------


def render_chat_tab(pipeline: RAGPipeline) -> None:
    st.subheader(":speech_balloon: Ask a question about your course documents")

    if not pipeline.is_ready():
        st.info(
            "No index loaded yet. Upload PDFs in the sidebar and click "
            "**Build / Rebuild Index** to start."
        )

    # Starter questions
    st.markdown("**Starter questions:**")
    cols = st.columns(len(STARTER_QUESTIONS))
    starter_click = None
    for col, q in zip(cols, STARTER_QUESTIONS):
        if col.button(q, use_container_width=True):
            starter_click = q

    question = st.text_input(
        "Your question",
        value=starter_click or "",
        placeholder="e.g. What is the late assignment policy?",
    )
    ask = st.button("Ask", type="primary")

    if ask and question.strip():
        if not pipeline.is_ready():
            st.error("Please build the index first (sidebar).")
        else:
            with st.spinner("Retrieving relevant chunks and generating answer..."):
                result = pipeline.answer(question)
            st.session_state.chat_history.append(
                {
                    "question": result.question,
                    "answer": result.answer,
                    "citations": result.citations,
                    "retrieved": result.retrieved,
                    "grounded": result.grounded,
                    "provider": result.provider,
                }
            )

    # Render chat history (newest first)
    for turn in reversed(st.session_state.chat_history):
        with st.container(border=True):
            st.markdown(f"**You:** {turn['question']}")
            st.markdown(f"**Assistant** _(provider: `{turn['provider']}`)_")
            st.write(turn["answer"])

            if turn["citations"]:
                st.markdown("**Citations:** " + " ".join(f"`{c}`" for c in turn["citations"]))
            elif not turn["grounded"]:
                st.warning("No citations - the assistant did not find strong supporting evidence.")

            with st.expander("Retrieved Chunks", expanded=False):
                if not turn["retrieved"]:
                    st.caption("No chunks were retrieved.")
                for i, r in enumerate(turn["retrieved"], start=1):
                    page_label = (
                        f"p.{r.page_start}"
                        if r.page_start == r.page_end
                        else f"p.{r.page_start}-{r.page_end}"
                    )
                    st.markdown(
                        f"**[{i}]** `{r.doc_name}` {page_label} "
                        f"| score: `{r.score:.3f}` "
                        f"| section: _{r.section_title or '—'}_"
                    )
                    st.text(r.raw_text)


# ---------------------------------------------------------------------------
# Documents tab
# ---------------------------------------------------------------------------


def render_documents_tab(pipeline: RAGPipeline) -> None:
    st.subheader(":page_facing_up: Indexed Documents")

    if not pipeline.is_ready():
        st.info("Index is empty. Upload PDFs and build the index first.")
        return

    summary = pipeline.document_summary()
    summary = summary.rename(
        columns={
            "doc_name": "Document",
            "num_chunks": "Chunks",
            "page_start": "First page indexed",
            "page_end": "Last page indexed",
        }
    )
    st.dataframe(summary, use_container_width=True)

    total_pages_est = int(summary["Last page indexed"].max()) if not summary.empty else 0
    total_chunks = int(summary["Chunks"].sum()) if not summary.empty else 0
    c1, c2, c3 = st.columns(3)
    c1.metric("Documents", len(summary))
    c2.metric("Total chunks", total_chunks)
    c3.metric("Max page seen", total_pages_est)

    st.markdown("#### Example chunks")
    examples = pipeline.example_chunks(n_per_doc=2)
    for _, row in examples.iterrows():
        with st.container(border=True):
            st.markdown(
                f"`{row['doc_name']}` — p.{row['page_start']}-{row['page_end']} "
                f"— _{row.get('section_title') or '—'}_"
            )
            st.text(str(row["raw_text"])[:600] + ("..." if len(str(row["raw_text"])) > 600 else ""))


# ---------------------------------------------------------------------------
# Evaluation tab
# ---------------------------------------------------------------------------


def render_evaluation_tab(pipeline: RAGPipeline) -> None:
    st.subheader(":bar_chart: Evaluation")

    st.markdown(
        "Upload a CSV of evaluation questions (columns: "
        "`question`, `gold_answer`, optional `gold_doc`, `gold_page`), "
        "or use the sample bundled in `eval/sample_eval_questions.csv`."
    )

    sample_path = EVAL_DIR / "sample_eval_questions.csv"
    col1, col2 = st.columns(2)
    with col1:
        use_sample = st.checkbox("Use bundled sample CSV", value=True)
    with col2:
        uploaded_csv = st.file_uploader("Or upload your own CSV", type=["csv"])

    df: pd.DataFrame | None = None
    if uploaded_csv is not None:
        df = pd.read_csv(uploaded_csv)
    elif use_sample and sample_path.exists():
        df = load_questions_csv(sample_path)

    if df is None:
        st.info("No evaluation CSV loaded yet.")
        return

    st.markdown("**Questions preview:**")
    st.dataframe(df, use_container_width=True)

    if st.button(":arrow_forward: Run evaluation", type="primary"):
        if not pipeline.is_ready():
            st.error("Build the index first before running evaluation.")
        else:
            with st.spinner("Running questions through the pipeline..."):
                results = run_evaluation(pipeline, df, top_k=pipeline.config.top_k)
            st.session_state.eval_results = results

    results = st.session_state.eval_results
    if results is None or results.empty:
        return

    st.markdown("#### Results — edit the `label` and `notes` columns as you review")

    label_options = VALID_LABELS
    edited = st.data_editor(
        results,
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "label": st.column_config.SelectboxColumn(
                "label", options=label_options, required=False
            ),
            "notes": st.column_config.TextColumn("notes"),
            "predicted_answer": st.column_config.TextColumn("predicted_answer", width="large"),
            "gold_answer": st.column_config.TextColumn("gold_answer", width="medium"),
            "hit_at_k": st.column_config.CheckboxColumn("hit_at_k"),
            "grounded_or_not": st.column_config.CheckboxColumn("grounded_or_not"),
        },
        key="eval_editor",
    )
    st.session_state.eval_results = edited

    # Summary metrics
    summary = summarize_evaluation(edited)
    st.markdown("#### Summary metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Questions", summary["n"])
    c2.metric(
        "Retrieval Hit@k",
        f"{summary['retrieval_hit_rate']:.0%}"
        if summary["retrieval_hit_rate"] is not None
        else "—",
    )
    c3.metric(
        "Correct rate (labeled)",
        f"{summary['correctness_rate']:.0%}"
        if summary["correctness_rate"] is not None
        else "—",
    )
    c4.metric(
        "Hallucination rate (labeled)",
        f"{summary['hallucination_rate']:.0%}"
        if summary["hallucination_rate"] is not None
        else "—",
    )

    st.caption(
        f"Grounded-by-guardrail rate: "
        f"{summary['grounded_rate']:.0%} "
        f"({summary.get('n_labeled', 0)} rows manually labeled)"
    )

    # Download buttons
    csv_bytes = edited.to_csv(index=False).encode("utf-8")
    st.download_button(
        ":arrow_down: Download results CSV",
        data=csv_bytes,
        file_name="evaluation_results.csv",
        mime="text/csv",
    )

    if QA_LOG_FILE.exists():
        st.download_button(
            ":arrow_down: Download QA log (JSONL)",
            data=QA_LOG_FILE.read_bytes(),
            file_name=QA_LOG_FILE.name,
            mime="application/jsonl",
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    _init_state()
    base_cfg = AppConfig()

    # Sidebar first, because its values drive pipeline construction.
    sidebar = render_sidebar(base_cfg)

    pipeline = _bootstrap_pipeline(
        embedding_model=base_cfg.embedding_model,
        chunk_size=sidebar["chunk_size"],
        chunk_overlap=sidebar["chunk_overlap"],
        min_chunk_size=sidebar["min_chunk_size"],
        top_k=sidebar["top_k"],
        min_score=sidebar["min_score"],
    )
    # Keep the pipeline config in sync with sidebar values even if cache hit.
    pipeline.config.chunk_size = sidebar["chunk_size"]
    pipeline.config.chunk_overlap = sidebar["chunk_overlap"]
    pipeline.config.min_chunk_size = sidebar["min_chunk_size"]
    pipeline.config.top_k = sidebar["top_k"]
    pipeline.config.min_score = sidebar["min_score"]

    provider_used = pipeline.set_llm_from_config(sidebar["provider"])

    # Handle build/rebuild click.
    if sidebar["build_clicked"]:
        saved = _persist_uploads(sidebar["uploaded"])
        existing_pdfs = sorted(DATA_DIR.glob("*.pdf"))
        pdf_paths = saved or existing_pdfs
        if not pdf_paths:
            st.sidebar.error("No PDFs found. Upload at least one PDF first.")
        else:
            with st.spinner(f"Building index from {len(pdf_paths)} PDF(s)..."):
                manifest = pipeline.build_index(pdf_paths)
            st.session_state.last_manifest = manifest
            if manifest.get("num_chunks", 0) == 0:
                st.sidebar.error("No text could be extracted from the provided PDFs.")
            else:
                st.sidebar.success(
                    f"Indexed {manifest['num_documents']} doc(s), "
                    f"{manifest['num_pages']} page(s), "
                    f"{manifest['num_chunks']} chunk(s)."
                )

    # Sidebar status
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Index status**")
    if pipeline.is_ready():
        n_docs = pipeline.store.metadata["doc_name"].nunique()
        n_chunks = len(pipeline.store)
        st.sidebar.success(
            f"Ready — {n_docs} document(s), {n_chunks} chunk(s)\n\n"
            f"Answering via: `{provider_used}`"
        )
    else:
        st.sidebar.warning("Index is empty — build it to enable chat.")

    # Main header + tabs
    st.title(":books: Course Syllabus QA Chatbot")
    st.caption(
        "A small RAG app that answers student questions using your uploaded "
        "syllabus, assignment, and policy PDFs — with citations."
    )

    tab_chat, tab_docs, tab_eval = st.tabs(["Chat", "Documents / Index", "Evaluation"])
    with tab_chat:
        render_chat_tab(pipeline)
    with tab_docs:
        render_documents_tab(pipeline)
    with tab_eval:
        render_evaluation_tab(pipeline)


if __name__ == "__main__":
    main()
