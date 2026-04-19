"""
Course Syllabus QA Chatbot - Streamlit UI.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

from config import (
    AppConfig,
    DATA_DIR,
    EVAL_DIR,
    METADATA_FILE,
    QA_LOG_FILE,
)
from src.evaluation import (
    VALID_LABELS,
    compute_retrieval_ranks,
    docs_match,
    load_questions_csv,
    run_evaluation,
    summarize_evaluation,
)
from src.llm_client import describe_available
from src.rag_pipeline import RAGPipeline
from src.utils import safe_filename

from reports.figures import (
    fig_chunks_per_doc,
    fig_heatmap,
    fig_hit_at_k,
    fig_label_distribution,
    fig_score_hist,
)


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
    if "eval_ranks" not in st.session_state:
        st.session_state.eval_ranks = None
    if "last_manifest" not in st.session_state:
        st.session_state.last_manifest = None


# ---------------------------------------------------------------------------
# Small helpers for corpus management + figure rendering
# ---------------------------------------------------------------------------


def _rebuild_from_data_dir(pipeline: RAGPipeline) -> dict:
    """Rebuild the index using exactly whatever PDFs currently live in `data/`.

    If `data/` is empty this resets the index so stale chunks don't linger.
    """
    pdf_paths = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_paths:
        pipeline.reset_index()
        return {"num_documents": 0, "num_pages": 0, "num_chunks": 0}
    return pipeline.build_index(pdf_paths)


def _fig_to_png_bytes(fig) -> bytes:
    """Render a matplotlib Figure to PNG bytes (for download buttons)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def _figures_to_zip_bytes(named_figs: list[tuple[str, object]]) -> bytes:
    """Bundle several figures into a single ZIP payload."""
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, fig in named_figs:
            z.writestr(name, _fig_to_png_bytes(fig))
    zip_buf.seek(0)
    return zip_buf.getvalue()


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
    clear_clicked = st.sidebar.button(
        ":wastebasket: Clear corpus & index",
        use_container_width=True,
        help="Delete every uploaded PDF from `data/` and wipe the FAISS index.",
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
        "clear_clicked": clear_clicked,
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

    ready = pipeline.is_ready()
    if not ready:
        st.info(
            "No index loaded yet. Upload PDFs in the sidebar and click "
            "**Build / Rebuild Index** to start."
        )

    # Top-of-tab controls: clear history.
    if st.session_state.chat_history:
        if st.button("Clear chat history", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()

    # Render the conversation using native chat widgets so Enter-to-send
    # works and the layout matches other Streamlit chat apps.
    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(turn["question"])
        with st.chat_message("assistant"):
            st.markdown(turn["answer"])
            if turn["citations"]:
                st.markdown(
                    "**Citations:** "
                    + " ".join(f"`{c}`" for c in turn["citations"])
                )
            elif not turn["grounded"]:
                st.caption(
                    "No citations — the assistant did not find strong supporting evidence."
                )
            with st.expander("Retrieved chunks", expanded=False):
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
            st.caption(f"provider: `{turn['provider']}`")

    placeholder = (
        "Ask a question about your course documents..."
        if ready
        else "Build the index first to enable chat"
    )
    question = st.chat_input(placeholder, disabled=not ready)
    if question:
        q = question.strip()
        if q:
            with st.spinner("Retrieving relevant chunks and generating answer..."):
                result = pipeline.answer(q)
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
            st.rerun()


# ---------------------------------------------------------------------------
# Documents tab
# ---------------------------------------------------------------------------


def render_documents_tab(pipeline: RAGPipeline) -> None:
    st.subheader(":page_facing_up: Indexed Documents")

    # --- Corpus on disk (so users can remove individual PDFs) ------------
    pdfs_on_disk = sorted(DATA_DIR.glob("*.pdf"))
    with st.container(border=True):
        st.markdown("**Uploaded PDFs on disk** (`data/`)")
        if not pdfs_on_disk:
            st.caption("No PDFs uploaded yet. Use the sidebar to upload.")
        else:
            st.caption(
                "Removing a file deletes it from `data/` and automatically "
                "rebuilds the index from whatever is left."
            )
            for p in pdfs_on_disk:
                c1, c2, c3 = st.columns([4, 2, 1])
                c1.write(f"`{p.name}`")
                size_kb = max(1, p.stat().st_size // 1024)
                c2.caption(f"{size_kb} KB")
                if c3.button("Remove", key=f"rm_{p.name}"):
                    try:
                        p.unlink()
                    except OSError as e:
                        st.error(f"Could not delete {p.name}: {e}")
                    else:
                        with st.spinner(f"Rebuilding index without {p.name}..."):
                            manifest = _rebuild_from_data_dir(pipeline)
                        st.session_state.last_manifest = manifest
                        st.session_state.eval_results = None
                        st.session_state.eval_ranks = None
                        st.success(
                            f"Removed `{p.name}`. "
                            f"Index now has {manifest.get('num_chunks', 0)} chunks "
                            f"from {manifest.get('num_documents', 0)} document(s)."
                        )
                        st.rerun()

    # --- Index contents --------------------------------------------------
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

    # --- Retrieval diagnostics: catch gold_doc/filename mismatches ------
    if pipeline.is_ready() and "gold_doc" in df.columns:
        indexed_docs: list[str] = sorted(
            pipeline.store.metadata["doc_name"].dropna().unique().tolist()
        )
        csv_docs = sorted(
            {
                str(x).strip()
                for x in df["gold_doc"].dropna().tolist()
                if str(x).strip()
            }
        )
        unmatched = [g for g in csv_docs if not any(docs_match(d, g) for d in indexed_docs)]

        header = (
            f"⚠  Retrieval diagnostics — {len(unmatched)} unmatched gold_doc value(s)"
            if unmatched
            else ":mag:  Retrieval diagnostics — all gold_doc values match"
        )
        with st.expander(header, expanded=bool(unmatched)):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Indexed documents (in `data/`)**")
                if not indexed_docs:
                    st.caption("(none)")
                for d in indexed_docs:
                    st.markdown(f"- `{d}`")
            with c2:
                st.markdown("**`gold_doc` values in CSV**")
                if not csv_docs:
                    st.caption("(none — Hit@k will be skipped)")
                for g in csv_docs:
                    ok = any(docs_match(d, g) for d in indexed_docs)
                    icon = "✅" if ok else "❌"
                    st.markdown(f"- {icon} `{g}`")

            if unmatched:
                st.warning(
                    "These `gold_doc` values in your CSV do not match any indexed "
                    "document name, so Hit@k will be 0 for those rows even if "
                    "retrieval is actually finding the right content.\n\n"
                    "Fix by either (a) renaming PDFs in `data/` and rebuilding, "
                    "or (b) editing the `gold_doc` column of your CSV to match an "
                    "indexed filename (substrings of the filename are OK — e.g. "
                    "`Syllabus.pdf` will match `CSE_434___CSE_534_Syllabus.pdf`)."
                )

    if st.button(":arrow_forward: Run evaluation", type="primary"):
        if not pipeline.is_ready():
            st.error("Build the index first before running evaluation.")
        else:
            with st.spinner("Running questions through the pipeline..."):
                results = run_evaluation(pipeline, df, top_k=pipeline.config.top_k)
                ranks = compute_retrieval_ranks(pipeline, df, max_k=10)
            st.session_state.eval_results = results
            st.session_state.eval_ranks = ranks

    results = st.session_state.eval_results
    ranks = st.session_state.eval_ranks
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

    # --- CSV / log downloads ---------------------------------------------
    csv_bytes = edited.to_csv(index=False).encode("utf-8")
    d1, d2 = st.columns(2)
    d1.download_button(
        ":arrow_down: Download results CSV",
        data=csv_bytes,
        file_name="evaluation_results.csv",
        mime="text/csv",
        use_container_width=True,
    )
    if QA_LOG_FILE.exists():
        d2.download_button(
            ":arrow_down: Download QA log (JSONL)",
            data=QA_LOG_FILE.read_bytes(),
            file_name=QA_LOG_FILE.name,
            mime="application/jsonl",
            use_container_width=True,
        )

    # --- Report figures --------------------------------------------------
    st.markdown("---")
    st.markdown("### :bar_chart: Report figures")
    st.caption(
        "Generated from the current evaluation run and your labels. "
        "Figures 3 and 5 appear once you label at least one row above."
    )

    chunks_df = pd.read_parquet(METADATA_FILE) if METADATA_FILE.exists() else pd.DataFrame()
    ranks_df = ranks if isinstance(ranks, pd.DataFrame) else pd.DataFrame()

    named_figs: list[tuple[str, object]] = []

    # Figure 1 - chunks per document
    with st.container(border=True):
        st.markdown("**Figure 1 — Chunks per document**")
        if chunks_df.empty:
            st.info("No index loaded; upload PDFs and build the index first.")
        else:
            f1 = fig_chunks_per_doc(chunks_df)
            st.pyplot(f1, use_container_width=True)
            st.download_button(
                "Download fig1_chunks_per_doc.png",
                data=_fig_to_png_bytes(f1),
                file_name="fig1_chunks_per_doc.png",
                mime="image/png",
                key="dl_fig1",
            )
            named_figs.append(("fig1_chunks_per_doc.png", f1))

    # Figure 2 - Hit@k
    with st.container(border=True):
        st.markdown("**Figure 2 — Retrieval Hit@k (k = 1, 3, 5, 10)**")
        if ranks_df.empty:
            st.info("Run evaluation to compute retrieval ranks.")
        elif ranks_df["gold_doc"].notna().sum() == 0:
            st.info("Add `gold_doc` (and optionally `gold_page`) in the eval CSV for Hit@k.")
        else:
            f2 = fig_hit_at_k(ranks_df)
            st.pyplot(f2, use_container_width=True)
            st.download_button(
                "Download fig2_hit_at_k.png",
                data=_fig_to_png_bytes(f2),
                file_name="fig2_hit_at_k.png",
                mime="image/png",
                key="dl_fig2",
            )
            named_figs.append(("fig2_hit_at_k.png", f2))

    # Figure 3 - Label distribution
    with st.container(border=True):
        st.markdown("**Figure 3 — Answer label distribution**")
        f3 = fig_label_distribution(edited)
        if f3 is None:
            st.info("Label at least one row above (Correct / Partially Correct / Unsupported / Hallucinated) to unlock this chart.")
        else:
            st.pyplot(f3, use_container_width=True)
            st.download_button(
                "Download fig3_label_distribution.png",
                data=_fig_to_png_bytes(f3),
                file_name="fig3_label_distribution.png",
                mime="image/png",
                key="dl_fig3",
            )
            named_figs.append(("fig3_label_distribution.png", f3))

    # Figure 4 - Top-1 score histogram
    with st.container(border=True):
        st.markdown("**Figure 4 — Top-1 retrieval score by Hit@1**")
        if ranks_df.empty or ranks_df["gold_doc"].notna().sum() == 0:
            st.info("Requires a `gold_doc` column and a completed evaluation run.")
        else:
            f4 = fig_score_hist(ranks_df)
            st.pyplot(f4, use_container_width=True)
            st.download_button(
                "Download fig4_score_hist.png",
                data=_fig_to_png_bytes(f4),
                file_name="fig4_score_hist.png",
                mime="image/png",
                key="dl_fig4",
            )
            named_figs.append(("fig4_score_hist.png", f4))

    # Figure 5 - Hit@k x label heatmap
    with st.container(border=True):
        st.markdown("**Figure 5 — Retrieval success × answer label heatmap**")
        f5 = fig_heatmap(edited, ranks_df, k=5)
        if f5 is None:
            st.info("Needs both (a) labeled rows and (b) `gold_doc` in the eval CSV.")
        else:
            st.pyplot(f5, use_container_width=True)
            st.download_button(
                "Download fig5_heatmap.png",
                data=_fig_to_png_bytes(f5),
                file_name="fig5_heatmap.png",
                mime="image/png",
                key="dl_fig5",
            )
            named_figs.append(("fig5_heatmap.png", f5))

    # Bundle download
    if named_figs:
        st.download_button(
            ":package: Download all figures (ZIP)",
            data=_figures_to_zip_bytes(named_figs),
            file_name="report_figures.zip",
            mime="application/zip",
            use_container_width=True,
            key="dl_figs_zip",
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

    # Handle "Clear corpus & index" click.
    if sidebar["clear_clicked"]:
        removed = 0
        for p in DATA_DIR.glob("*.pdf"):
            try:
                p.unlink()
                removed += 1
            except OSError:
                pass
        pipeline.reset_index()
        st.session_state.last_manifest = None
        st.session_state.eval_results = None
        st.session_state.eval_ranks = None
        st.sidebar.success(f"Cleared {removed} PDF(s) and reset the index.")

    # Handle build/rebuild click. Always rebuild from the full `data/` folder
    # (after saving any new uploads) so the index matches what is actually on
    # disk - removing a PDF truly removes it from retrieval.
    if sidebar["build_clicked"]:
        _persist_uploads(sidebar["uploaded"])
        pdf_paths = sorted(DATA_DIR.glob("*.pdf"))
        if not pdf_paths:
            st.sidebar.error("No PDFs found. Upload at least one PDF first.")
            pipeline.reset_index()
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
                st.session_state.eval_results = None
                st.session_state.eval_ranks = None

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
