"""
Generative AI RAG Document Q&A Chatbot - Streamlit UI.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# IMPORTANT: defuse macOS OpenMP / MKL conflicts before any heavy import.
#
# faiss, torch (via sentence-transformers), and bert-score each ship their own
# OpenMP runtime. When all three are loaded into the same Python process on
# macOS / arm64 they collide and the process segfaults (exit 139) — typically
# the moment "Run evaluation" is clicked with "Compute BERTScore" enabled.
# These three env vars must be set BEFORE numpy / faiss / torch are imported,
# so they live at the very top of this module.
# ---------------------------------------------------------------------------
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import importlib.util
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

from config import (
    AppConfig,
    DATA_DIR,
    EMBEDDING_MODEL_OPTIONS,
    EVAL_DIR,
    MANIFEST_FILE,
    METADATA_FILE,
    QA_LOG_FILE,
)
from src.evaluation import (
    VALID_LABELS,
    calibration_table,
    compute_retrieval_ranks,
    docs_match,
    load_questions_csv,
    run_evaluation,
    summarize_evaluation,
)
from src.llm_client import describe_available
from src.rag_pipeline import RAGPipeline
from src.retriever import Retriever
from src.utils import safe_filename

from reports.figures import (
    fig_calibration,
    fig_chunks_per_doc,
    fig_heatmap,
    fig_hit_at_k,
    fig_label_distribution,
    fig_score_hist,
)


PROJECT_ROOT = Path(__file__).resolve().parent
LORA_EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "lora_qlora"
LORA_DATA_DIR = LORA_EXPERIMENT_DIR / "data"
LORA_TRAIN_FILE = LORA_DATA_DIR / "train.jsonl"
LORA_EVAL_FILE = LORA_DATA_DIR / "eval.jsonl"
LORA_ADAPTER_DIR = LORA_EXPERIMENT_DIR / "adapters" / "flan-t5-small-lora"
LORA_RESULTS_DIR = LORA_EXPERIMENT_DIR / "results"
LORA_EVAL_SUMMARY_FILE = LORA_RESULTS_DIR / "adapter_eval_summary.json"


st.set_page_config(
    page_title="Generative AI RAG Document Q&A Chatbot",
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
    cache_version: int,
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


def _reset_eval_editor_state() -> None:
    """Clear the results data_editor widget state.

    Call this whenever the underlying `eval_results` DataFrame is being
    replaced (new evaluation run, corpus change, etc.). Without this, the
    editor's diff (stored under key `eval_editor`) tries to apply old edits
    to the new rows and can silently reset or swap values.
    """
    for k in ("eval_editor",):
        if k in st.session_state:
            del st.session_state[k]


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


def _count_jsonl(path: Path) -> int:
    """Count non-empty JSONL rows without loading the full file."""
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _load_json_file(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _tail_text(text: str, limit: int = 6000) -> str:
    text = str(text or "").strip()
    if len(text) <= limit:
        return text
    return "... output truncated ...\n" + text[-limit:]


def _run_lora_script(script_name: str, args: list[str] | None = None) -> subprocess.CompletedProcess:
    """Run one of the local LoRA experiment scripts in the app's venv."""
    cmd = [sys.executable, str(LORA_EXPERIMENT_DIR / script_name)]
    cmd.extend(args or [])
    return subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _render_process_result(result: subprocess.CompletedProcess) -> None:
    if result.returncode == 0:
        st.success("Command completed successfully.")
    else:
        st.error(f"Command failed with exit code {result.returncode}.")
    if result.stdout.strip():
        st.markdown("**stdout**")
        st.code(_tail_text(result.stdout), language="text")
    if result.stderr.strip():
        st.markdown("**stderr**")
        st.code(_tail_text(result.stderr), language="text")


def _package_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


# ---------------------------------------------------------------------------
# LoRA inference helpers (used by the "Try the adapter" UI section)
# ---------------------------------------------------------------------------


def _format_lora_prompt(instruction: str, context: str, question: str) -> str:
    """Match the training prompt format used in train_lora.format_source."""
    return (
        f"Instruction: {instruction}\n"
        f"Context: {context}\n"
        f"Question: {question}\n"
        "Answer:"
    )


def _adapter_signature(adapter_dir: Path) -> str:
    """A cache-key string that changes whenever the saved adapter weights change."""
    weights = adapter_dir / "adapter_model.safetensors"
    if not weights.exists():
        return "missing"
    return f"{int(weights.stat().st_mtime)}-{weights.stat().st_size}"


@st.cache_resource(show_spinner="Loading FLAN-T5 base model...")
def _load_flan_base(model_id: str):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    model.eval()
    return tokenizer, model


@st.cache_resource(show_spinner="Loading LoRA adapter on top of FLAN-T5...")
def _load_flan_with_adapter(model_id: str, adapter_dir: str, adapter_signature: str):
    """`adapter_signature` is part of the cache key so retraining busts the cache."""
    from peft import PeftModel
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    adapter_path = Path(adapter_dir)
    tokenizer_src = (
        adapter_path
        if (adapter_path / "tokenizer_config.json").exists()
        else model_id
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src)
    base = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return tokenizer, model


def _generate_text(tokenizer, model, prompt: str, max_new_tokens: int = 160) -> str:
    import torch

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(generated[0], skip_special_tokens=True)


def _jaccard_overlap(a: str, b: str) -> float:
    import re

    def toks(text: str) -> set[str]:
        return {
            t for t in re.sub(r"[^a-z0-9]+", " ", str(text).lower()).split() if len(t) > 2
        }

    left, right = toks(a), toks(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


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

    st.sidebar.subheader("Embedding model")
    # Build the list of selectable models. If the value from `.env` isn't in
    # the catalog, surface it as a first "custom" row so the user's config
    # is never silently overridden.
    catalog_ids = [mid for mid, _, _ in EMBEDDING_MODEL_OPTIONS]
    labels_by_id = {mid: lbl for mid, lbl, _ in EMBEDDING_MODEL_OPTIONS}
    descs_by_id = {mid: desc for mid, _, desc in EMBEDDING_MODEL_OPTIONS}
    options: list[str] = list(catalog_ids)
    if base_cfg.embedding_model and base_cfg.embedding_model not in options:
        options.insert(0, base_cfg.embedding_model)
        labels_by_id[base_cfg.embedding_model] = f"{base_cfg.embedding_model} (from .env)"
        descs_by_id[base_cfg.embedding_model] = "Custom model id loaded from your .env file."

    default_idx = (
        options.index(base_cfg.embedding_model)
        if base_cfg.embedding_model in options
        else 0
    )
    embedding_model = st.sidebar.selectbox(
        "Embedding model",
        options=options,
        index=default_idx,
        format_func=lambda mid: labels_by_id.get(mid, mid),
        help=(
            "The model used to embed both chunks and questions. Changing it "
            "invalidates any existing index — you'll be prompted to rebuild."
        ),
    )
    st.sidebar.caption(descs_by_id.get(embedding_model, ""))

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
    retrieval_options = ["hybrid", "dense", "bm25"]
    mode_labels = {
        "hybrid": "Hybrid BM25 + FAISS",
        "dense": "Dense FAISS only",
        "bm25": "BM25 keyword only",
    }
    default_mode = (
        base_cfg.retrieval_mode
        if base_cfg.retrieval_mode in retrieval_options
        else "hybrid"
    )
    retrieval_mode = st.sidebar.selectbox(
        "Retrieval mode",
        options=retrieval_options,
        index=retrieval_options.index(default_mode),
        format_func=lambda m: mode_labels[m],
        help="Hybrid mode combines semantic FAISS search with keyword BM25 search.",
    )
    hybrid_alpha = float(base_cfg.hybrid_alpha)
    if retrieval_mode == "hybrid":
        hybrid_alpha = st.sidebar.slider(
            "Hybrid dense weight",
            min_value=0.0,
            max_value=1.0,
            value=float(max(0.0, min(1.0, base_cfg.hybrid_alpha))),
            step=0.05,
            help="Higher values favor FAISS semantic search; lower values favor BM25 keyword search.",
        )
    top_k = st.sidebar.slider("Top-k chunks", min_value=1, max_value=10, value=base_cfg.top_k)
    min_score = st.sidebar.slider(
        "Min retrieval score",
        min_value=0.0, max_value=1.0,
        value=float(base_cfg.min_score), step=0.05,
        help="If the top retrieved chunk scores below this, the assistant refuses to answer.",
    )

    st.sidebar.subheader("Documents")
    uploaded = st.sidebar.file_uploader(
        "Upload PDFs",
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
        "embedding_model": embedding_model,
        "chunk_size": int(chunk_size),
        "chunk_overlap": int(chunk_overlap),
        "min_chunk_size": int(min_chunk_size),
        "top_k": int(top_k),
        "min_score": float(min_score),
        "retrieval_mode": retrieval_mode,
        "hybrid_alpha": float(hybrid_alpha),
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
    st.subheader(":speech_balloon: Ask a question about your documents")

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
                    score_parts = [f"score: `{r.score:.3f}`"]
                    retrieval_source = getattr(r, "retrieval_source", "dense")
                    if retrieval_source == "hybrid":
                        score_parts.append(f"dense: `{(getattr(r, 'dense_score', 0) or 0):.3f}`")
                        score_parts.append(f"bm25: `{(getattr(r, 'bm25_score', 0) or 0):.3f}`")
                    st.markdown(
                        f"**[{i}]** `{r.doc_name}` {page_label} "
                        f"| {' | '.join(score_parts)} "
                        f"| mode: `{retrieval_source}` "
                        f"| section: _{r.section_title or '—'}_"
                    )
                    st.text(r.raw_text)

    placeholder = (
        "Ask a question about your documents..."
        if ready
        else "Build the index first to enable chat"
    )
    question = st.chat_input(placeholder, disabled=not ready)
    if question:
        q = question.strip()
        if q:
            # Render the user's question immediately so they can see what
            # they asked while the pipeline is still working, then show a
            # thinking spinner inside the assistant bubble. After the
            # pipeline returns we save the turn and rerun so the final
            # answer (with citations + retrieved chunks) takes the
            # placeholder's place.
            with st.chat_message("user"):
                st.markdown(q)
            with st.chat_message("assistant"):
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
                        _reset_eval_editor_state()
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

    # Show exactly which embedding model these chunks were embedded with.
    # We prefer the on-disk manifest value because that is the model that
    # actually produced the current FAISS vectors; fall back to the
    # pipeline's live config if the manifest is missing.
    indexed_model: str | None = None
    if MANIFEST_FILE.exists():
        try:
            import json as _json
            indexed_model = (
                _json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
                .get("embedding_model")
            )
        except Exception:  # noqa: BLE001
            indexed_model = None
    indexed_model = indexed_model or pipeline.config.embedding_model
    st.caption(f"Embedding model used for this index: `{indexed_model}`")

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

    compute_bertscore = st.checkbox(
        "Compute BERTScore during evaluation",
        value=False,
        help=(
            "Advanced metric for graduate analysis. It is slower and may download "
            "a small transformer model the first time it runs."
        ),
    )

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
                results = run_evaluation(
                    pipeline,
                    df,
                    top_k=pipeline.config.top_k,
                    compute_bertscore=compute_bertscore,
                )
                ranks = compute_retrieval_ranks(pipeline, df, max_k=10)
            st.session_state.eval_results = results
            st.session_state.eval_ranks = ranks
            if compute_bertscore and results["bertscore_f1"].isna().all():
                st.warning(
                    "BERTScore was requested but could not be computed. "
                    "Install `bert-score` and make sure the model can be downloaded."
                )
            # New rows - clear the editor's diff so old label edits don't try
            # to apply to different questions.
            _reset_eval_editor_state()

    if st.session_state.eval_results is None or st.session_state.eval_results.empty:
        return

    # Everything below is wrapped in a Streamlit fragment so that editing a
    # cell in the results data-editor (typically the `label` column) only
    # re-runs THIS block. The rest of the page - sidebar, diagnostics,
    # questions preview, etc. - is not re-executed, so the scroll position
    # no longer jumps to the bottom on every label change, and the figures
    # simply update in place.
    _render_results_fragment()


@st.fragment
def _render_results_fragment() -> None:
    """Results table + metrics + downloads + report figures.

    Runs as a Streamlit fragment so in-table edits do NOT reflow the whole
    page. Reads the latest results/ranks from session_state on each partial
    rerun so figure 3 (label distribution) and figure 5 (Hit@k x label)
    stay in sync with the editor above them.
    """
    results = st.session_state.eval_results
    ranks = st.session_state.eval_ranks
    if results is None or results.empty:
        return

    st.markdown("#### Results — edit the `label` and `notes` columns as you review")

    # Column order: put the two columns the grader actually edits
    # (`label`, `notes`) immediately after `predicted_answer`, so they are
    # visible without horizontal scrolling. Other columns follow in a useful
    # reading order.
    preferred_order = [
        "question",
        "gold_answer",
        "predicted_answer",
        "label",
        "notes",
        "hit_at_k",
        "grounded_or_not",
        "retrieved_doc_names",
        "retrieved_pages",
        "top_score",
        "overlap_pred_vs_gold",
        "semantic_similarity_pred_vs_gold",
        "bertscore_f1",
        "bertscore_precision",
        "bertscore_recall",
        "overlap_pred_vs_context",
    ]
    column_order = [c for c in preferred_order if c in results.columns]
    # Include any extra columns at the end so nothing is silently hidden.
    column_order += [c for c in results.columns if c not in column_order]

    label_options = VALID_LABELS

    st.caption(
        "Tip: edit labels in the normal (non-fullscreen) view. "
        "Streamlit's fullscreen overlay for tables resets on every keystroke; "
        "in the normal view your edits persist and the charts below update in place."
    )

    # IMPORTANT: pass the baseline `results` to the editor and do NOT write
    # the returned `edited` back into st.session_state.eval_results.
    # `st.data_editor` stores edits as a diff against its `data` arg (keyed by
    # "eval_editor"); if we mutate the baseline each rerun, the diff and the
    # new data go out of sync and Streamlit resets the widget - which both
    # loses your just-picked label and closes the fullscreen overlay.
    edited = st.data_editor(
        results,
        use_container_width=True,
        num_rows="fixed",
        column_order=column_order,
        column_config={
            "label": st.column_config.SelectboxColumn(
                "label", options=label_options, required=False, width="small"
            ),
            "notes": st.column_config.TextColumn("notes", width="medium"),
            "predicted_answer": st.column_config.TextColumn("predicted_answer", width="large"),
            "gold_answer": st.column_config.TextColumn("gold_answer", width="medium"),
            "hit_at_k": st.column_config.CheckboxColumn("hit_at_k"),
            "grounded_or_not": st.column_config.CheckboxColumn("grounded_or_not"),
        },
        key="eval_editor",
    )

    # Summary metrics
    summary = summarize_evaluation(edited)
    st.markdown("#### Summary metrics")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
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
    c5.metric(
        "Semantic sim.",
        f"{summary['semantic_similarity_mean']:.3f}"
        if summary["semantic_similarity_mean"] is not None
        else "—",
    )
    c6.metric(
        "Calibration ECE",
        f"{summary['calibration_ece']:.3f}"
        if summary["calibration_ece"] is not None
        else "—",
    )

    bert_note = (
        f" | Mean BERTScore F1: {summary['bertscore_f1_mean']:.3f}"
        if summary.get("bertscore_f1_mean") is not None
        else ""
    )
    st.caption(
        f"Grounded-by-guardrail rate: {summary['grounded_rate']:.0%} "
        f"({summary.get('n_labeled', 0)} rows manually labeled)"
        f"{bert_note}"
    )

    calib_table, calib_ece = calibration_table(edited)
    if calib_ece is not None:
        with st.expander("Calibration buckets", expanded=False):
            st.dataframe(calib_table, use_container_width=True)

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
        "Figures 3, 5, and 6 appear once you label at least one row above. "
        "Editing the table above updates these charts in place."
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

    # Figure 4 - Top-1 confidence histogram
    with st.container(border=True):
        st.markdown("**Figure 4 — Top-1 retrieval confidence by Hit@1**")
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

    # Figure 6 - Calibration curve
    with st.container(border=True):
        st.markdown("**Figure 6 — Calibration by retrieval confidence**")
        f6 = fig_calibration(edited)
        if f6 is None:
            st.info("Label at least one row above to compute calibration buckets and ECE.")
        else:
            st.pyplot(f6, use_container_width=True)
            st.download_button(
                "Download fig6_calibration.png",
                data=_fig_to_png_bytes(f6),
                file_name="fig6_calibration.png",
                mime="image/png",
                key="dl_fig6",
            )
            named_figs.append(("fig6_calibration.png", f6))

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
# LoRA / QLoRA tab
# ---------------------------------------------------------------------------


def render_lora_tab(pipeline: RAGPipeline) -> None:
    st.subheader(":microscope: LoRA / QLoRA Experiment")

    st.caption(
        "Research extension for CSE 534: generate document-grounded QA pairs "
        "from the indexed PDF chunks, fine-tune `google/flan-t5-small` with "
        "LoRA, evaluate the adapter, and check whether the machine can support QLoRA."
    )

    ready = pipeline.is_ready()
    n_docs = int(pipeline.store.metadata["doc_name"].nunique()) if ready else 0
    n_chunks = int(len(pipeline.store)) if ready else 0
    train_rows = _count_jsonl(LORA_TRAIN_FILE)
    eval_rows = _count_jsonl(LORA_EVAL_FILE)
    adapter_ready = (LORA_ADAPTER_DIR / "adapter_config.json").exists()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Indexed documents", n_docs)
    c2.metric("Indexed chunks", n_chunks)
    c3.metric("QA examples", f"{train_rows} train / {eval_rows} eval")
    c4.metric("LoRA adapter", "Ready" if adapter_ready else "Not trained")

    deps = {
        "transformers": _package_available("transformers"),
        "datasets": _package_available("datasets"),
        "peft": _package_available("peft"),
        "accelerate": _package_available("accelerate"),
        "bitsandbytes": _package_available("bitsandbytes"),
    }
    missing_required = [name for name in ("transformers", "datasets", "peft", "accelerate") if not deps[name]]
    with st.expander("Dependency status", expanded=bool(missing_required)):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Package": name,
                        "Installed": "yes" if available else "no",
                        "Purpose": (
                            "QLoRA 4-bit loading"
                            if name == "bitsandbytes"
                            else "LoRA training/evaluation"
                        ),
                    }
                    for name, available in deps.items()
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
        if missing_required:
            st.warning(
                "Install the missing LoRA packages from `requirements.txt` "
                "before running training."
            )

    st.markdown("#### 1. Build the fine-tuning dataset")
    d1, d2 = st.columns(2)
    with d1:
        examples_per_chunk = st.slider(
            "Examples per chunk",
            min_value=1,
            max_value=5,
            value=3,
            help="More examples gives the adapter more practice but increases training time.",
        )
    with d2:
        eval_fraction = st.slider(
            "Eval split",
            min_value=0.05,
            max_value=0.50,
            value=0.20,
            step=0.05,
            help="Fraction of generated QA pairs reserved for adapter evaluation.",
        )

    if st.button(
        "Generate QA training dataset",
        disabled=not ready,
        use_container_width=True,
    ):
        with st.spinner("Creating template-based QA examples from indexed chunks..."):
            result = _run_lora_script(
                "prepare_qa_dataset.py",
                [
                    "--examples-per-chunk",
                    str(int(examples_per_chunk)),
                    "--eval-fraction",
                    str(float(eval_fraction)),
                ],
            )
        _render_process_result(result)
        if result.returncode == 0:
            st.rerun()

    if not ready:
        st.info("Build the document index first; the LoRA dataset is generated from `indexes/chunks.parquet`.")

    if LORA_TRAIN_FILE.exists():
        with st.expander("Preview generated training rows", expanded=False):
            preview_rows: list[dict] = []
            with LORA_TRAIN_FILE.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        preview_rows.append(json.loads(line))
                    if len(preview_rows) >= 5:
                        break
            cols = ["question", "answer", "doc_name", "page_start", "page_end"]
            preview_df = pd.DataFrame(preview_rows)
            if preview_df.empty:
                st.caption("No preview rows available.")
            else:
                st.dataframe(
                    preview_df[[c for c in cols if c in preview_df.columns]],
                    use_container_width=True,
                )

    st.markdown("#### 2. Train LoRA on FLAN-T5-small")
    model_id = st.text_input("Base model", value="google/flan-t5-small")
    t1, t2, t3, t4 = st.columns(4)
    with t1:
        epochs = st.number_input("Epochs", min_value=0.1, max_value=10.0, value=3.0, step=0.5)
    with t2:
        batch_size = st.number_input("Batch size", min_value=1, max_value=16, value=2, step=1)
    with t3:
        learning_rate = st.number_input(
            "Learning rate",
            min_value=0.00001,
            max_value=0.01,
            value=0.0005,
            step=0.0001,
            format="%.5f",
        )
    with t4:
        max_steps = st.number_input(
            "Max steps",
            min_value=-1,
            max_value=5000,
            value=-1,
            step=1,
            help="Use -1 for the full epoch-based run; small positive values are useful for a quick smoke test.",
        )

    can_train = train_rows > 0 and eval_rows > 0 and not missing_required
    if st.button("Train LoRA adapter", disabled=not can_train, use_container_width=True):
        with st.spinner("Training LoRA adapter. The first run may download the base model and take several minutes..."):
            result = _run_lora_script(
                "train_lora.py",
                [
                    "--model",
                    model_id,
                    "--epochs",
                    str(float(epochs)),
                    "--batch-size",
                    str(int(batch_size)),
                    "--learning-rate",
                    str(float(learning_rate)),
                    "--max-steps",
                    str(int(max_steps)),
                ],
            )
        _render_process_result(result)
        if result.returncode == 0:
            st.rerun()
    if train_rows == 0 or eval_rows == 0:
        st.info("Generate the QA training dataset before starting LoRA training.")

    training_summary = _load_json_file(LORA_ADAPTER_DIR / "training_summary.json")
    if training_summary:
        with st.expander("Latest LoRA training summary", expanded=False):
            st.json(training_summary)

    st.markdown("#### 3. Evaluate the adapter")
    eval_limit = st.number_input(
        "Evaluation row limit",
        min_value=0,
        max_value=500,
        value=0,
        step=5,
        help="Use 0 to evaluate every generated eval row.",
    )
    if st.button("Evaluate LoRA adapter", disabled=not adapter_ready, use_container_width=True):
        with st.spinner("Generating adapter answers and scoring overlap..."):
            result = _run_lora_script(
                "evaluate_adapter.py",
                [
                    "--model",
                    model_id,
                    "--limit",
                    str(int(eval_limit)),
                ],
            )
        _render_process_result(result)

    adapter_eval_summary = _load_json_file(LORA_EVAL_SUMMARY_FILE)
    if adapter_eval_summary:
        st.markdown("**Latest adapter evaluation summary**")
        st.json(adapter_eval_summary)
        adapter_eval_file = LORA_RESULTS_DIR / "adapter_eval.jsonl"
        if adapter_eval_file.exists():
            st.download_button(
                "Download adapter evaluation JSONL",
                data=adapter_eval_file.read_bytes(),
                file_name="adapter_eval.jsonl",
                mime="application/jsonl",
                use_container_width=True,
            )

    st.markdown("#### 4. Check QLoRA readiness")
    st.caption(
        "QLoRA requires 4-bit quantization support. On most Mac/CPU-only setups, "
        "this check will report that full QLoRA training is not available locally."
    )
    if st.button("Check QLoRA readiness", use_container_width=True):
        with st.spinner("Checking CUDA and bitsandbytes availability..."):
            result = _run_lora_script("train_qlora.py", ["--model", model_id, "--check-only"])
        _render_process_result(result)

    st.markdown("#### 5. Try the adapter")
    st.caption(
        "Ask the trained LoRA adapter your own question. By default the app "
        "auto-retrieves context from the same FAISS index the Chat tab uses, "
        "so you can just type a question. Switch the **Context source** below "
        "to manually pick a chunk or paste your own text for debugging."
    )

    inference_blockers: list[str] = []
    if not adapter_ready:
        inference_blockers.append(
            "Train the LoRA adapter first (step 2) — `adapter_config.json` is missing."
        )
    if not deps["transformers"] or not deps["peft"]:
        inference_blockers.append(
            "Install `transformers` and `peft` to run adapter inference."
        )

    if inference_blockers:
        for msg in inference_blockers:
            st.info(msg)
    else:
        chunk_options: list[dict] = []
        if ready and n_chunks > 0:
            preview_df = pipeline.store.metadata.copy()
            for i, row in preview_df.iterrows():
                text = str(row.get("raw_text", "")).strip()
                if not text:
                    continue
                snippet = text[:90].replace("\n", " ")
                chunk_options.append(
                    {
                        "idx": int(i),
                        "doc_name": str(row.get("doc_name", "")),
                        "page_start": int(row.get("page_start", 1) or 1),
                        "page_end": int(row.get("page_end", row.get("page_start", 1)) or 1),
                        "section_title": str(row.get("section_title") or "this section"),
                        "raw_text": text,
                        "label": (
                            f"{row.get('doc_name', '')} p.{row.get('page_start', '?')} — {snippet}..."
                        ),
                    }
                )

        source_options: list[str] = []
        if ready and n_chunks > 0:
            source_options.append("Ask a question (auto-retrieve from index)")
        if chunk_options:
            source_options.append("Pick a chunk manually")
        source_options.append("Paste my own context")

        source_mode = st.radio(
            "Context source",
            options=source_options,
            horizontal=True,
            key="lora_try_source_mode",
            help=(
                "Auto-retrieve = closest to the Chat tab: just ask a question and "
                "the FAISS index supplies the context. Pick / Paste are useful for "
                "debugging the adapter on a known passage."
            ),
        )

        default_instruction = (
            "Answer using only the provided context. If the answer is unsupported, "
            "say you could not find a supported answer. Cite the source."
        )

        # Where context will come from for this run.
        context_text: str = ""
        question_text: str = ""
        retrieved_for_display: list = []  # populated only in auto-retrieve mode
        retrieval_top_k: int = pipeline.config.top_k

        if source_mode == "Ask a question (auto-retrieve from index)":
            question_text = st.text_input(
                "Your question",
                value=st.session_state.get("lora_try_auto_question", "Can I submit the project late?"),
                key="lora_try_auto_question",
                placeholder="Ask anything that should be answerable from your indexed PDFs...",
            )
            retrieval_top_k = st.slider(
                "Context chunks to retrieve (top-k)",
                min_value=1,
                max_value=min(8, max(1, n_chunks)),
                value=min(2, max(1, n_chunks)),
                step=1,
                help="How many top retrieved chunks to concatenate as context. FLAN-T5-small only handles ~512 tokens, so 1-2 chunks is usually right.",
                key="lora_try_auto_topk",
            )
            instruction = default_instruction
        elif source_mode == "Pick a chunk manually" and chunk_options:
            chosen_idx = st.selectbox(
                "Choose a chunk",
                options=list(range(len(chunk_options))),
                format_func=lambda i: chunk_options[i]["label"],
                key="lora_try_chunk_pick",
            )
            chosen = chunk_options[chosen_idx]
            instruction = st.text_input(
                "Instruction",
                value=default_instruction,
                key="lora_try_pick_instruction",
            )
            context_text = st.text_area(
                "Context (auto-filled from the chosen chunk; editable)",
                value=chosen["raw_text"][:1400],
                height=180,
                key=f"lora_try_pick_context_{chosen_idx}",
            )
            question_text = st.text_input(
                "Question",
                value=f"What does {chosen['doc_name']} say about {chosen['section_title']}?",
                key=f"lora_try_pick_question_{chosen_idx}",
            )
        else:
            instruction = st.text_input(
                "Instruction",
                value=default_instruction,
                key="lora_try_paste_instruction",
            )
            context_text = st.text_area(
                "Context",
                value=(
                    "Late submissions for assignments are accepted up to 48 hours after "
                    "the deadline with a 20% penalty per day. Project deadlines do not "
                    "allow late submissions."
                ),
                height=180,
                key="lora_try_paste_context",
            )
            question_text = st.text_input(
                "Question",
                value="Can I submit the project late?",
                key="lora_try_paste_question",
            )

        g1, g2, g3 = st.columns([1, 1, 1])
        with g1:
            max_new_tokens = st.slider(
                "Max new tokens",
                min_value=32,
                max_value=320,
                value=160,
                step=16,
                key="lora_try_max_new",
            )
        with g2:
            compare_with_base = st.checkbox(
                "Compare against base model",
                value=True,
                help="Generate the same prompt with the unmodified base FLAN-T5 so you can see what fine-tuning changed.",
                key="lora_try_compare",
            )
        with g3:
            if st.button("Reload models", help="Clear cached weights (use after retraining)."):
                _load_flan_base.clear()
                _load_flan_with_adapter.clear()
                st.success("Model cache cleared. Next generation will reload weights.")

        if st.button(
            "Generate answer",
            type="primary",
            use_container_width=True,
            key="lora_try_generate",
        ):
            question_clean = (question_text or "").strip()
            if not question_clean:
                st.warning("Please type a question.")
            else:
                # Auto-retrieve mode: fetch context from FAISS using the same
                # retriever the Chat tab uses, then concatenate the top chunks.
                if source_mode == "Ask a question (auto-retrieve from index)":
                    with st.spinner("Retrieving relevant chunks from the index..."):
                        retrieved_for_display = pipeline.retriever.retrieve(
                            question_clean,
                            top_k=int(retrieval_top_k),
                            mode=pipeline.config.retrieval_mode,
                            hybrid_alpha=pipeline.config.hybrid_alpha,
                        )
                    if not retrieved_for_display:
                        st.warning(
                            "No chunks were retrieved for this question. The "
                            "adapter would have nothing to read from. Try a "
                            "different phrasing or build the index first."
                        )
                        return
                    pieces: list[str] = []
                    for r in retrieved_for_display:
                        cite = r.citation  # e.g. "[Syllabus.pdf p.3]"
                        pieces.append(f"{cite}\n{r.raw_text}")
                    context_text = "\n\n---\n\n".join(pieces)[:1800]

                if not (context_text or "").strip():
                    st.warning("No context available — please paste or pick one.")
                    return

                prompt = _format_lora_prompt(instruction, context_text, question_clean)
                with st.expander("Prompt sent to the model", expanded=False):
                    st.code(prompt, language="text")

                if retrieved_for_display:
                    with st.expander(
                        f"Retrieved {len(retrieved_for_display)} chunk(s) used as context",
                        expanded=False,
                    ):
                        for i, r in enumerate(retrieved_for_display, start=1):
                            st.markdown(
                                f"**{i}.** `{r.citation}` — score "
                                f"{r.score:.3f} ({r.retrieval_source})"
                            )
                            preview = r.raw_text.strip().replace("\n", " ")
                            st.caption(preview[:280] + ("..." if len(preview) > 280 else ""))

                adapter_pred: str | None = None
                base_pred: str | None = None
                signature = _adapter_signature(LORA_ADAPTER_DIR)

                try:
                    with st.spinner("Generating with LoRA adapter..."):
                        adapter_tok, adapter_model = _load_flan_with_adapter(
                            model_id,
                            str(LORA_ADAPTER_DIR),
                            signature,
                        )
                        adapter_pred = _generate_text(
                            adapter_tok, adapter_model, prompt, max_new_tokens=int(max_new_tokens)
                        )
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Adapter generation failed: {exc}")

                if compare_with_base:
                    try:
                        with st.spinner("Generating with base FLAN-T5 (no adapter)..."):
                            base_tok, base_model = _load_flan_base(model_id)
                            base_pred = _generate_text(
                                base_tok, base_model, prompt, max_new_tokens=int(max_new_tokens)
                            )
                    except Exception as exc:  # noqa: BLE001
                        st.warning(f"Base model generation failed: {exc}")

                if adapter_pred is not None and base_pred is not None:
                    col_b, col_a = st.columns(2)
                    with col_b:
                        st.markdown("**Base FLAN-T5 (no adapter)**")
                        st.success(base_pred or "(empty output)")
                        st.caption(
                            f"Token overlap with context: "
                            f"{_jaccard_overlap(base_pred or '', context_text):.2f}"
                        )
                    with col_a:
                        st.markdown("**LoRA adapter**")
                        st.success(adapter_pred or "(empty output)")
                        st.caption(
                            f"Token overlap with context: "
                            f"{_jaccard_overlap(adapter_pred or '', context_text):.2f}"
                        )
                elif adapter_pred is not None:
                    st.markdown("**LoRA adapter answer**")
                    st.success(adapter_pred or "(empty output)")
                    st.caption(
                        f"Token overlap with context: "
                        f"{_jaccard_overlap(adapter_pred or '', context_text):.2f}"
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
        embedding_model=sidebar["embedding_model"],
        chunk_size=sidebar["chunk_size"],
        chunk_overlap=sidebar["chunk_overlap"],
        min_chunk_size=sidebar["min_chunk_size"],
        top_k=sidebar["top_k"],
        min_score=sidebar["min_score"],
        cache_version=2,
    )
    # Keep the pipeline config in sync with sidebar values even if cache hit.
    pipeline.config.chunk_size = sidebar["chunk_size"]
    pipeline.config.chunk_overlap = sidebar["chunk_overlap"]
    pipeline.config.min_chunk_size = sidebar["min_chunk_size"]
    pipeline.config.top_k = sidebar["top_k"]
    pipeline.config.min_score = sidebar["min_score"]
    pipeline.config.retrieval_mode = sidebar["retrieval_mode"]
    pipeline.config.hybrid_alpha = sidebar["hybrid_alpha"]
    # Streamlit can keep a cached pipeline object across hot reloads. Rebuild
    # the lightweight retriever wrapper so cached sessions pick up the current
    # Retriever.retrieve signature and hybrid/BM25 behavior.
    pipeline.retriever = Retriever(pipeline.embedder, pipeline.store)

    # If the persisted index was built with a different embedding model than
    # the one currently selected, its vectors live in a different space and
    # search results would be garbage. Reset the index and tell the user.
    # We read the manifest from disk (not session_state) so this also covers
    # the case where the user restarts the app and then picks a new model.
    persisted_model: str | None = None
    if MANIFEST_FILE.exists():
        try:
            import json as _json
            persisted_model = (
                _json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
                .get("embedding_model")
            )
        except Exception:  # noqa: BLE001
            persisted_model = None

    if (
        pipeline.is_ready()
        and persisted_model
        and persisted_model != sidebar["embedding_model"]
    ):
        pipeline.reset_index()
        st.session_state.last_manifest = None
        st.session_state.eval_results = None
        st.session_state.eval_ranks = None
        _reset_eval_editor_state()
        st.sidebar.warning(
            f"Embedding model changed from `{persisted_model}` to "
            f"`{sidebar['embedding_model']}`. The old index was cleared — "
            "click **Build / Rebuild Index** to re-embed your PDFs."
        )

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
        _reset_eval_editor_state()
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
                _reset_eval_editor_state()

    # Sidebar status
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Index status**")
    if pipeline.is_ready():
        n_docs = pipeline.store.metadata["doc_name"].nunique()
        n_chunks = len(pipeline.store)
        st.sidebar.success(
            f"Ready — {n_docs} document(s), {n_chunks} chunk(s)\n\n"
            f"Embedding: `{pipeline.config.embedding_model}`\n\n"
            f"Retrieval: `{pipeline.config.retrieval_mode}`\n\n"
            f"Answering via: `{provider_used}`"
        )
    else:
        st.sidebar.warning("Index is empty — build it to enable chat.")
        st.sidebar.caption(f"Selected embedding model: `{sidebar['embedding_model']}`")

    # Main header + tabs
    st.title(":books: Generative AI RAG Document Q&A Chatbot")
    st.caption(
        "A RAG application that answers your questions from uploaded PDF "
        "documents using retrieval-augmented generation — with citations."
    )

    tab_chat, tab_docs, tab_eval, tab_lora = st.tabs(
        ["Chat", "Documents / Index", "Evaluation", "LoRA / QLoRA"]
    )
    with tab_chat:
        render_chat_tab(pipeline)
    with tab_docs:
        render_documents_tab(pipeline)
    with tab_eval:
        render_evaluation_tab(pipeline)
    with tab_lora:
        render_lora_tab(pipeline)


if __name__ == "__main__":
    main()
