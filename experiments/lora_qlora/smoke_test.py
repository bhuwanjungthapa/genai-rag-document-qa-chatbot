"""
End-to-end smoke test for the LoRA / QLoRA tab.

This script reproduces, in code, what the four buttons in the Streamlit
"LoRA / QLoRA" tab do, plus the new "Try the adapter" generate flow with
all three context-source modes. It is the fastest way to confirm that the
whole section works from start to finish without clicking through the UI.

What it verifies
----------------
[A] Dependency status      — transformers, peft, datasets, accelerate present
[B] Index is built         — chunks.parquet exists and has rows
[C] QA dataset             — train.jsonl + eval.jsonl exist with rows
[D] Trained adapter        — adapter_model.safetensors exists
[E] Adapter held-out eval  — adapter_eval_summary.json exists with a number
[F] QLoRA readiness        — train_qlora.py --check-only returns valid JSON
[G] Inference: base model  — base FLAN-T5 generates non-empty text
[H] Inference: adapter     — base + LoRA adapter generates non-empty text
[I] Mode 1 — auto-retrieve — pulls chunks from FAISS, runs adapter on them
[J] Mode 2 — pick a chunk  — adapter generates from a known chunk
[K] Mode 3 — pasted text   — adapter generates from arbitrary context

Run from project root:
    .venv/bin/python3.11 experiments/lora_qlora/smoke_test.py
"""

from __future__ import annotations

# Defuse macOS OpenMP conflicts before importing torch / faiss / transformers.
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

LORA_DIR = PROJECT_ROOT / "experiments" / "lora_qlora"
ADAPTER_DIR = LORA_DIR / "adapters" / "flan-t5-small-lora"
TRAIN_JSONL = LORA_DIR / "data" / "train.jsonl"
EVAL_JSONL = LORA_DIR / "data" / "eval.jsonl"
ADAPTER_EVAL_SUMMARY = LORA_DIR / "results" / "adapter_eval_summary.json"
INDEX_CHUNKS = PROJECT_ROOT / "indexes" / "chunks.parquet"

BASE_MODEL_ID = "google/flan-t5-small"


# ---------------------------------------------------------------------------
# Tiny test framework (no pytest needed; one binary green/red per check)
# ---------------------------------------------------------------------------

class Reporter:
    def __init__(self) -> None:
        self.results: list[tuple[str, str, str, float]] = []  # (id, status, detail, secs)

    def add(self, check_id: str, status: str, detail: str, secs: float = 0.0) -> None:
        self.results.append((check_id, status, detail, secs))
        symbol = {"PASS": "[ok]", "FAIL": "[FAIL]", "SKIP": "[skip]"}.get(status, "[?]")
        timing = f" ({secs:.1f}s)" if secs >= 0.1 else ""
        print(f"{symbol} {check_id}: {detail}{timing}")

    def summary(self) -> int:
        passed = sum(1 for _, s, *_ in self.results if s == "PASS")
        failed = sum(1 for _, s, *_ in self.results if s == "FAIL")
        skipped = sum(1 for _, s, *_ in self.results if s == "SKIP")
        total = len(self.results)
        print()
        print("=" * 64)
        print(f"SUMMARY: {passed}/{total} passed   ({failed} failed, {skipped} skipped)")
        print("=" * 64)
        return failed


def run_check(reporter: Reporter, check_id: str, fn) -> object | None:
    print(f"\n--- {check_id} ---")
    t0 = time.time()
    try:
        result = fn()
        secs = time.time() - t0
        if isinstance(result, tuple) and len(result) == 2:
            status, detail = result
        else:
            status, detail = "PASS", str(result) if result is not None else "ok"
        reporter.add(check_id, status, detail, secs)
        return result
    except _SkipCheck as exc:
        reporter.add(check_id, "SKIP", str(exc), time.time() - t0)
        return None
    except Exception as exc:  # noqa: BLE001
        reporter.add(check_id, "FAIL", f"{type(exc).__name__}: {exc}", time.time() - t0)
        traceback.print_exc()
        return None


class _SkipCheck(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_a_dependencies():
    import importlib.util

    required = ["transformers", "peft", "datasets", "accelerate", "torch"]
    missing = [m for m in required if importlib.util.find_spec(m) is None]
    if missing:
        return "FAIL", f"Missing: {', '.join(missing)}"
    return "PASS", "All required packages importable"


def check_b_index():
    if not INDEX_CHUNKS.exists():
        return "FAIL", f"Index missing at {INDEX_CHUNKS.relative_to(PROJECT_ROOT)}. Build it in the Streamlit app first."
    import pandas as pd

    df = pd.read_parquet(INDEX_CHUNKS)
    if df.empty:
        return "FAIL", "Index parquet is empty"
    n_docs = df["doc_name"].nunique()
    return "PASS", f"{len(df)} chunks across {n_docs} document(s)"


def check_c_qa_dataset():
    if not TRAIN_JSONL.exists() or not EVAL_JSONL.exists():
        return "FAIL", "train.jsonl or eval.jsonl missing — run prepare_qa_dataset.py first"
    n_train = sum(1 for line in TRAIN_JSONL.open() if line.strip())
    n_eval = sum(1 for line in EVAL_JSONL.open() if line.strip())
    if n_train == 0 or n_eval == 0:
        return "FAIL", f"Empty dataset (train={n_train}, eval={n_eval})"
    return "PASS", f"{n_train} train rows, {n_eval} eval rows"


def check_d_adapter():
    weights = ADAPTER_DIR / "adapter_model.safetensors"
    cfg = ADAPTER_DIR / "adapter_config.json"
    if not weights.exists() or not cfg.exists():
        return "FAIL", "adapter not trained yet — click 'Train LoRA adapter' in the UI"
    return "PASS", f"adapter weights = {weights.stat().st_size / 1024:.0f} KB"


def check_e_adapter_eval_summary():
    if not ADAPTER_EVAL_SUMMARY.exists():
        return "FAIL", "adapter_eval_summary.json missing — click 'Evaluate LoRA adapter'"
    summary = json.loads(ADAPTER_EVAL_SUMMARY.read_text())
    n = summary.get("examples", 0)
    jaccard = summary.get("mean_jaccard_pred_vs_gold")
    if not n or jaccard is None:
        return "FAIL", f"Bad summary contents: {summary}"
    return "PASS", f"{n} eval examples, mean Jaccard = {jaccard}"


def check_f_qlora_readiness():
    cmd = [sys.executable, str(LORA_DIR / "train_qlora.py"), "--model", BASE_MODEL_ID, "--check-only"]
    res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False, timeout=60)
    if res.returncode != 0:
        return "FAIL", f"exit={res.returncode}; stderr={res.stderr.strip()[-200:]}"
    # train_qlora.py prints a multi-line indented JSON object to stdout.
    payload = json.loads(res.stdout.strip())
    expected_keys = {"bitsandbytes_installed", "cuda_available", "ready_for_qlora"}
    if not expected_keys.issubset(payload):
        return "FAIL", f"unexpected payload: {payload}"
    return "PASS", (
        f"cuda={payload['cuda_available']} bnb={payload['bitsandbytes_installed']} "
        f"ready_for_qlora={payload['ready_for_qlora']}"
    )


# Cache loaded models across the inference checks (matches Streamlit caching).
_BASE_CACHE: tuple | None = None
_LORA_CACHE: tuple | None = None


def _load_base():
    global _BASE_CACHE
    if _BASE_CACHE is None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_ID)
        model.eval()
        _BASE_CACHE = (tok, model)
    return _BASE_CACHE


def _load_adapter():
    global _LORA_CACHE
    if _LORA_CACHE is None:
        from peft import PeftModel
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        tok_src = ADAPTER_DIR if (ADAPTER_DIR / "tokenizer_config.json").exists() else BASE_MODEL_ID
        tok = AutoTokenizer.from_pretrained(tok_src)
        base = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_ID)
        model = PeftModel.from_pretrained(base, ADAPTER_DIR)
        model.eval()
        _LORA_CACHE = (tok, model)
    return _LORA_CACHE


def _generate(tok, model, prompt: str, max_new_tokens: int = 120) -> str:
    import torch

    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tok.decode(out[0], skip_special_tokens=True)


def _format_prompt(instruction: str, context: str, question: str) -> str:
    return (
        f"Instruction: {instruction}\n"
        f"Context: {context}\n"
        f"Question: {question}\n"
        "Answer:"
    )


SAMPLE_INSTRUCTION = (
    "Answer using only the provided context. If the answer is unsupported, "
    "say you could not find a supported answer. Cite the source."
)
SAMPLE_CONTEXT = (
    "Late submissions for assignments are accepted up to 48 hours after the "
    "deadline with a 20% penalty per day. Project deadlines do not allow "
    "late submissions. [policies.pdf p.4]"
)
SAMPLE_QUESTION = "Can I submit the project late?"


def check_g_inference_base():
    if not (ADAPTER_DIR / "adapter_config.json").exists():
        raise _SkipCheck("adapter not trained — base model load is still possible but skipped to keep the suite focused")
    tok, model = _load_base()
    prompt = _format_prompt(SAMPLE_INSTRUCTION, SAMPLE_CONTEXT, SAMPLE_QUESTION)
    out = _generate(tok, model, prompt).strip()
    if not out:
        return "FAIL", "base model returned empty output"
    return "PASS", f'base output ({len(out)} chars): "{out[:90]}..."' if len(out) > 90 else f'base output: "{out}"'


def check_h_inference_adapter():
    if not (ADAPTER_DIR / "adapter_config.json").exists():
        raise _SkipCheck("adapter not trained")
    tok, model = _load_adapter()
    prompt = _format_prompt(SAMPLE_INSTRUCTION, SAMPLE_CONTEXT, SAMPLE_QUESTION)
    out = _generate(tok, model, prompt).strip()
    if not out:
        return "FAIL", "adapter returned empty output"
    return "PASS", f'adapter output ({len(out)} chars): "{out[:90]}..."' if len(out) > 90 else f'adapter output: "{out}"'


def check_i_mode_auto_retrieve():
    """Mirrors UI mode: 'Ask a question (auto-retrieve from index)'."""
    if not (ADAPTER_DIR / "adapter_config.json").exists():
        raise _SkipCheck("adapter not trained")
    if not INDEX_CHUNKS.exists():
        raise _SkipCheck("index not built")

    from config import AppConfig
    from src.rag_pipeline import RAGPipeline

    pipeline = RAGPipeline(AppConfig())
    pipeline.load_index_if_exists()
    if not pipeline.is_ready():
        return "FAIL", "pipeline not ready after load"

    question = "What is the late submission policy?"
    retrieved = pipeline.retriever.retrieve(
        question,
        top_k=2,
        mode=pipeline.config.retrieval_mode,
        hybrid_alpha=pipeline.config.hybrid_alpha,
    )
    if not retrieved:
        return "FAIL", "retrieval returned 0 chunks"

    context = "\n\n---\n\n".join(f"{r.citation}\n{r.raw_text}" for r in retrieved)[:1800]
    tok, model = _load_adapter()
    out = _generate(tok, model, _format_prompt(SAMPLE_INSTRUCTION, context, question))
    if not out.strip():
        return "FAIL", "adapter produced empty answer for retrieved context"
    sources = ", ".join({r.doc_name for r in retrieved})
    return "PASS", f"retrieved {len(retrieved)} chunk(s) from {sources}; got {len(out)}-char answer"


def check_j_mode_pick_chunk():
    """Mirrors UI mode: 'Pick a chunk manually'."""
    if not (ADAPTER_DIR / "adapter_config.json").exists():
        raise _SkipCheck("adapter not trained")
    if not INDEX_CHUNKS.exists():
        raise _SkipCheck("index not built")

    import pandas as pd

    df = pd.read_parquet(INDEX_CHUNKS)
    if df.empty:
        return "FAIL", "index parquet is empty"
    row = df.iloc[0]
    context = str(row["raw_text"])[:1400]
    question = f"What does {row['doc_name']} say about {row.get('section_title') or 'this section'}?"

    tok, model = _load_adapter()
    out = _generate(tok, model, _format_prompt(SAMPLE_INSTRUCTION, context, question))
    if not out.strip():
        return "FAIL", "adapter produced empty answer for picked chunk"
    return "PASS", f'answered from "{row["doc_name"]}" p.{row["page_start"]}, {len(out)} chars'


def check_k_mode_pasted_text():
    """Mirrors UI mode: 'Paste my own context'."""
    if not (ADAPTER_DIR / "adapter_config.json").exists():
        raise _SkipCheck("adapter not trained")
    tok, model = _load_adapter()
    out = _generate(tok, model, _format_prompt(SAMPLE_INSTRUCTION, SAMPLE_CONTEXT, SAMPLE_QUESTION))
    if not out.strip():
        return "FAIL", "adapter produced empty answer for pasted context"
    return "PASS", f'answered from pasted text, {len(out)} chars: "{out[:80]}..."'


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> int:
    print("LoRA / QLoRA end-to-end smoke test")
    print(f"Project root: {PROJECT_ROOT}")
    reporter = Reporter()

    run_check(reporter, "A. dependencies", check_a_dependencies)
    run_check(reporter, "B. index built", check_b_index)
    run_check(reporter, "C. QA dataset", check_c_qa_dataset)
    run_check(reporter, "D. trained adapter", check_d_adapter)
    run_check(reporter, "E. adapter eval summary", check_e_adapter_eval_summary)
    run_check(reporter, "F. QLoRA readiness check", check_f_qlora_readiness)
    run_check(reporter, "G. inference: base FLAN-T5", check_g_inference_base)
    run_check(reporter, "H. inference: base + LoRA adapter", check_h_inference_adapter)
    run_check(reporter, "I. UI mode: auto-retrieve", check_i_mode_auto_retrieve)
    run_check(reporter, "J. UI mode: pick chunk", check_j_mode_pick_chunk)
    run_check(reporter, "K. UI mode: paste context", check_k_mode_pasted_text)

    failures = reporter.summary()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
