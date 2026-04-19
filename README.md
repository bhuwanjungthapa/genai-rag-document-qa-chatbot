# Course Syllabus Question-Answering Chatbot

A small, local **RAG (Retrieval-Augmented Generation)** chatbot that answers
student questions using your own course PDFs — syllabi, assignment handouts,
and policy documents. Answers are grounded in retrieved context and always
shown with citations like `[Syllabus.pdf p.3]`.

This project is intentionally kept simple and implementable. It is meant as a
student-level project that still covers every real RAG concern: ingestion,
section-aware chunking, embeddings, FAISS vector search, grounded generation,
logging, and evaluation.

---

## 1. Project Overview

**What it does**

- Upload one or more course PDFs in the UI.
- The app extracts page-level text, cleans it, and chunks it (section-aware,
  with recursive fallback and overlap).
- Chunks are embedded with `sentence-transformers/all-MiniLM-L6-v2` and
  stored in a local **FAISS** index that persists to disk.
- When you ask a question, the app retrieves the top-k most similar chunks,
  stuffs them into a strict "answer only from this context" prompt, and asks
  **Gemini** (default) or **OpenAI** to generate the answer.
- If retrieval is too weak, the app refuses to answer instead of hallucinating.
- Every Q&A is logged to `logs/qa_log.jsonl`.
- An evaluation tab runs a CSV of gold questions through the pipeline and
  computes Hit@k + lets you label answers as Correct / Partially Correct /
  Unsupported / Hallucinated.

---

## 2. Architecture / Pipeline

```
            ┌───────────────┐
  PDFs ───▶ │  pdf_loader   │ ── per-page text + metadata
            └───────┬───────┘
                    ▼
            ┌───────────────┐
            │   chunker     │ ── section-aware + recursive, with overlap
            └───────┬───────┘
                    ▼
            ┌───────────────┐
            │   embedder    │ ── all-MiniLM-L6-v2, L2-normalized
            └───────┬───────┘
                    ▼
            ┌───────────────┐
            │ FAISS store   │ ── IndexFlatIP + chunks.parquet
            └───────┬───────┘
                    ▼

User Question
  → Embed Query
  → Retrieve Top-k Chunks from FAISS
  → Combine Context + Strict Grounded Prompt
  → LLM (Gemini / OpenAI) Generates Grounded Answer
  → Show Answer + Citations + Retrieved Chunks
```

Key modules (all in `src/`):

| File | Purpose |
| --- | --- |
| `pdf_loader.py`   | Extract page-level text with `pypdf`, clean whitespace, keep metadata. |
| `chunker.py`      | Heading-aware splitting with recursive fallback + tiny-chunk merging. |
| `embedder.py`     | `sentence-transformers` wrapper that returns L2-normalized vectors. |
| `vector_store.py` | FAISS `IndexFlatIP` + aligned pandas metadata + save/load. |
| `retriever.py`    | Combines embedder + store into a `top_k` retriever. |
| `llm_client.py`   | Provider abstraction: Gemini, OpenAI, or a safe NullClient. |
| `rag_pipeline.py` | High-level API: build index, answer, log, weak-retrieval guardrail. |
| `evaluation.py`   | Hit@k, overlap helper, summary metrics, results dataframe. |
| `utils.py`        | Text cleaning, heading detection, JSONL logging. |

---

## 3. Setup

Tested with **Python 3.11**.

```bash
# 1. Create a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate         # macOS / Linux
# .venv\Scripts\activate          # Windows

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

The first run of the app will download the `all-MiniLM-L6-v2` embedding model
(~90 MB) from Hugging Face and cache it locally.

---

## 4. Add API keys

Copy `.env.example` to `.env` and fill in the key(s) you want to use:

```bash
cp .env.example .env
```

Edit `.env`:

```
GEMINI_API_KEY=your_gemini_key_here
GEMINI_MODEL=gemini-1.5-flash

OPENAI_API_KEY=your_openai_key_here
OPENAI_MODEL=gpt-4o-mini
```

- You only need **one** of the two keys.
- If **neither** key is set, the app still runs — you can ingest, index, and
  retrieve — but the "Ask" button will return a friendly message explaining
  that generation is disabled.
- The UI sidebar lets you pick which provider to use at runtime.

---

## 5. Run the app

```bash
streamlit run app.py
```

Then:

1. In the sidebar, **upload one or more PDFs** (syllabus, assignment, policy).
2. Click **Build / Rebuild Index**. Status will show number of documents and
   chunks. The index is persisted to `indexes/`, so the next run reloads it
   automatically.
3. In the **Chat** tab, click one of the starter questions or type your own.
4. The answer panel shows the response, citations, and an expandable
   "Retrieved Chunks" section with similarity scores.

### Managing your corpus

- The **Documents / Index** tab lists every PDF currently in `data/`.
  Click **Remove** next to a file to delete it from disk and **automatically
  rebuild the index** from the remaining PDFs.
- The sidebar has a **Clear corpus & index** button that wipes `data/` and the
  FAISS index in one click.
- "Build / Rebuild Index" always rebuilds from whatever is in `data/` at that
  moment — so the index and the files you see always match.

---

## 6. How evaluation works

Open the **Evaluation** tab and either:

- check "Use bundled sample CSV" to load `eval/sample_eval_questions.csv`, or
- upload your own CSV with columns:
  - `question` (required)
  - `gold_answer` (required)
  - `gold_doc` (optional: the filename that should support the answer)
  - `gold_page` (optional: the page number inside that document)

Click **Run evaluation**. For each question the pipeline will:

1. Embed the question, retrieve top-k chunks from FAISS.
2. Compute **Hit@k** — whether `gold_doc`/`gold_page` appears in retrieved results.
3. Generate a predicted answer using the selected LLM.
4. Compute a rough token-overlap signal between predicted answer, gold answer,
   and retrieved context (for grader prioritization — not a final score).

The results table is editable: mark each row with one of
**Correct / Partially Correct / Unsupported / Hallucinated** and add notes.
The summary shows:

- retrieval hit rate (based on rows with a `gold_doc`)
- correctness rate (based on rows you labeled)
- hallucination rate (based on rows you labeled)
- grounded-by-guardrail rate (share of answers that passed the min-score gate)

You can download both the results CSV and the full JSONL Q&A log.

**Important:** we do not fabricate benchmark scores. The numbers you see are
exactly what your questions, your PDFs, and your manual labels produce.

---

## 6b. Generating report figures

There are two ways to produce the five report figures:

### Option A — inside the app (recommended)

In the **Evaluation** tab, click **Run evaluation**. Below the results table
the app now renders the five figures inline and shows a download button under
each one. There is also a **Download all figures (ZIP)** button at the bottom.

- Figures 1, 2, and 4 appear as soon as evaluation finishes.
- Figures 3 and 5 unlock the moment you start labeling rows
  (Correct / Partially Correct / Unsupported / Hallucinated) — the charts
  update live as you edit.

### Option B — command line

For batch reporting or CI, the same figures can be produced with a script:

```bash
python reports/make_figures.py \
    --eval-csv eval/sample_eval_questions.csv \
    --results-csv path/to/evaluation_results.csv \
    --out reports/figures
```

Both paths use the same figure builders (`reports/figures.py`), so the UI
charts and the saved PNGs are identical.

The files produced are:

| File | What it shows |
| --- | --- |
| `fig1_chunks_per_doc.png`     | Corpus description: number of chunks per uploaded document. |
| `fig2_hit_at_k.png`           | Retrieval quality: Hit@k for k = 1, 3, 5, 10. |
| `fig3_label_distribution.png` | Answer quality: counts of Correct / Partially Correct / Unsupported / Hallucinated (needs a labeled CSV). |
| `fig4_score_hist.png`         | Top-1 cosine similarity histogram split by Hit@1 (justifies the `MIN_SCORE` guardrail). |
| `fig5_heatmap.png`            | Hit@k × answer label heatmap (ties retrieval success to answer quality). |

Notes:

- Figures 2 and 4 need a **built FAISS index** in `indexes/` (build it in the Streamlit app once).
- Figures 3 and 5 need a **labeled** `evaluation_results.csv` — open the Evaluation tab,
  click **Run evaluation**, label each row, then click **Download results CSV** and
  pass that path via `--results-csv`.
- If any input is missing the script prints a friendly warning and skips just that figure.

---

## 7. Screenshots (placeholders)

Add your own screenshots here once you run the app:

- `docs/screenshot_chat.png` — the Chat tab with a grounded answer and citations.
- `docs/screenshot_documents.png` — the Documents / Index tab.
- `docs/screenshot_evaluation.png` — the Evaluation tab with metrics.

---

## 8. Chunking strategy

The chunker tries to preserve the logical structure of a course document:

1. **Heading detection.** Each page's text is scanned line-by-line. Lines that
   look like headings ("Week 3", "Section 4.2", ALL-CAPS titles, numbered
   headings) start a new **section**.
2. **Per-section splitting.** If a section is short enough (`<= chunk_size`)
   it becomes one chunk. Otherwise, a **recursive character splitter** splits
   on `\n\n` → `\n` → `. ` → ` ` with configurable overlap
   (default `chunk_size=800`, `chunk_overlap=150`).
3. **Tiny-chunk merging.** If a chunk ends up below `min_chunk_size`
   (default 250 chars) and shares a section with the previous chunk, it is
   merged backward. This prevents lonely two-sentence chunks.
4. **Metadata.** Every chunk keeps `chunk_id`, `doc_name`, `page_start`,
   `page_end`, `section_title`, and `raw_text`.

All three thresholds are configurable from `.env`, `config.py`, or live in the
Streamlit sidebar.

---

## 9. Retrieval + grounded answer generation

- **Embeddings.** Queries and chunks are embedded with
  `sentence-transformers/all-MiniLM-L6-v2` and L2-normalized, so FAISS inner
  product is equivalent to cosine similarity.
- **Index.** A simple `faiss.IndexFlatIP` — exact search is perfectly fine at
  student-project scale and it avoids the ANN recall tradeoff.
- **Top-k retrieval** with `top_k=4` by default.
- **Weak-retrieval guardrail.** If the top result's similarity is below
  `MIN_SCORE` (default 0.25), the pipeline short-circuits and returns:
  _"I could not find a supported answer in the uploaded course documents."_
- **Strict prompt.** Generation uses the system prompt in
  `config.ANSWER_SYSTEM_PROMPT`, which forbids invented policies, dates, or
  grades, forces citations, and hides chain-of-thought.
- **Citations.** The assistant is instructed to cite `[filename p.PAGE]`. The
  UI also shows all retrieved chunks with their scores so students can verify.

---

## 10. Limitations and future improvements

**Known limits**

- PDF text extraction is only as good as `pypdf` — scanned PDFs without OCR
  will produce empty or garbled text. Run OCR beforehand if needed.
- Heading detection is heuristic. Unusual layouts (multi-column tables,
  slide decks) may produce awkward sections.
- `all-MiniLM-L6-v2` is a small, fast model; retrieval quality on very long or
  technical content can be limited. Upgrading to a larger model (e.g. `bge-small-en`)
  typically improves Hit@k.
- The token-overlap "overlap" metric in evaluation is a prioritization signal,
  not a real correctness score. Manual labels remain the ground truth.
- No user authentication, no database — this is intentional for a local
  single-user project.

**Future improvements**

- Add OCR fallback (e.g., `ocrmypdf`) for scanned documents.
- Hybrid retrieval: combine BM25 + embeddings for better coverage.
- Re-ranking the retrieved chunks with a small cross-encoder.
- Per-course indexes (namespaces) with an easy switcher in the UI.
- Automatic answer-grounding checker that flags claims not supported by any
  retrieved chunk.
- Caching of LLM responses by (question, retrieved-ids) to cut costs during
  evaluation sweeps.

---

## Project layout

```
.
├── app.py                       # Streamlit UI
├── config.py                    # Paths, prompts, defaults, dataclass
├── requirements.txt
├── .env.example
├── README.md
├── data/                        # uploaded PDFs land here
├── indexes/                     # persisted FAISS index + metadata
├── logs/                        # qa_log.jsonl (all questions + retrieved chunks)
├── eval/
│   └── sample_eval_questions.csv
├── reports/
│   ├── figures.py               # pure figure builders (shared by app + script)
│   ├── make_figures.py          # CLI: save the 5 report figures to disk
│   └── figures/                 # output PNGs land here (when using the CLI)
└── src/
    ├── __init__.py
    ├── pdf_loader.py
    ├── chunker.py
    ├── embedder.py
    ├── vector_store.py
    ├── retriever.py
    ├── llm_client.py
    ├── rag_pipeline.py
    ├── evaluation.py
    └── utils.py
```
