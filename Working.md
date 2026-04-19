# How the Course Syllabus QA Chatbot Works

This document explains what the project actually *does* end to end — not the
language or libraries under the hood, but the **flow of information** from the
moment you drop a PDF into the sidebar to the moment a chart appears in the
Evaluation tab.

If you want to know *"when I click X, what happens behind the scenes?"* — this
is the file to read.

---

## 1. The 30-second picture

The chatbot is a **Retrieval-Augmented Generation (RAG)** system.

Instead of asking a large language model (LLM) a raw question and hoping it
remembers the right policy, we:

1. **Break the uploaded PDFs into small text pieces ("chunks").**
2. **Turn each chunk into a numerical fingerprint (an "embedding").**
3. **Store all the fingerprints in a fast searchable index.**
4. When a student asks a question, **turn the question into the same kind
   of fingerprint** and ask the index *"which chunks are closest?"*.
5. **Hand those few chunks to the LLM** along with the question and tell it:
   *"answer only from this. If it is not here, refuse."*
6. **Attach citations** pointing back at the document and page each chunk
   came from.

Everything the app shows on screen is a consequence of that pipeline.

---

## 2. Architecture at a glance

```
                           ┌──────────────────────────────┐
                           │          Sidebar             │
                           │  - Upload PDFs               │
                           │  - Choose LLM (Gemini/OpenAI)│
                           │  - chunk_size, overlap, k    │
                           │  - Build / Rebuild Index     │
                           │  - Clear corpus & index      │
                           └──────────────┬───────────────┘
                                          │
                                          ▼
                ┌──────────────────────────────────────────┐
                │             INDEX BUILD                  │
                │   PDFs  →  pages  →  chunks  →  vectors  │
                │                                          │
                │          stored on disk in:              │
                │            indexes/faiss.index           │
                │            indexes/chunks.parquet        │
                │            indexes/manifest.json         │
                └──────────────────────────────────────────┘
                                          │
     ┌────────────────────────────────────┼───────────────────────────────┐
     ▼                                    ▼                               ▼
 ┌────────────┐                  ┌──────────────────┐              ┌──────────────┐
 │ Chat tab   │                  │ Documents / Index│              │ Evaluation   │
 │            │                  │ tab              │              │ tab          │
 │ Q → embed  │                  │ - list of PDFs   │              │ - load CSV   │
 │ → search   │                  │ - chunks/doc     │              │ - run Qs     │
 │ → LLM      │                  │ - chunk previews │              │ - label      │
 │ → answer + │                  │ - remove a PDF   │              │ - charts     │
 │  citations │                  └──────────────────┘              └──────────────┘
 └────────────┘
```

The three tabs are three different *views* on top of one shared index.

---

## 3. The lifecycle of your data

### Step 1 — Configure in the sidebar

Before anything is indexed, the sidebar lets you set:

| Control           | What it controls                                                |
| ----------------- | --------------------------------------------------------------- |
| **Upload PDFs**   | Which course documents will become the chatbot's knowledge.     |
| **LLM provider**  | Which model writes the final answer (Gemini or OpenAI).         |
| **chunk_size**    | Maximum characters per chunk. Bigger = more context per chunk.  |
| **chunk_overlap** | Characters shared with the neighboring chunk. Prevents cut-offs.|
| **min_chunk_size**| Chunks smaller than this are merged into a neighbor.            |
| **top_k**         | How many chunks to retrieve for every question.                 |
| **min_score**     | Retrieval confidence floor. Below this, the bot refuses to answer.|

None of those knobs do anything on their own — they are **remembered** and
applied the next time you click **Build / Rebuild Index** (for chunking) or
ask a question (for retrieval).

The LLM choice is different: it is applied **live**. If you switch from
Gemini to OpenAI between two questions, the next answer is written by the
new provider. If neither API key is set, the app silently falls back to a
"no-LLM" mode where retrieval and citations still work but the answer
itself is a notice asking you to configure a key.

### Step 2 — Click "Build / Rebuild Index"

This is the single action that turns your PDFs into something searchable.
Under the hood the app does five things in order:

1. **Save uploads.** Any new files from the sidebar uploader are copied
   into the project's `data/` folder so they survive a rerun.
2. **Read every PDF page-by-page.** Each page produces a record carrying
   the document name and the page number. Text is cleaned up — repeated
   whitespace is collapsed, blank fragments are dropped.
3. **Chunk every document.** This is not a dumb character split. The
   chunker first tries to detect **headings** and split the document into
   logical sections (e.g. "Grading Policy", "Schedule"). Long sections are
   broken down further using paragraph → line → character boundaries so
   that no chunk is bigger than `chunk_size`. Tiny tail chunks are merged
   into their neighbor so the index is not cluttered with 40-character
   stubs. Each chunk keeps:
   - `chunk_id` (unique),
   - `doc_name`,
   - `page_start` and `page_end`,
   - `section_title` (if one was detected),
   - `raw_text`.
4. **Embed the chunks.** A small local sentence-embedding model
   (`all-MiniLM-L6-v2`) converts every chunk's text into a vector of
   numbers (a "fingerprint"). Similar passages end up as vectors pointing
   in similar directions.
5. **Index and persist.** All the vectors go into a FAISS index for fast
   similarity search, and three files are written to disk:
   - `indexes/faiss.index` — the numeric index,
   - `indexes/chunks.parquet` — the chunk metadata + raw text,
   - `indexes/manifest.json` — a snapshot of how the index was built
     (which PDFs, chunk size, how many chunks, timestamp).

After this step the sidebar status flips to **Ready — N document(s),
M chunk(s)**, and the Chat and Evaluation tabs come alive.

### Step 3 — Managing your corpus

The project treats the `data/` folder as the single source of truth:

- Uploading new PDFs adds files to `data/`.
- **Remove** buttons in the Documents tab delete the file **and** rebuild
  the index from whatever is left.
- **Clear corpus & index** in the sidebar wipes `data/` and all three
  index files so you can start clean.

Any of those actions clears cached evaluation results as well, so you are
never looking at metrics computed against a stale index.

---

## 4. What happens when you ask a question

This is the core of the project. When you type a question in the **Chat**
tab and press Enter:

```
┌─────────────────────────┐
│  Student question text  │
└────────────┬────────────┘
             │
             ▼
      [embed the question]        ← same model used during indexing
             │
             ▼
   [similarity search in FAISS]   ← compare question vector against every
             │                      chunk vector; keep the top `top_k`
             ▼
   ┌─────────────────────┐
   │  top_k chunks +     │
   │  similarity scores  │
   └──────────┬──────────┘
              │
              ▼
      [confidence guardrail]      ← if top score < min_score, refuse
              │                     and do NOT call the LLM
              ▼
   [build a context block from    ← each chunk is labeled with its
    the retrieved chunks]           doc/page/section so the LLM can cite
              │
              ▼
   ┌────────────────────────┐
   │  LLM (Gemini / OpenAI) │   system prompt:
   │  receives:             │   "answer ONLY from this context,
   │    - system rules      │    refuse otherwise, cite [file p.PAGE]"
   │    - question          │
   │    - context block     │
   └────────────┬───────────┘
                │
                ▼
        ┌───────────────┐
        │  Final answer │
        │  + citations  │
        └───────────────┘
```

### What you see in the chat bubble

After the pipeline finishes, the Chat tab shows one message with
several parts. Each part maps to a stage above:

| Piece of output             | Where it comes from                                         |
| --------------------------- | ----------------------------------------------------------- |
| **Answer text**             | The LLM's response to the question + context block.         |
| **"Answered via: X" caption** | Tells you which provider actually wrote this answer (Gemini, OpenAI, or the no-LLM fallback). |
| **Citations `[file p.N]`**  | Deduplicated list of the documents/pages behind the retrieved chunks. Only shown when the answer is grounded. |
| **Retrieved chunks expander** | The exact chunks handed to the LLM, with similarity scores, page ranges, and section titles. This is your trust layer — you can see what the model read before it answered. |

### The "I could not find a supported answer" case

You will see this fixed sentence whenever **any** of these happen:

- Retrieval returned nothing (index empty or question is gibberish).
- The top similarity score is below `min_score` (the guardrail).
- The LLM itself decided there was not enough evidence in the context.

When this message appears, the citations list is intentionally empty —
there are no sources the bot will stand behind. This is the project's way
of **preferring silence over hallucination**.

### The logs

Every question, answer, and the chunks that were retrieved are appended
to `logs/qa_log.jsonl`. That file is what gets offered as a download in
the Evaluation tab, and it is the audit trail if you ever need to
reconstruct why the bot said something.

---

## 5. The "Documents / Index" tab

This tab is your window into **what the chatbot actually knows right
now**. It has three sections:

### Files on disk (`data/`)

A simple list of every PDF currently in the `data/` folder with its size
in KB. Each row has a **Remove** button. Clicking it:

1. Deletes the PDF from disk.
2. Rebuilds the index from whatever is left.
3. Clears cached evaluation results.

If no PDFs are left, the index resets and the app goes back to "not
ready".

### Per-document summary

A table with one row per indexed document, showing:

- `doc_name`
- `num_chunks` — how many pieces the chunker produced for that document.
- `page_start` / `page_end` — the first and last page found in the
  extracted text.

This is a fast sanity check. If you uploaded a 30-page syllabus and the
summary says 1 chunk, something went wrong (probably a scanned PDF with
no extractable text). If chunk counts look wildly different between two
documents of similar length, `chunk_size` is probably worth revisiting.

### Example chunk previews

A few sample chunks per document, with their page range, section title,
and text. Useful for seeing whether the chunker is cutting in sensible
places and whether headings are being detected.

---

## 6. The "Evaluation" tab

The Evaluation tab is where the project earns its reporting story. It
answers two separate questions:

1. **Is retrieval finding the right material?** (objective, automated)
2. **Is the final answer actually correct?** (subjective, manual label)

### 6.1 The evaluation CSV

A tiny CSV with at most four columns:

| Column       | Required | Meaning                                                    |
| ------------ | -------- | ---------------------------------------------------------- |
| `question`   | yes      | The student question to ask.                               |
| `gold_answer`| yes      | A short reference answer you consider correct.             |
| `gold_doc`   | no       | Filename you expect to contain the supporting text.        |
| `gold_page`  | no       | Page number (inside `gold_doc`) that has the answer.       |

You can edit this file outside the app, upload it from the Evaluation
tab, or just load the bundled `eval/sample_eval_questions.csv`.

### 6.2 Retrieval diagnostics

Before running the questions, the tab compares the `gold_doc` values in
your CSV against the document filenames that are actually indexed and
shows a short checklist with ✅ / ❌. Because real filenames are often
long (`CSE_434___CSE_534_Syllabus.pdf`) while CSV authors write short
ones (`Syllabus.pdf`), the app uses a **forgiving matcher**:

1. Lower-case both names.
2. Drop `.pdf`.
3. Replace anything that is not a letter or digit with `_`.
4. Accept equal names, or a substring match in either direction (with a
   4-character minimum).

If a `gold_doc` does not match anything in the index, the diagnostics
panel auto-opens and warns you — that is the single most common reason
a Hit@k chart shows 0 even though answers look correct.

### 6.3 Running the evaluation

The **Run evaluation** button does two passes over the questions:

- **Pass A — full pipeline.** For each question, run the exact same
  chat flow: retrieve → guardrail → LLM → citations. This produces the
  results table with `predicted_answer`, `hit_at_k`, `grounded_or_not`,
  token-overlap signals, etc.
- **Pass B — retrieval only.** For each question, retrieve the top 10
  chunks *without* calling the LLM, and record the rank at which the
  gold chunk first appears (1 = top result, … , 10 = last; blank if the
  gold chunk never showed up in the top 10). This is cheap and is what
  the Hit@k and score-distribution charts are built from.

### 6.4 The results table

Editable, with the column order tuned for manual grading:

```
question | gold_answer | predicted_answer | label | notes |
hit_at_k | grounded_or_not | retrieved_doc_names | retrieved_pages |
top_score | overlap_pred_vs_gold | overlap_pred_vs_context
```

The two columns you actually edit are **label** and **notes**:

- `label` is a dropdown with four values:
  - **Correct** — answer fully matches the gold answer.
  - **Partially Correct** — gets part of it, misses or muddles the rest.
  - **Unsupported** — the bot refused to answer (intentional silence).
  - **Hallucinated** — the bot made something up that is not in the
    documents.
- `notes` is free text for anything future-you will want to remember.

**Important implementation detail.** The table is inside a Streamlit
fragment and the baseline data is kept immutable across reruns. That
means editing a `label` only re-renders the fragment — not the whole
page — and your selection is preserved even as the charts below refresh.
(Streamlit's fullscreen *overlay* for tables still resets on each
rerun; that is a platform limitation, so the UI gently suggests editing
in the normal view.)

### 6.5 The summary metric row

Four numbers shown above the charts:

| Metric                         | What it tells you                                    |
| ------------------------------ | ---------------------------------------------------- |
| **Questions**                  | Total rows in the evaluation run.                    |
| **Retrieval Hit@k**            | Fraction of rows whose `gold_doc`/`gold_page` was inside the top `top_k` retrieved chunks. Purely objective; ignores the LLM. |
| **Correct rate (labeled)**     | Of the rows you have labeled, the share marked **Correct**. |
| **Hallucination rate (labeled)** | Of the rows you have labeled, the share marked **Hallucinated**. |

A small caption below shows the **grounded-by-guardrail rate** — the
fraction of answers that were *not* the "I could not find…" refusal.
This is a sanity check: if it's near 100% but your Correct rate is low,
the bot is confidently wrong; if it's near 0%, retrieval is probably
broken.

### 6.6 The five report figures

Each figure lives inside its own card, can be downloaded as a PNG, and
all five can be bundled as a single ZIP for your report.

#### Figure 1 — Chunks per document

*What it plots:* a horizontal bar per document with the number of chunks
produced for it.

*Why it matters:* shows whether the chunker treated documents in a
balanced way. A tiny, scanned PDF with 0–1 chunks is a red flag for
failed text extraction. Wildly uneven counts for similarly-sized PDFs
often signal a `chunk_size` or heading-detection problem.

#### Figure 2 — Retrieval Hit@k (k = 1, 3, 5, 10)

*What it plots:* four bars. Bar `k` is the fraction of evaluation rows
whose gold chunk appeared in the top `k` retrieved chunks. Each bar is
labeled with both the raw count and the percentage, e.g. `7/10 / 70%`.

*Why it matters:* this is the single most honest measure of retrieval
quality, and it is **independent of the LLM**. If Hit@10 is high but
Hit@1 is low, retrieval is finding the right document but ranking it
poorly — a hint that chunk size or embedding model quality is the
bottleneck. If all bars are low, the diagnostics panel above usually
reveals a `gold_doc` mismatch. A warning box is drawn inside the chart
when every bar is zero.

#### Figure 3 — Answer label distribution

*What it plots:* one bar per manual label (`Correct`, `Partially
Correct`, `Unsupported`, `Hallucinated`) with the count of rows you gave
that label.

*Why it matters:* this is the human side of quality. It only appears
once you label at least one row, and it updates live as you grade more.
A healthy distribution has most of the mass on `Correct` and
`Unsupported` (appropriate refusals), with `Hallucinated` close to zero.

#### Figure 4 — Top-1 retrieval score by Hit@1

*What it plots:* two overlaid histograms of the top-1 cosine similarity
score, one for questions where Hit@1 was true (green) and one where it
was false (red).

*Why it matters:* it tells you whether **the similarity score is a
useful signal for trust**. If the green histogram lives at high scores
and the red one at low scores, your `min_score` threshold is doing
real work — refusing when the top chunk is weak. If the two histograms
overlap heavily, the score alone is not separating hits from misses and
you should not trust any single-chunk threshold.

#### Figure 5 — Retrieval success × answer label heatmap

*What it plots:* a 2×4 grid. Rows are `Hit@5 = True` / `Hit@5 = False`.
Columns are the four labels. Each cell is the number of rows that fall
into that combination.

*Why it matters:* this is the figure that separates **retrieval errors**
from **generation errors**. The interesting cells are:

- `Hit@5=True, Hallucinated` — retrieval worked, the model
  hallucinated anyway. That is a prompt / model problem.
- `Hit@5=False, Correct` — retrieval "missed", yet the answer is
  correct. Usually means your `gold_doc`/`gold_page` is wrong or the
  chunk spanned a different page than you expected.
- `Hit@5=False, Unsupported` — retrieval missed and the model correctly
  refused. The guardrail earned its keep.

Like Figure 3, this one needs at least one manual label and a
`gold_doc` column to render.

---

## 7. Where reliability comes from

A few design choices that are easy to miss but do most of the heavy
lifting:

- **One embedding model on both sides.** Because the same model is used
  to embed chunks and to embed questions, their vectors live in the same
  space, so cosine similarity is meaningful.
- **L2-normalized vectors + inner-product FAISS.** Inner product on
  normalized vectors equals cosine similarity, which gives us scores
  bounded roughly in `[-1, 1]` and a clean threshold to talk about.
- **Confidence guardrail before the LLM.** If the top chunk scores
  below `min_score`, the LLM is never even called. This stops the most
  confident kind of hallucination before it starts.
- **Strict, terse system prompt.** The LLM is told to answer *only* from
  the provided context, to cite with `[filename p.PAGE]`, and to return a
  specific refusal sentence when evidence is missing. The app then
  detects that exact refusal sentence to flip `grounded_or_not`.
- **Everything on disk is reproducible.** The `manifest.json` records
  which documents, chunk sizes, and embedding model produced the index.
  If a new teammate clones the repo and builds with the same settings,
  they get the same index.
- **`data/` is the source of truth.** Uploads, removals, and the "Clear
  corpus" button all operate on `data/` first, then rebuild from what
  remains. You can never end up with an index pointing at a file that no
  longer exists.

---

## 8. End-to-end flow (single diagram)

```
┌──────────────────────────────────────────────────────────────────────┐
│                               USER                                   │
└────────────────────┬─────────────────────────────┬───────────────────┘
                     │ upload + build              │ ask a question
                     ▼                             ▼
    ┌────────────────────────────┐   ┌──────────────────────────────┐
    │        INGESTION           │   │          QUERY TIME          │
    │                            │   │                              │
    │  PDF  ──▶ pages            │   │  question ─▶ embedding       │
    │         ──▶ cleaned text   │   │              │               │
    │         ──▶ chunks         │   │              ▼               │
    │              │             │   │    FAISS similarity search   │
    │              ▼             │   │              │               │
    │        embeddings          │   │              ▼               │
    │              │             │   │     top-k chunks + scores    │
    │              ▼             │   │              │               │
    │         FAISS index ◀──────┼───┘              │               │
    │      + chunks.parquet      │                  ▼               │
    │      + manifest.json       │     min_score guardrail          │
    └────────────────────────────┘                  │               │
                                                    ▼               │
                                  ┌─────────────────────────────┐   │
                                  │  LLM: Gemini or OpenAI      │   │
                                  │  system: "answer from       │   │
                                  │  context only, cite, refuse"│   │
                                  └──────────┬──────────────────┘   │
                                             │                      │
                                             ▼                      │
                                  ┌────────────────────────────┐    │
                                  │  answer + citations        │    │
                                  │  + retrieved chunks        │    │
                                  │  + log to qa_log.jsonl     │────┘
                                  └────────────────────────────┘
```

That one diagram is essentially the entire project.

---

## 9. TL;DR per UI surface

- **Sidebar** — pick a model, pick your knobs, build the index.
- **Chat tab** — ask a question; you always see the answer, the
  provider, citations, and the exact chunks that were used.
- **Documents / Index tab** — inspect and curate what the bot knows.
- **Evaluation tab** — run a question set, grade the answers, and
  read five charts that separate *retrieval quality* from *answer
  quality* so you can tell which part of the pipeline to improve next.
