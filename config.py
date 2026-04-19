"""
Central configuration for the Course Syllabus QA Chatbot.

All tunable knobs live here. Values are read from environment variables
(via python-dotenv) with safe defaults, so the app works out of the box.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Project paths ---
PROJECT_ROOT: Path = Path(__file__).parent.resolve()
DATA_DIR: Path = PROJECT_ROOT / "data"
INDEX_DIR: Path = PROJECT_ROOT / "indexes"
LOG_DIR: Path = PROJECT_ROOT / "logs"
EVAL_DIR: Path = PROJECT_ROOT / "eval"

for _p in (DATA_DIR, INDEX_DIR, LOG_DIR, EVAL_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# --- Persisted index filenames ---
FAISS_INDEX_FILE: Path = INDEX_DIR / "faiss.index"
METADATA_FILE: Path = INDEX_DIR / "chunks.parquet"   # chunk metadata + raw text
MANIFEST_FILE: Path = INDEX_DIR / "manifest.json"    # ingestion settings snapshot

# --- Logs ---
QA_LOG_FILE: Path = LOG_DIR / "qa_log.jsonl"


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


@dataclass
class AppConfig:
    """Runtime configuration. Instantiate fresh to pick up env/UI overrides."""

    # Chunking
    chunk_size: int = field(default_factory=lambda: _get_int("CHUNK_SIZE", 800))
    chunk_overlap: int = field(default_factory=lambda: _get_int("CHUNK_OVERLAP", 150))
    min_chunk_size: int = field(default_factory=lambda: _get_int("MIN_CHUNK_SIZE", 250))

    # Retrieval
    top_k: int = field(default_factory=lambda: _get_int("TOP_K", 4))
    min_score: float = field(default_factory=lambda: _get_float("MIN_SCORE", 0.25))

    # Embeddings
    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )

    # LLM providers
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    gemini_model: str = field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    )
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    )

    # Default provider preference order
    default_provider: str = "gemini"

    def available_providers(self) -> list[str]:
        available = []
        if self.gemini_api_key:
            available.append("gemini")
        if self.openai_api_key:
            available.append("openai")
        return available


# The strict, grounded answer prompt used by the generation step.
ANSWER_SYSTEM_PROMPT = (
    "You are a course document assistant. Answer the user's question using ONLY "
    "the provided retrieved context from the uploaded course documents. "
    "If the answer is not clearly supported by the context, reply exactly: "
    "\"I could not find a supported answer in the uploaded course documents.\" "
    "Do not invent policies, dates, grades, or deadlines. "
    "Be concise, factual, and student-friendly. "
    "At the end of your answer, cite the supporting source(s) in square brackets "
    "using the format [filename p.PAGE], for example [Syllabus.pdf p.3]. "
    "Do not reveal your reasoning or chain-of-thought."
)

ANSWER_USER_PROMPT_TEMPLATE = """\
User question:
{question}

Retrieved context (each block is labeled with its source):
{context}

Instructions:
- Use only the retrieved context above.
- Cite sources using [filename p.PAGE].
- If the context does not contain the answer, reply exactly:
  "I could not find a supported answer in the uploaded course documents."
"""

# Starter questions shown in the UI.
STARTER_QUESTIONS: list[str] = [
    "What is the grading policy for this course?",
    "When are the midterm and final exams?",
    "What is the late assignment policy?",
    "Who is the instructor and what are their office hours?",
    "What topics are covered in week 3?",
]
