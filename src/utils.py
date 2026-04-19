"""Small shared utilities: text cleaning, logging, file helpers."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

# --- Text cleaning ---------------------------------------------------------

_WS_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Normalize whitespace and line breaks without destroying structure."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS_RE.sub(" ", text)
    # Preserve paragraph breaks but collapse 3+ newlines into 2.
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    lines = [ln.strip() for ln in text.split("\n")]
    # Drop empty fragments but keep blank lines as paragraph separators.
    cleaned: list[str] = []
    for ln in lines:
        if ln or (cleaned and cleaned[-1] != ""):
            cleaned.append(ln)
    return "\n".join(cleaned).strip()


# --- Heading detection -----------------------------------------------------

# Rough patterns for common syllabus/assignment headings.
_HEADING_PATTERNS = [
    re.compile(r"^\s*(week\s+\d+.*)$", re.IGNORECASE),
    re.compile(r"^\s*(module\s+\d+.*)$", re.IGNORECASE),
    re.compile(r"^\s*(section\s+\d+.*)$", re.IGNORECASE),
    re.compile(r"^\s*(chapter\s+\d+.*)$", re.IGNORECASE),
    re.compile(r"^\s*(\d+\.\s+[A-Z][^\n]{2,80})$"),
    re.compile(r"^\s*([A-Z][A-Z0-9 \-&:]{4,60})$"),  # ALL-CAPS headings
]


def looks_like_heading(line: str) -> bool:
    """Heuristic: is this line a likely section heading?"""
    if not line or len(line) > 120:
        return False
    for pat in _HEADING_PATTERNS:
        if pat.match(line.strip()):
            return True
    return False


# --- JSONL logging ---------------------------------------------------------


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one JSON record to a JSONL file, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


# --- Misc ------------------------------------------------------------------


def safe_filename(name: str) -> str:
    """Return a filesystem-friendly version of a filename."""
    name = name.strip().replace("\\", "/").split("/")[-1]
    return re.sub(r"[^A-Za-z0-9._\-]+", "_", name)


def batched(seq: list, size: int) -> Iterable[list]:
    """Yield successive chunks of `size` from `seq`."""
    for i in range(0, len(seq), size):
        yield seq[i : i + size]
