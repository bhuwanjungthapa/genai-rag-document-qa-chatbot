"""PDF ingestion: read PDFs page-by-page and return cleaned text with metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader

from .utils import clean_text


@dataclass
class PageRecord:
    """One page of one document."""

    doc_name: str
    page_number: int        # 1-indexed
    text: str
    source_path: str        # absolute path on disk (for debugging)


def load_pdf(path: str | Path) -> list[PageRecord]:
    """Extract text from a single PDF. Returns an empty list on failure."""
    path = Path(path)
    records: list[PageRecord] = []
    try:
        reader = PdfReader(str(path))
    except Exception as e:  # noqa: BLE001 - surface any parsing error as empty result
        print(f"[pdf_loader] Failed to open {path.name}: {e}")
        return records

    for i, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception as e:  # noqa: BLE001
            print(f"[pdf_loader] Failed page {i} of {path.name}: {e}")
            raw = ""

        text = clean_text(raw)
        if not text:
            continue
        records.append(
            PageRecord(
                doc_name=path.name,
                page_number=i,
                text=text,
                source_path=str(path.resolve()),
            )
        )
    return records


def load_pdfs(paths: Iterable[str | Path]) -> list[PageRecord]:
    """Extract text from many PDFs."""
    all_pages: list[PageRecord] = []
    for p in paths:
        all_pages.extend(load_pdf(p))
    return all_pages
