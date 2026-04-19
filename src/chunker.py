"""
Section-aware chunking with recursive fallback.

Strategy:
1. Group consecutive pages of the same document together.
2. Split each document by detected headings into logical sections.
3. For each section, if it fits inside `chunk_size`, keep it as one chunk.
   Otherwise, recursively split on paragraph -> line -> character boundaries
   with `chunk_overlap`.
4. Merge tiny tail chunks (< `min_chunk_size`) into their previous neighbor
   when they share the same section.

Every chunk carries the metadata required by the retriever.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

from .pdf_loader import PageRecord
from .utils import looks_like_heading


@dataclass
class Chunk:
    chunk_id: str
    doc_name: str
    page_start: int
    page_end: int
    section_title: str
    raw_text: str

    def as_dict(self) -> dict:
        return asdict(self)


# --- Recursive character splitter -----------------------------------------

_SEPARATORS = ["\n\n", "\n", ". ", " "]


def _recursive_split(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split `text` into chunks of roughly `chunk_size` chars with overlap."""
    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []

    # Find the best separator that actually appears.
    separator = ""
    for sep in _SEPARATORS:
        if sep in text:
            separator = sep
            break

    pieces: list[str] = text.split(separator) if separator else list(text)

    chunks: list[str] = []
    buf = ""
    for piece in pieces:
        candidate = (buf + separator + piece) if buf else piece
        if len(candidate) <= chunk_size:
            buf = candidate
            continue

        # Flush current buffer.
        if buf:
            chunks.append(buf)

        # If a single piece is still too big, recurse with a finer separator.
        if len(piece) > chunk_size:
            chunks.extend(_recursive_split(piece, chunk_size, chunk_overlap))
            buf = ""
        else:
            buf = piece

    if buf:
        chunks.append(buf)

    # Apply overlap by prepending the tail of the previous chunk to the next one.
    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped: list[str] = [chunks[0]]
        for prev, curr in zip(chunks, chunks[1:]):
            tail = prev[-chunk_overlap:] if len(prev) > chunk_overlap else prev
            overlapped.append((tail + " " + curr).strip())
        chunks = overlapped

    return [c.strip() for c in chunks if c.strip()]


# --- Section detection ----------------------------------------------------


@dataclass
class _Section:
    title: str
    text: str
    page_start: int
    page_end: int


def _split_pages_into_sections(pages: list[PageRecord]) -> list[_Section]:
    """Scan pages line-by-line and group text into sections by heading."""
    sections: list[_Section] = []
    current_title = "Introduction"
    current_lines: list[str] = []
    current_page_start = pages[0].page_number if pages else 1
    current_page_end = current_page_start

    def flush():
        nonlocal current_lines, current_title, current_page_start, current_page_end
        text = "\n".join(current_lines).strip()
        if text:
            sections.append(
                _Section(
                    title=current_title,
                    text=text,
                    page_start=current_page_start,
                    page_end=current_page_end,
                )
            )
        current_lines = []

    for page in pages:
        for line in page.text.split("\n"):
            if looks_like_heading(line):
                flush()
                current_title = line.strip()
                current_page_start = page.page_number
                current_page_end = page.page_number
            else:
                if not current_lines:
                    current_page_start = page.page_number
                current_lines.append(line)
                current_page_end = page.page_number

    flush()
    return sections


# --- Public API -----------------------------------------------------------


def chunk_pages(
    pages: Iterable[PageRecord],
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    min_chunk_size: int = 250,
) -> list[Chunk]:
    """Turn a list of pages (possibly from multiple PDFs) into retrieval chunks."""
    by_doc: dict[str, list[PageRecord]] = {}
    for p in pages:
        by_doc.setdefault(p.doc_name, []).append(p)

    all_chunks: list[Chunk] = []

    for doc_name, doc_pages in by_doc.items():
        doc_pages.sort(key=lambda r: r.page_number)
        sections = _split_pages_into_sections(doc_pages)

        # If no headings were detected, fall back to whole-doc recursive splitting.
        if not sections:
            joined = "\n\n".join(p.text for p in doc_pages)
            sections = [
                _Section(
                    title="Document",
                    text=joined,
                    page_start=doc_pages[0].page_number,
                    page_end=doc_pages[-1].page_number,
                )
            ]

        doc_chunks: list[Chunk] = []
        for sec in sections:
            sub_texts = _recursive_split(sec.text, chunk_size, chunk_overlap)
            for j, sub in enumerate(sub_texts):
                cid = f"{doc_name}::p{sec.page_start}-{sec.page_end}::s{len(doc_chunks)}"
                doc_chunks.append(
                    Chunk(
                        chunk_id=cid,
                        doc_name=doc_name,
                        page_start=sec.page_start,
                        page_end=sec.page_end,
                        section_title=sec.title[:120],
                        raw_text=sub,
                    )
                )

        # Merge tiny tail chunks into their previous neighbor (same section only).
        merged: list[Chunk] = []
        for ch in doc_chunks:
            if (
                merged
                and len(ch.raw_text) < min_chunk_size
                and merged[-1].section_title == ch.section_title
                and len(merged[-1].raw_text) + len(ch.raw_text) + 1
                <= chunk_size + chunk_overlap
            ):
                prev = merged[-1]
                prev.raw_text = (prev.raw_text + "\n" + ch.raw_text).strip()
                prev.page_end = max(prev.page_end, ch.page_end)
            else:
                merged.append(ch)

        all_chunks.extend(merged)

    return all_chunks
