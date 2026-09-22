"""
Notes corpus — ingest and retrieve a school's "shared drive" of prior years'
student notes (the recycled notes on the same lectures, semester after
semester), so card generation can lean on what past students actually found
worth writing down, not just the raw transcript/slides.

Two halves:
  - Ingestion:  chunk a notes document (usually one big combined doc/export)
                into sections and store them, tagged by school.
  - Retrieval:  given a lecture's course/topic/transcript, find the most
                relevant prior-notes chunks for that school via BM25 keyword
                search (no embeddings/API calls needed — fast, free, and
                good enough for "find notes that mention the same terms").

Nothing here calls Claude or costs money; it's pure local search over
whatever has been ingested. If nothing has been ingested yet for a school,
retrieval just returns [] and generation proceeds exactly as before.
"""

import re
from pathlib import Path
from typing import List, Optional, Tuple

from sqlalchemy.orm import Session

from database import SessionLocal
from models import NoteChunk

CHUNK_WORDS = 700
CHUNK_OVERLAP_WORDS = 100

SUPPORTED_NOTE_EXTS = {".docx", ".txt", ".pdf", ".pptx", ".ppt"}


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _body_font_size(doc) -> float:
    """Modal font size (pt) across the doc's runs — the 'normal text' size."""
    from collections import Counter
    sizes = Counter()
    for p in doc.paragraphs:
        for r in p.runs:
            if r.font.size:
                sizes[r.font.size.pt] += 1
    return sizes.most_common(1)[0][0] if sizes else 12.0


def _is_heading_paragraph(para, body_size: float) -> bool:
    """
    Student notes/study guides are usually one flat 'Normal' style throughout,
    with section titles marked by manual formatting (bold + a bigger font,
    and/or ALL CAPS) rather than Word's built-in Heading styles. Word-style
    detection (below) catches the minority that do use real heading styles;
    this catches the much more common manual-formatting convention.
    """
    text = para.text.strip()
    if not text or len(text) > 90:
        return False
    runs = [r for r in para.runs if r.text.strip()]
    if not runs:
        return False
    all_bold = all(r.bold for r in runs)
    if not all_bold:
        return False
    sizes_here = [r.font.size.pt for r in runs if r.font.size]
    bigger = bool(sizes_here) and max(sizes_here) > body_size
    return bigger or (text.isupper() and len(text.split()) <= 12)


def _extract_docx_sections(path: str) -> List[Tuple[Optional[str], str]]:
    """
    Return [(heading_or_None, text_block), ...], splitting on section
    headings so a giant combined doc still yields per-topic sections instead
    of one undifferentiated blob. Tries Word's built-in heading styles first,
    then falls back to the bold/larger-font/ALL-CAPS convention most student
    notes actually use.
    """
    from docx import Document
    doc = Document(path)
    body_size = _body_font_size(doc)

    sections: List[Tuple[Optional[str], str]] = []
    current_heading: Optional[str] = None
    buffer: List[str] = []

    def flush():
        text = "\n".join(buffer).strip()
        if text:
            sections.append((current_heading, text))
        buffer.clear()

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style = (para.style.name or "") if para.style else ""
        is_style_heading = style.lower().startswith("heading") or style.lower() in ("title", "subtitle")
        if is_style_heading or _is_heading_paragraph(para, body_size):
            flush()
            current_heading = text
        else:
            buffer.append(text)
    flush()

    if not sections:
        # No detectable headings at all — treat the whole doc as one section.
        full_text = "\n".join(p.text.strip() for p in doc.paragraphs if p.text.strip())
        sections = [(None, full_text)]
    return sections


def _extract_plain_sections(path: str, ext: str) -> List[Tuple[Optional[str], str]]:
    """Fallback for .txt / .pdf / .pptx — reuses the main pipeline's extractors."""
    from pipeline import _extract_pdf_text, _extract_pptx_text
    if ext == ".txt":
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
    elif ext == ".pdf":
        text = _extract_pdf_text(path)
    elif ext in {".pptx", ".ppt"}:
        text = _extract_pptx_text(path)
    else:
        raise ValueError(f"Unsupported notes file type: {ext}")
    return [(None, text)]


def _chunk_words(heading: Optional[str], text: str) -> List[Tuple[Optional[str], str]]:
    """Split one section's text into overlapping ~CHUNK_WORDS-word pieces."""
    words = text.split()
    if len(words) <= CHUNK_WORDS:
        return [(heading, text)] if text.strip() else []

    step = CHUNK_WORDS - CHUNK_OVERLAP_WORDS
    pieces = []
    for i in range(0, len(words), step):
        piece = " ".join(words[i:i + CHUNK_WORDS])
        if piece.strip():
            pieces.append((heading, piece))
        if i + CHUNK_WORDS >= len(words):
            break
    return pieces


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def ingest_notes_file(file_path: str, school: str, source_label: Optional[str] = None) -> int:
    """
    Extract, chunk, and store one notes document for a school. Returns the
    number of chunks stored.

    Safe to re-run: chunks are keyed by (school, source_label), so ingesting
    an updated export of the same file replaces its old chunks rather than
    duplicating them.
    """
    ext = Path(file_path).suffix.lower()
    if ext not in SUPPORTED_NOTE_EXTS:
        raise ValueError(f"Unsupported notes file type '{ext}'. Use .docx, .txt, .pdf, or .pptx.")

    label = source_label or Path(file_path).name

    if ext == ".docx":
        sections = _extract_docx_sections(file_path)
    else:
        sections = _extract_plain_sections(file_path, ext)

    chunks: List[Tuple[Optional[str], str]] = []
    for heading, text in sections:
        chunks.extend(_chunk_words(heading, text))

    db = SessionLocal()
    try:
        db.query(NoteChunk).filter(
            NoteChunk.school == school, NoteChunk.source_doc == label
        ).delete()
        for i, (heading, text) in enumerate(chunks):
            db.add(NoteChunk(
                school=school, source_doc=label, heading=heading,
                chunk_index=i, chunk_text=text,
            ))
        db.commit()
    finally:
        db.close()

    _invalidate_cache(school)
    return len(chunks)


# ---------------------------------------------------------------------------
# Retrieval — BM25 over a school's chunks, index cached in-process
# ---------------------------------------------------------------------------

_index_cache = {}  # school -> (BM25Okapi | None, [NoteChunk, ...])


def _invalidate_cache(school: str):
    _index_cache.pop(school, None)


def _get_index(db: Session, school: str):
    if school in _index_cache:
        return _index_cache[school]

    rows = db.query(NoteChunk).filter(NoteChunk.school == school).all()
    if not rows:
        _index_cache[school] = (None, [])
        return _index_cache[school]

    from rank_bm25 import BM25Okapi
    tokenized = [_tokenize(f"{r.heading or ''} {r.chunk_text}") for r in rows]
    bm25 = BM25Okapi(tokenized)
    _index_cache[school] = (bm25, rows)
    return _index_cache[school]


def retrieve_relevant_notes(
    query_text: str,
    school: str,
    top_k: int = 4,
    max_chars_each: int = 1500,
) -> List[str]:
    """
    Return up to `top_k` prior-notes excerpts (most relevant to `query_text`)
    for `school`. `query_text` should combine whatever signal you have —
    course name, lecture topic, and/or a snippet of the actual transcript —
    since a giant unsorted notes corpus has no per-lecture metadata to match
    on directly.

    Returns [] if nothing has been ingested for this school yet, or if
    nothing scores as an actual keyword match (no forced-in filler).
    """
    if not school or not query_text.strip():
        return []

    db = SessionLocal()
    try:
        bm25, rows = _get_index(db, school)
    finally:
        db.close()

    if bm25 is None or not rows:
        return []

    query_tokens = _tokenize(query_text)
    if not query_tokens:
        return []
    query_token_set = set(query_tokens)

    scores = bm25.get_scores(query_tokens)
    ranked = sorted(range(len(rows)), key=lambda i: scores[i], reverse=True)

    # BM25's raw score sign isn't a reliable relevance cutoff on small/skewed
    # corpora (common terms can get negative IDF), so gate on actual shared
    # *distinctive* vocabulary instead. Word length alone doesn't separate
    # signal from filler (generic terms like "clinical", "their", "types"
    # are all >3 chars and show up in nearly every chunk), so use the
    # corpus's own IDF: a shared word only counts if it's rare enough in
    # this corpus to be topically distinctive, not boilerplate.
    IDF_FLOOR = 2.0
    MIN_DISTINCTIVE_TERMS = 2  # one shared rare word can be a coincidental mention, not a topic match
    results = []
    for i in ranked:
        if len(results) >= top_k:
            break
        row = rows[i]
        chunk_tokens = set(_tokenize(f"{row.heading or ''} {row.chunk_text}"))
        shared = query_token_set & chunk_tokens
        distinctive_overlap = {t for t in shared if bm25.idf.get(t, 0) >= IDF_FLOOR}
        if len(distinctive_overlap) < MIN_DISTINCTIVE_TERMS:
            continue
        text = row.chunk_text[:max_chars_each]
        prefix = f"[{row.heading}] " if row.heading else ""
        results.append(f"{prefix}{text}")
    return results


def corpus_stats(school: str) -> dict:
    """Quick sanity-check helper: how much has been ingested for a school."""
    db = SessionLocal()
    try:
        rows = db.query(NoteChunk).filter(NoteChunk.school == school).all()
    finally:
        db.close()
    sources = sorted({r.source_doc for r in rows})
    return {"school": school, "chunk_count": len(rows), "sources": sources}
