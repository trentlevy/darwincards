"""
DarwinCards – Core pipeline
Supports: .mp4/.mov/.mkv (video), .mp3/.m4a/.wav (audio),
          .txt (transcript), .pdf (slides), .pptx/.ppt (slides)

Route
-----
video  → ffmpeg extract audio → Whisper → Claude → .apkg
audio  → Whisper → Claude → .apkg
txt    → Claude (no Whisper needed, free!) → .apkg
pdf    → pdfplumber extract text → Claude → .apkg
pptx   → python-pptx extract text → Claude → .apkg
"""

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Callable, List, Dict

import anthropic
import genanki
import pdfplumber
from pptx import Presentation

# ---------------------------------------------------------------------------
WORDS_PER_CHUNK = 1500

SLIDE_EXTS = {".pdf", ".pptx", ".ppt"}
TEXT_EXTS  = {".txt"}
ALLOWED_EXTS = SLIDE_EXTS | TEXT_EXTS

SYSTEM_PROMPT = """\
You are a medical education expert who creates high-quality Anki flashcards \
from lecture content. Your cards follow best practices:
- One concept per card (minimum information principle)
- Clear, unambiguous questions
- Answers that are concise but complete
- Cloze cards use {{c1::...}} syntax around the key term(s)

You MUST respond with valid JSON only — no prose, no markdown fences.
The JSON should be an array of card objects, each with:
  "type"  : "basic" or "cloze"
  "front" : question string (or cloze-formatted string for cloze cards)
  "back"  : answer / extra context string
"""

BASIC_MODEL_ID = 1607392319
CLOZE_MODEL_ID = 1607392320

BASIC_MODEL = genanki.Model(
    BASIC_MODEL_ID,
    "DarwinCards – Basic",
    fields=[{"name": "Front"}, {"name": "Back"}],
    templates=[{
        "name": "Card 1",
        "qfmt": "{{Front}}",
        "afmt": "{{FrontSide}}<hr id=answer>{{Back}}",
    }],
)

CLOZE_MODEL = genanki.Model(
    CLOZE_MODEL_ID,
    "DarwinCards – Cloze",
    fields=[{"name": "Text"}, {"name": "Extra"}],
    templates=[{
        "name": "Cloze",
        "qfmt": "{{cloze:Text}}",
        "afmt": "{{cloze:Text}}<br><br>{{Extra}}",
    }],
    model_type=genanki.Model.CLOZE,
)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

# Cost per million tokens by model (USD)
_COST_PER_MILLION = {
    "claude-opus-4-6":           {"input": 15.0,  "output": 75.0},
    "claude-sonnet-4-6":         {"input":  3.0,  "output": 15.0},
    "claude-haiku-4-5-20251001": {"input":  0.80, "output":  4.0},
}


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = _COST_PER_MILLION.get(model, {"input": 3.0, "output": 15.0})
    return (input_tokens / 1_000_000) * rates["input"] + (output_tokens / 1_000_000) * rates["output"]


def generate_cards_from_file(
    file_paths,
    anthropic_api_key: str,
    deck_name: str = "DarwinCards",
    card_type: str = "both",
    cards_per_chunk: int = 8,
    language: str = "en",
    claude_model: str = "claude-sonnet-4-6",
    tags: List[str] = None,
    progress: Callable[[str], None] = lambda _: None,
) -> tuple:
    """
    Accepts one or more files: transcripts (.txt), PDF slides (.pdf), or PowerPoint (.pptx/.ppt).
    Multiple files are merged into a single transcript before card generation.
    Returns (apkg_path, usage_stats).
    """
    tags = tags or []
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    sections = []
    for file_path in file_paths:
        ext = Path(file_path).suffix.lower()
        if ext not in ALLOWED_EXTS:
            raise ValueError(f"Unsupported file type '{ext}'. Please upload .txt, .pdf, or .pptx files.")
        if ext in TEXT_EXTS:
            progress(f"Reading transcript ({Path(file_path).name})…")
            text = Path(file_path).read_text(encoding="utf-8", errors="ignore")
        elif ext == ".pdf":
            progress(f"Extracting text from PDF ({Path(file_path).name})…")
            text = _extract_pdf_text(file_path)
        elif ext in {".pptx", ".ppt"}:
            progress(f"Extracting text from PowerPoint ({Path(file_path).name})…")
            text = _extract_pptx_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        if text.strip():
            sections.append(text.strip())

    if not sections:
        raise ValueError("No text could be extracted from the uploaded file(s).")

    # Merge all sources with a clear divider so Claude sees them as one body of material
    transcript = "\n\n━━━ NEW SOURCE ━━━\n\n".join(sections)

    progress("Generating Anki cards with Claude…")
    cards, input_tokens, output_tokens = _generate_cards(
        transcript, anthropic_api_key, card_type,
        cards_per_chunk, claude_model, progress,
    )

    progress(f"Building .apkg deck with {len(cards)} card(s)…")
    apkg_path = _build_apkg(cards, deck_name, tags)

    usage_stats = {
        "model": claude_model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": _estimate_cost(claude_model, input_tokens, output_tokens),
    }
    return apkg_path, usage_stats


# ---------------------------------------------------------------------------
# Slide text extraction
# ---------------------------------------------------------------------------

def _extract_pdf_text(pdf_path: str) -> str:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text.strip())
    return "\n\n".join(pages)


def _extract_pptx_text(pptx_path: str) -> str:
    prs = Presentation(pptx_path)
    slides = []
    for slide in prs.slides:
        texts = []
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip():
                texts.append(shape.text.strip())
        if texts:
            slides.append("\n".join(texts))
    return "\n\n---\n\n".join(slides)


# ---------------------------------------------------------------------------
# Claude card generation
# ---------------------------------------------------------------------------

def _generate_cards(transcript, api_key, card_type, cards_per_chunk, model, progress):
    client = anthropic.Anthropic(api_key=api_key)
    type_instruction = {
        "both":  "Generate a mix of 'basic' (Q&A) and 'cloze' (fill-in-the-blank) cards.",
        "basic": "Generate only 'basic' (Q&A) cards.",
        "cloze": "Generate only 'cloze' (fill-in-the-blank) cards.",
    }[card_type]

    chunks = _chunk_text(transcript, WORDS_PER_CHUNK)
    all_cards = []
    total_input_tokens = 0
    total_output_tokens = 0

    for i, chunk in enumerate(chunks, 1):
        progress(f"Generating cards for section {i}/{len(chunks)}…")
        msg = (
            f"{type_instruction} "
            f"Generate up to {cards_per_chunk} cards from the following lecture content. "
            f"Focus on high-yield medical facts, mechanisms, definitions, and clinical pearls.\n\n"
            f"CONTENT:\n{chunk}"
        )
        resp = client.messages.create(
            model=model, max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": msg}],
        )
        total_input_tokens  += resp.usage.input_tokens
        total_output_tokens += resp.usage.output_tokens
        all_cards.extend(_parse_cards(resp.content[0].text.strip()))

    return all_cards, total_input_tokens, total_output_tokens


# ---------------------------------------------------------------------------
# Build .apkg
# ---------------------------------------------------------------------------

def _stable_id(text: str) -> int:
    """SHA-256-based stable integer ID from a string (survives process restarts)."""
    return int(hashlib.sha256(text.encode()).hexdigest(), 16) % (10 ** 10)


def _build_apkg(cards: List[Dict], deck_name: str, tags: List[str]) -> str:
    deck_id = _stable_id(deck_name)
    deck = genanki.Deck(deck_id, deck_name)

    for card in cards:
        ctype = card.get("type", "basic")
        front = card.get("front", "").strip()
        back  = card.get("back", "").strip()
        if not front:
            continue
        # Stable GUID: same card front+back always maps to the same note ID,
        # so re-importing into Anki updates rather than duplicates the card.
        guid = _stable_id(f"{ctype}:{front}:{back}")
        if ctype == "cloze":
            note = genanki.Note(model=CLOZE_MODEL, fields=[front, back], tags=tags, guid=guid)
        else:
            note = genanki.Note(model=BASIC_MODEL, fields=[front, back], tags=tags, guid=guid)
        deck.add_note(note)

    tmp = tempfile.NamedTemporaryFile(suffix=".apkg", delete=False)
    tmp.close()
    genanki.Package(deck).write_to_file(tmp.name)
    return tmp.name


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk_text(text: str, max_words: int) -> List[str]:
    words = text.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)] or [""]


def _parse_cards(raw: str) -> List[Dict]:
    raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\n?```$", "", raw, flags=re.MULTILINE).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group())
            except json.JSONDecodeError:
                return []
        else:
            return []
    if not isinstance(data, list):
        return []
    return [
        {"type": d.get("type", "basic"), "front": str(d.get("front", "")), "back": str(d.get("back", ""))}
        for d in data if isinstance(d, dict) and d.get("front")
    ]


def _safe_remove(path: str):
    try:
        if path:
            os.remove(path)
    except OSError:
        pass
