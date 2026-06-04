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

import json
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, List, Dict

import anthropic
import genanki
import openai
import pdfplumber
from pptx import Presentation

# ---------------------------------------------------------------------------
WHISPER_MAX_BYTES = 24 * 1024 * 1024
WORDS_PER_CHUNK = 1500

VIDEO_EXTS  = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}
AUDIO_EXTS  = {".mp3", ".m4a", ".wav", ".ogg", ".flac"}
SLIDE_EXTS  = {".pdf", ".pptx", ".ppt"}
TEXT_EXTS   = {".txt"}

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

def generate_cards_from_file(
    file_path: str,
    openai_api_key: str,
    anthropic_api_key: str,
    deck_name: str = "DarwinCards Deck",
    card_type: str = "both",
    cards_per_chunk: int = 8,
    language: str = "en",
    claude_model: str = "claude-opus-4-6",
    tags: List[str] = None,
    progress: Callable[[str], None] = lambda _: None,
) -> str:
    """
    Auto-detects file type and runs the appropriate pipeline.
    Returns path to a .apkg file (caller must delete when done).
    """
    tags = tags or []
    ext = Path(file_path).suffix.lower()

    if ext in VIDEO_EXTS:
        transcript = _video_to_transcript(file_path, openai_api_key, language, progress)
    elif ext in AUDIO_EXTS:
        transcript = _audio_to_transcript(file_path, openai_api_key, language, progress)
    elif ext in TEXT_EXTS:
        progress("Reading transcript…")
        transcript = Path(file_path).read_text(encoding="utf-8", errors="ignore")
    elif ext == ".pdf":
        progress("Extracting text from PDF slides…")
        transcript = _extract_pdf_text(file_path)
    elif ext in {".pptx", ".ppt"}:
        progress("Extracting text from PowerPoint slides…")
        transcript = _extract_pptx_text(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    if not transcript.strip():
        raise ValueError("No text could be extracted from the file.")

    progress("Generating Anki cards with Claude…")
    cards = _generate_cards(
        transcript, anthropic_api_key, card_type,
        cards_per_chunk, claude_model, progress,
    )

    progress(f"Building .apkg deck with {len(cards)} card(s)…")
    return _build_apkg(cards, deck_name, tags)


# ---------------------------------------------------------------------------
# Video → transcript
# ---------------------------------------------------------------------------

def _video_to_transcript(video_path, api_key, language, progress):
    progress("Extracting audio from video…")
    audio_path = _extract_audio(video_path)
    try:
        progress("Splitting audio into segments…")
        chunks = _split_audio(audio_path)
        return _transcribe(chunks, api_key, language, progress)
    finally:
        _safe_remove(audio_path)


def _audio_to_transcript(audio_path, api_key, language, progress):
    progress("Splitting audio into segments…")
    chunks = _split_audio(audio_path)
    return _transcribe(chunks, api_key, language, progress)


# ---------------------------------------------------------------------------
# ffmpeg helpers
# ---------------------------------------------------------------------------

def _extract_audio(video_path: str) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tmp.close()
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "libmp3lame",
        "-ar", "16000", "-ac", "1", "-b:a", "64k", tmp.name,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg error:\n{r.stderr[-2000:]}")
    return tmp.name


def _split_audio(audio_path: str) -> List[str]:
    size = os.path.getsize(audio_path)
    if size <= WHISPER_MAX_BYTES:
        return [audio_path]

    n_parts = math.ceil(size / WHISPER_MAX_BYTES)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
        capture_output=True, text=True,
    )
    duration = float(probe.stdout.strip())
    seg_dur = math.ceil(duration / n_parts)

    tmp_dir = tempfile.mkdtemp()
    pattern = os.path.join(tmp_dir, "chunk_%03d.mp3")
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", audio_path,
         "-f", "segment", "-segment_time", str(seg_dur), "-c", "copy", pattern],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg split error:\n{r.stderr[-2000:]}")
    return sorted(str(p) for p in Path(tmp_dir).glob("chunk_*.mp3"))


# ---------------------------------------------------------------------------
# Whisper transcription
# ---------------------------------------------------------------------------

def _transcribe(chunks, api_key, language, progress):
    client = openai.OpenAI(api_key=api_key)
    parts = []
    for i, path in enumerate(chunks, 1):
        progress(f"Transcribing segment {i}/{len(chunks)}…")
        with open(path, "rb") as f:
            resp = client.audio.transcriptions.create(
                model="whisper-1", file=f, language=language,
            )
        parts.append(resp.text)
    return "\n\n".join(parts)


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
        all_cards.extend(_parse_cards(resp.content[0].text.strip()))

    return all_cards


# ---------------------------------------------------------------------------
# Build .apkg
# ---------------------------------------------------------------------------

def _build_apkg(cards: List[Dict], deck_name: str, tags: List[str]) -> str:
    deck_id = abs(hash(deck_name)) % (10 ** 10)
    deck = genanki.Deck(deck_id, deck_name)

    for card in cards:
        ctype = card.get("type", "basic")
        front = card.get("front", "").strip()
        back  = card.get("back", "").strip()
        if not front:
            continue
        if ctype == "cloze":
            note = genanki.Note(model=CLOZE_MODEL, fields=[front, back], tags=tags)
        else:
            note = genanki.Note(model=BASIC_MODEL, fields=[front, back], tags=tags)
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
