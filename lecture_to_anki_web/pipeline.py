"""
Core pipeline: MP4 → audio → Whisper transcript → Claude cards → .apkg
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

# ---------------------------------------------------------------------------
WHISPER_MAX_BYTES = 24 * 1024 * 1024
WORDS_PER_CHUNK = 1500

SYSTEM_PROMPT = """\
You are a medical education expert who creates high-quality Anki flashcards \
from lecture transcripts. Your cards follow best practices:
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

# Stable model IDs (arbitrary but must be consistent for deck merging)
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

def generate_cards_from_video(
    video_path: str,
    openai_api_key: str,
    anthropic_api_key: str,
    deck_name: str = "Lecture Cards",
    card_type: str = "both",
    cards_per_chunk: int = 8,
    language: str = "en",
    claude_model: str = "claude-opus-4-6",
    tags: List[str] = None,
    progress: Callable[[str], None] = lambda _: None,
) -> str:
    """
    Runs the full pipeline and returns the path to a .apkg file.
    Caller is responsible for deleting the file when done.
    """
    tags = tags or []
    audio_chunks: List[str] = []
    audio_path = ""

    try:
        progress("Extracting audio from video…")
        audio_path = _extract_audio(video_path)

        progress("Splitting audio into segments…")
        audio_chunks = _split_audio(audio_path)

        progress(f"Transcribing {len(audio_chunks)} audio segment(s) with Whisper…")
        transcript = _transcribe(audio_chunks, openai_api_key, language, progress)

        progress("Generating Anki cards with Claude…")
        cards = _generate_cards(
            transcript, anthropic_api_key, card_type,
            cards_per_chunk, claude_model, progress,
        )

        progress(f"Building .apkg deck with {len(cards)} card(s)…")
        apkg_path = _build_apkg(cards, deck_name, tags)

    finally:
        _safe_remove(audio_path)
        for c in audio_chunks:
            _safe_remove(c)

    return apkg_path


# ---------------------------------------------------------------------------
# Step 1 – Extract audio
# ---------------------------------------------------------------------------

def _extract_audio(video_path: str) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tmp.close()
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "libmp3lame",
        "-ar", "16000", "-ac", "1", "-b:a", "64k",
        tmp.name,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg error:\n{r.stderr[-2000:]}")
    return tmp.name


# ---------------------------------------------------------------------------
# Step 2 – Split audio
# ---------------------------------------------------------------------------

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
# Step 3 – Transcribe
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
# Step 4 – Generate cards with Claude
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
            f"Generate up to {cards_per_chunk} cards from the following lecture excerpt. "
            f"Focus on high-yield medical facts, mechanisms, definitions, and clinical pearls.\n\n"
            f"TRANSCRIPT:\n{chunk}"
        )
        resp = client.messages.create(
            model=model, max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": msg}],
        )
        all_cards.extend(_parse_cards(resp.content[0].text.strip()))

    return all_cards


# ---------------------------------------------------------------------------
# Step 5 – Build .apkg
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
            note = genanki.Note(
                model=CLOZE_MODEL,
                fields=[front, back],
                tags=tags,
            )
        else:
            note = genanki.Note(
                model=BASIC_MODEL,
                fields=[front, back],
                tags=tags,
            )
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
        {"type": d.get("type","basic"), "front": str(d.get("front","")), "back": str(d.get("back",""))}
        for d in data if isinstance(d, dict) and d.get("front")
    ]


def _safe_remove(path: str):
    try:
        if path:
            os.remove(path)
    except OSError:
        pass
