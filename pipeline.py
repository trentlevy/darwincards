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

import base64
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Callable, List, Dict, Tuple, Optional

import anthropic
import fitz  # pymupdf
import genanki
import pdfplumber
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

# ---------------------------------------------------------------------------
WORDS_PER_CHUNK = 1500

SLIDE_EXTS = {".pdf", ".pptx", ".ppt"}
TEXT_EXTS  = {".txt"}
ALLOWED_EXTS = SLIDE_EXTS | TEXT_EXTS

SYSTEM_PROMPT = """\
You are a medical education expert who creates high-yield Anki flashcards in the style of AnKing.

Rules:
- One concept per card (minimum information principle)
- Prefer CLOZE cards for facts, definitions, mechanisms, and values — they are more effective for memorization
- Use BASIC cards for "what/why/how" questions that need a full explanation
- Cloze syntax: {{c1::term}} for the key fact. Use {{c2::...}} for a second blank on the same card only when the two facts are closely linked
- Cloze "back" (Extra) field: add a brief explanation or clinical pearl to reinforce the answer
- Keep answers concise — one to two sentences max
- Focus on high-yield: mechanisms, drug classes, classic presentations, values, pathophysiology
- Decide the NUMBER of cards yourself, per section of content, based on how much is actually
  high-yield and testable — never pad to hit a target count, and never skip a genuinely
  high-yield fact to stay under one. A section that's mostly administrative framing, a story,
  or repetition of an already-covered point can honestly warrant zero cards; a section dense
  with distinct testable facts can warrant more than a "typical" section. Quality and
  non-redundancy over quantity — a student reviewing this deck should not hit filler.

You MUST respond with valid JSON only — no prose, no markdown fences.
The JSON should be an array of card objects, each with:
  "type"  : "basic" or "cloze"
  "front" : question (basic) or sentence with {{c1::...}} blanks (cloze)
  "back"  : answer (basic) or extra context / clinical pearl (cloze)
"""

BASIC_MODEL_ID = 1607392319
CLOZE_MODEL_ID = 1607392320

_CARD_CSS = """
.card {
  font-family: 'Helvetica Neue', Arial, sans-serif;
  font-size: 20px;
  line-height: 1.6;
  color: #1a1a1a;
  background: #ffffff;
  max-width: 680px;
  margin: 0 auto;
  padding: 24px 28px;
  text-align: center;
}
.front { font-size: 22px; font-weight: 500; }
hr#answer { border: none; border-top: 2px solid #e0e0e0; margin: 20px 0; }
.back { font-size: 20px; color: #1a1a1a; }
.extra {
  font-size: 15px; color: #555; margin-top: 14px;
  padding-top: 12px; border-top: 1px solid #eee;
  text-align: left;
}
/* Cloze */
.cloze { font-weight: bold; color: #0070f3; }
.cloze b { font-weight: bold; color: #0070f3; }

/* Night Mode — Anki desktop (2.1.20+), AnkiDroid, and AnkiMobile all add a
   "night_mode" class to the card element when the user has dark mode on.
   Without this, light-mode colors above would stay hardcoded and either be
   invisible (white-on-white) or unreadable (dark-on-dark) depending on which
   way it was hardcoded — so both palettes are defined and Anki picks
   whichever one applies via this class, per user, automatically. */
.card.night_mode {
  color: #f2f2f2;
  background: #2f2f31;
}
.night_mode .back { color: #f2f2f2; }
.night_mode .extra { color: #b0b0b0; border-top-color: #4a4a4c; }
.night_mode hr#answer { border-top-color: #4a4a4c; }
.night_mode .cloze, .night_mode .cloze b { color: #58a6ff; }
"""

BASIC_MODEL = genanki.Model(
    BASIC_MODEL_ID,
    "DarwinCards – Basic",
    fields=[{"name": "Front"}, {"name": "Back"}],
    templates=[{
        "name": "Card 1",
        "qfmt": '<div class="front">{{Front}}</div>',
        "afmt": '<div class="front">{{Front}}</div><hr id=answer><div class="back">{{Back}}</div>',
    }],
    css=_CARD_CSS,
)

CLOZE_MODEL = genanki.Model(
    CLOZE_MODEL_ID,
    "DarwinCards – Cloze",
    fields=[{"name": "Text"}, {"name": "Extra"}],
    templates=[{
        "name": "Cloze",
        "qfmt": '<div class="front">{{cloze:Text}}</div>',
        "afmt": '<div class="front">{{cloze:Text}}</div><div class="extra">{{Extra}}</div>',
    }],
    model_type=genanki.Model.CLOZE,
    css=_CARD_CSS,
)

IMAGE_CARD_MODEL_ID = 1607392321
IMAGE_CARD_MODEL = genanki.Model(
    IMAGE_CARD_MODEL_ID,
    "DarwinCards – Image",
    fields=[{"name": "Question"}, {"name": "Image"}, {"name": "Answer"}],
    templates=[{
        "name": "Image Card",
        "qfmt": '<div class="front">{{Question}}</div><div class="img-wrap">{{Image}}</div>',
        "afmt": '<div class="front">{{Question}}</div><div class="img-wrap">{{Image}}</div><hr id=answer><div class="back">{{Answer}}</div>',
    }],
    css=_CARD_CSS + """
.img-wrap img { max-width: 100%; height: auto; border-radius: 8px; margin: 10px 0; }
""",
)

IMAGE_SYSTEM_PROMPT = """\
You are a medical education expert analyzing a slide image to create Anki flashcards.

If the image contains a labeled diagram, anatomical illustration, chart, or table with medical content:
- Create 2-4 cards that test knowledge of specific elements visible in the image
- For anatomical diagrams: ask "What structure is labeled X?" or "What is the function of [visible structure]?"
- For charts/graphs: ask about specific values, trends, or relationships
- For tables: ask about specific cell values or comparisons
- Keep questions focused on ONE element per card
- Keep answers concise (one sentence)

If the image is decorative, a title slide, or contains no testable medical content, return [].

Respond with valid JSON only:
[{"question": "...", "answer": "..."}]
"""

# Max images to process per file (to control cost)
MAX_IMAGES_PER_FILE = 8
# Skip images smaller than this (likely icons/logos)
MIN_IMAGE_BYTES = 15_000


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


def _slugify(text: str) -> str:
    """Convert a label to a safe Anki tag segment (no spaces, no special chars)."""
    import re as _re
    text = text.strip()
    text = _re.sub(r'[^\w\s-]', '', text)   # strip special chars
    text = _re.sub(r'[\s]+', '-', text)       # spaces → hyphens
    return text or "Unknown"


def generate_cards_from_file(
    file_paths,
    anthropic_api_key: str,
    deck_name: str = "DarwinCards",
    card_type: str = "both",
    include_images: bool = True,
    auto_tags: bool = True,
    cards_per_chunk: int = 15,  # safety ceiling per ~1500-word section, not a target — see _generate_cards
    language: str = "en",
    claude_model: str = "claude-sonnet-4-6",
    course: str = "",
    lecture_topic: str = "",
    school: str = "Perelman School of Medicine",
    progress: Callable[[str], None] = lambda _: None,
) -> tuple:
    """
    Accepts one or more files: transcripts (.txt), PDF slides (.pdf), or PowerPoint (.pptx/.ppt).
    Multiple files are merged into a single transcript before card generation.
    Returns (cards, tags, usage_stats) — call build_apkg(cards, deck_name, tags)
    once the caller has run its own keep/discard review step, or immediately
    with the full list to skip review.
    """
    # Build hierarchical tag base: DeckName::Course::LectureTopic
    root = _slugify(deck_name)
    tag_parts = [root]
    if course.strip():
        tag_parts.append(_slugify(course))
    if lecture_topic.strip():
        tag_parts.append(_slugify(lecture_topic))
    tag_base = "::".join(tag_parts)
    tags = [tag_base]
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

    # Pull in matching excerpts from this school's prior-years' student notes
    # (if any have been ingested) to help Claude gauge what's historically
    # high-yield for this topic, alongside the actual lecture content.
    reference_notes: List[str] = []
    if school:
        try:
            from notes_corpus import retrieve_relevant_notes
            query = f"{course} {lecture_topic}\n{transcript[:1000]}"
            reference_notes = retrieve_relevant_notes(query, school=school, top_k=4)
            if reference_notes:
                progress(f"Found {len(reference_notes)} relevant excerpt(s) from prior student notes…")
        except Exception as e:
            progress(f"Reference notes lookup skipped: {e}")

    progress("Generating Anki cards with Claude…")
    cards, input_tokens, output_tokens = _generate_cards(
        transcript, anthropic_api_key, card_type,
        cards_per_chunk, claude_model, progress,
        reference_notes=reference_notes,
    )

    # Extract images from slide files and generate image-based cards
    image_cards = []
    img_input_tokens = img_output_tokens = 0
    if include_images:
        for file_path in file_paths:
            ext = Path(file_path).suffix.lower()
            if ext in {".pdf", ".pptx", ".ppt"}:
                try:
                    images = _extract_images(file_path, ext)
                    if images:
                        progress(f"Generating image cards from {len(images)} slide image(s)…")
                        ic, it, ot = _generate_image_cards(images, anthropic_api_key, claude_model, progress)
                        image_cards.extend(ic)
                        img_input_tokens  += it
                        img_output_tokens += ot
                except Exception as e:
                    progress(f"Image extraction skipped: {e}")

    # Auto-generate subject tags with Claude — nested under the base path
    if auto_tags:
        try:
            progress("Auto-tagging with AI…")
            ai_tags, at_in, at_out = _generate_tags(transcript, anthropic_api_key, claude_model)
            # Each AI tag becomes DeckName::Course::LectureTopic::ai-tag
            nested = [f"{tag_base}::{t}" for t in ai_tags]
            tags = list(dict.fromkeys(tags + nested))
            input_tokens  += at_in
            output_tokens += at_out
        except Exception as e:
            progress(f"Auto-tagging skipped: {e}")

    all_cards = cards + image_cards
    progress(f"Generated {len(all_cards)} card(s) ({len(image_cards)} image) — ready for review")

    total_in  = input_tokens  + img_input_tokens
    total_out = output_tokens + img_output_tokens
    usage_stats = {
        "model": claude_model,
        "input_tokens":  total_in,
        "output_tokens": total_out,
        "estimated_cost_usd": _estimate_cost(claude_model, total_in, total_out),
    }
    # NOTE: this used to also build the .apkg here and return its path. It now
    # stops after generating cards so the caller can run a keep/discard review
    # step first — see build_apkg() below, called once the user has picked
    # which cards to keep.
    return all_cards, tags, usage_stats


def build_apkg(cards: List[Dict], deck_name: str, tags: List[str]) -> str:
    """
    Build the final .apkg from a (possibly review-filtered) list of cards.
    Split out from generate_cards_from_file() so a review step can sit
    between "cards generated" and "deck packaged".
    """
    return _build_apkg(cards, deck_name, tags)


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
# Auto-tagging
# ---------------------------------------------------------------------------

def _generate_tags(transcript: str, api_key: str, model: str) -> Tuple[List[str], int, int]:
    """Ask Claude to suggest 3-8 Anki tags for the content."""
    client = anthropic.Anthropic(api_key=api_key)
    # Trim transcript to keep cost minimal
    snippet = transcript[:4000]
    resp = client.messages.create(
        model=model,
        max_tokens=128,
        messages=[{
            "role": "user",
            "content": (
                "You are a medical educator tagging Anki flashcard decks for US med students.\n"
                "Given the lecture excerpt below, reply with ONLY a JSON array of 3-8 lowercase "
                "single-word or hyphenated tags (e.g. [\"cardiology\", \"step1\", \"pharmacology\"]).\n"
                "No explanation — just the JSON array.\n\n"
                f"EXCERPT:\n{snippet}"
            ),
        }],
    )
    raw = resp.content[0].text.strip()
    # Parse the JSON array; fall back gracefully
    import re as _re
    match = _re.search(r'\[.*?\]', raw, _re.DOTALL)
    if match:
        import json as _json
        ai_tags = _json.loads(match.group())
        ai_tags = [t.lower().replace(" ", "-") for t in ai_tags if isinstance(t, str)][:8]
    else:
        ai_tags = []
    return ai_tags, resp.usage.input_tokens, resp.usage.output_tokens


# ---------------------------------------------------------------------------
# Image extraction
# ---------------------------------------------------------------------------

def _extract_images(file_path: str, ext: str) -> List[Tuple[bytes, str]]:
    """Return list of (image_bytes, media_type) for meaningful images in the file."""
    images = []
    if ext == ".pdf":
        doc = fitz.open(file_path)
        for page in doc:
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                img_bytes = base_image["image"]
                if len(img_bytes) < MIN_IMAGE_BYTES:
                    continue
                raw_ext = base_image.get("ext", "png")
                media_type = "image/jpeg" if raw_ext in ("jpg", "jpeg") else f"image/{raw_ext}"
                if media_type not in ("image/jpeg", "image/png", "image/webp", "image/gif"):
                    media_type = "image/png"
                images.append((img_bytes, media_type))
                if len(images) >= MAX_IMAGES_PER_FILE:
                    break
            if len(images) >= MAX_IMAGES_PER_FILE:
                break
        doc.close()
    elif ext in {".pptx", ".ppt"}:
        prs = Presentation(file_path)
        for slide in prs.slides:
            for shape in slide.shapes:
                if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                    img = shape.image
                    img_bytes = img.blob
                    if len(img_bytes) < MIN_IMAGE_BYTES:
                        continue
                    content_type = img.content_type or "image/png"
                    images.append((img_bytes, content_type))
                    if len(images) >= MAX_IMAGES_PER_FILE:
                        break
            if len(images) >= MAX_IMAGES_PER_FILE:
                break
    return images


def _generate_image_cards(
    images: List[Tuple[bytes, str]],
    api_key: str,
    model: str,
    progress: Callable,
) -> Tuple[List[Dict], int, int]:
    """Send each image to Claude Vision, return image-based card dicts."""
    client = anthropic.Anthropic(api_key=api_key)
    all_cards: List[Dict] = []
    total_in = total_out = 0

    for i, (img_bytes, media_type) in enumerate(images, 1):
        try:
            img_b64 = base64.b64encode(img_bytes).decode()
            resp = client.messages.create(
                model=model,
                max_tokens=1024,
                system=IMAGE_SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": media_type, "data": img_b64},
                        },
                        {"type": "text", "text": "Generate flashcards for this medical slide image."},
                    ],
                }],
            )
            total_in  += resp.usage.input_tokens
            total_out += resp.usage.output_tokens
            raw = resp.content[0].text.strip()
            parsed = _parse_cards(raw)
            # Tag these as image cards and attach the embedded image HTML
            img_tag = f'<img src="data:{media_type};base64,{img_b64}">'
            for card in parsed:
                card["type"]  = "image"
                card["image"] = img_tag
            all_cards.extend(parsed)
        except Exception as e:
            print(f"[image card error] image {i}: {e}")

    return all_cards, total_in, total_out


# ---------------------------------------------------------------------------
# Claude card generation
# ---------------------------------------------------------------------------

def _generate_cards(transcript, api_key, card_type, cards_per_chunk, model, progress, reference_notes: Optional[List[str]] = None):
    client = anthropic.Anthropic(api_key=api_key)
    type_instruction = {
        "both":  "Generate a mix of 'basic' (Q&A) and 'cloze' (fill-in-the-blank) cards.",
        "basic": "Generate only 'basic' (Q&A) cards.",
        "cloze": "Generate only 'cloze' (fill-in-the-blank) cards.",
    }[card_type]

    reference_block = ""
    if reference_notes:
        joined = "\n\n---\n\n".join(reference_notes)
        reference_block = (
            "\n\nREFERENCE — excerpts from prior years' student notes on related topics at this school, "
            "found via keyword search and NOT guaranteed to be on-topic. These reflect what past students "
            "found worth writing down (and, by extension, what tends to get tested). Use them as a relevancy "
            "signal for HOW MANY cards this content deserves, not just their wording: if these notes show past "
            "students flagged a point as worth writing down or testing, that's a signal to make a card for it "
            "even if the lecture only mentions it briefly; if a lecture passage reads as filler/administrative "
            "and nothing here corroborates it as commonly tested, that's a signal to skip it rather than force a "
            "card. The CONTENT above is authoritative for facts — do not copy the reference wording verbatim — "
            "and if an excerpt turns out to be about a different topic than the lecture (keyword search can "
            "surface tangential matches), ignore it entirely rather than forcing a connection:\n"
            f"{joined}"
        )

    chunks = _chunk_text(transcript, WORDS_PER_CHUNK)
    all_cards = []
    total_input_tokens = 0
    total_output_tokens = 0

    for i, chunk in enumerate(chunks, 1):
        progress(f"Generating cards for section {i}/{len(chunks)}…")
        msg = (
            f"{type_instruction} "
            f"Decide for yourself how many cards this content actually warrants — judge by how much is "
            f"genuinely high-yield and testable, not by any target number. Do not pad with filler cards to "
            f"reach a count, and do not omit a genuinely high-yield fact to stay under one. As a hard safety "
            f"ceiling only (not a target), do not exceed {cards_per_chunk} cards for this section. "
            f"Focus on high-yield medical facts, mechanisms, definitions, and clinical pearls.\n\n"
            f"CONTENT:\n{chunk}"
            f"{reference_block}"
        )
        resp = client.messages.create(
            model=model, max_tokens=6144,
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
        image = card.get("image", "")
        if not front and not image:
            continue
        guid = _stable_id(f"{ctype}:{front}:{back}:{image[:64]}")
        if ctype == "cloze":
            note = genanki.Note(model=CLOZE_MODEL, fields=[front, back], tags=tags, guid=guid)
        elif ctype == "image":
            question = front or card.get("question", "")
            answer   = back  or card.get("answer", "")
            note = genanki.Note(model=IMAGE_CARD_MODEL, fields=[question, image, answer], tags=tags, guid=guid)
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
