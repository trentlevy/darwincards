"""
DarwinCards – Web App
FastAPI backend: upload file → Claude → .apkg download

Supported inputs: .mp4/.mov/.mkv (video), .mp3/.m4a/.wav (audio),
                  .txt (transcript), .pdf (slides), .pptx (slides)

Endpoints
---------
POST /api/generate          Upload file + options, returns {job_id}
GET  /api/progress/{job_id} SSE stream of progress messages + final status
GET  /api/download/{job_id} Download the generated .apkg file
GET  /                      Serve the frontend
"""

import asyncio
import os
import tempfile
import uuid
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from pipeline import generate_cards_from_file, VIDEO_EXTS, AUDIO_EXTS

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="DarwinCards")

STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# In-memory job store
jobs: Dict[str, Any] = {}

# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return HTMLResponse(index.read_text())
    return HTMLResponse("<h1>DarwinCards API is running</h1>")


# ---------------------------------------------------------------------------
# Generate endpoint
# ---------------------------------------------------------------------------

@app.post("/api/generate")
async def start_generation(
    file: UploadFile = File(...),
    openai_api_key: str = Form(""),
    anthropic_api_key: str = Form(...),
    deck_name: str = Form("DarwinCards Deck"),
    card_type: str = Form("both"),
    cards_per_chunk: int = Form(8),
    language: str = Form("en"),
    claude_model: str = Form("claude-opus-4-6"),
    tags: str = Form(""),
):
    ext = Path(file.filename).suffix.lower()

    # OpenAI key only required for video/audio
    needs_openai = ext in VIDEO_EXTS or ext in AUDIO_EXTS
    if needs_openai and not openai_api_key.strip():
        raise HTTPException(
            status_code=400,
            detail="OpenAI API key is required for video and audio files."
        )
    if not anthropic_api_key.strip():
        raise HTTPException(status_code=400, detail="Anthropic API key is required.")

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "running", "messages": [], "apkg_path": None, "error": None}

    # Save uploaded file to temp
    suffix = ext or ".tmp"
    tmp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    contents = await file.read()
    tmp_file.write(contents)
    tmp_file.close()

    tag_list = [t.strip() for t in tags.split() if t.strip()]

    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        None,
        _run_pipeline,
        job_id, tmp_file.name,
        openai_api_key.strip(), anthropic_api_key.strip(),
        deck_name, card_type, cards_per_chunk,
        language, claude_model, tag_list,
    )

    return {"job_id": job_id}


def _run_pipeline(
    job_id, file_path, openai_key, anthropic_key,
    deck_name, card_type, cards_per_chunk, language, claude_model, tags,
):
    job = jobs[job_id]

    def progress(msg: str):
        job["messages"].append(msg)

    try:
        apkg_path = generate_cards_from_file(
            file_path=file_path,
            openai_api_key=openai_key,
            anthropic_api_key=anthropic_key,
            deck_name=deck_name,
            card_type=card_type,
            cards_per_chunk=cards_per_chunk,
            language=language,
            claude_model=claude_model,
            tags=tags,
            progress=progress,
        )
        job["apkg_path"] = apkg_path
        job["status"] = "done"
        job["messages"].append("__DONE__")
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        job["messages"].append(f"__ERROR__{e}")
    finally:
        try:
            os.remove(file_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# SSE progress stream
# ---------------------------------------------------------------------------

@app.get("/api/progress/{job_id}")
async def progress_stream(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator():
        sent = 0
        while True:
            job = jobs[job_id]
            messages = job["messages"]
            while sent < len(messages):
                msg = messages[sent]
                sent += 1
                yield {"data": msg}
                if msg.startswith("__DONE__") or msg.startswith("__ERROR__"):
                    return
            if job["status"] in ("done", "error") and sent >= len(messages):
                return
            await asyncio.sleep(0.5)

    return EventSourceResponse(event_generator())


# ---------------------------------------------------------------------------
# Download endpoint
# ---------------------------------------------------------------------------

@app.get("/api/download/{job_id}")
async def download_apkg(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    if job["status"] != "done" or not job["apkg_path"]:
        raise HTTPException(status_code=400, detail="Job not complete or failed")
    apkg_path = job["apkg_path"]
    if not os.path.exists(apkg_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path=apkg_path,
        filename="darwincards_deck.apkg",
        media_type="application/octet-stream",
    )


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
