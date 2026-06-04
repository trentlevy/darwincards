"""
Lecture to Anki – Web App
FastAPI backend: upload MP4 → Whisper → Claude → .apkg download

Endpoints
---------
POST /api/generate          Upload MP4 + options, returns {job_id}
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

from pipeline import generate_cards_from_video

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Lecture to Anki")

STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# In-memory job store  {job_id: {"status": ..., "messages": [], "apkg_path": ...}}
jobs: Dict[str, Any] = {}

# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return HTMLResponse(index.read_text())
    return HTMLResponse("<h1>Lecture to Anki API is running</h1>")


# ---------------------------------------------------------------------------
# Generate endpoint
# ---------------------------------------------------------------------------

@app.post("/api/generate")
async def start_generation(
    video: UploadFile = File(...),
    openai_api_key: str = Form(...),
    anthropic_api_key: str = Form(...),
    deck_name: str = Form("Lecture Cards"),
    card_type: str = Form("both"),
    cards_per_chunk: int = Form(8),
    language: str = Form("en"),
    claude_model: str = Form("claude-opus-4-6"),
    tags: str = Form(""),
):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "running", "messages": [], "apkg_path": None, "error": None}

    # Save uploaded video to a temp file
    suffix = Path(video.filename).suffix or ".mp4"
    tmp_video = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    contents = await video.read()
    tmp_video.write(contents)
    tmp_video.close()

    tag_list = [t.strip() for t in tags.split() if t.strip()]

    # Run pipeline in a background thread so we don't block the event loop
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        None,
        _run_pipeline,
        job_id, tmp_video.name,
        openai_api_key, anthropic_api_key,
        deck_name, card_type, cards_per_chunk,
        language, claude_model, tag_list,
    )

    return {"job_id": job_id}


def _run_pipeline(
    job_id, video_path, openai_key, anthropic_key,
    deck_name, card_type, cards_per_chunk, language, claude_model, tags,
):
    job = jobs[job_id]

    def progress(msg: str):
        job["messages"].append(msg)

    try:
        apkg_path = generate_cards_from_video(
            video_path=video_path,
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
            os.remove(video_path)
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

    deck_name = "lecture_cards"
    filename = f"{deck_name.replace(' ', '_')}.apkg"
    return FileResponse(
        path=apkg_path,
        filename=filename,
        media_type="application/octet-stream",
    )


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
