"""
DarwinCards – Web App (open access, cost-capped)

Anyone can generate cards using the owner's API keys.
Two guards prevent runaway spend:
  1. Global monthly cap  – stops all generation once estimated spend hits $95.
  2. Per-IP daily limit  – max 5 generations per IP per 24 hours.

Endpoints
---------
GET  /                          Frontend
POST /api/generate              Upload file, returns {job_id}
GET  /api/progress/{job_id}     SSE progress stream
GET  /api/download/{job_id}     Download .apkg
GET  /api/status                Current monthly cost / capacity info
"""

import asyncio
import datetime
import os
import tempfile
import uuid
from pathlib import Path
from typing import Dict, Any, List

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Depends, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse
from sqlalchemy.orm import Session

from database import Base, engine, get_db, SessionLocal
from models import GlobalUsage, GenerationLog
from pipeline import generate_cards_from_file

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

Base.metadata.create_all(bind=engine)

app = FastAPI(title="DarwinCards")

STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

jobs: Dict[str, Any] = {}

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# Cost guard — stop accepting jobs once estimated monthly spend reaches this
MONTHLY_COST_LIMIT_USD = 95.0

# Per-IP rate limit — max generations per rolling 24-hour window
IP_DAILY_LIMIT = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_client_ip(request: Request) -> str:
    """Resolve real client IP, accounting for Railway's proxy."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def current_month() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m")


def get_or_create_monthly(db: Session) -> GlobalUsage:
    m = current_month()
    row = db.query(GlobalUsage).filter(GlobalUsage.month == m).first()
    if not row:
        row = GlobalUsage(month=m, estimated_cost_usd=0.0, generation_count=0)
        db.add(row)
        db.commit()
        db.refresh(row)
    return row


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return HTMLResponse(index.read_text())
    return HTMLResponse("<h1>DarwinCards</h1>")


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

@app.get("/api/status")
def get_status(db: Session = Depends(get_db)):
    monthly = get_or_create_monthly(db)
    return {
        "at_capacity": monthly.estimated_cost_usd >= MONTHLY_COST_LIMIT_USD,
        "estimated_cost_usd": round(monthly.estimated_cost_usd, 4),
        "generation_count": monthly.generation_count,
        "month": monthly.month,
        "limit_usd": MONTHLY_COST_LIMIT_USD,
    }


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------

@app.post("/api/generate")
async def start_generation(
    request: Request,
    files: List[UploadFile] = File(...),
    deck_name: str = Form("DarwinCards"),
    card_type: str = Form("both"),
    cards_per_chunk: int = Form(8),
    language: str = Form("en"),
    claude_model: str = Form("claude-sonnet-4-6"),
    tags: str = Form(""),
    db: Session = Depends(get_db),
):
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="Server not configured. Contact the site owner.")

    ip = get_client_ip(request)

    # Global monthly cost check
    monthly = get_or_create_monthly(db)
    if monthly.estimated_cost_usd >= MONTHLY_COST_LIMIT_USD:
        raise HTTPException(
            status_code=503,
            detail="DarwinCards is at capacity for this month. Check back next month."
        )

    # Per-IP daily limit
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(hours=24)
    ip_count = db.query(GenerationLog).filter(
        GenerationLog.ip_address == ip,
        GenerationLog.created_at >= cutoff,
    ).count()
    if ip_count >= IP_DAILY_LIMIT:
        raise HTTPException(
            status_code=429,
            detail=f"You've reached the daily limit ({IP_DAILY_LIMIT} generations per day). Check back tomorrow."
        )

    # Save all uploads to temp files
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "running", "messages": [], "apkg_path": None, "error": None}

    tmp_paths = []
    for file in files:
        ext = Path(file.filename or "upload.tmp").suffix.lower()
        tmp = tempfile.NamedTemporaryFile(suffix=ext or ".tmp", delete=False)
        tmp.write(await file.read())
        tmp.close()
        tmp_paths.append(tmp.name)

    tag_list = [t.strip() for t in tags.split() if t.strip()]

    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        None, _run_pipeline,
        job_id, ip, tmp_paths,
        ANTHROPIC_API_KEY,
        deck_name, card_type, cards_per_chunk,
        language, claude_model, tag_list,
    )

    return {"job_id": job_id}


def _run_pipeline(job_id, ip_address, file_paths,
                  anthropic_key,
                  deck_name, card_type, cards_per_chunk,
                  language, claude_model, tags):
    job = jobs[job_id]

    def progress(msg: str):
        job["messages"].append(msg)

    try:
        apkg_path, usage_stats = generate_cards_from_file(
            file_paths=file_paths,
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

        # Log cost (non-fatal)
        try:
            db = SessionLocal()
            try:
                db.add(GenerationLog(
                    ip_address=ip_address,
                    estimated_cost_usd=usage_stats["estimated_cost_usd"],
                    input_tokens=usage_stats["input_tokens"],
                    output_tokens=usage_stats["output_tokens"],
                    model=usage_stats["model"],
                ))
                m = current_month()
                monthly = db.query(GlobalUsage).filter(GlobalUsage.month == m).first()
                if not monthly:
                    monthly = GlobalUsage(month=m, estimated_cost_usd=0.0, generation_count=0)
                    db.add(monthly)
                monthly.estimated_cost_usd = (monthly.estimated_cost_usd or 0) + usage_stats["estimated_cost_usd"]
                monthly.generation_count   = (monthly.generation_count or 0) + 1
                monthly.updated_at         = datetime.datetime.utcnow()
                db.commit()
            finally:
                db.close()
        except Exception as cost_err:
            print(f"[cost logging error] {cost_err}")

    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        job["messages"].append(f"__ERROR__{e}")
    finally:
        for p in file_paths:
            try:
                os.remove(p)
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
        last_ping = asyncio.get_event_loop().time()
        while True:
            job = jobs[job_id]
            messages = job["messages"]
            while sent < len(messages):
                msg = messages[sent]
                sent += 1
                yield {"data": msg}
                last_ping = asyncio.get_event_loop().time()
                if msg.startswith("__DONE__") or msg.startswith("__ERROR__"):
                    return
            if job["status"] in ("done", "error") and sent >= len(messages):
                return
            # Send a keepalive ping every 10s — Railway's nginx proxy drops
            # idle SSE connections; X-Accel-Buffering: no disables buffering.
            now = asyncio.get_event_loop().time()
            if now - last_ping > 10:
                yield {"event": "ping", "data": ""}
                last_ping = now
            await asyncio.sleep(0.5)

    return EventSourceResponse(
        event_generator(),
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

@app.get("/api/download/{job_id}")
async def download_apkg(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    if job["status"] != "done" or not job["apkg_path"]:
        raise HTTPException(status_code=400, detail="Job not complete")
    return FileResponse(
        path=job["apkg_path"],
        filename="darwincards_deck.apkg",
        media_type="application/octet-stream",
    )


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
