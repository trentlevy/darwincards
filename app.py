"""
DarwinCards – Web App (open access, cost-capped)

Anyone can generate cards using the owner's API keys.
Two guards prevent runaway spend:
  1. Global monthly cap  – stops all generation once estimated spend hits $95.
  2. Per-IP daily limit  – max 5 generations per IP per 24 hours.

Job state is stored in PostgreSQL so it survives container restarts.

Endpoints
---------
GET  /                          Frontend
POST /api/generate              Upload file, returns {job_id}
GET  /api/progress/{job_id}     Poll for progress (JSON)
GET  /api/download/{job_id}     Download .apkg
GET  /api/status                Current monthly cost / capacity info
"""

import asyncio
import datetime
import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Depends, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from database import Base, engine, get_db, SessionLocal
from models import GlobalUsage, GenerationLog, Job
from pipeline import generate_cards_from_file

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

Base.metadata.create_all(bind=engine)

app = FastAPI(title="DarwinCards")

STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MONTHLY_COST_LIMIT_USD = 95.0
IP_DAILY_LIMIT = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_client_ip(request: Request) -> str:
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


def _job_append_message(job_id: str, msg: str):
    """Append a progress message to the job row — called from the pipeline thread."""
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if job:
            msgs = json.loads(job.messages_json or "[]")
            msgs.append(msg)
            job.messages_json = json.dumps(msgs)
            db.commit()
    finally:
        db.close()


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
    image_occlusion: str = Form("true"),
    auto_tags: str = Form("true"),
    cards_per_chunk: int = Form(8),
    language: str = Form("en"),
    claude_model: str = Form("claude-sonnet-4-6"),
    course: str = Form(""),
    lecture_topic: str = Form(""),
    db: Session = Depends(get_db),
):
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="Server not configured. Contact the site owner.")

    ip = get_client_ip(request)

    monthly = get_or_create_monthly(db)
    if monthly.estimated_cost_usd >= MONTHLY_COST_LIMIT_USD:
        raise HTTPException(status_code=503,
            detail="DarwinCards is at capacity for this month. Check back next month.")

    cutoff = datetime.datetime.utcnow() - datetime.timedelta(hours=24)
    ip_count = db.query(GenerationLog).filter(
        GenerationLog.ip_address == ip,
        GenerationLog.created_at >= cutoff,
    ).count()
    if ip_count >= IP_DAILY_LIMIT:
        raise HTTPException(status_code=429,
            detail=f"You've reached the daily limit ({IP_DAILY_LIMIT} generations per day). Check back tomorrow.")

    # Create job row in DB
    job_id = str(uuid.uuid4())
    db.add(Job(id=job_id, status="running", messages_json="[]"))
    db.commit()

    # Save uploads to temp files
    tmp_paths = []
    for file in files:
        ext = Path(file.filename or "upload.tmp").suffix.lower()
        tmp = tempfile.NamedTemporaryFile(suffix=ext or ".tmp", delete=False)
        tmp.write(await file.read())
        tmp.close()
        tmp_paths.append(tmp.name)


    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        None, _run_pipeline,
        job_id, ip, tmp_paths,
        ANTHROPIC_API_KEY,
        deck_name, card_type,
        image_occlusion.lower() == "true",
        auto_tags.lower() == "true",
        cards_per_chunk, language, claude_model, course, lecture_topic,
    )

    return {"job_id": job_id}


def _run_pipeline(job_id, ip_address, file_paths,
                  anthropic_key,
                  deck_name, card_type, include_images, auto_tags,
                  cards_per_chunk, language, claude_model, course, lecture_topic):

    def progress(msg: str):
        _job_append_message(job_id, msg)

    try:
        apkg_path, usage_stats = generate_cards_from_file(
            file_paths=file_paths,
            anthropic_api_key=anthropic_key,
            deck_name=deck_name,
            card_type=card_type,
            include_images=include_images,
            auto_tags=auto_tags,
            cards_per_chunk=cards_per_chunk,
            language=language,
            claude_model=claude_model,
            course=course,
            lecture_topic=lecture_topic,
            progress=progress,
        )

        # Read apkg bytes and store in DB so download survives restarts
        with open(apkg_path, "rb") as f:
            apkg_data = f.read()
        try:
            os.remove(apkg_path)
        except OSError:
            pass

        db = SessionLocal()
        try:
            job = db.query(Job).filter(Job.id == job_id).first()
            if job:
                msgs = json.loads(job.messages_json or "[]")
                msgs.append("__DONE__")
                job.messages_json = json.dumps(msgs)
                job.status    = "done"
                job.apkg_data = apkg_data
            # Log cost
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

    except Exception as e:
        print(f"[pipeline error] {e}")
        db = SessionLocal()
        try:
            job = db.query(Job).filter(Job.id == job_id).first()
            if job:
                msgs = json.loads(job.messages_json or "[]")
                msgs.append(f"__ERROR__{e}")
                job.messages_json = json.dumps(msgs)
                job.status = "error"
                job.error  = str(e)
                db.commit()
        finally:
            db.close()
    finally:
        for p in file_paths:
            try:
                os.remove(p)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Progress polling
# ---------------------------------------------------------------------------

@app.get("/api/progress/{job_id}")
def get_progress(job_id: str, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "status":   job.status,
        "messages": json.loads(job.messages_json or "[]"),
        "error":    job.error,
    }


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

@app.get("/api/download/{job_id}")
def download_apkg(job_id: str, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "done" or not job.apkg_data:
        raise HTTPException(status_code=400, detail="Job not complete")
    return Response(
        content=job.apkg_data,
        media_type="application/octet-stream",
        headers={"Content-Disposition": "attachment; filename=darwincards_deck.apkg"},
    )


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
