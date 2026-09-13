"""
DarwinCards – Web App
FastAPI backend with auth, usage limits, and Stripe billing.

Plans
-----
  free : 3 lectures/month
  pro  : unlimited, $9/month via Stripe

Endpoints
---------
POST /api/auth/register         Create account
POST /api/auth/login            Login, returns JWT
GET  /api/auth/me               Current user info + usage

POST /api/billing/checkout      Create Stripe checkout session (upgrade to Pro)
POST /api/billing/portal        Stripe customer portal (manage subscription)
POST /api/billing/webhook       Stripe webhook handler

POST /api/generate              Upload file, returns {job_id}
GET  /api/progress/{job_id}     SSE progress stream
GET  /api/download/{job_id}     Download .apkg

GET  /                          Frontend
"""

import asyncio
import datetime
import os
import tempfile
import uuid
from pathlib import Path
from typing import Dict, Any, Optional

import stripe
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Depends, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from sse_starlette.sse import EventSourceResponse
from sqlalchemy.orm import Session

from database import Base, engine, get_db
from models import User, UsageLog
from auth import (
    hash_password, verify_password, create_access_token,
    get_current_user, get_current_user_optional,
)
from stripe_utils import create_checkout_session, create_portal_session, handle_webhook
from pipeline import generate_cards_from_file, VIDEO_EXTS, AUDIO_EXTS

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

Base.metadata.create_all(bind=engine)

app = FastAPI(title="DarwinCards")

STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

jobs: Dict[str, Any] = {}

FREE_LIMIT = 3   # lectures per month on free plan

OPENAI_API_KEY     = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY  = os.environ.get("ANTHROPIC_API_KEY", "")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def usage_this_month(user: User, db: Session) -> int:
    start = datetime.datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return db.query(UsageLog).filter(
        UsageLog.user_id == user.id,
        UsageLog.created_at >= start,
    ).count()


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
# Auth routes
# ---------------------------------------------------------------------------

class RegisterRequest(BaseModel):
    email: str
    password: str

@app.post("/api/auth/register")
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == req.email.lower()).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    if len(req.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    user = User(email=req.email.lower(), hashed_password=hash_password(req.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    token = create_access_token(user.id)
    return {"access_token": token, "token_type": "bearer", "plan": user.plan}


@app.post("/api/auth/login")
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form.username.lower()).first()
    if not user or not verify_password(form.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    token = create_access_token(user.id)
    return {"access_token": token, "token_type": "bearer", "plan": user.plan}


@app.get("/api/auth/me")
def me(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    used = usage_this_month(user, db)
    return {
        "email": user.email,
        "plan": user.plan,
        "usage_this_month": used,
        "limit": FREE_LIMIT if user.plan == "free" else None,
        "can_generate": user.plan == "pro" or used < FREE_LIMIT,
    }


# ---------------------------------------------------------------------------
# Billing routes
# ---------------------------------------------------------------------------

@app.post("/api/billing/checkout")
def billing_checkout(user: User = Depends(get_current_user)):
    if user.plan == "pro":
        raise HTTPException(status_code=400, detail="Already on Pro plan")
    url = create_checkout_session(user.email, user.id)
    return {"url": url}


@app.post("/api/billing/portal")
def billing_portal(user: User = Depends(get_current_user)):
    if not user.stripe_customer_id:
        raise HTTPException(status_code=400, detail="No billing account found")
    url = create_portal_session(user.stripe_customer_id)
    return {"url": url}


@app.post("/api/billing/webhook")
async def billing_webhook(request: Request, db: Session = Depends(get_db)):
    payload    = await request.body()
    sig_header = request.headers.get("stripe-signature", "")
    event      = handle_webhook(payload, sig_header)

    etype = event["type"]
    data  = event["data"]["object"]

    if etype == "checkout.session.completed":
        user_id     = int(data["metadata"]["user_id"])
        customer_id = data["customer"]
        sub_id      = data["subscription"]
        user = db.query(User).filter(User.id == user_id).first()
        if user:
            user.plan = "pro"
            user.stripe_customer_id     = customer_id
            user.stripe_subscription_id = sub_id
            db.commit()

    elif etype in ("customer.subscription.deleted", "customer.subscription.paused"):
        sub_id = data["id"]
        user = db.query(User).filter(User.stripe_subscription_id == sub_id).first()
        if user:
            user.plan = "free"
            user.stripe_subscription_id = None
            db.commit()

    elif etype == "customer.subscription.updated":
        sub_id = data["id"]
        status = data["status"]
        user = db.query(User).filter(User.stripe_subscription_id == sub_id).first()
        if user:
            user.plan = "pro" if status == "active" else "free"
            db.commit()

    return {"received": True}


# ---------------------------------------------------------------------------
# Generate endpoint
# ---------------------------------------------------------------------------

@app.post("/api/generate")
async def start_generation(
    file: UploadFile = File(...),
    deck_name: str = Form("DarwinCards Deck"),
    card_type: str = Form("both"),
    cards_per_chunk: int = Form(8),
    language: str = Form("en"),
    claude_model: str = Form("claude-opus-4-6"),
    tags: str = Form(""),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # Check API keys are configured server-side
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="Server not configured. Contact support.")

    # Check usage limit
    if current_user.plan == "free":
        used = usage_this_month(current_user, db)
        if used >= FREE_LIMIT:
            raise HTTPException(
                status_code=402,
                detail=f"Free limit reached ({FREE_LIMIT} lectures/month). Please upgrade to Pro."
            )

    # Log usage immediately
    log = UsageLog(user_id=current_user.id)
    db.add(log)
    db.commit()

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "running", "messages": [], "apkg_path": None, "error": None}

    ext = Path(file.filename).suffix.lower()
    suffix = ext or ".tmp"
    tmp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    contents = await file.read()
    tmp_file.write(contents)
    tmp_file.close()

    tag_list = [t.strip() for t in tags.split() if t.strip()]

    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        None, _run_pipeline,
        job_id, tmp_file.name,
        OPENAI_API_KEY, ANTHROPIC_API_KEY,
        deck_name, card_type, cards_per_chunk,
        language, claude_model, tag_list,
    )

    return {"job_id": job_id}


def _run_pipeline(job_id, file_path, openai_key, anthropic_key,
                  deck_name, card_type, cards_per_chunk, language, claude_model, tags):
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
# SSE + Download
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


@app.get("/api/download/{job_id}")
async def download_apkg(job_id: str, current_user: User = Depends(get_current_user)):
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
