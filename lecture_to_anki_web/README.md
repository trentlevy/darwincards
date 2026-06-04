# DarwinCards – Web App

Upload a medical lecture video → get a ready-to-import Anki deck.

**Stack:** FastAPI · OpenAI Whisper · Anthropic Claude · genanki

---

## Run locally

### 1. Install prerequisites
- Python 3.10+
- ffmpeg: `brew install ffmpeg` (Mac) or `sudo apt install ffmpeg` (Linux)

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 3. Start the server
```bash
python app.py
```

Open http://localhost:8000 in your browser.

---

## Deploy to Railway (recommended, ~2 min)

1. Push this folder to a GitHub repo
2. Go to https://railway.app → New Project → Deploy from GitHub
3. Select your repo — Railway auto-detects the Dockerfile
4. Your app gets a public URL instantly

No environment variables needed — users supply their own API keys in the UI.

---

## Deploy to Render

1. Push this folder to a GitHub repo
2. Go to https://render.com → New → Web Service
3. Connect your repo, choose "Docker" as runtime
4. Deploy — Render uses `render.yaml` automatically

---

## File structure

```
lecture_to_anki_web/
├── app.py           # FastAPI app (upload, SSE progress, download)
├── pipeline.py      # MP4 → audio → Whisper → Claude → .apkg
├── static/
│   └── index.html   # Single-page frontend
├── requirements.txt
├── Dockerfile       # Used by Railway and Render
├── railway.toml     # Railway config
├── render.yaml      # Render config
└── README.md
```

---

## How it works

1. User uploads an MP4 and their API keys via the browser
2. Server extracts audio with ffmpeg (mono 16kHz mp3, much smaller than the video)
3. Audio is transcribed with OpenAI Whisper (split into <25MB chunks for long videos)
4. Transcript is chunked and sent to Claude with a medical education prompt
5. Claude returns JSON arrays of Basic and Cloze card objects
6. Cards are packaged into a `.apkg` file using genanki
7. User downloads the `.apkg` and double-clicks to import into Anki

---

## Cost estimate

A 1-hour lecture typically costs:
- Whisper: ~$0.06 (at $0.006/min)
- Claude Opus: ~$0.15–0.25 depending on transcript length
- **Total: ~$0.20–0.30 per lecture**

---

## Notes

- API keys are sent directly to OpenAI and Anthropic — they are never logged or stored
- Uploaded videos are deleted from the server immediately after processing
- Generated .apkg files are held in memory until downloaded, then cleaned up
- For production use, consider adding authentication and a proper job queue (Celery + Redis)
