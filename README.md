# English spellchecker

Correct English spelling and grammar locally with [LFM2.5-Encoder-350M-Spellchecker](https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M-Spellchecker)
(Liquid AI): a 350M bidirectional encoder running on CPU via `transformers`. FastAPI backend + static
frontend, no Docker — a plain local Python project.

Voice input (record in the browser → transcribe with Whisper → auto-correct) goes through an OpenAI-compatible
local server, same setup as sentence-checker.

## Setup

```bash
python -m venv .venv
source .venv/activate

# CPU-only torch wheel first (the default PyPI wheel is CUDA and huge)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Voice input requirements (optional)

- An OpenAI-compatible server on `http://localhost:8000/v1` serving the whisper model
  (`whisper-large-v3-turbo-int4-ov`) — the same one already used by sentence-checker.
- System `ffmpeg` (converts the browser's webm recording to 16 kHz mono wav).

Both are overridable without touching code:

| Env var | Default |
|---------|---------|
| `LOCAL_LLM` | `http://localhost:8000/v1` |
| `WHISPER_MODEL` | `whisper-large-v3-turbo-int4-ov` |
| `SPELLCHECKER_MODEL` | `LiquidAI/LFM2.5-Encoder-350M-Spellchecker` |

## Run

```bash
python server.py            # or: uvicorn server:app --host 0.0.0.0 --port 5000
```

Open <http://localhost:5000>. The console prints model loading + a self-test line; the page shows
"Model ready" once `/api/health` answers.

## How it works

- Type/paste text (or click an example) → correction fires after a 650 ms debounce; corrections render
  as inline diff segments (green highlight = edit, red strikethrough = removed word).
- Click **Start Recording** → speak → **Stop Recording**: audio is uploaded to `/api/transcribe`,
  converted via ffmpeg, transcribed by Whisper, and dropped into the input textarea — which then runs
  through the same correction pipeline automatically.
- Note on the engine: this model is a GECToR-style encoder tagger whose `correct()` loop lives in HF
  remote code; it cannot run through llama.cpp's HTTP API (only the Whisper side of voice input does).
