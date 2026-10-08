"""FastAPI backend for the LFM2.5 Spellchecker demo (local project).

    python server.py            # runs uvicorn on 0.0.0.0:5000
    uvicorn server:app --host 0.0.0.0 --port 5000
"""
import difflib
import io
import os
import re
import subprocess

import httpx
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoModel


def _detect_cpus():
    """CPU cores available to this process (affinity when exposed, else os.cpu_count())."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return os.cpu_count() or 1


_CPUS = _detect_cpus()
torch.set_num_threads(_CPUS)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

MODEL_ID = os.environ.get("SPELLCHECKER_MODEL", "LiquidAI/LFM2.5-Encoder-350M-Spellchecker")
MODEL_REV = os.environ.get("SPELLCHECKER_REVISION", "main")
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# OpenAI-compatible local server used for voice transcription (same setup as sentence-checker).
LOCAL_LLM = os.environ.get("LOCAL_LLM", "http://localhost:8000/v1")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "whisper-large-v3-turbo-int4-ov")

print(f"[server] loading {MODEL_ID}@{MODEL_REV} on {_CPUS} CPU thread(s) ...", flush=True)
_model = AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REV, trust_remote_code=True,
                                   token=os.environ.get("HF_TOKEN")).float().eval()
_mem_bytes = sum(t.numel() * t.element_size() for t in (*_model.parameters(), *_model.buffers()))
_mem_human = (f"{_mem_bytes / 1024**3:.2f} GB" if _mem_bytes >= 1024**3
              else f"{_mem_bytes / 1024**2:.0f} MB")
print(f"[server] model ready ({_mem_human} in memory)", flush=True)


# The model is trained on spaCy-tokenized text (punctuation separated by spaces AND contractions split:
# "let's"->"let 's", "don't"->"do n't"). We tokenize natural input into that form before the model and
# detokenize for display, so users type/read natural text while the model sees its training form. Feeding
# JOINED contractions is out-of-distribution and made the model edit them erratically (the "let 's" /
# "That's 's" artifacts) — splitting+rejoining contractions is the fix.
_PUNCT = re.compile(r'([.,!?;:()\[\]{}"«»…])')
_ATTACH_LEFT = re.compile(r'\s+([.,!?;:%)\]}»…])')
_ATTACH_RIGHT = re.compile(r'([(\[{«])\s+')
_NT = re.compile(r"(\w+?)(n['’]t)\b", re.I)                       # don't->do n't, can't->ca n't
_CONTR = re.compile(r"(\w)(['’](?:s|re|ve|ll|d|m))\b", re.I)      # let's->let 's, I'm->I 'm
_REJOIN_NT = re.compile(r"\s+(n['’]t)\b", re.I)                   # do n't->don't
_REJOIN_CONTR = re.compile(r"\s+(['’](?:s|re|ve|ll|d|m))\b", re.I)  # let 's->let's
_REJOIN_SPLIT_APOSTROPHE = re.compile(r"(?<=\w)\s+(['’])\s+(?=(?:s|t|d|m|re|ve|ll)\b)", re.I)


def tokenize(text: str) -> str:
    text = _PUNCT.sub(r" \1 ", text)
    text = _NT.sub(r"\1 \2", text)                               # split n't (own apostrophe) first
    text = _CONTR.sub(r"\1 \2", text)                            # then 's/'re/'ve/'ll/'d/'m
    return re.sub(r"\s+", " ", text).strip()


def detok(text: str) -> str:
    text = _ATTACH_LEFT.sub(r"\1", text)
    text = _ATTACH_RIGHT.sub(r"\1", text)
    text = _REJOIN_NT.sub(r"\1", text)                           # do n't->don't
    text = _REJOIN_CONTR.sub(r"\1", text)                        # let 's->let's
    text = _REJOIN_SPLIT_APOSTROPHE.sub(r"\1", text)              # don ' t->don't
    return re.sub(r"\s+", " ", text).strip()


_PROBE_IN = "That's a fair point, let's discuss it tomorrow."
try:
    _PROBE_OUT = detok(_model.correct([tokenize(_PROBE_IN)], max_iter=3)[0])
    _RERANK_ACTIVE = (_PROBE_OUT == _PROBE_IN)
except Exception as e:                                           # pragma: no cover
    _PROBE_OUT, _RERANK_ACTIVE = f"ERROR: {e}", None
print(f"[server] self-test ok={_RERANK_ACTIVE}: {_PROBE_IN!r} -> {_PROBE_OUT!r}", flush=True)


def diff_segments(source: str, corrected: str):
    """Word-level diff -> [{text, kind}] segments, kind in {keep, edit, del}, for inline rendering of
    the corrected text: `keep` unchanged, `edit` inserted/replaced (highlight green), `del` removed
    (shown struck-through red — these words are NOT part of the corrected text)."""
    s, c = source.split(), corrected.split()
    seg = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, s, c, autojunk=False).get_opcodes():
        if op == "equal":
            seg.append({"text": " ".join(c[j1:j2]), "kind": "keep"})
        elif op == "insert":
            seg.append({"text": " ".join(c[j1:j2]), "kind": "edit"})
        elif op == "delete":
            seg.append({"text": " ".join(s[i1:i2]), "kind": "del"})
        elif op == "replace":                          # new words highlighted; removed words shown struck
            seg.append({"text": " ".join(s[i1:i2]), "kind": "del"})
            seg.append({"text": " ".join(c[j1:j2]), "kind": "edit"})
    return seg or [{"text": corrected, "kind": "keep"}]


class CorrectRequest(BaseModel):
    text: str
    min_error_prob: float = 0.0
    max_iter: int = 3


app = FastAPI(title="LFM2.5 Spellchecker")


@app.get("/api/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "revision": MODEL_REV,
            "mem_bytes": _mem_bytes, "mem_human": _mem_human,
            "rerank_active": _RERANK_ACTIVE, "self_test": {"in": _PROBE_IN, "out": _PROBE_OUT}}


@app.post("/api/correct")
@torch.no_grad()
def correct(req: CorrectRequest):
    text = (req.text or "").strip()
    if not text:
        return {"corrected": "", "segments": [], "changed": False}
    src = tokenize(text)                                       # natural -> spaced (the model's form)
    out = _model.correct([src], min_error_prob=float(req.min_error_prob),
                         max_iter=int(req.max_iter))[0]
    # diff on spaced tokens (accurate, word-level); the frontend detokenizes spacing for display.
    return {"corrected": detok(out), "segments": diff_segments(src, out), "changed": out != src}


@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """Send browser-recorded webm to Whisper via the local OpenAI-compatible server and return text."""
    audio_bytes = await audio.read()

    # Convert webm to wav using ffmpeg
    try:
        proc = subprocess.run(
            [
                "ffmpeg", "-i", "pipe:0",
                "-ar", "16000",
                "-ac", "1",
                "-f", "wav",
                "pipe:1",
            ],
            input=audio_bytes,
            capture_output=True,
            timeout=30,
        )
        if proc.returncode != 0:
            return JSONResponse(
                status_code=400,
                content={"error": f"ffmpeg error: {proc.stderr.decode()}"},
            )
        wav_bytes = proc.stdout
    except FileNotFoundError:
        return JSONResponse(
            status_code=500,
            content={"error": "ffmpeg not installed. Install it with: sudo apt install ffmpeg"},
        )
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": f"Audio conversion failed: {str(e)}"},
        )

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{LOCAL_LLM}/audio/transcriptions",
            files={"file": ("audio.wav", wav_bytes, "audio/wav")},
            data={"model": WHISPER_MODEL, "language": "en"},
        )

    if resp.status_code != 200:
        return JSONResponse(
            status_code=502,
            content={"error": f"Whisper error: {resp.text}"},
        )

    data = resp.json()
    return {"text": data.get("text", "")}


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


app.mount("/", StaticFiles(directory=STATIC), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
