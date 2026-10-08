# Implementation Plan — spellchecker: local project + voice input

Context: bring sentence-checker (`/home/sergio/Code/sentence-checker`) patterns into spellchecker
(`/home/sergio/Code/spellchecker`). Decisions already made with the user:

1. **Correction engine stays the HF transformers encoder** (`LiquidAI/LFM2.5-Encoder-350M-Spellchecker`,
   torch + `AutoModel` + `trust_remote_code`). Reason: llama.cpp CANNOT run this model. It is a
   bidirectional encoder with a GECToR-style tagging head; correction = iterative masked-token loop
   implemented in the model's Python remote code (`model.correct()`). llama.cpp supports LFM2 decoders/audio/VL/embeddings
   but not this architecture, and there is no llama-server endpoint for it. The litert-community repo
   the user mentioned is a LiteRT/.tflite conversion (on-device), not GGUF — also unusable here.
2. **Port 5000**, host `0.0.0.0`, sentence-checker-style `if __name__ == "__main__": uvicorn.run(...)`.
3. Voice recording + Whisper transcription go through the OpenAI-compatible local server at
   `http://localhost:8000/v1` exactly like sentence-checker (`/audio/transcriptions`, ffmpeg webm→wav).
4. **Text-to-speech uses the browser's Web Speech API** (`speechSynthesis`), no backend endpoint, no
   new dependency, OS-provided voices. Considered and rejected: a local TTS model (parler-tts/kokoro)
   on the llama-server at :8000 via `/audio/speech` — higher quality but a second model to install and
   serve; revisit only if the OS voices disappoint.

---

## Phase 1 — Remove Docker / make it a plain local project

Files: `Dockerfile`, `.dockerignore`, `README.md`, `server.py`, `requirements.txt`, `static/index.html`

- Delete `Dockerfile` and `.dockerignore`. Leave `.gitattributes` alone.
- `README.md`: replace the HF Spaces YAML frontmatter with a real README: what the app is, setup
  (venv + `pip install -r requirements.txt`; torch CPU wheel via
  `pip install torch --index-url https://download.pytorch.org/whl/cpu`), how to run
  (`python server.py` → open http://localhost:5000), and that voice input requires an
  OpenAI-compatible llama-server on :8000 serving the whisper model.
- `server.py`:
  - Delete `_effective_cpus()` cgroup-quota logic entirely (that exists only for the 2-vCPU HF Space).
    Replace with simply `os.sched_getaffinity(0)` length or `os.cpu_count()` fallback before
    `torch.set_num_threads(...)`. Keep `set_num_interop_threads(1)`.
  - Keep `MODEL_ID`/`MODEL_REV` env override but drop reliance on `HF_TOKEN` (model is public;
    `token=os.environ.get("HF_TOKEN")` may stay as-is harmlessly — passing None is fine).
  - Add at bottom the sentence-checker entrypoint:
    ```python
    if __name__ == "__main__":
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=5000)
    ```
  - Keep everything else: tokenize/detok, diff_segments, /api/correct, /api/health, static mount.
- `requirements.txt`: keep `transformers`, `safetensors`, `fastapi`, `uvicorn[standard]`,
  `huggingface_hub`; ADD `httpx` and `python-multipart` (needed by Phase 2 upload endpoint).
  Note torch CPU wheel install in README (not in requirements, same as old Dockerfile did it separately).

## Phase 2 — Voice recording (port from sentence-checker)

### Backend (`server.py`)

Copy sentence-checker's `/api/transcribe` verbatim-in-spirit:

- New imports: `io`, `subprocess`, `httpx`, plus `from fastapi import ... File, UploadFile` and
  `JSONResponse` from `fastapi.responses`.
- Constants near top:
  ```python
  LOCAL_LLM = os.environ.get("LOCAL_LLM", "http://localhost:8000/v1")
  WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "whisper-large-v3-turbo-int4-ov")
  ```
  (env-overridable so the user can retune without editing code)
- `@app.post("/api/transcribe")` `async def transcribe(audio: UploadFile = File(...))`:
  1. read upload bytes; run ffmpeg subprocess `["ffmpeg","-i","pipe:0","-ar","16000","-ac","1","-f","wav","pipe:1"]`,
     timeout 30s; on failure return JSONResponse 400/500 with `{error}` (same messages as sentence-checker).
  2. `httpx.AsyncClient(timeout=60)` POST `{LOCAL_LLM}/audio/transcriptions` with
     multipart file `("audio.wav", wav_bytes, "audio/wav")`, `data={"model": WHISPER_MODEL, "language": "en"}`.
     NOTE: sentence-checker passes `io.BytesIO(wav_bytes)` — pass raw `wav_bytes` instead (BytesIO works too, but plain bytes are simpler and correct for httpx files tuple).
  3. non-200 → JSONResponse 502 `{error: "Whisper error: ..."}`; else `return {"text": data.get("text","")}`.

### Frontend (`static/index.html`) — integrate into the EXISTING fancy UI, do not replace it

The spellchecker page is a polished design (input panel + output panel + stats). Add voice next to the
Input textarea in the `input-panel` article:

- Markup: inside `.editor-stage input-stage` area (below the textarea) add
  `<button id="record" class="record-button">Start Recording</button>` plus a live status span
  (`aria-live="polite"`), e.g. `<span id="voice-status" role="status"></span>`.
- JS: port sentence-checker's `static/app.js` recorder logic (getUserMedia audio:true, MediaRecorder,
  prefer `audio/webm;codecs=opus`, chunks → Blob):
  - Toggle button: click while recording = stop; while idle = start. Mirror button text/class states
    ("Start Recording" ↔ "Stop Recording", `.recording` class with pulse animation like sentence-checker's).
  - On `mediaRecorder.onstop`: build blob, POST FormData to `/api/transcribe`
    (field name `audio`, filename `recording.webm`).
  - CRITICAL GOTCHA: setting `input.value` programmatically does **not** fire the textarea's
    `"input"` event (that event only fires for user typing — that's why example buttons call
    `schedule(0)` explicitly). So after successful transcription do:
    ```js
    input.value = data.text || "";
    updateCount();
    clearActiveExample();
    schedule(0);   // reuses the existing debounce/correct pipeline unchanged
    ```
  - Errors: mic denied / connection error / `{error}` from server → show in `voice-status`, never throw into UI.
  - Remove the HF Space `__sign` block: delete `const __sign` + `signedUrl()` and use plain
    `fetch("/api/correct")` / `fetch("/api/health")`. That whole private-Space-cookie mechanism is dead locally.

### CSS (`static/style.css`)

- Add `.record-button` styles matching the page's design tokens (it currently has none of these):
  idle button style, a `.is-recording` state (red-ish background + subtle pulse keyframes, respect
  `prefers-reduced-motion` like the rest of the page does), and `.voice-status` typography similar to
  small meta text. Keep it minimal (~30–50 lines) appended near the input-panel rules.

## Phase 3 — LLM API wiring summary (what "same implementation as sentence-checker" means here)

- Text correction does NOT go through llama.cpp (impossible for this encoder model — see context above).
- Only Whisper/audio uses the OpenAI-compatible server: requires the user's existing setup — start their
  llama-server-style binary on port **8000** serving `whisper-large-v3-turbo-int4-ov` (same one already
  working for sentence-checker today; nothing new to install if sentence-checker works).
- ffmpeg is a system dependency (document in README, same note as sentence-checker AGENTS.md).

## Phase 4 — Verification checklist

1. `python -m venv .venv && pip install -r requirements.txt` (+ torch CPU wheel per README).
2. `python server.py` → console prints loading/self-test line (`[server] self-test ok=True ...`).
3. Open http://localhost:5000 — status strip shows "Model ready"; type text → corrections render with
   green/strikethrough diff segments (unchanged pipeline).
4. With llama-server running on :8000: click Start Recording → Stop → textarea auto-fills with transcript
   → correction fires automatically after 650 ms debounce path (`schedule(0)`).
5. Negative tests: no mic permission → friendly error; whisper server down → `/api/transcribe` returns
   502 `{error}` and UI shows it without breaking the typed-text flow.

## Phase 5 — Text-to-speech (Listen button)

Frontend-only; `server.py` and `requirements.txt` untouched.

- Button placement: output panel header, next to Copy — it reads the **corrected** text, so it
  enables/disables with the same `lastCorrected` logic. Both buttons are wrapped in a new
  `.output-actions` div (flex, 8px gap) so the header's `space-between` layout is preserved.
- `static/index.html`:
  - Markup: `<button type="button" class="copy-button speak-button" id="speak" disabled>Listen</button>`
    inside `.output-actions` in `.output-heading`.
  - JS (after the copy handler): `hasTTS = "speechSynthesis" in window`; the button removes itself if
    unsupported. `stopSpeaking()` = `speechSynthesis.cancel()` + reset text to "Listen" + drop `.active`.
    Click toggles: while speaking — cancel; otherwise `new SpeechSynthesisUtterance(lastCorrected)`
    with `lang="en"`, `onend`/`onerror` → `stopSpeaking()`, text to "Stop" + `.active` class.
  - Lifecycle wiring: `speakBtn.disabled = !lastCorrected` after a successful correction (alongside
    `copy.disabled`); `resetResult()` sets `speakBtn.disabled = true` and calls `stopSpeaking()`;
    `correct()` calls `stopSpeaking()` at the start of every check so it never reads stale text.
- `static/style.css`: `.output-actions` (flex/gap), `.speak-button.active` (purple border/background +
  `speak-pulse` keyframes, same 1.5s opacity pulse as `.record-button.recording`). The page's global
  `prefers-reduced-motion` rule already neutralizes the pulse.
- Caveat: Chrome has a long-standing quirk where `speechSynthesis` can pause mid-utterance on very
  long text. Fine for spellchecker-length inputs; if whole pages ever get spoken, chunk by sentence.
- Verification (done): Listen renders beside Copy — click flips to "Stop" + purple pulse — end handler
  resets to "Listen" — auto-cancels on a new check — no JS errors on the page.

## Files touched

| File | Action |
|------|--------|
| `Dockerfile`, `.dockerignore` | delete |
| `README.md` | rewrite as local-project README |
| `requirements.txt` | add `httpx`, `python-multipart` |
| `server.py` | drop cgroup logic, add uvicorn entrypoint port 5000, add `/api/transcribe` + constants/imports |
| `static/index.html` | add record button + recorder JS + transcribe-fill wiring; remove `__sign`; call schedule(0) after transcription; add Listen button + speechSynthesis TTS wiring |
| `static/style.css` | append `.record-button` / `.voice-status` styles; add `.output-actions` / `.speak-button.active` / `speak-pulse` |
