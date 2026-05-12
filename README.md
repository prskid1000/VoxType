# VoxType

Local voice dictation overlay for Windows, written in **pure Python +
PySide6**. Press a hotkey, speak, release — cleaned text appears at
your cursor in any app. No cloud, no telemetry, no account.

STT and TTS both run **in-process via ONNX Runtime** — no separate
sidecar servers, no extra venvs. STT goes through HuggingFace
`transformers` + `optimum` (truly generic — any HF Whisper-family ONNX
repo loads); TTS goes through `sherpa-onnx` (best Kokoro / VITS-Piper
support). An embedded aiohttp server exposes both on a single
OpenAI-compatible HTTP port (default `:6600`) so external clients can
call `/v1/audio/transcriptions` and `/v1/audio/speech`.

**Default models** (~355 MB total disk):
- **STT**: `onnx-community/whisper-base-ONNX` q4f16 — 99 languages,
  ~85 MB (encoder 14 MB + decoders 68 MB + tokenizer ~3 MB)
- **TTS**: `csukuangfj/kokoro-multi-lang-v1_0` — 53 voices, English +
  Chinese, ~270 MB

Sibling project of [telecode](https://github.com/prskid1000/telecode).
LLM transcript cleanup is routed through telecode's dual-protocol proxy
at `http://127.0.0.1:1235`, so any model telecode serves (llama.cpp,
Qwen-VL, etc.) becomes a dictation backend automatically. There is no
direct LM Studio dependency.

---

## Quick start

```powershell
git clone https://github.com/prskid1000/voxtype.git "$env:USERPROFILE\.voxtype"
cd "$env:USERPROFILE\.voxtype"
.\setup.ps1
```

`setup.ps1` will:

1. Verify **Python 3.10+**, **git**, **ffmpeg** (optional), GPU support
2. Create `voxtype-venv/` and `pip install -r voxtype/requirements.txt`
   into one venv. STT uses `transformers` + `optimum[onnxruntime]`
   (truly generic for any HF Whisper-family export). TTS uses
   `sherpa-onnx` (best Kokoro / VITS-Piper handling). Both target the
   same ONNX Runtime under the hood.
3. If `-GpuSupport $true` (default): swap CPU `onnxruntime` for
   `onnxruntime-gpu` so `device='cuda'` lands on the GPU for both STT
   and TTS (falls back to CPU automatically if CUDA isn't usable)
4. Pre-download the default STT + TTS models into the HuggingFace cache
   (`~/.cache/huggingface/hub`) using **selective `allow_patterns`** so
   only the variants the engines actually load are fetched — STT pulls
   the q4f16 ONNX files only (~85 MB, not the ~290 MB fp32 weights);
   TTS pulls model + voices + lexicons (~270 MB, not the test WAVs).
   Skipped silently if a model is already cached — re-runs cost nothing.
5. Register a single scheduled task `VoxType` that launches
   `pythonw.exe -m voxtype` at logon (no console window)
6. Seed `voxtype/data/settings.json` with defaults
7. Start VoxType immediately

Re-running `setup.ps1` is idempotent at every phase: venvs reuse
existing site-packages, the model pre-download skips cached files, the
scheduled task is recreated cleanly.

Look for the tray icon (bottom-right). Press **Ctrl+Win**, speak,
release.

### Setup options

```powershell
.\setup.ps1                            # full install (STT + TTS, GPU)
.\setup.ps1 -GpuSupport $false         # CPU-only ONNX Runtime
.\setup.ps1 -InstallDir "D:\voxtype"   # custom location
```

Re-running `setup.ps1` is idempotent.

### Picking models

Both engines ship with **sensible defaults** pre-filled in the model
field — clear it to fall back to the same built-in default, or type a
different HuggingFace repo ID / local path.

**STT default:** `onnx-community/whisper-base-ONNX` (q4f16 quant)
- 99 languages, ~85 MB on disk
- Loaded via `optimum.onnxruntime.ORTModelForSpeechSeq2Seq` + a HF
  `WhisperProcessor` — truly generic, any HF Whisper-family ONNX repo
  works the same way.

**STT alternatives** (any HF Whisper-family ONNX export):
- `onnx-community/whisper-small-ONNX` — bigger, more accurate
- `onnx-community/whisper-large-v3-turbo-ONNX` — best quality, ~1.6 GB
- `onnx-community/distil-whisper-distil-large-v3-ONNX` — distilled
- Set `stt_quant` in settings to pick `q4f16` (default) / `q4` / `int8`
  / `fp16` / empty (fp32) — whichever variant the repo ships.

**TTS default:** `csukuangfj/kokoro-multi-lang-v1_0`
- Kokoro multilingual v1.0, **53 voices**, Chinese + English, ~270 MB
- Pick a voice by passing an integer `voice` field on
  `/v1/audio/speech` (0–52), or set `tts_speaker` in settings.

**TTS alternatives** (sherpa-onnx-compatible models):
- `csukuangfj/kokoro-multi-lang-v1_1` — 103 voices, ~395 MB
- `csukuangfj/kokoro-en-v0_19` — 11 English voices, ~85 MB
- Any Piper / VITS / Matcha-TTS sherpa-onnx export

The **Check** button next to each model field verifies the value —
local stat for paths, HuggingFace API for repo IDs.

---

## Prerequisites

| Dependency | Required for | Where to get it |
|---|---|---|
| **Windows 10/11** | Target OS | — |
| **Python 3.10+** | Everything | https://python.org |
| **git** | Cloning the repo | https://git-scm.com |
| **ffmpeg** (optional) | Non-WAV audio uploads to the embedded server | `winget install ffmpeg` |
| **NVIDIA GPU + CUDA driver** | Optional — STT + TTS fall back to CPU | https://nvidia.com/drivers |
| **telecode** (optional) | LLM transcript cleanup | https://github.com/prskid1000/telecode |

Without telecode running, dictation still works — you just get raw
STT transcripts (no filler-word cleanup, no punctuation fixes).
Set `enhance_enabled = false` in settings to silence the "proxy
unreachable" warnings.

---

## How it works

```
Hotkey down (pynput)
    → recorder.start() — sounddevice opens a 16 kHz mono int16 PCM stream
    → pill = recording

Hotkey up
    → recorder.stop() → raw PCM buffer
    → VAD gate (RMS energy) — drop pure silence
    → pill = processing
    → stt.transcribe() — DIRECT call into stt_engine.STTEngine
                         (no HTTP — that's only for external clients)

if enhance_enabled:
    → pill = enhancing
    → if screen_context: capture active display + paint red cursor
      marker → JPEG base64
    → llm.enhance() — OpenAI-shape POST to telecode proxy (:1235)
                      with JSON-schema response_format
    → 4-stage JSON recovery for malformed responses
    → LRU cache (50 entries) keyed on (transcript, screenshot fingerprint)

→ pill = typing
→ typer.type_text() — write to clipboard, send Ctrl+V via PowerShell
                      SendKeys, restore previous clipboard contents
→ history.add() — append to data/history.json (last 500)
→ pill = idle
```

### Embedded HTTP server

Lives in `voxtype/server.py`, starts on port 6600 (configurable). Routes:

```
POST /v1/audio/transcriptions  →  STT (multipart upload)
POST /v1/audio/speech          →  TTS (JSON in, WAV out)
GET  /v1/models                →  engine list
GET  /health                   →  engine readiness snapshot
```

The hot path inside VoxType calls the engines directly — this server
exists so external clients (telecode, MCP tools, any OpenAI-shape API
consumer) can reach VoxType over standard HTTP.

### Threading

- **Main thread**: Qt event loop (tray, pill, settings window)
- **Worker thread**: dedicated asyncio loop for HTTP server + inference
- **Inference thread pools**: one single-thread executor per engine —
  serialises model calls so we never OOM from concurrent inference
- **Pynput thread**: raw keyboard input hook

Quit uses an `os._exit(0)` watchdog (5 s). Engine models are
deallocated and CUDA caches flushed in `process.stop_all()`.

---

## Tray menu

```
⬡/⬢ STT     ▸ status + model + Load / Unload / Reload
⬡/⬢ TTS     ▸ status + model + Load / Unload / Reload
⬡/⬢ LLM     ▸ proxy model + Test Proxy Connection
⬢   Pill    ▸ Hide Pill / Show Pill + Reset Position
─
Open Settings Window   (default left-click)
─
Quit VoxType
```

The Settings window has these sections:

- **Dictation** — hotkey mode, live **Rebind** button, auto-stop on
  silence, VAD, append mode, save history
- **Services** — three cards:
  - **OpenAI HTTP Server** — enable + port for the embedded server
  - **STT** — enable, auto-start, idle unload, model (free text accepting
    HF repo or local path + Browse + HF Check button), device, language,
    Reload
  - **TTS** — enable, auto-start, idle unload, model path (with Browse
    file dialog), device, speaker, length scale, Reload
- **LLM** — enhance on/off, screen context, proxy URL + model, Test
  Proxy Connection
- **History** — saved transcripts with 📋 Raw / 📋 Final copy icons
- **Logs** — live-tailing `voxtype.log` / `voxtype.log.prev`

Every toggle writes through to `data/settings.json` atomically. Engine
settings (model, device, compute type) trigger an automatic reload on
next inference call.

---

## Data layout

All user state lives under the repo:

```
voxtype/data/
  settings.json      # AppSettings — auto-created on first run
  history.json       # last 500 transcripts (if save_history=true)
  voxtype.log        # current session
  voxtype.log.prev   # previous session (rotated on restart)
```

Override with `VOXTYPE_DATA_DIR=C:\some\other\path` if you want storage
outside the repo. `voxtype/data/` is gitignored.

---

## LLM enhancement

`settings.json` fields:

```json
{
  "enhance_enabled": true,
  "screen_context":  true,
  "proxy_url":       "http://127.0.0.1:1235",
  "proxy_model":     "qwen3.5-35b"
}
```

`proxy_model` can be anything telecode's llamacpp registry recognises,
OR anything in `proxy.model_mapping`. VoxType sends OpenAI-shape
`/v1/chat/completions` with `response_format: json_schema` — the model
returns structured output and VoxType extracts the `output` field.

If the request fails, the **original STT transcript** is returned
unchanged — dictation keeps working when the LLM is unreachable.

---

## Hotkey

Defaults to **Ctrl + Win** (hold). Use the **Rebind** button in
Settings → Dictation to capture a new combo.

`hotkey_mode` can be `"hold"` (dictate while held) or `"toggle"` (tap
to start, tap to stop).

---

## Uninstall

```powershell
.\uninstall.ps1
```

Removes the scheduled task and (interactively) offers to delete the
install directory and repo-local `voxtype/data/`.

---

## Known limitations

- **Windows-only.** Typer uses PowerShell SendKeys; screen capture
  uses Win32 `GetCursorPos`.
- **No live mic device picker.** sounddevice picks the system default.
- **TTS isn't wired into the dictation pipeline** — it's served via the
  HTTP endpoint for external clients. Speak-back is not part of the
  hotkey flow.
