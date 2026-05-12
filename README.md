# VoxType

Local voice dictation overlay for Windows, written in **pure Python +
PySide6**. Press a hotkey, speak, release — cleaned text appears at
your cursor in any app. No cloud, no telemetry, no account.

STT and TTS both run **in-process via PyTorch** — one ML backend, one
venv. STT uses HuggingFace `transformers` (any Whisper-family repo
works); TTS uses the `kokoro` PyPI package wrapping Kokoro-82M. An
embedded aiohttp server exposes both on a single OpenAI-compatible
HTTP port (default `:6600`) so external clients can call
`/v1/audio/transcriptions` and `/v1/audio/speech`.

**Default models** (~475 MB total disk):
- **STT**: `openai/whisper-base` — 99 languages, ~145 MB
- **TTS**: `hexgrad/Kokoro-82M` — 54 voices in 9 language families
  (American + British English, Spanish, French, Hindi, Italian,
  Japanese, Brazilian Portuguese, Mandarin Chinese), ~327 MB

Sibling project of [telecode](https://github.com/prskid1000/telecode).
LLM transcript cleanup is routed through telecode's dual-protocol proxy
at `http://127.0.0.1:1235`, so any model telecode serves becomes a
dictation backend automatically.

---

## Quick start

```powershell
git clone https://github.com/prskid1000/voxtype.git "$env:USERPROFILE\.voxtype"
cd "$env:USERPROFILE\.voxtype"
.\setup.ps1
```

`setup.ps1` will:

1. Verify **Python 3.10+**, **git**, **ffmpeg** (optional), GPU support
2. Create `voxtype-venv/` and install:
   - `torch` (CUDA 13 nightly wheel if `-GpuSupport`, CPU wheel otherwise)
   - `transformers` (STT)
   - `kokoro` (TTS)
   - PySide6 / pynput / sounddevice / soundfile / aiohttp / numpy / pywin32 / Pillow / mss
3. Pre-download the default STT + TTS models into the HuggingFace
   cache (`~/.cache/huggingface/hub`). Skipped silently if already
   cached — re-runs cost nothing.
4. Register a scheduled task `VoxType` that launches
   `pythonw.exe -m voxtype` at logon (no console window)
5. Seed `voxtype/data/settings.json` with defaults
6. Start VoxType immediately

Re-running `setup.ps1` is idempotent at every phase.

Look for the tray icon (bottom-right). Press **Ctrl+Win**, speak,
release.

### Setup options

```powershell
.\setup.ps1                              # full install (CUDA 13 nightly torch)
.\setup.ps1 -CudaVersion cu124           # CUDA 12.4 stable torch (recommended if you don't have CUDA 13)
.\setup.ps1 -GpuSupport $false           # CPU-only torch
.\setup.ps1 -InstallDir "D:\voxtype"     # custom location
```

### Picking models

Both engines ship with sensible defaults pre-filled in the model
field — clear it to fall back to the same built-in default, or type a
different HuggingFace repo ID / local path.

**STT default:** `openai/whisper-base`
- 99 languages, ~145 MB on disk
- Loaded via `transformers.WhisperForConditionalGeneration` — any HF
  Whisper-family repo works.

**STT alternatives:**
- `openai/whisper-small` / `whisper-medium` / `whisper-large-v3` — bigger, more accurate
- `openai/whisper-large-v3-turbo` — fast + accurate, ~1.6 GB
- `distil-whisper/distil-large-v3` — distilled, ~756 MB
- Any community fine-tune on HF.

**TTS default:** `hexgrad/Kokoro-82M`
- 54 voices, 9 language families, ~327 MB
- The **Voice** field accepts the Kokoro voice name string:
  - `a{f,m}_*` — American English (`af_heart`, `am_adam`)
  - `b{f,m}_*` — British English  (`bf_emma`, `bm_george`)
  - `e_`, `f_`, `h_`, `i_` — Spanish, French, Hindi, Italian
  - `j{f,m}_*` — Japanese (`jf_alpha`, `jm_kumo`)
  - `p{f,m}_*` — Brazilian Portuguese
  - `z{f,m}_*` — Mandarin Chinese (`zf_xiaobei`, `zm_yunjian`)

The **Check** button next to each model field verifies the value —
local stat for paths, HuggingFace API for repo IDs.

---

## Prerequisites

| Dependency | Required for | Where to get it |
|---|---|---|
| **Windows 10/11** | Target OS | — |
| **Python 3.10–3.12** | Everything (kokoro pins <3.13) | https://python.org |
| **git** | Cloning the repo | https://git-scm.com |
| **ffmpeg** (optional) | Non-WAV audio uploads to the embedded server | `winget install ffmpeg` |
| **NVIDIA GPU + recent driver** | Optional — STT + TTS fall back to CPU. torch ships its own CUDA runtime, so no separate CUDA toolkit install is needed. | https://nvidia.com/drivers |
| **espeak-ng** (recommended for non-English TTS) | Phonemizer fallback for languages misaki doesn't cover | `winget install eSpeak-NG.eSpeak-NG` |
| **telecode** (optional) | LLM transcript cleanup | https://github.com/prskid1000/telecode |

Without telecode running, dictation still works — you just get raw
STT transcripts (no filler-word cleanup, no punctuation fixes).

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
    → LRU cache (50 entries) keyed on (transcript, screenshot fingerprint)

→ pill = typing
→ typer.type_text() — clipboard + Ctrl+V via PowerShell SendKeys
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

The `model` and `voice` request fields are **accepted for OpenAI API
compatibility but ignored** — VoxType controls which model is loaded
and which voice is used through its own settings. External clients
only address VoxType by host + port; they don't pick the model.

---

## Tray menu

```
⬡/⬢ STT     ▸ status + model + Load / Unload / Reload
⬡/⬢ TTS     ▸ status + voice + Load / Unload / Reload
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
- **Services** — three cards (each with a footer row containing a live
  status pill and Load / Unload / Reload buttons):
  - **OpenAI HTTP Server** — enable + port for the embedded server
  - **STT** — enable, auto-start, idle unload, model, device, language
  - **TTS** — enable, auto-start, idle unload, model, device, voice
    (Kokoro voice name), speed
- **LLM** — enhance on/off, screen context, proxy URL + model, Test
  Proxy Connection
- **History** — saved transcripts with 📋 Raw / 📋 Final copy icons
- **Logs** — live-tailing `voxtype.log` / `voxtype.log.prev`

Every toggle writes through to `data/settings.json` atomically. Engine
settings (model, device, voice) trigger an automatic reload on the
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
`/v1/chat/completions` with `response_format: json_schema`.

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
- **CUDA 13 torch wheels are nightly.** Stable cu130 wheels haven't
  shipped yet. Use `-CudaVersion cu124` for the stable channel.
