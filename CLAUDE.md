# VoxType — engineering notes

User-facing docs: [README.md](README.md).

## What this is

VoxType is a **pure-Python / PySide6** voice-dictation overlay for
Windows. Hold a hotkey, speak, release — the cleaned transcript is
pasted at the cursor.

STT and TTS both run **in-process via PyTorch through a single
generic backend per modality**. Paste any HuggingFace repo id (or
local path); the backend reads `config.json`, auto-detects the
family, and dispatches to the right loader. One transformers
install covers Whisper, Wav2Vec2/HuBERT/WavLM, MMS, SeamlessM4T,
Moonshine, Speech2Text, SpeechT5 (ASR + TTS), Voxtral,
Granite-Speech, Phi-4-Multimodal, Qwen2-Audio, VITS / MMS-TTS,
Bark, Parler-TTS, XTTS, Orpheus, CSM, Higgs-Audio, VibeVoice,
plus a `transformers.pipeline()` fallback for anything else HF
registers as ASR / TTS.

## Family coverage

### STT families (handler + main per-call opts)

| Family            | model_type             | Handler / loader class                     | Per-call opts                                                  | Example repo                       |
|-------------------|------------------------|---------------------------------------------|----------------------------------------------------------------|------------------------------------|
| Whisper           | `whisper`              | `WhisperForConditionalGeneration`           | `task`, `num_beams`, `temperature`, `repetition_penalty`, `initial_prompt` | `openai/whisper-large-v3`         |
| Wav2Vec2 / HuBERT | `wav2vec2`/`hubert`/`wavlm` | `AutoModelForCTC` (argmax CTC)         | —                                                              | `facebook/wav2vec2-large-960h-lv60-self` |
| MMS               | `wav2vec2` + MMS tag   | `Wav2Vec2ForCTC` + `load_adapter(<lang3>)`  | (driven by universal Language)                                | `facebook/mms-1b-all`              |
| SeamlessM4T(v2)   | `seamless_m4t*`        | `SeamlessM4Tv2ForSpeechToText`              | `task`, `num_beams`, `tgt_lang`                                | `facebook/seamless-m4t-v2-large`   |
| Moonshine         | `moonshine`            | `AutoModelForSpeechSeq2Seq`                 | `num_beams`                                                    | `UsefulSensors/moonshine-base`     |
| Speech2Text       | `speech_to_text`       | `Speech2TextForConditionalGeneration`       | `num_beams`                                                    | `facebook/s2t-small-librispeech-asr` |
| SpeechT5 ASR      | `speecht5`             | generic pipeline                            | —                                                              | `microsoft/speecht5_asr`           |
| Voxtral           | `voxtral`              | `VoxtralForConditionalGeneration` (prompted ASR shared base) | `task`, `temperature`, `prompt`             | `mistralai/Voxtral-Mini-3B-2507`   |
| Granite-Speech    | `granite_speech`       | `GraniteSpeechForConditionalGeneration` (prompted) | `task`, `prompt`                                      | `ibm-granite/granite-speech-3.3-8b`|
| Phi-4-Multimodal  | `phi4_multimodal`      | `Phi4MultimodalForCausalLM` (prompted)      | `prompt`, `temperature`                                        | `microsoft/Phi-4-multimodal-instruct` |
| Qwen2-Audio       | `qwen2_audio`          | `Qwen2AudioForConditionalGeneration` (prompted) | `prompt`, `temperature`, `top_p`                          | `Qwen/Qwen2-Audio-7B-Instruct`     |
| VibeVoice ASR     | `vibevoice_*`          | generic pipeline                            | —                                                              | `microsoft/VibeVoice-*-ASR`        |
| Generic ASR       | (anything pipeline-supported) | `pipeline("automatic-speech-recognition")` | —                                                       | any HF ASR repo                    |

### TTS families

| Family       | model_type     | Handler / loader class                        | Per-call opts                                                        | Voices                          | Example repo                    |
|--------------|----------------|-----------------------------------------------|----------------------------------------------------------------------|---------------------------------|---------------------------------|
| Kokoro       | (PyPI `kokoro`)| `kokoro.KPipeline`                            | `voice_blend`                                                        | 54 curated (9 langs)            | `hexgrad/Kokoro-82M`            |
| VITS / MMS-TTS | `vits`       | `VitsModel` + `AutoTokenizer`                 | `noise_scale`, `noise_scale_duration`, `seed`                        | implicit (per-language repo)    | `facebook/mms-tts-eng`          |
| SpeechT5     | `speecht5`     | `SpeechT5ForTextToSpeech` + HifiGan vocoder   | `speaker_embedding` (HF dataset:row)                                 | 4 curated x-vectors             | `microsoft/speecht5_tts`        |
| Bark         | `bark`         | `BarkModel`                                   | `semantic_temperature`, `coarse_temperature`, `min_eos_p`            | 11 preset speakers              | `suno/bark`                     |
| Parler-TTS   | (PyPI `parler_tts`) | `ParlerTTSForConditionalGeneration`      | `style`, `temperature`, `max_new_tokens`                             | 5 style presets                 | `parler-tts/parler-tts-mini-v1` |
| XTTS         | (PyPI `TTS`)   | `TTS.api.TTS`                                 | `reference_audio`, `language`, `temperature`, `top_p`, `top_k`, `repetition_penalty`, `length_penalty` | clone from reference  | `tts_models/multilingual/multi-dataset/xtts_v2` |
| Orpheus      | `llama` + SNAC | `orpheus_tts.OrpheusModel`                    | `temperature`, `top_p`, `emotion_tags`                               | 8 named speakers                | `canopylabs/orpheus-3b-0.1-ft`  |
| CSM          | `csm`          | `CsmForConditionalGeneration` (native ≥ trans-formers 5.x) | `temperature`                                                  | implicit conversational         | `sesame/csm-1b`                 |
| Higgs-Audio  | (`AutoModelForCausalLM`) | `AutoModelForCausalLM` (trust_remote_code)  | `temperature`, `reference_audio`                                     | zero-shot (reference clip)      | `bosonai/higgs-audio-v2-generation-3B-base` |
| Qwen3-TTS    | (pipeline)     | generic pipeline                              | `temperature`, `top_p`                                               | model-dependent                 | `Qwen/Qwen3-TTS-*`              |
| VibeVoice    | `vibevoice_*`  | `pipeline("text-to-speech")` (trust_remote_code) | `temperature`                                                  | multi-speaker                   | `microsoft/VibeVoice-1.5B`      |
| Generic TTS  | (anything pipeline-supported) | `pipeline("text-to-speech")`   | —                                                                    | implicit                        | any HF TTS repo                 |

An embedded aiohttp server exposes both engines on one
OpenAI-compatible port (`:6600` by default) so external clients reach
VoxType over standard HTTP. LLM transcript cleanup is routed through
**telecode's dual-protocol proxy** (`http://127.0.0.1:1235`).

## Project layout

```
voxtype/
├── setup.ps1                     # Idempotent installer
├── uninstall.ps1                 # Reverse of setup
├── README.md                     # User-facing docs
├── CLAUDE.md                     # This file
├── tests/                        # unittest-style tests, stdlib-only
│   ├── _isolate.py               # per-test VOXTYPE_DATA_DIR sandbox
│   ├── test_family_detect.py     # repo-id → family heuristics
│   ├── test_settings_migration.py# AppSettings legacy-key migration
│   ├── test_config_patch.py      # stt_opts.* / tts_opts.* dotted writes
│   ├── test_backends_registry.py # registry returns generic backend
│   ├── test_alias_catalog.py     # resources/models.json integrity
│   └── test_engine_opts_filter.py# engine filters opts by spec
└── voxtype/                      # The Python package
    ├── __init__.py
    ├── __main__.py               # `python -m voxtype` entry
    ├── main.py                   # Qt loop + asyncio worker + pynput
    ├── types.py                  # AppSettings (+ stt_opts/tts_opts)
    ├── config.py                 # JSON I/O + dotted patch
    ├── debug_log.py
    │
    ├── audio.py                  # sounddevice → 16 kHz mono int16 PCM
    ├── hotkey.py                 # pynput keyboard listener
    ├── vad.py                    # numpy RMS energy gate
    ├── screen_capture.py
    ├── typer.py                  # Clipboard + Ctrl+V via SendKeys
    ├── history.py
    │
    ├── stt_engine.py             # STT orchestrator (lifecycle)
    ├── tts_engine.py             # TTS orchestrator (lifecycle)
    ├── server.py                 # Embedded aiohttp /v1/audio/* server
    ├── stt.py                    # Shim → stt_engine
    ├── llm.py                    # OpenAI-shape POST to telecode
    ├── process.py                # Facade over engines for tray UI
    │
    ├── qt_theme.py               # Dark QSS
    ├── tray_menu.py              # QSystemTrayIcon + submenus
    ├── pill_window.py            # Frameless always-on-top status pill
    ├── settings_window.py        # Spec-driven settings UI
    │
    ├── requirements.txt
    ├── resources/
    │   ├── icon.png
    │   ├── system-prompt.md      # LLM cleanup instructions
    │   └── models.json           # Curated friendly-name → repo-id catalog
    ├── data/                     # User state — gitignored
    │   ├── settings.json
    │   ├── history.json
    │   ├── voxtype.log
    │   └── voxtype.log.prev
    └── backends/
        ├── __init__.py           # Registry — registers `generic` only
        ├── stt_base.py           # STTBackend ABC + LoadConfig + OptionSpec
        ├── tts_base.py           # TTSBackend ABC + TTSLoadConfig + OptionSpec + VoiceEntry
        ├── shared.py             # Whisper 99-language table
        ├── family_detect.py      # Family detection + per-family options + voice catalogs
        ├── generic_stt.py        # The one STT backend; dispatches to family handlers
        └── generic_tts.py        # The one TTS backend; dispatches to family handlers
```

## Runtime architecture

```
Main thread          Qt event loop — widgets, tray, pill, signal delivery
voxtype-asyncio      asyncio loop — HTTP server, llm.enhance, engine load/unload
voxtype-stt          single-thread executor — torch STT inference
voxtype-tts          single-thread executor — torch TTS synthesis
pynput thread        raw keyboard hook
```

Cross-thread handoff uses Qt signals (pill state) or
`QTimer.singleShot(0, lambda: …)` (pulling async results back to the
Qt thread).

Engine status callbacks fire on the executor thread — **never touch
Qt widgets from them directly.** The settings window polls
`engine.get_backend().detected_family()` from a Qt-thread QTimer
instead. The Detect button uses a `QObject` + `Signal` bridge to
marshal worker-thread results into the GUI thread.

## Generic backend dispatcher

`backends/generic_stt.py` and `backends/generic_tts.py` are the only
backends registered. Each is a thin dispatcher:

```python
class GenericSTTBackend(STTBackend):
    def load_sync(self, cfg):
        family = fd.detect_stt_family(cfg.model_id)   # config.json + HF API
        cls = _HANDLERS.get(family, _GenericPipelineHandler)
        self._handler = cls()
        try:
            self._handler.load(cfg)
            self._family = family
        except Exception:
            # Family-specific loader failed (missing optional dep,
            # exotic model). Fall through to pipeline() universal.
            self._handler = _GenericPipelineHandler()
            self._family = fd.STT_GENERIC
            self._handler.load(cfg)
```

Family handlers (all inside `generic_stt.py` / `generic_tts.py`):

| File | STT handlers | TTS handlers |
|---|---|---|
| `generic_stt.py` | `_WhisperHandler`, `_Wav2Vec2Handler`, `_MMSHandler`, `_SeamlessHandler`, `_MoonshineHandler`, `_S2THandler`, `_GenericPipelineHandler` | — |
| `generic_tts.py` | — | `_KokoroHandler`, `_VitsHandler`, `_SpeechT5Handler`, `_BarkHandler`, `_ParlerHandler`, `_PipelineTTSHandler` |

Each handler owns its own model + processor and the inference
loop. No shared state across families.

## Family detection (`backends/family_detect.py`)

Three layers, fast → slow:

1. **Local config.json** — for paths that exist on disk
   (`_read_local_config`).
2. **Repo-id substring heuristic** — synchronous, ~0 ms.
   `_stt_from_repo_id` / `_tts_from_repo_id`. Used by the UI on every
   `textChanged` so the family pill + voice picker update without
   blocking on network I/O.
3. **HuggingFace API** — `huggingface.co/api/models/<id>` +
   `/resolve/main/config.json`. 3 s timeout. Triggered by the
   **Detect** button for verification.

Family identifiers are module-level constants
(`STT_WHISPER`, `STT_WAV2VEC2`, `STT_MMS`, `STT_SEAMLESS`,
`STT_MOONSHINE`, `STT_S2T`, `STT_SPEECHT5`, `STT_PARAKEET`,
`STT_QWEN_AUDIO`, `STT_GENERIC`; `TTS_KOKORO`, `TTS_VITS`,
`TTS_SPEECHT5`, `TTS_BARK`, `TTS_PARLER`, `TTS_XTTS`, `TTS_QWEN_TTS`,
`TTS_GENERIC`).

Per-family metadata helpers:

| Helper | Used by |
|---|---|
| `stt_capabilities(family)` / `tts_capabilities(family)` | UI to gate universal widgets (dtype, torch.compile, speed, stream) |
| `stt_runtime_options(family)` → `list[OptionSpec]` | UI to render the per-family Advanced section |
| `tts_runtime_options(family)` → `list[OptionSpec]` | same |
| `tts_voices_for_family(family)` → `list[VoiceEntry]` | UI to populate the voice picker pre-load (Kokoro/Bark/Parler/SpeechT5 catalogs are static) |
| `stt_family_label(family)` / `tts_family_label(family)` | UI status pill (`"Whisper · 99 langs · translate"`) |

## Option-spec UI

Every UI knob is described by `OptionSpec` (defined in `stt_base.py`
and `tts_base.py`):

```python
@dataclass
class OptionSpec:
    key: str          # storage key inside opts dict
    kind: str         # "enum" | "bool" | "int" | "float" | "str" | "text"
    label: str
    default: Any
    help: str = ""
    choices: list[tuple[str, str]] = ...   # for enum
    min: float | None = None
    max: float | None = None
    step: float | None = None
    rebuild: bool = False                  # forces engine reload
```

`settings_window._render_option(spec, "stt_opts" | "tts_opts")` maps
the spec to a Qt widget bound to `<bag>.<spec.key>` via
`config.patch()`. Adding a new family option requires editing only
`family_detect.py` — no UI code changes.

## Settings shape (`types.py`)

```python
@dataclass
class AppSettings:
    # Universal STT (every family honours these)
    stt_model_path:    str = "openai/whisper-base"
    stt_device:        str = "cpu"
    stt_language:      str = "en"
    stt_dtype:         str = "auto"
    stt_warmup:        bool = True
    stt_torch_compile: bool = False
    stt_idle_unload_sec: int = 300

    # Family-specific per-call opts — dynamic shape.
    # Whisper / Seamless:  {"task": "translate", "num_beams": 5,
    #                       "initial_prompt": "VoxType"}
    # Wav2Vec2 / MMS / generic: {}
    stt_opts: dict = field(default_factory=dict)

    # Universal TTS
    tts_model_path:    str = "hexgrad/Kokoro-82M"
    tts_device:        str = "cpu"
    tts_voice:         str = "af_heart"
    tts_speed:         float = 1.0
    tts_warmup:        bool = True
    tts_torch_compile: bool = False
    tts_stream:        bool = False
    # Parler:    {"style": "A calm female voice"}
    # SpeechT5:  {"speaker_embedding": "dataset:row"}
    # Bark:      {"temperature": 0.7}
    tts_opts: dict = field(default_factory=dict)
```

`AppSettings.from_json()` migrates pre-opts-bag settings:
`stt_task` → `stt_opts.task`, `stt_num_beams` → `stt_opts.num_beams`,
`stt_initial_prompt` → `stt_opts.initial_prompt`,
`tts_speaker` → `tts_voice` (renamed),
`tts_length_scale` → `tts_speed` (renamed).
Legacy `stt_backend` / `tts_backend` values collapse to `"generic"`.

`config.patch("stt_opts.task", "translate")` style dotted writes
land in the opts dict; flat keys still work for top-level fields.

## Embedded HTTP server (`server.py`)

aiohttp app, starts in `_boot_engines()` via `server.start(port=...)`.
Routes:

```
POST /v1/audio/transcriptions   →  stt_engine.transcribe (multipart)
POST /v1/audio/speech           →  tts_engine.synthesize (JSON in, WAV out)
GET  /v1/models                 →  engine list
GET  /health                    →  engine readiness snapshot
GET  /                          →  liveness probe
```

The `model` field is accepted-but-ignored (VoxType controls the
loaded model). The `voice` field on `/v1/audio/speech` IS honoured if
it matches the loaded backend's catalog; otherwise the configured
default is used.

## Tests

Stdlib-only `unittest` (works under `pytest` too):

```powershell
.\voxtype-venv\Scripts\python.exe -m unittest discover tests
```

Coverage:
- `test_family_detect.py` — config-blob + repo-id heuristics for
  every family.
- `test_settings_migration.py` — legacy-key migration paths.
- `test_config_patch.py` — flat keys, `stt_opts.*` / `tts_opts.*`
  dotted writes, persistence to disk.
- `test_backends_registry.py` — registry only registers `generic`,
  resolves any model id to the same backend.
- `test_alias_catalog.py` — `resources/models.json` integrity +
  coverage.
- `test_engine_opts_filter.py` — engine forwards only family-relevant
  opts to the backend.

Each test gets an isolated `VOXTYPE_DATA_DIR` so the real
`voxtype/data/settings.json` is never touched.

## Setup script (`setup.ps1`)

Single venv. Idempotent. Parameters:

| Flag | Default | What it does |
|---|---|---|
| `-InstallDir <path>` | `~/.voxtype` | Where the venv + scheduled task land |
| `-GpuSupport $true\|$false` | `$true` | Install GPU torch wheel vs CPU wheel |
| `-CudaVersion cu130\|cu124\|cpu` | `cu130` | torch CUDA wheel index. `cu130` = nightly, `cu124` = stable |

Phases:

1. Prereqs: Python 3.10–3.12, git, ffmpeg (warn-only), GPU detect.
2. Single venv `voxtype-venv/`. Installs `torch` first from the right
   wheel index, then `voxtype/requirements.txt`.
3. Pre-download default models (Whisper-base + Kokoro-82M) via
   `huggingface_hub.snapshot_download`. Idempotent; non-fatal on
   network failure.
4. Scheduled task `VoxType` runs `pythonw.exe -m voxtype` at logon.
5. Seed `data/settings.json` with AppSettings defaults.

## Dependencies (`voxtype/requirements.txt`)

- **torch** — bundled CUDA runtime.
- **transformers** — covers every HF speech family.
- **sentencepiece** — needed by Seamless / Speech2Text tokenizers.
- **datasets** — SpeechT5 speaker-embedding loading.
- **huggingface_hub** — model auto-download.
- **kokoro** — the one TTS family that uses a non-HF loader.
- **PySide6**, **pynput**, **sounddevice**, **soundfile**, **aiohttp**,
  **numpy**, **Pillow**, **mss**, **pywin32**.

Commented-out optional deps in requirements.txt:
- **parler-tts** — Parler's style-prompt family. Without it, the
  generic backend falls through to `pipeline("text-to-speech")`.
- **phonemizer** + **espeak-ng** — some VITS/Bark/Parler voices.

## Testing the running app

```powershell
Stop-ScheduledTask -TaskName VoxType
.\voxtype-venv\Scripts\python.exe -m voxtype
```

Smoke test:
- Tray icon appears
- `data/voxtype.log` starts filling
- First hotkey: pill goes red → amber (loading) → green text
- Settings → Services → STT card: model field shows
  `openai/whisper-base`, status pill shows `✓ Whisper · 99 langs · translate`
- Settings → Services → TTS card: voice picker is a dropdown with 54
  Kokoro voices
- `curl http://127.0.0.1:6600/health` returns engine status JSON
- `curl http://127.0.0.1:6600/v1/models` lists `whisper-1` + `tts-1`

## What changed in the generic-backend refactor

Removed:
- The `Backend` dropdown in the STT/TTS cards (there's only one
  backend now).
- Per-backend modules `backends/whisper.py` and `backends/kokoro.py`
  (folded into `generic_stt.py` / `generic_tts.py` as family
  handlers).
- The `TranscribeOptions` dataclass (`opts` is now `dict[str, Any]`
  on the ABC, so adding a new family knob never breaks other
  backends).
- AppSettings fields `stt_task`, `stt_num_beams`, `stt_initial_prompt`,
  `tts_speaker`, `tts_length_scale` (migrated into the opts bags).

Added:
- `backends/family_detect.py` — fast + slow detection + per-family
  metadata (capabilities, option specs, voice catalogs, labels).
- `backends/generic_stt.py` and `backends/generic_tts.py` — the
  generic dispatchers with all family handlers inline.
- `voxtype/resources/models.json` — curated friendly-name catalog of
  recommended HF model ids per family.
- `tests/` — stdlib unittest suite.
- `AppSettings.stt_opts` / `tts_opts` — free-form per-family opts
  bags with on-load migration of legacy top-level keys.
- `config.patch("stt_opts.task", "translate")` style dotted writes.
- Spec-driven UI in `settings_window.py`:
  - `_render_option(spec, "stt_opts" | "tts_opts")` maps an
    `OptionSpec` to a Qt widget.
  - The model row's status pill shows the auto-detected family.
    A synchronous repo-id heuristic runs on every `textChanged`;
    the **Detect** button verifies against the HF API
    (worker thread → `QObject` Signal bridge → GUI thread).
  - The voice picker is populated from the family's static catalog
    (`fd.tts_voices_for_family(family)`) the moment a family is
    known — no model load required.
  - Universal widgets (Language, Dtype, torch.compile, Speed,
    Stream) auto-hide via `supports(feature)` when the family
    doesn't honour them.
