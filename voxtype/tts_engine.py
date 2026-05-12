"""Direct in-process TTS inference via sherpa-onnx (ONNX Runtime).

Same library as the STT engine — one dependency, both engines. The
Kokoro multilingual model (Chinese + English, 103 speakers, 82M params,
~395 MB) is the built-in default. Any sherpa-onnx-compatible Kokoro
export works.

Model source: a local model directory OR a HuggingFace repo ID, which
is auto-downloaded via `huggingface_hub.snapshot_download()` on first
load. Expected layout (matches `csukuangfj/kokoro-multi-lang-v1_1`):

    <model>/model.onnx
    <model>/voices.bin
    <model>/tokens.txt
    <model>/lexicon-us-en.txt   (optional, per-language)
    <model>/lexicon-gb-en.txt
    <model>/lexicon-zh.txt
    <model>/dict/               (optional)
    <model>/espeak-ng-data/     (optional)

CPU / GPU switching is purely an ONNX Runtime concern — sherpa-onnx
takes a `provider` string. `device='cuda'` falls back to CPU
automatically if onnxruntime-gpu isn't usable.

Speaker selection: the `voice` field in the OpenAI /v1/audio/speech
request is parsed as an integer speaker ID (0-102 for the default
Kokoro multi-lang model). Out-of-range falls back to `tts_speaker`
from settings.
"""
from __future__ import annotations

import asyncio
import gc
import io
import logging
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger("voxtype.tts_engine")


# ── Default model ────────────────────────────────────────────────────
# Kokoro multi-lang v1.0: 53 speakers, Chinese + English, ~270 MB
# minimal footprint. Picked over v1.1 (103 speakers, ~395 MB) to keep
# disk usage closer to the project's small-install target. v1.1 is
# better if you want the extra voices — just point the model field at it.
DEFAULT_MODEL = "csukuangfj/kokoro-multi-lang-v1_0"


# ── Status type ──────────────────────────────────────────────────────

@dataclass
class TTSStatus:
    running: bool = False
    ready: bool = False
    pid: int | None = None
    last_error: str = ""

    @property
    def name(self) -> str:
        return "tts"


class TTSEngine:
    """Singleton — call `get_engine()`. Thread-safe."""

    def __init__(self) -> None:
        self._tts: Any = None
        self._model_lock = asyncio.Lock()
        self._exec = ThreadPoolExecutor(max_workers=1, thread_name_prefix="voxtype-tts")
        self._loaded_key: tuple | None = None
        self._status = TTSStatus()
        self._listeners: list[Callable[[TTSStatus], None]] = []
        self._last_used = 0.0
        self._idle_unload_sec = 0
        self._idle_watch_started = False
        self._sample_rate = 24000     # sherpa-onnx Kokoro default
        self._num_speakers = 0

        # Current settings.
        self._model_path = ""
        self._device = "cpu"
        self._speaker = 0
        self._length_scale = 1.0

    # ── Listener wiring ──────────────────────────────────────────────

    def on_status_change(self, fn: Callable[[TTSStatus], None]) -> None:
        self._listeners.append(fn)

    def get_status(self) -> TTSStatus:
        return TTSStatus(
            running=self._status.running,
            ready=self._status.ready,
            pid=None,
            last_error=self._status.last_error,
        )

    def _notify(self) -> None:
        for fn in list(self._listeners):
            try:
                fn(self.get_status())
            except Exception:
                pass

    # ── Configuration ────────────────────────────────────────────────

    def _effective_model(self) -> str:
        """Empty setting → use the built-in default."""
        return self._model_path or DEFAULT_MODEL

    def _key(self) -> tuple:
        return (self._effective_model(), self._device)

    async def configure(self, s) -> None:
        self._model_path = str(getattr(s, "tts_model_path", "") or "")
        self._device = getattr(s, "tts_device", "cpu")
        self._speaker = int(getattr(s, "tts_speaker", 0))
        self._length_scale = float(getattr(s, "tts_length_scale", 1.0))
        self._idle_unload_sec = int(getattr(s, "tts_idle_unload_sec", 0))

        if self._loaded_key is not None and self._loaded_key != self._key():
            log.info("tts config changed — unloading current model")
            await self.unload()

    # ── Load / unload ────────────────────────────────────────────────

    async def ensure_loaded(self) -> None:
        if self._tts is not None and self._loaded_key == self._key():
            return
        async with self._model_lock:
            if self._tts is not None and self._loaded_key == self._key():
                return
            if self._tts is not None:
                await self._do_unload_locked()
            await self._do_load_locked()

    async def _do_load_locked(self) -> None:
        model = self._effective_model()
        log.info("tts loading model=%s device=%s", model, self._device)
        self._status.last_error = ""
        self._status.running = False
        self._status.ready = False
        self._notify()
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(self._exec, self._build_tts, model)
            self._loaded_key = self._key()
            self._status.running = True
            self._status.ready = True
            self._last_used = time.monotonic()
            log.info("tts ready (%d speakers, %d Hz)", self._num_speakers, self._sample_rate)
            self._notify()
            self._ensure_idle_watcher()
        except Exception as exc:
            log.error("tts load failed: %s", exc)
            self._tts = None
            self._loaded_key = None
            self._status.running = False
            self._status.ready = False
            self._status.last_error = str(exc)
            self._notify()
            raise

    def _build_tts(self, model_path: str) -> None:
        """Sync — runs in the executor. Resolves the model (local OR HF)
        and builds a sherpa-onnx OfflineTts."""
        import sherpa_onnx
        model_dir = resolve_model_dir(model_path)

        model_onnx = _pick(model_dir, ("model.onnx", "model.int8.onnx"))
        voices_bin = _pick(model_dir, ("voices.bin",))
        tokens = _pick(model_dir, ("tokens.txt",))
        if not (model_onnx and voices_bin and tokens):
            raise RuntimeError(
                f"Kokoro TTS files not found under {model_dir} — expected "
                "model.onnx + voices.bin + tokens.txt"
            )

        # Optional lexicons / dict / espeak-ng-data: pass when present.
        lexicons = sorted(model_dir.glob("lexicon-*.txt"))
        lexicon_arg = ",".join(str(p) for p in lexicons) if lexicons else ""
        dict_dir = model_dir / "dict"
        data_dir = model_dir / "espeak-ng-data"

        provider = "cuda" if self._device == "cuda" else "cpu"
        config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                    model=str(model_onnx),
                    voices=str(voices_bin),
                    tokens=str(tokens),
                    lexicon=lexicon_arg,
                    data_dir=str(data_dir) if data_dir.is_dir() else "",
                    dict_dir=str(dict_dir) if dict_dir.is_dir() else "",
                    length_scale=self._length_scale,
                ),
                num_threads=2,
                provider=provider,
            ),
            max_num_sentences=1,
        )
        self._tts = sherpa_onnx.OfflineTts(config)
        # sherpa-onnx exposes sample_rate + num_speakers post-init.
        self._sample_rate = int(getattr(self._tts, "sample_rate", 24000))
        self._num_speakers = int(getattr(self._tts, "num_speakers", 0))

    async def unload(self) -> None:
        async with self._model_lock:
            await self._do_unload_locked()

    async def _do_unload_locked(self) -> None:
        if self._tts is None:
            return
        log.info("tts unloading")
        self._tts = None
        self._loaded_key = None
        self._status.running = False
        self._status.ready = False
        self._notify()
        gc.collect()

    # ── Synthesis ────────────────────────────────────────────────────

    async def synthesize(self, text: str,
                          voice: str | int | None = None,
                          speed: float | None = None) -> bytes:
        """Return WAV bytes (16-bit mono, sherpa-onnx Kokoro sample rate).

        `voice`: speaker ID. Accepts an int directly, or a string that
            parses to int. Out-of-range / unparseable → falls back to
            `tts_speaker` from settings.
        `speed`: OpenAI-shape (1.0 = normal). Maps to sherpa-onnx
            `speed` arg directly (sherpa uses the same convention).
        """
        await self.ensure_loaded()
        self._last_used = time.monotonic()
        sid = self._resolve_sid(voice)
        spd = float(speed) if (speed and speed > 0) else 1.0
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._exec, self._do_synthesize, text, sid, spd,
        )

    def _resolve_sid(self, voice) -> int:
        if voice is None or voice == "":
            return self._speaker
        try:
            sid = int(voice)
        except (TypeError, ValueError):
            return self._speaker
        if self._num_speakers > 0 and not (0 <= sid < self._num_speakers):
            log.warning("voice id %d out of range [0, %d) — using %d",
                        sid, self._num_speakers, self._speaker)
            return self._speaker
        return sid

    def _do_synthesize(self, text: str, sid: int, speed: float) -> bytes:
        """Sync — runs in the executor. Returns WAV bytes."""
        import numpy as np
        result = self._tts.generate(text, sid=sid, speed=speed)
        samples = np.asarray(result.samples, dtype=np.float32)
        # float32 [-1, 1] → int16
        np.clip(samples, -1.0, 1.0, out=samples)
        int16 = (samples * 32767.0).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(int(result.sample_rate or self._sample_rate))
            wf.writeframes(int16.tobytes())
        return buf.getvalue()

    # ── Idle unload watcher ──────────────────────────────────────────

    def _ensure_idle_watcher(self) -> None:
        if self._idle_watch_started:
            return
        self._idle_watch_started = True

        def _loop_thread() -> None:
            INTERVAL = 30.0
            while True:
                time.sleep(INTERVAL)
                if self._tts is None:
                    continue
                if self._idle_unload_sec <= 0:
                    continue
                idle = time.monotonic() - (self._last_used or 0.0)
                if idle < self._idle_unload_sec:
                    continue
                log.info("tts idle for %.0fs ≥ %ds — unloading",
                         idle, self._idle_unload_sec)
                threading.Thread(
                    target=lambda: asyncio.run(self.unload()),
                    daemon=True,
                ).start()

        threading.Thread(target=_loop_thread, daemon=True,
                         name="voxtype-tts-idle").start()


# ── Helpers ──────────────────────────────────────────────────────────

def _pick(directory: Path, candidates: tuple[str, ...]) -> Path | None:
    """Return the first existing candidate file inside `directory`.
    Walks one level deep so models packed in a sub-folder still work."""
    for name in candidates:
        p = directory / name
        if p.exists():
            return p
    for sub in directory.iterdir() if directory.is_dir() else []:
        if sub.is_dir():
            for name in candidates:
                p = sub / name
                if p.exists():
                    return p
    return None


def resolve_model_dir(model_path: str) -> Path:
    """Accept either a local path or a HuggingFace repo ID.

    Returns a local `Path` to the model directory:
      - Local file/dir exists → return its dir (or parent)
      - Looks like `org/repo` → snapshot_download via huggingface_hub
        and return the cached dir
    """
    if not model_path:
        model_path = DEFAULT_MODEL
    p = Path(model_path).expanduser()
    if p.exists():
        return p if p.is_dir() else p.parent
    if "/" in model_path and not p.is_absolute():
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "huggingface_hub not installed — `pip install huggingface_hub` "
                "or enter a local path to the model directory"
            ) from exc
        log.info("tts downloading HF repo %s …", model_path)
        cached = snapshot_download(repo_id=model_path)
        return Path(cached)
    raise RuntimeError(f"model not found: {model_path}")


# ── Module singleton ─────────────────────────────────────────────────

_ENGINE: TTSEngine | None = None


def get_engine() -> TTSEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = TTSEngine()
    return _ENGINE
