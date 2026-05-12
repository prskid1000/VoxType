"""Direct in-process STT inference via ONNX Runtime (sherpa-onnx).

The recognizer object lives on the voxtype process heap; a single-thread
`ThreadPoolExecutor` serialises inference so the asyncio loop is never
blocked.

Model source: either a local directory or a **HuggingFace repo ID**.
If the user enters something like `csukuangfj/sherpa-onnx-whisper-small`,
the engine snapshot_downloads the repo to the HF cache on first load
and re-uses the cached files thereafter. Expected layout:

    <model>/encoder.onnx     (or encoder.int8.onnx)
    <model>/decoder.onnx     (or decoder.int8.onnx)
    <model>/tokens.txt

CPU / GPU switching is purely an ONNX Runtime concern — sherpa-onnx
takes a `provider` string. `device='cuda'` falls back to CPU
automatically if onnxruntime-gpu isn't usable.

Lifecycle, status callbacks and idle unload mirror tts_engine.py one-
for-one so the tray + settings UI use the same code paths.
"""
from __future__ import annotations

import asyncio
import gc
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger("voxtype.stt_engine")


# ── Default model ────────────────────────────────────────────────────
# Whisper Large V3 Turbo (sherpa-onnx export): the multilingual STT
# default sweet spot in 2026. Distilled from large-v3, ~6× faster than
# large-v3 while staying within ~1-2% WER, ~809M params, 99+ languages.
# Popular + latest + multilingual + fast. Ships in csukuangfj's well-
# maintained sherpa-onnx repos.
DEFAULT_MODEL = "csukuangfj/sherpa-onnx-whisper-turbo"


# ── Status type ──────────────────────────────────────────────────────

@dataclass
class EngineStatus:
    running: bool = False
    ready: bool = False
    pid: int | None = None
    last_error: str = ""

    @property
    def name(self) -> str:
        return "stt"


# ── Engine ───────────────────────────────────────────────────────────

class STTEngine:
    """Singleton — call `get_engine()` to access. Thread-safe."""

    def __init__(self) -> None:
        self._recognizer: Any = None
        self._model_lock = asyncio.Lock()
        self._exec = ThreadPoolExecutor(max_workers=1, thread_name_prefix="voxtype-stt")
        self._loaded_key: tuple | None = None
        self._status = EngineStatus()
        self._listeners: list[Callable[[EngineStatus], None]] = []
        self._last_used = 0.0
        self._idle_unload_sec = 0
        self._idle_watch_started = False

        # Current settings.
        self._model_path = ""
        self._device = "cpu"
        self._language = "en"

    # ── Listener wiring ──────────────────────────────────────────────

    def on_status_change(self, fn: Callable[[EngineStatus], None]) -> None:
        self._listeners.append(fn)

    def get_status(self) -> EngineStatus:
        return EngineStatus(
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

    def _key(self) -> tuple:
        return (self._effective_model(), self._device)

    async def configure(self, s) -> None:
        """Apply settings from `AppSettings`. If the key changed and a
        recognizer is loaded, unload so the next call picks up new config."""
        self._model_path = str(getattr(s, "stt_model_path", "") or "")
        self._device = str(getattr(s, "stt_device", "cpu"))
        self._language = str(getattr(s, "stt_language", "en"))
        self._idle_unload_sec = int(getattr(s, "stt_idle_unload_sec", 0))

        if self._loaded_key is not None and self._loaded_key != self._key():
            log.info("stt config changed — unloading current recognizer")
            await self.unload()

    # ── Load / unload ────────────────────────────────────────────────

    def _effective_model(self) -> str:
        """Empty setting → use the built-in default (matches docgraph's
        reranker pattern: empty = use default)."""
        return self._model_path or DEFAULT_MODEL

    async def ensure_loaded(self) -> None:
        if self._recognizer is not None and self._loaded_key == self._key():
            return
        async with self._model_lock:
            if self._recognizer is not None and self._loaded_key == self._key():
                return
            if self._recognizer is not None:
                await self._do_unload_locked()
            await self._do_load_locked()

    async def _do_load_locked(self) -> None:
        model = self._effective_model()
        log.info("stt loading model=%s device=%s", model, self._device)
        self._status.last_error = ""
        self._status.running = False
        self._status.ready = False
        self._notify()
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(self._exec, self._build_recognizer, model)
            self._loaded_key = self._key()
            self._status.running = True
            self._status.ready = True
            self._last_used = time.monotonic()
            log.info("stt ready")
            self._notify()
            self._ensure_idle_watcher()
        except Exception as exc:
            log.error("stt load failed: %s", exc)
            self._recognizer = None
            self._loaded_key = None
            self._status.running = False
            self._status.ready = False
            self._status.last_error = str(exc)
            self._notify()
            raise

    def _build_recognizer(self, model_path: str) -> None:
        """Sync — runs in the executor. Resolves the model (local path
        OR HuggingFace repo) and builds a sherpa-onnx recognizer."""
        import sherpa_onnx
        model_dir = resolve_model_dir(model_path)
        # Resolve encoder / decoder. Prefer fp32; fall back to quantized.
        encoder = _pick(model_dir,
                         ("encoder.onnx", "encoder.int8.onnx",
                          "encoder-fp16.onnx", "encoder-int8.onnx"))
        decoder = _pick(model_dir,
                         ("decoder.onnx", "decoder.int8.onnx",
                          "decoder-fp16.onnx", "decoder-int8.onnx"))
        tokens = _pick(model_dir, ("tokens.txt",))
        if encoder is None or decoder is None or tokens is None:
            raise RuntimeError(
                f"sherpa-onnx model files not found under {model_dir} — "
                "expected encoder*.onnx + decoder*.onnx + tokens.txt"
            )
        provider = "cuda" if self._device == "cuda" else "cpu"
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=str(encoder),
            decoder=str(decoder),
            tokens=str(tokens),
            language=self._language,
            task="transcribe",
            provider=provider,
        )

    async def unload(self) -> None:
        async with self._model_lock:
            await self._do_unload_locked()

    async def _do_unload_locked(self) -> None:
        if self._recognizer is None:
            return
        log.info("stt unloading")
        self._recognizer = None
        self._loaded_key = None
        self._status.running = False
        self._status.ready = False
        self._notify()
        gc.collect()

    # ── Transcription ────────────────────────────────────────────────

    async def transcribe(self, pcm: bytes, language: str | None = None,
                          beam_size: int | None = None) -> str:
        """Run STT on raw 16 kHz mono int16 PCM. Returns the text."""
        await self.ensure_loaded()
        self._last_used = time.monotonic()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._exec, self._do_transcribe, pcm)

    def _do_transcribe(self, pcm: bytes) -> str:
        """Sync — runs in the executor."""
        import numpy as np
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        stream = self._recognizer.create_stream()
        stream.accept_waveform(16000, audio)
        self._recognizer.decode_stream(stream)
        return (stream.result.text or "").strip()

    # ── Idle unload watcher ──────────────────────────────────────────

    def _ensure_idle_watcher(self) -> None:
        if self._idle_watch_started:
            return
        self._idle_watch_started = True

        def _loop_thread() -> None:
            INTERVAL = 30.0
            while True:
                time.sleep(INTERVAL)
                if self._recognizer is None:
                    continue
                if self._idle_unload_sec <= 0:
                    continue
                idle = time.monotonic() - (self._last_used or 0.0)
                if idle < self._idle_unload_sec:
                    continue
                log.info("stt idle for %.0fs ≥ %ds — unloading",
                         idle, self._idle_unload_sec)
                threading.Thread(
                    target=lambda: asyncio.run(self.unload()),
                    daemon=True,
                ).start()

        threading.Thread(target=_loop_thread, daemon=True,
                         name="voxtype-stt-idle").start()


def _pick(directory: Path, candidates: tuple[str, ...]) -> Path | None:
    """Return the first existing candidate file inside `directory`.
    Walks one level deep so models packed inside a sub-folder still work."""
    for name in candidates:
        p = directory / name
        if p.exists():
            return p
    # One-level deep fallback — HF snapshots sometimes nest files inside a dir
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
      - If `model_path` exists locally (file or dir), return its parent dir
        (or the dir itself).
      - Otherwise treat it as `org/repo` and `snapshot_download()` it via
        `huggingface_hub`. The cached dir is returned.
    """
    if not model_path:
        # Should never happen — STTEngine._effective_model() substitutes
        # DEFAULT_MODEL before calling. Guard anyway.
        model_path = DEFAULT_MODEL
    p = Path(model_path).expanduser()
    if p.exists():
        return p if p.is_dir() else p.parent
    # Looks like an HF repo ID? (contains '/', not absolute)
    if "/" in model_path and not p.is_absolute():
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "huggingface_hub not installed — `pip install huggingface_hub` "
                "or enter a local path to the model directory"
            ) from exc
        log.info("stt downloading HF repo %s …", model_path)
        cached = snapshot_download(repo_id=model_path)
        return Path(cached)
    raise RuntimeError(f"model not found: {model_path}")


# ── Module singleton ─────────────────────────────────────────────────

_ENGINE: STTEngine | None = None


def get_engine() -> STTEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = STTEngine()
    return _ENGINE
