"""Direct in-process STT inference via HuggingFace transformers + torch.

Any HF Whisper-family repo works (multilingual or English-only,
distilled or full). torch picks the CUDA / CPU device automatically;
on CUDA we use fp16 for ~2× speed at negligible accuracy loss.

Default: `openai/whisper-base` — 99 languages, ~145 MB on disk.

Model source is either a local path OR a HF repo ID (auto-downloaded
via the HF cache on first load).
"""
from __future__ import annotations

import asyncio
import gc
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

log = logging.getLogger("voxtype.stt_engine")


# ── Default model ────────────────────────────────────────────────────
# `openai/whisper-base`: 99 languages, ~145 MB. The official HF Whisper
# weights — broadest compatibility across the transformers ecosystem.
# Override per-install via settings.stt_model_path (any HF Whisper repo
# or a local fine-tune). Quantization is left to the user — torch's
# fp16 on GPU is the standard fast path; CPU runs fp32.
DEFAULT_MODEL = "openai/whisper-base"


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


class STTEngine:
    """Singleton — call `get_engine()` to access. Thread-safe."""

    def __init__(self) -> None:
        self._model: Any = None
        self._processor: Any = None
        self._torch_device: str = "cpu"
        self._torch_dtype: Any = None
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

    def _effective_model(self) -> str:
        """Empty setting → use the built-in default."""
        return self._model_path or DEFAULT_MODEL

    def _key(self) -> tuple:
        return (self._effective_model(), self._device)

    async def configure(self, s) -> None:
        self._model_path = str(getattr(s, "stt_model_path", "") or "")
        self._device = str(getattr(s, "stt_device", "cpu"))
        self._language = str(getattr(s, "stt_language", "en"))
        self._idle_unload_sec = int(getattr(s, "stt_idle_unload_sec", 0))

        if self._loaded_key is not None and self._loaded_key != self._key():
            log.info("stt config changed — unloading current model")
            await self.unload()

    # ── Load / unload ────────────────────────────────────────────────

    async def ensure_loaded(self) -> None:
        if self._model is not None and self._loaded_key == self._key():
            return
        async with self._model_lock:
            if self._model is not None and self._loaded_key == self._key():
                return
            if self._model is not None:
                await self._do_unload_locked()
            await self._do_load_locked()

    async def _do_load_locked(self) -> None:
        model_id = self._effective_model()
        log.info("stt loading model=%s device=%s", model_id, self._device)
        self._status.last_error = ""
        self._status.running = False
        self._status.ready = False
        self._notify()
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(self._exec, self._build_model, model_id)
            self._loaded_key = self._key()
            self._status.running = True
            self._status.ready = True
            self._last_used = time.monotonic()
            log.info("stt ready (device=%s dtype=%s)",
                     self._torch_device, self._torch_dtype)
            self._notify()
            self._ensure_idle_watcher()
        except Exception as exc:
            log.error("stt load failed: %s", exc)
            self._model = None
            self._processor = None
            self._loaded_key = None
            self._status.running = False
            self._status.ready = False
            self._status.last_error = str(exc)
            self._notify()
            raise

    def _build_model(self, model_id: str) -> None:
        """Sync — runs in the executor.

        Resolves the torch device with graceful CPU fallback:
        device='cuda' but torch.cuda.is_available() == False → CPU.
        On GPU we use fp16 for ~2× speedup at negligible quality cost.
        """
        import torch
        from transformers import WhisperForConditionalGeneration, AutoProcessor

        if self._device == "cuda" and torch.cuda.is_available():
            self._torch_device = "cuda"
            self._torch_dtype = torch.float16
        else:
            if self._device == "cuda":
                log.warning("stt: device=cuda requested but torch.cuda.is_available()=False — using CPU")
            self._torch_device = "cpu"
            self._torch_dtype = torch.float32

        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = WhisperForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=self._torch_dtype,
        ).to(self._torch_device)
        self._model.eval()

    async def unload(self) -> None:
        async with self._model_lock:
            await self._do_unload_locked()

    async def _do_unload_locked(self) -> None:
        if self._model is None:
            return
        log.info("stt unloading")
        self._model = None
        self._processor = None
        self._loaded_key = None
        self._status.running = False
        self._status.ready = False
        self._notify()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

    # ── Transcription ────────────────────────────────────────────────

    async def transcribe(self, pcm: bytes, language: str | None = None) -> str:
        """Run STT on raw 16 kHz mono int16 PCM. Returns the text."""
        await self.ensure_loaded()
        self._last_used = time.monotonic()
        lang = (language or self._language or "en").strip() or "en"
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._exec, self._do_transcribe, pcm, lang)

    def _do_transcribe(self, pcm: bytes, language: str) -> str:
        """Sync — runs in the executor."""
        import numpy as np
        import torch

        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        inputs = self._processor(
            audio, sampling_rate=16000, return_tensors="pt",
        )
        input_features = inputs.input_features.to(
            self._torch_device, dtype=self._torch_dtype,
        )
        with torch.no_grad():
            generated = self._model.generate(
                input_features,
                language=language,
                task="transcribe",
                max_new_tokens=440,
            )
        text = self._processor.batch_decode(generated, skip_special_tokens=True)[0]
        return (text or "").strip()

    # ── Idle unload watcher ──────────────────────────────────────────

    def _ensure_idle_watcher(self) -> None:
        if self._idle_watch_started:
            return
        self._idle_watch_started = True

        def _loop_thread() -> None:
            INTERVAL = 30.0
            while True:
                time.sleep(INTERVAL)
                if self._model is None:
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


# ── Module singleton ─────────────────────────────────────────────────

_ENGINE: STTEngine | None = None


def get_engine() -> STTEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = STTEngine()
    return _ENGINE
