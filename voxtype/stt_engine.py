"""Direct in-process STT inference — truly generic, via HuggingFace
transformers + optimum.onnxruntime.

Any HF-exported Whisper-family ONNX repo works out of the box. The
engine uses `optimum.onnxruntime.ORTModelForSpeechSeq2Seq` which knows
the Whisper encoder/decoder/decoder_with_past split, drives the
autoregressive decoding loop, and binds CUDA / CPU execution providers.

Default: `onnx-community/whisper-base-ONNX` loaded with the q4f16
quantization variant — ~85 MB total (encoder 14 MB + decoder 68 MB
+ tokenizer/configs ~3 MB), 99-language multilingual, accuracy close
to fp32 thanks to fp16 activations + 4-bit weights.

Model source: a local model directory OR a HuggingFace repo ID. The
HF cache stores it under `~/.cache/huggingface/hub/` and re-uses on
subsequent loads (idempotent). For the default, only the q4f16
variants are downloaded via `setup.ps1`'s `allow_patterns` filter —
the full fp32 weights are skipped.

CPU / GPU switching: ONNX Runtime provider. `device='cuda'` falls back
to CPU automatically if onnxruntime-gpu isn't usable.
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
# `onnx-community/whisper-base-ONNX` with q4f16 quantization:
#   * ~85 MB total (encoder 14 MB + decoder 68 MB + decoder_with_past
#     66 MB + tokenizer/configs ~3 MB — we share the decoder_with_past
#     buffer with the main decoder so the working set is ~85 MB)
#   * 99-language multilingual
#   * Quantization-aware: 4-bit weights, fp16 activations preserve
#     attention precision better than plain int8.
# Best balance of {small, accurate, multilingual, latest} in 2026
# under the 200 MB ceiling.
DEFAULT_MODEL = "onnx-community/whisper-base-ONNX"
# Quantization variant to load. The setup script's allow_patterns
# filter downloads only these files (saves ~200 MB on disk vs. fp32).
DEFAULT_QUANT = "q4f16"


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
        self._quant = DEFAULT_QUANT

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
        return (self._effective_model(), self._device, self._quant)

    async def configure(self, s) -> None:
        self._model_path = str(getattr(s, "stt_model_path", "") or "")
        self._device = str(getattr(s, "stt_device", "cpu"))
        self._language = str(getattr(s, "stt_language", "en"))
        self._idle_unload_sec = int(getattr(s, "stt_idle_unload_sec", 0))
        # Quantization variant. Allowed values match the file-name suffix
        # used by `onnx-community` repos: "q4f16", "fp16", "int8",
        # "q4", "bnb4", "uint8", "" (full fp32).
        self._quant = str(getattr(s, "stt_quant", DEFAULT_QUANT) or DEFAULT_QUANT)

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
        log.info("stt loading model=%s device=%s quant=%s",
                 model_id, self._device, self._quant)
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
            log.info("stt ready")
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
        """Sync — runs in the executor. Loads via optimum.onnxruntime.

        optimum's `ORTModelForSpeechSeq2Seq` handles the Whisper-shape
        ONNX export (split encoder + decoder + decoder_with_past) and
        drives the autoregressive generate() loop. Provider choice lands
        the inference on CPU or CUDA; the CPU provider is the silent
        fallback if CUDA init fails.
        """
        from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
        from transformers import AutoProcessor

        provider = ("CUDAExecutionProvider" if self._device == "cuda"
                    else "CPUExecutionProvider")
        subfolder, file_names = _split_for_quant(model_id, self._quant)

        # Build optimum kwargs. The quantization-specific files are
        # nested under `subfolder=onnx` in `onnx-community` repos.
        kwargs: dict[str, Any] = {"provider": provider}
        if subfolder:
            kwargs["subfolder"] = subfolder
        if file_names:
            kwargs.update(file_names)

        try:
            self._model = ORTModelForSpeechSeq2Seq.from_pretrained(model_id, **kwargs)
        except Exception as exc:
            # If the requested quant variant isn't present, fall through
            # to the un-suffixed (fp32) default. Quietly logged — the
            # model still loads, just bigger.
            log.warning("stt: quant=%s load failed (%s) — retrying default files",
                        self._quant, exc)
            self._model = ORTModelForSpeechSeq2Seq.from_pretrained(
                model_id, provider=provider,
                **({"subfolder": subfolder} if subfolder else {}),
            )
        self._processor = AutoProcessor.from_pretrained(model_id)

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
        gc.collect()

    # ── Transcription ────────────────────────────────────────────────

    async def transcribe(self, pcm: bytes, language: str | None = None,
                          beam_size: int | None = None) -> str:
        """Run STT on raw 16 kHz mono int16 PCM. Returns the text."""
        await self.ensure_loaded()
        self._last_used = time.monotonic()
        lang = (language or self._language or "en").strip() or "en"
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._exec, self._do_transcribe, pcm, lang)

    def _do_transcribe(self, pcm: bytes, language: str) -> str:
        """Sync — runs in the executor."""
        import numpy as np
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        inputs = self._processor(
            audio, sampling_rate=16000, return_tensors="pt",
        )
        # forced_decoder_ids steers Whisper toward the target language
        # without invoking the model's own language-detection pass.
        forced = self._processor.get_decoder_prompt_ids(
            language=language, task="transcribe",
        )
        generated = self._model.generate(
            inputs.input_features,
            forced_decoder_ids=forced,
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


# ── Helpers ──────────────────────────────────────────────────────────

def _split_for_quant(model_id: str, quant: str) -> tuple[str, dict]:
    """Return (subfolder, file-name kwargs) for the requested quant.

    `onnx-community/*-ONNX` repos lay out variants under `onnx/`:
        onnx/encoder_model.onnx              # fp32
        onnx/encoder_model_q4f16.onnx
        onnx/encoder_model_fp16.onnx
        ...

    Optimum's `from_pretrained` takes `encoder_file_name`,
    `decoder_file_name`, `decoder_with_past_file_name`. We pass them
    when `quant` is non-empty; otherwise let optimum pick the default.
    """
    is_onnx_community = "/" in model_id and "onnx" in model_id.split("/")[0].lower()
    subfolder = "onnx" if is_onnx_community else ""
    if not quant:
        return subfolder, {}
    suffix = quant.lower().strip("_-")
    if not suffix:
        return subfolder, {}
    return subfolder, {
        "encoder_file_name": f"encoder_model_{suffix}.onnx",
        "decoder_file_name": f"decoder_model_{suffix}.onnx",
        "decoder_with_past_file_name": f"decoder_with_past_model_{suffix}.onnx",
    }


# ── Module singleton ─────────────────────────────────────────────────

_ENGINE: STTEngine | None = None


def get_engine() -> STTEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = STTEngine()
    return _ENGINE
