"""Whisper-family STT via faster-whisper (CTranslate2).

Same Whisper architecture, ~4× faster on GPU and lower VRAM than
the transformers reference. CTranslate2 picks its compute_type
independently of torch's dtype: float16 / int8_float16 on CUDA,
int8 on CPU.

Loaded models live in ~/.cache/huggingface/hub like the transformers
backend, so switching backends doesn't re-download weights.
"""
from __future__ import annotations

import gc
import logging
from typing import Any

from voxtype.backends.stt_base import (
    LoadConfig, STTBackend, TranscribeOptions,
)

log = logging.getLogger("voxtype.backends.faster_whisper")


class FasterWhisperBackend(STTBackend):
    name = "faster-whisper"
    default_model = "openai/whisper-base"
    family_tags = ("whisper",)

    def __init__(self) -> None:
        self._model: Any = None
        self._device: str = "cpu"
        self._compute_type: str = "int8"

    def supports(self, feature: str) -> bool:
        # faster-whisper handles its own dtype via compute_type; torch.compile
        # doesn't apply (CT2 has its own optimized kernels). Everything
        # else maps to Whisper-family semantics.
        if feature in {"torch_compile", "bf16"}:
            return False
        return feature in {"task_translate", "initial_prompt", "num_beams"}

    def load_sync(self, cfg: LoadConfig) -> None:
        from faster_whisper import WhisperModel

        # Resolve device + compute_type
        if cfg.device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    log.warning("faster-whisper: cuda requested but unavailable — using CPU")
                    self._device = "cpu"
                else:
                    self._device = "cuda"
            except Exception:
                self._device = "cpu"
        else:
            self._device = "cpu"

        # Map UI dtype hint to a CT2 compute_type. fp16/auto on GPU,
        # int8 on CPU for the speed boost (CT2's strength).
        pref = (cfg.dtype or "auto").lower()
        if self._device == "cuda":
            self._compute_type = "float16" if pref in {"auto", "fp16"} else \
                                  "bfloat16" if pref == "bf16" else "float32"
        else:
            self._compute_type = "int8" if pref in {"auto", "fp16"} else "float32"

        log.info("faster-whisper loading model=%s device=%s compute=%s",
                 cfg.model_id, self._device, self._compute_type)
        self._model = WhisperModel(
            cfg.model_id, device=self._device, compute_type=self._compute_type,
        )

        if cfg.warmup:
            try:
                import numpy as np
                dummy = np.zeros(16000, dtype=np.int16).tobytes()
                self.transcribe_sync(dummy, TranscribeOptions(
                    language="en", task="transcribe",
                    num_beams=1, initial_prompt="",
                ))
                log.info("faster-whisper warmup ok")
            except Exception as exc:  # noqa: BLE001
                log.warning("faster-whisper: warmup failed (%s)", exc)

    def unload_sync(self) -> None:
        self._model = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

    def transcribe_sync(self, pcm: bytes, opts: TranscribeOptions) -> str:
        import numpy as np

        # CT2 expects float32 mono at 16 kHz.
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        lang = (opts.language or "").lower()
        language_arg = None if (not lang or lang == "auto") else lang

        kwargs: dict = {
            "beam_size": max(1, int(opts.num_beams or 1)),
            "language": language_arg,
            "task": opts.task or "transcribe",
        }
        if opts.initial_prompt:
            kwargs["initial_prompt"] = opts.initial_prompt

        segments, _info = self._model.transcribe(audio, **kwargs)
        # `segments` is a generator — materialise it.
        return " ".join(s.text.strip() for s in segments).strip()

    def runtime_info(self) -> dict:
        return {"device": self._device, "compute_type": self._compute_type}
