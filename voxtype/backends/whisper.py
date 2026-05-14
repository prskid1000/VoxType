"""Whisper-family STT via HuggingFace transformers.

Handles any model that exports `WhisperForConditionalGeneration` —
the official OpenAI ladder, distil-whisper, large-v3-turbo, fine-tunes.
"""
from __future__ import annotations

import gc
import logging
from typing import Any

from voxtype.backends.stt_base import (
    LoadConfig, STTBackend, TranscribeOptions,
)

log = logging.getLogger("voxtype.backends.whisper")


class WhisperBackend(STTBackend):
    name = "whisper"
    default_model = "openai/whisper-base"
    family_tags = ("whisper",)

    def __init__(self) -> None:
        self._model: Any = None
        self._processor: Any = None
        self._torch_device: str = "cpu"
        self._torch_dtype: Any = None

    def load_sync(self, cfg: LoadConfig) -> None:
        import torch
        from transformers import WhisperForConditionalGeneration, AutoProcessor

        on_cuda = cfg.device == "cuda" and torch.cuda.is_available()
        if cfg.device == "cuda" and not on_cuda:
            log.warning("whisper: device=cuda requested but torch.cuda.is_available()=False — using CPU")
        self._torch_device = "cuda" if on_cuda else "cpu"

        pref = (cfg.dtype or "auto").lower()
        if pref == "auto":
            self._torch_dtype = torch.float16 if on_cuda else torch.float32
        elif pref == "fp16":
            if not on_cuda:
                log.warning("whisper: fp16 requested on CPU — falling back to fp32")
                self._torch_dtype = torch.float32
            else:
                self._torch_dtype = torch.float16
        elif pref == "bf16":
            self._torch_dtype = torch.bfloat16
        else:
            self._torch_dtype = torch.float32

        self._processor = AutoProcessor.from_pretrained(cfg.model_id)
        self._model = WhisperForConditionalGeneration.from_pretrained(
            cfg.model_id, torch_dtype=self._torch_dtype,
        ).to(self._torch_device)
        self._model.eval()

        if cfg.torch_compile:
            try:
                log.info("whisper torch.compile() — first call will pause for JIT")
                self._model = torch.compile(self._model, mode="reduce-overhead")
            except Exception as exc:  # noqa: BLE001
                log.warning("whisper: torch.compile failed (%s) — running uncompiled", exc)

        if cfg.warmup:
            try:
                import numpy as np
                dummy = np.zeros(16000, dtype=np.int16).tobytes()
                self.transcribe_sync(dummy, TranscribeOptions(
                    language="en", task="transcribe",
                    num_beams=1, initial_prompt="",
                ))
                log.info("whisper warmup ok")
            except Exception as exc:  # noqa: BLE001
                log.warning("whisper: warmup failed (%s)", exc)

    def unload_sync(self) -> None:
        self._model = None
        self._processor = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

    def transcribe_sync(self, pcm: bytes, opts: TranscribeOptions) -> str:
        import numpy as np
        import torch

        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        inputs = self._processor(
            audio, sampling_rate=16000, return_tensors="pt",
        )
        input_features = inputs.input_features.to(
            self._torch_device, dtype=self._torch_dtype,
        )

        gen_kwargs: dict = {
            "task": opts.task or "transcribe",
            "max_new_tokens": 440,
            "num_beams": max(1, int(opts.num_beams or 1)),
        }
        lang = (opts.language or "").lower()
        if lang and lang != "auto":
            gen_kwargs["language"] = lang
        if opts.initial_prompt:
            try:
                prompt_ids = self._processor.get_prompt_ids(
                    opts.initial_prompt, return_tensors="pt",
                ).to(self._torch_device)
                gen_kwargs["prompt_ids"] = prompt_ids
            except Exception as exc:  # noqa: BLE001
                log.debug("whisper: prompt_ids unsupported (%s)", exc)

        with torch.no_grad():
            generated = self._model.generate(input_features, **gen_kwargs)
        text = self._processor.batch_decode(generated, skip_special_tokens=True)[0]
        return (text or "").strip()

    def runtime_info(self) -> dict:
        return {"device": self._torch_device, "dtype": str(self._torch_dtype)}
