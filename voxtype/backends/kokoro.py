"""Kokoro TTS via the official `kokoro` PyPI package.

Kokoro-82M with 54 voices across 9 lang_codes. Lightweight (~327 MB),
fast on CPU, faster on GPU. Per-sentence streaming is native to
KPipeline so we yield as the model produces.
"""
from __future__ import annotations

import gc
import logging
from typing import Any, Iterator

from voxtype.backends.tts_base import (
    TTSBackend, TTSLoadConfig, VoiceEntry,
)

log = logging.getLogger("voxtype.backends.kokoro")


# ── Voice catalog ────────────────────────────────────────────────────
# Same canonical list, kept here so the backend module is self-contained.

_VOICES: dict[str, list[tuple[str, str, str]]] = {
    "American English": [
        ("af_alloy",   "F", "Alloy"),   ("af_aoede",   "F", "Aoede"),
        ("af_bella",   "F", "Bella"),   ("af_heart",   "F", "Heart"),
        ("af_jessica", "F", "Jessica"), ("af_kore",    "F", "Kore"),
        ("af_nicole",  "F", "Nicole"),  ("af_nova",    "F", "Nova"),
        ("af_river",   "F", "River"),   ("af_sarah",   "F", "Sarah"),
        ("af_sky",     "F", "Sky"),
        ("am_adam",    "M", "Adam"),    ("am_echo",    "M", "Echo"),
        ("am_eric",    "M", "Eric"),    ("am_fenrir",  "M", "Fenrir"),
        ("am_liam",    "M", "Liam"),    ("am_michael", "M", "Michael"),
        ("am_onyx",    "M", "Onyx"),    ("am_puck",    "M", "Puck"),
        ("am_santa",   "M", "Santa"),
    ],
    "British English": [
        ("bf_alice",    "F", "Alice"),    ("bf_emma",   "F", "Emma"),
        ("bf_isabella", "F", "Isabella"), ("bf_lily",   "F", "Lily"),
        ("bm_daniel",   "M", "Daniel"),   ("bm_fable",  "M", "Fable"),
        ("bm_george",   "M", "George"),   ("bm_lewis",  "M", "Lewis"),
    ],
    "Spanish": [
        ("ef_dora",  "F", "Dora"),
        ("em_alex",  "M", "Alex"), ("em_santa", "M", "Santa"),
    ],
    "French": [("ff_siwis", "F", "Siwis")],
    "Hindi": [
        ("hf_alpha", "F", "Alpha"), ("hf_beta",  "F", "Beta"),
        ("hm_omega", "M", "Omega"), ("hm_psi",   "M", "Psi"),
    ],
    "Italian": [
        ("if_sara", "F", "Sara"), ("im_nicola", "M", "Nicola"),
    ],
    "Japanese": [
        ("jf_alpha", "F", "Alpha"), ("jf_gongitsune", "F", "Gongitsune"),
        ("jf_nezumi", "F", "Nezumi"), ("jf_tebukuro", "F", "Tebukuro"),
        ("jm_kumo", "M", "Kumo"),
    ],
    "Brazilian Portuguese": [
        ("pf_dora",  "F", "Dora"),
        ("pm_alex",  "M", "Alex"), ("pm_santa", "M", "Santa"),
    ],
    "Mandarin Chinese": [
        ("zf_xiaobei",  "F", "Xiaobei"),  ("zf_xiaoni",   "F", "Xiaoni"),
        ("zf_xiaoxiao", "F", "Xiaoxiao"), ("zf_xiaoyi",   "F", "Xiaoyi"),
        ("zm_yunjian",  "M", "Yunjian"),  ("zm_yunxi",    "M", "Yunxi"),
        ("zm_yunxia",   "M", "Yunxia"),   ("zm_yunyang",  "M", "Yunyang"),
    ],
}


class KokoroBackend(TTSBackend):
    name = "kokoro"
    default_model = "hexgrad/Kokoro-82M"
    default_voice = "af_heart"
    family_tags = ("kokoro",)
    sample_rate = 24000

    def __init__(self) -> None:
        self._pipeline: Any = None
        self._torch_device: str = "cpu"

    def voices(self) -> list[VoiceEntry]:
        out: list[VoiceEntry] = []
        for lang, items in _VOICES.items():
            for vid, gender, name in items:
                out.append(VoiceEntry(vid, lang, gender, name))
        return out

    def supports(self, feature: str) -> bool:
        return feature in {"speed", "stream", "torch_compile"}

    def load_sync(self, cfg: TTSLoadConfig) -> None:
        import torch
        from kokoro import KPipeline

        on_cuda = cfg.device == "cuda" and torch.cuda.is_available()
        if cfg.device == "cuda" and not on_cuda:
            log.warning("kokoro: cuda requested but unavailable — using CPU")
        self._torch_device = "cuda" if on_cuda else "cpu"

        self._pipeline = KPipeline(
            lang_code="a", repo_id=cfg.model_id, device=self._torch_device,
        )

        if cfg.torch_compile:
            try:
                inner = getattr(self._pipeline, "model", None)
                if inner is not None:
                    log.info("kokoro torch.compile() — first synth will pause for JIT")
                    self._pipeline.model = torch.compile(inner, mode="reduce-overhead")
            except Exception as exc:  # noqa: BLE001
                log.warning("kokoro: torch.compile failed (%s)", exc)

        if cfg.warmup:
            try:
                # Drain one synthesis with the default voice.
                for _ in self.synth_chunks_sync("Voxtype ready.", self.default_voice, 1.0):
                    pass
                log.info("kokoro warmup ok")
            except Exception as exc:  # noqa: BLE001
                log.warning("kokoro: warmup failed (%s)", exc)

    def unload_sync(self) -> None:
        self._pipeline = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

    def synth_chunks_sync(self, text: str, voice: str, speed: float) -> Iterator[bytes]:
        import numpy as np
        import torch

        v = voice or self.default_voice
        spd = float(speed) if speed and speed > 0 else 1.0
        for _, _, audio in self._pipeline(text, voice=v, speed=spd):
            if audio is None:
                continue
            if isinstance(audio, torch.Tensor):
                arr = audio.detach().cpu().to(torch.float32).numpy()
            else:
                arr = np.asarray(audio, dtype=np.float32)
            arr = arr.reshape(-1)
            np.clip(arr, -1.0, 1.0, out=arr)
            yield (arr * 32767.0).astype(np.int16).tobytes()

    def runtime_info(self) -> dict:
        return {"device": self._torch_device, "sample_rate": self.sample_rate}
