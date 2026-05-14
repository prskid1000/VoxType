"""Abstract base for TTS backends.

The engine wrapper (`voxtype.tts_engine`) calls these methods; each
concrete backend implements them with its library of choice (kokoro,
piper-tts, coqui-tts, etc.).

Voice catalog: every backend ships its own list. The UI rebuilds the
voice picker whenever the backend changes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator


@dataclass
class TTSLoadConfig:
    model_id: str          # HF repo id OR local path OR a backend-specific token
    device: str            # "cpu" | "cuda"
    warmup: bool
    torch_compile: bool


@dataclass
class VoiceEntry:
    """One row in the voice picker."""
    voice_id: str          # backend-specific key passed back on synthesise
    language: str          # human label, e.g. "American English"
    gender: str            # "F" | "M" | "" (unknown / non-binary)
    display_name: str      # short proper noun, e.g. "Heart"


class TTSBackend(ABC):
    """A concrete TTS engine implementation."""

    # ── Identity ─────────────────────────────────────────────────────

    name: str = ""
    default_model: str = ""
    default_voice: str = ""
    family_tags: tuple[str, ...] = ()
    sample_rate: int = 24000   # most backends; override in subclass if not

    # ── Catalog ──────────────────────────────────────────────────────

    @abstractmethod
    def voices(self) -> list[VoiceEntry]:
        """Full voice catalog."""

    def voice_ids(self) -> set[str]:
        return {v.voice_id for v in self.voices()}

    def voice_combo_options(self) -> list[tuple[str, str]]:
        """(value, label) tuples for a QComboBox."""
        return [
            (v.voice_id,
             f"{v.voice_id}  ·  {v.language} · {v.gender or '—'} · {v.display_name}")
            for v in self.voices()
        ]

    def supports(self, feature: str) -> bool:
        """UI capability flag. Features:
          - "speed": adjustable synthesis rate
          - "stream": yields per-chunk audio
          - "torch_compile"
        """
        return feature in {"speed", "stream", "torch_compile"}

    # ── Lifecycle ────────────────────────────────────────────────────

    @abstractmethod
    def load_sync(self, cfg: TTSLoadConfig) -> None:
        """Build the pipeline on the requested device. Blocking."""

    @abstractmethod
    def unload_sync(self) -> None:
        """Drop weights + clear CUDA cache. Blocking."""

    @abstractmethod
    def synth_chunks_sync(self, text: str, voice: str, speed: float) -> Iterator[bytes]:
        """Yield raw int16 PCM chunks (mono, self.sample_rate Hz). Blocking
        generator. The engine wraps this into WAV (or streams it via the
        HTTP server's chunked transfer)."""

    def runtime_info(self) -> dict:
        return {}
