"""Abstract base for TTS backends.

The engine wrapper (`voxtype.tts_engine`) calls these methods; each
concrete backend implements them with its library of choice (kokoro,
transformers, piper-tts, coqui-tts, etc.).

Voice catalog: every backend ships its own list. The UI rebuilds the
voice picker whenever the backend changes. For generic backends the
voice list is *post-load* — `voices()` returns `[]` until the model
is loaded.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class TTSLoadConfig:
    model_id: str          # HF repo id OR local path OR backend-specific token
    device: str            # "cpu" | "cuda"
    warmup: bool
    torch_compile: bool
    attn_impl: str = "auto"  # "auto" | "sdpa" | "flash_attention_2" | "eager"


@dataclass
class VoiceEntry:
    """One row in the voice picker."""
    voice_id: str          # backend-specific key passed back on synthesise
    language: str          # human label, e.g. "American English"
    gender: str            # "F" | "M" | "" (unknown / non-binary)
    display_name: str      # short proper noun, e.g. "Heart"


@dataclass
class OptionSpec:
    """One UI option declared by a TTS backend. Mirrors the STT spec
    so the settings window uses a single renderer for both modalities."""
    key: str
    kind: str                         # "enum" | "bool" | "int" | "float" | "str" | "text"
    label: str
    default: Any
    help: str = ""
    choices: list[tuple[str, str]] = field(default_factory=list)
    min: float | None = None
    max: float | None = None
    step: float | None = None
    rebuild: bool = False


class TTSBackend(ABC):
    """A concrete TTS engine implementation."""

    # ── Identity ─────────────────────────────────────────────────────

    name: str = ""
    default_model: str = ""
    default_voice: str = ""
    priority: int = 0           # higher = preferred for a given model id
    sample_rate: int = 24000    # most backends; override per family

    # ── Catalog ──────────────────────────────────────────────────────

    def voices(self) -> list[VoiceEntry]:
        """Full voice catalog. For backends where voices come from the
        loaded model (Coqui, SpeechT5, Parler) this returns `[]` until
        `load_sync()` runs."""
        return []

    def voice_ids(self) -> set[str]:
        return {v.voice_id for v in self.voices()}

    def voice_combo_options(self) -> list[tuple[str, str]]:
        """(value, label) tuples for the voice QComboBox."""
        out: list[tuple[str, str]] = []
        for v in self.voices():
            label = v.voice_id
            if v.language or v.gender or v.display_name:
                label = (f"{v.voice_id}  ·  "
                          f"{v.language} · {v.gender or '—'} · {v.display_name}")
            out.append((v.voice_id, label))
        return out

    def supports(self, feature: str) -> bool:
        """UI capability flag. Recognised:
          - "speed"           — adjustable rate
          - "stream"          — native chunked synthesis
          - "torch_compile"
          - "multilingual"    — single model spans multiple languages
          - "voice_clone"     — accepts a reference audio file
          - "style_prompt"    — free-text style/description (Parler)
        """
        return False

    def load_options(self) -> list[OptionSpec]:
        return []

    def runtime_options(self) -> list[OptionSpec]:
        """Per-call knobs. Populated after load for families whose
        options depend on the actual loaded model."""
        return []

    # ── Lifecycle ────────────────────────────────────────────────────

    @abstractmethod
    def load_sync(self, cfg: TTSLoadConfig) -> None:
        """Build the pipeline on the requested device. Blocking."""

    @abstractmethod
    def unload_sync(self) -> None:
        """Drop weights + clear CUDA cache. Blocking."""

    @abstractmethod
    def synth_chunks_sync(self, text: str, voice: str,
                          opts: dict[str, Any]) -> Iterator[bytes]:
        """Yield raw int16 PCM chunks (mono, self.sample_rate Hz).

        Blocking generator. `opts` is the per-call dict from
        `AppSettings.tts_opts`; implementations read only the keys they
        understand. The engine wraps yielded chunks into a WAV (or
        streams them via the HTTP server's chunked transfer)."""

    def runtime_info(self) -> dict:
        return {}

    def detected_family(self) -> str:
        """Human-readable family identifier shown in the UI after load."""
        return self.name
