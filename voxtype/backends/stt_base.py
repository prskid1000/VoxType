"""Abstract base for STT backends.

The engine wrapper (`voxtype.stt_engine`) calls the methods on this
ABC; each concrete backend implements them with its library of
choice (transformers, faster-whisper, NeMo, etc.).

Threading: `load_sync()` and `transcribe_sync()` are CALLED from
the engine's single-thread executor, so they MUST be blocking
implementations — no asyncio inside. The engine is in charge of
the async glue.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LoadConfig:
    """Bundle of load-time options passed to backend.load_sync()."""
    model_id: str          # HF repo id OR local path
    device: str            # "cpu" | "cuda"
    dtype: str             # "auto" | "fp32" | "fp16" | "bf16"
    warmup: bool           # run a dummy inference after load
    torch_compile: bool    # JIT compile the model where supported


@dataclass
class TranscribeOptions:
    """Bundle of per-call inference options."""
    language: str          # ISO 639-1 code, or "auto"
    task: str              # "transcribe" | "translate"
    num_beams: int         # >=1; 1 = greedy
    initial_prompt: str    # decoder bias text


class STTBackend(ABC):
    """A concrete STT engine implementation."""

    # ── Identity ─────────────────────────────────────────────────────

    name: str = ""              # e.g. "whisper" / "faster-whisper"
    default_model: str = ""     # model id to use when settings is empty
    family_tags: tuple[str, ...] = ()
    # Tags any HF model in this family should match (used by the Check
    # button's family validator). Empty tuple = skip family check.

    # ── Catalog (UI introspection) ───────────────────────────────────

    def language_options(self) -> list[tuple[str, str]]:
        """(code, label) tuples for a QComboBox. Override if the backend
        supports a different set than Whisper's 99 + auto."""
        from voxtype.backends.shared import WHISPER_LANGUAGES
        return WHISPER_LANGUAGES

    def valid_language_codes(self) -> set[str]:
        return {c for c, _ in self.language_options()}

    def supports(self, feature: str) -> bool:
        """UI capability flag. Features:
          - "task_translate": Whisper-style translate-to-EN mode
          - "initial_prompt": decoder bias text
          - "num_beams": beam search width >1
          - "torch_compile": torch.compile(model)
          - "bf16": bfloat16 dtype
        Override per backend; the base assumes Whisper-family capability set.
        """
        return feature in {
            "task_translate", "initial_prompt", "num_beams",
            "torch_compile", "bf16",
        }

    # ── Lifecycle ────────────────────────────────────────────────────

    @abstractmethod
    def load_sync(self, cfg: LoadConfig) -> None:
        """Build the model on the requested device. Blocking."""

    @abstractmethod
    def unload_sync(self) -> None:
        """Drop weights + clear CUDA cache. Blocking."""

    @abstractmethod
    def transcribe_sync(self, pcm: bytes, opts: TranscribeOptions) -> str:
        """Transcribe 16 kHz mono int16 PCM. Blocking. Returns plain text."""

    # ── Optional introspection (for status / diagnostics) ────────────

    def runtime_info(self) -> dict:
        """Free-form dict surfaced by /health and the tray pill."""
        return {}
