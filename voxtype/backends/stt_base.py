"""Abstract base for STT backends.

The engine wrapper (`voxtype.stt_engine`) calls the methods on this
ABC; each concrete backend implements them with its library of
choice (transformers, NeMo, faster-whisper, etc.).

Threading: `load_sync()` and `transcribe_sync()` are CALLED from
the engine's single-thread executor, so they MUST be blocking
implementations — no asyncio inside. The engine is in charge of
the async glue.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LoadConfig:
    """Bundle of load-time options passed to backend.load_sync()."""
    model_id: str          # HF repo id OR local path
    device: str            # "cpu" | "cuda"
    dtype: str             # "auto" | "fp32" | "fp16" | "bf16"
    warmup: bool           # run a dummy inference after load
    torch_compile: bool    # JIT compile the model where supported
    attn_impl: str = "auto"  # "auto" | "sdpa" | "flash_attention_2" | "eager"


@dataclass
class OptionSpec:
    """One UI option, declared by a backend. The settings window
    renders a widget from this spec — no per-backend UI code needed."""
    key: str                          # storage key inside opts dict
    kind: str                         # "enum" | "bool" | "int" | "float" | "str" | "text"
    label: str                        # row label
    default: Any                      # initial value
    help: str = ""                    # row help text under the label
    choices: list[tuple[str, str]] = field(default_factory=list)  # for enum: (value, label)
    min: float | None = None          # for int/float
    max: float | None = None
    step: float | None = None
    rebuild: bool = False             # changing this forces engine reload


class STTBackend(ABC):
    """A concrete STT engine implementation."""

    # ── Identity ─────────────────────────────────────────────────────

    name: str = ""              # registry key, e.g. "generic" / "whisper"
    default_model: str = ""     # model id to use when settings is empty
    priority: int = 0           # higher = preferred when multiple backends
                                # claim a model id. 100 = specialist,
                                # 0 = universal fallback.

    # ── Catalog (UI introspection) ───────────────────────────────────

    def language_options(self) -> list[tuple[str, str]]:
        """(code, label) tuples for the language picker. Default = the
        Whisper 99-language table + Auto-detect. Backends that don't
        support multilingual decoding should override to return their
        single supported language (or `[]` to hide the picker)."""
        from voxtype.backends.shared import WHISPER_LANGUAGES
        return WHISPER_LANGUAGES

    def valid_language_codes(self) -> set[str]:
        return {c for c, _ in self.language_options()}

    def supports(self, feature: str) -> bool:
        """UI capability flag. Recognised features:
          - "dtype"           — `dtype` setting honoured
          - "torch_compile"   — torch.compile(model) supported
          - "multilingual"    — language picker meaningful
          - "task_translate"  — Whisper-style translate-to-EN mode
          - "initial_prompt"  — decoder bias text
          - "num_beams"       — beam search width >1
          - "bf16"            — bfloat16 dtype supported
          - "streaming"       — partial transcripts via chunk feed
        """
        return False

    def load_options(self) -> list[OptionSpec]:
        """Pre-load knobs that live above the [Load] button. Override
        if you have backend-specific load-time knobs (most backends
        don't — model/device/dtype are universal and live in
        AppSettings as first-class fields)."""
        return []

    def runtime_options(self) -> list[OptionSpec]:
        """Per-call knobs (language, beams, prompt, etc). The generic
        backend returns family-specific specs after the model is loaded
        and the family is known; before load it returns []."""
        return []

    # ── Lifecycle ────────────────────────────────────────────────────

    @abstractmethod
    def load_sync(self, cfg: LoadConfig) -> None:
        """Build the model on the requested device. Blocking."""

    @abstractmethod
    def unload_sync(self) -> None:
        """Drop weights + clear CUDA cache. Blocking."""

    @abstractmethod
    def transcribe_sync(self, pcm: bytes, opts: dict[str, Any]) -> str:
        """Transcribe 16 kHz mono int16 PCM. Blocking. Returns plain text.

        `opts` is the per-call dict from `AppSettings.stt_opts`.
        Implementations read only the keys they understand and ignore
        the rest, so adding a new family-specific option never breaks
        other backends."""

    # ── Optional introspection (for status / diagnostics) ────────────

    def runtime_info(self) -> dict:
        """Free-form dict surfaced by /health and the tray pill."""
        return {}

    def detected_family(self) -> str:
        """Human-readable family identifier shown in the UI after load.
        For the generic backend this is dynamic (e.g. "whisper",
        "wav2vec2", "mms", "seamless"); single-family backends just
        return their own name."""
        return self.name
