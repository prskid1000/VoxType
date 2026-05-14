"""Pluggable STT / TTS backends.

One generic STT backend + one generic TTS backend cover every
supported model family. Each backend sniffs the model's config.json
on load and dispatches internally to a family-specific handler
(Whisper, Wav2Vec2, MMS, Seamless, Moonshine, Kokoro, VITS, SpeechT5,
Bark, Parler, plus a `transformers.pipeline()` universal fallback).
"""
from __future__ import annotations

from voxtype.backends.generic_stt import GenericSTTBackend
from voxtype.backends.generic_tts import GenericTTSBackend
from voxtype.backends.stt_base import STTBackend
from voxtype.backends.tts_base import TTSBackend


def get_stt_backend() -> STTBackend:
    return GenericSTTBackend()


def get_tts_backend() -> TTSBackend:
    return GenericTTSBackend()
