"""Model-family detection for the generic STT/TTS backends.

The generic backends accept *any* HuggingFace repo id or local model
path. This module figures out which architectural family the model
belongs to so the dispatcher can pick the right loader class and so
the UI can render the right per-family option widgets.

Detection sources (in priority order):
  1. Local config.json `model_type` / `architectures`
  2. Cached HF metadata (model_type / pipeline_tag / tags)
  3. Repo-id heuristics (last-resort substring match)

Detection is sniff-only — no model weights are downloaded. The HF
config.json is ~2 KB and is fetched on demand.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from voxtype.backends.stt_base import OptionSpec as STTOptionSpec
from voxtype.backends.tts_base import OptionSpec as TTSOptionSpec, VoiceEntry

log = logging.getLogger("voxtype.family_detect")


# ── Family names (constants used by generic backends + UI) ───────────

# STT families
STT_WHISPER     = "whisper"        # WhisperForConditionalGeneration (seq2seq)
STT_WAV2VEC2    = "wav2vec2"       # Wav2Vec2/HuBERT/WavLM CTC
STT_MMS         = "mms"            # Wav2Vec2-CTC w/ language adapter
STT_SEAMLESS    = "seamless"       # SeamlessM4T(v2) speech-to-text
STT_MOONSHINE   = "moonshine"      # Moonshine encoder-decoder
STT_S2T         = "speech_to_text" # Speech2Text legacy seq2seq
STT_SPEECHT5    = "speecht5_asr"   # SpeechT5 ASR head
STT_QWEN_AUDIO  = "qwen_audio"     # Qwen2-Audio / Qwen3-ASR multimodal
STT_VOXTRAL     = "voxtral"        # Mistral Voxtral — VoxtralForConditionalGeneration
STT_GRANITE     = "granite_speech" # IBM Granite-Speech
STT_PHI4MM      = "phi4_multimodal"# Microsoft Phi-4-Multimodal
STT_VIBEVOICE   = "vibevoice_asr"  # Microsoft VibeVoice ASR
STT_GENERIC     = "generic_asr"    # transformers.pipeline fallback

# TTS families
TTS_KOKORO      = "kokoro"         # kokoro PyPI package (KPipeline)
TTS_VITS        = "vits"           # VITS / MMS-TTS
TTS_SPEECHT5    = "speecht5_tts"   # SpeechT5 + speaker embeddings
TTS_BARK        = "bark"           # Bark
TTS_PARLER      = "parler"         # Parler-TTS (free-text style)
TTS_XTTS        = "xtts"           # Coqui XTTS (voice cloning)
TTS_QWEN_TTS    = "qwen_tts"       # Qwen3-TTS audio decoder
TTS_ORPHEUS     = "orpheus"        # Canopy Orpheus — Llama backbone + SNAC vocoder
TTS_CSM         = "csm"            # Sesame CSM — native transformers model_type=csm
TTS_HIGGS       = "higgs"          # Boson Higgs-Audio v2
TTS_VIBEVOICE   = "vibevoice_tts"  # Microsoft VibeVoice
TTS_GENERIC     = "generic_tts"    # transformers.pipeline fallback


# ── Detection ────────────────────────────────────────────────────────

def _read_local_config(model_id: str) -> dict[str, Any]:
    """Look for a `config.json` alongside the entered path (file or
    directory). Returns an empty dict on any failure."""
    p = Path(model_id).expanduser()
    if not p.exists():
        return {}
    cfg_path = p / "config.json" if p.is_dir() else p.parent / "config.json"
    try:
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        pass
    # Kokoro local checkouts usually have a `voices/` dir but no
    # transformers-style config.json. Encode that hint.
    if p.is_dir() and (p / "voices").is_dir():
        return {"_kokoro_local": True}
    return {}


def _hf_get_json(url: str) -> tuple[int, dict[str, Any] | None, str]:
    """GET a JSON URL with 3s timeout. Returns (status, body or None,
    error). Uses HTTPError.code on 4xx/5xx so callers can distinguish
    not-found / gated / network failure."""
    req = urllib.request.Request(url, headers={
        "Accept": "application/json", "User-Agent": "voxtype/1.0",
    })
    try:
        with urllib.request.urlopen(req, timeout=3) as r:
            if r.status == 200:
                return 200, json.loads(r.read().decode("utf-8")), ""
            return r.status, None, f"http {r.status}"
    except urllib.error.HTTPError as e:
        return e.code, None, f"http {e.code}"
    except Exception as e:  # noqa: BLE001
        return 0, None, str(e) or e.__class__.__name__


def _fetch_hf_config(repo_id: str) -> dict[str, Any]:
    """Pull the HF model card metadata + config.json. Swallows
    errors — returns an empty dict on failure."""
    out: dict[str, Any] = {}
    repo = repo_id.strip("/")
    status, body, _ = _hf_get_json(f"https://huggingface.co/api/models/{repo}")
    if status == 200 and isinstance(body, dict):
        out["card"] = body
    status, body, _ = _hf_get_json(
        f"https://huggingface.co/{repo}/resolve/main/config.json")
    if status == 200 and isinstance(body, dict):
        out["config"] = body
    return out


@dataclass
class ModelCheck:
    """Result of verifying a model id / local path."""
    valid: bool          # exists and is loadable as a model
    source: str          # "local" | "hf" | "none"
    gated: bool = False  # HF repo requires sign-in
    error: str = ""      # short reason when valid is False
    family: str = ""     # detected family ("" = unknown architecture)


def verify_model_id(model_id: str, *, stt: bool) -> ModelCheck:
    """Confirm the entered model id / path is real and detect its
    family in one shot. Used by the UI's Detect button.

    Order:
      1. Local path? — must exist AND have a `config.json` (or a
         `voices/` dir for Kokoro). Return local + family.
      2. HF repo? — hit `/api/models/<id>`. 200 = valid, 404 = not
         found, 401/403 = gated, else network error. If valid, also
         pull config.json so family detection can use it.
    """
    mid = (model_id or "").strip()
    if not mid:
        return ModelCheck(False, "none", error="empty")

    # ── Local path ──────────────────────────────────────────────────
    p = Path(mid).expanduser()
    if p.exists():
        cfg = _read_local_config(mid)
        if not cfg:
            return ModelCheck(False, "local",
                               error="no config.json in directory")
        fam = _family_from_config(cfg, mid, stt=stt)
        return ModelCheck(True, "local", family=fam)

    # ── Looks like a path that doesn't exist? ──────────────────────
    # Reject anything with backslash, drive-letter, or absolute "/" —
    # those are clearly meant as local paths, not HF repo ids.
    if ("\\" in mid or mid.startswith("/")
            or (len(mid) >= 2 and mid[1] == ":")):
        return ModelCheck(False, "local", error="path not found")

    # ── HuggingFace repo ────────────────────────────────────────────
    repo = mid.strip("/")
    # HF repo ids are "<org>/<name>" with no whitespace.
    if "/" not in repo or " " in repo:
        return ModelCheck(False, "none", error="not a valid HF repo id")

    status, card, err = _hf_get_json(f"https://huggingface.co/api/models/{repo}")
    if status == 404:
        return ModelCheck(False, "hf", error="repo not found on HuggingFace")
    if status in (401, 403):
        return ModelCheck(False, "hf", gated=True,
                           error="gated — sign-in required (huggingface-cli login)")
    if status != 200 or not isinstance(card, dict):
        return ModelCheck(False, "hf",
                           error=f"unreachable ({err or 'network error'})")

    # Repo exists — fetch config.json best-effort for family detect.
    bundle: dict[str, Any] = {"card": card}
    cfg_status, cfg_body, _ = _hf_get_json(
        f"https://huggingface.co/{repo}/resolve/main/config.json")
    if cfg_status == 200 and isinstance(cfg_body, dict):
        bundle["config"] = cfg_body
    fam = _family_from_config(bundle, mid, stt=stt)
    return ModelCheck(True, "hf", family=fam)


def _family_from_config(cfg: dict[str, Any], repo_id: str, *, stt: bool) -> str:
    """Match a config.json blob to a family. `cfg` may be the bare HF
    config.json or {'card': ..., 'config': ...} from `_fetch_hf_config`."""
    rid = (repo_id or "").lower()

    # Unwrap if it's the {'card':..., 'config':...} bundle.
    card = cfg.get("card") if isinstance(cfg.get("card"), dict) else {}
    inner = cfg.get("config") if isinstance(cfg.get("config"), dict) else cfg

    model_type = str(inner.get("model_type") or "").lower()
    arch_list = inner.get("architectures") or []
    if isinstance(arch_list, list):
        archs = [str(a).lower() for a in arch_list]
    else:
        archs = [str(arch_list).lower()]
    archs_blob = " ".join(archs)

    pipeline_tag = str(card.get("pipeline_tag") or "").lower()
    tags = [str(t).lower() for t in (card.get("tags") or [])]
    library = str(card.get("library_name") or "").lower()

    if inner.get("_kokoro_local"):
        return TTS_KOKORO

    if stt:
        # ── STT ─────────────────────────────────────────────────────
        if model_type == "whisper" or "whisper" in archs_blob:
            return STT_WHISPER
        if model_type == "moonshine" or "moonshine" in archs_blob:
            return STT_MOONSHINE
        if model_type.startswith("seamless") or "seamless" in archs_blob:
            return STT_SEAMLESS
        if model_type == "speech_to_text" or "speech2text" in archs_blob:
            return STT_S2T
        if model_type == "speecht5" or "speecht5" in archs_blob:
            # SpeechT5 has both ASR and TTS heads. Disambiguate via
            # architectures (ForSpeechToText vs ForTextToSpeech).
            if "forspeechtotext" in archs_blob or "asr" in tags:
                return STT_SPEECHT5
            return ""
        if model_type in {"wav2vec2", "hubert", "wavlm", "unispeech",
                           "unispeech_sat", "wav2vec2_conformer"}:
            # MMS = Wav2Vec2 with per-language adapters. Detect by tag.
            if "mms" in rid or "mms" in tags:
                return STT_MMS
            return STT_WAV2VEC2
        if model_type == "voxtral" or "voxtral" in archs_blob:
            return STT_VOXTRAL
        if model_type == "granite_speech" or "granitespeech" in archs_blob:
            return STT_GRANITE
        if model_type in {"phi4_multimodal", "phi4mm"} or "phi4multimodal" in archs_blob:
            return STT_PHI4MM
        if "vibevoice" in model_type or "vibevoice" in archs_blob:
            return STT_VIBEVOICE
        if model_type in {"qwen2_audio", "qwen_audio"} or "qwen2audio" in archs_blob:
            return STT_QWEN_AUDIO
        # Whisper-ish fallback: any HF model tagged
        # automatic-speech-recognition that we can't pin to a family.
        if pipeline_tag == "automatic-speech-recognition":
            return STT_GENERIC
        # Heuristic last-resort
        if "whisper" in rid:
            return STT_WHISPER
        if "wav2vec2" in rid:
            return STT_WAV2VEC2
        return ""

    # ── TTS ────────────────────────────────────────────────────────
    if "kokoro" in rid or "kokoro" in tags or library == "kokoro":
        return TTS_KOKORO
    if model_type == "vits" or "vits" in archs_blob or "mms-tts" in rid:
        return TTS_VITS
    if model_type == "speecht5" or "speecht5" in archs_blob:
        if "fortexttospeech" in archs_blob or "tts" in tags or pipeline_tag == "text-to-speech":
            return TTS_SPEECHT5
        return ""
    if model_type == "bark" or "bark" in archs_blob:
        return TTS_BARK
    if "parler" in rid or "parler" in archs_blob or model_type.startswith("parler"):
        return TTS_PARLER
    if "xtts" in rid or "coqui" in tags:
        return TTS_XTTS
    if "orpheus" in rid or "orpheus" in archs_blob or "orpheus" in tags:
        return TTS_ORPHEUS
    if model_type == "csm" or "csm" in archs_blob or ("csm" in rid and "sesame" in rid):
        return TTS_CSM
    if "higgs" in rid or "higgs" in archs_blob:
        return TTS_HIGGS
    if "vibevoice" in model_type or "vibevoice" in archs_blob or "vibevoice" in rid:
        return TTS_VIBEVOICE
    if "qwen" in rid and ("tts" in rid or pipeline_tag == "text-to-speech"):
        return TTS_QWEN_TTS
    if pipeline_tag == "text-to-speech":
        return TTS_GENERIC
    return ""


def _stt_from_repo_id(model_id: str) -> str:
    """Pure repo-id substring heuristic. Cheap; runs synchronously
    on every textChanged in the UI."""
    rid = (model_id or "").lower()
    # Order matters: MMS is a wav2vec2-derivative, check it first.
    if "mms" in rid and ("asr" in rid or "/mms-" in rid or "1b" in rid):
        return STT_MMS
    if "whisper" in rid:
        return STT_WHISPER
    if "moonshine" in rid:
        return STT_MOONSHINE
    if "seamless" in rid:
        return STT_SEAMLESS
    if "voxtral" in rid:
        return STT_VOXTRAL
    if "granite" in rid and "speech" in rid:
        return STT_GRANITE
    if "phi-4-multimodal" in rid or "phi4-multimodal" in rid or "phi-4-mm" in rid:
        return STT_PHI4MM
    if "vibevoice" in rid:
        return STT_VIBEVOICE
    if "wav2vec2" in rid or "hubert" in rid or "wavlm" in rid:
        return STT_WAV2VEC2
    if "speecht5" in rid and "asr" in rid:
        return STT_SPEECHT5
    if "speech_to_text" in rid or "s2t" in rid:
        return STT_S2T
    if "qwen" in rid and "audio" in rid:
        return STT_QWEN_AUDIO
    return ""


def _tts_from_repo_id(model_id: str) -> str:
    rid = (model_id or "").lower()
    if "kokoro" in rid:
        return TTS_KOKORO
    if "mms-tts" in rid or "mms_tts" in rid or "/vits-" in rid or rid.endswith("-vits"):
        return TTS_VITS
    if "speecht5" in rid and "tts" in rid:
        return TTS_SPEECHT5
    if "bark" in rid:
        return TTS_BARK
    if "parler" in rid:
        return TTS_PARLER
    if "xtts" in rid or "coqui" in rid:
        return TTS_XTTS
    if "orpheus" in rid:
        return TTS_ORPHEUS
    if "sesame" in rid and "csm" in rid:
        return TTS_CSM
    if "higgs" in rid and "audio" in rid:
        return TTS_HIGGS
    if "vibevoice" in rid:
        return TTS_VIBEVOICE
    if "qwen" in rid and "tts" in rid:
        return TTS_QWEN_TTS
    return ""


def detect_stt_family_fast(model_id: str) -> str:
    """Synchronous repo-id + local-config heuristic. No HTTP call.
    Used by the UI to populate the family pill / voice picker the
    instant the user types a model id."""
    if not model_id:
        return ""
    local = _read_local_config(model_id)
    if local:
        fam = _family_from_config(local, model_id, stt=True)
        if fam:
            return fam
    # Repo-id heuristic — covers the common-case HF ids cheaply.
    fam = _stt_from_repo_id(model_id)
    if fam:
        return fam
    # Try the config-driven matcher with empty config (last resort).
    return _family_from_config({}, model_id, stt=True)


def detect_tts_family_fast(model_id: str) -> str:
    if not model_id:
        return ""
    local = _read_local_config(model_id)
    if local:
        fam = _family_from_config(local, model_id, stt=False)
        if fam:
            return fam
    fam = _tts_from_repo_id(model_id)
    if fam:
        return fam
    return _family_from_config({}, model_id, stt=False)


def detect_stt_family(model_id: str) -> str:
    """Detect family for an STT model id. Empty string = unknown.
    Tries local config first, then HF metadata."""
    if not model_id:
        return ""
    local = _read_local_config(model_id)
    if local:
        fam = _family_from_config(local, model_id, stt=True)
        if fam:
            return fam
    remote = _fetch_hf_config(model_id)
    if remote:
        fam = _family_from_config(remote, model_id, stt=True)
        if fam:
            return fam
    return ""


def detect_tts_family(model_id: str) -> str:
    if not model_id:
        return ""
    local = _read_local_config(model_id)
    if local:
        fam = _family_from_config(local, model_id, stt=False)
        if fam:
            return fam
    remote = _fetch_hf_config(model_id)
    if remote:
        fam = _family_from_config(remote, model_id, stt=False)
        if fam:
            return fam
    return ""


# ── Per-family option specs (UI driver) ──────────────────────────────

# Whisper task choices
_TASK_CHOICES = [("transcribe", "Transcribe"), ("translate", "Translate → EN")]


def stt_runtime_options(family: str) -> list[STTOptionSpec]:
    """Family-specific per-call options shown in the STT "Advanced"
    section. The universal fields (language, device, dtype, etc.)
    are still rendered first-class above this."""
    if family == STT_WHISPER:
        return [
            STTOptionSpec("task", "enum", "Task",
                "transcribe",
                help="transcribe = output source language. translate = "
                     "output English regardless of source (Whisper's "
                     "built-in translation mode).",
                choices=_TASK_CHOICES),
            STTOptionSpec("num_beams", "int", "Beams", 1,
                help="Beam-search width. 1 = greedy decoding, fastest. "
                     "Higher = lower WER but ~N× slower.",
                min=1, max=10),
            STTOptionSpec("temperature", "float", "Temperature", 0.0,
                help="0.0 = deterministic (recommended). Raise to 0.2-0.4 "
                     "as a fallback for noisy mics where greedy decoding "
                     "stalls.",
                min=0.0, max=1.0, step=0.05),
            STTOptionSpec("repetition_penalty", "float", "Rep. Penalty", 1.0,
                help="1.0 = off. Bump to 1.1-1.3 if Whisper loops on "
                     "filler sounds (um, hum, breath).",
                min=1.0, max=2.0, step=0.05),
            STTOptionSpec("initial_prompt", "str", "Initial Prompt", "",
                help="Free text fed to the decoder to bias decoding. "
                     "Useful for jargon, acronyms, proper names. "
                     "Empty = no bias."),
        ]
    if family == STT_SEAMLESS:
        return [
            STTOptionSpec("task", "enum", "Task",
                "transcribe",
                help="transcribe = source language. translate → EN.",
                choices=_TASK_CHOICES),
            STTOptionSpec("num_beams", "int", "Beams", 5,
                help="Seamless defaults to beam=5 (recommended).",
                min=1, max=10),
            STTOptionSpec("tgt_lang", "str", "Target Lang (ISO-639-3)", "",
                help="Override the output language code. Empty = derive "
                     "from universal Language + Task. Use ISO-639-3 "
                     "(eng, fra, spa, cmn, …)."),
        ]
    if family == STT_MOONSHINE:
        return [
            STTOptionSpec("num_beams", "int", "Beams", 1,
                help="Moonshine is English-only; beams >1 buys little.",
                min=1, max=5),
        ]
    if family == STT_VOXTRAL:
        return [
            STTOptionSpec("task", "enum", "Task", "transcribe",
                help="transcribe = source language. translate → EN.",
                choices=_TASK_CHOICES),
            STTOptionSpec("temperature", "float", "Temperature", 0.0,
                help="Sampling temperature. 0.0 = greedy decoding.",
                min=0.0, max=1.0, step=0.05),
            STTOptionSpec("prompt", "text", "Instruction", "",
                help="Optional system instruction. Voxtral is an "
                     "instruction-tuned audio LLM — leave empty for "
                     "plain transcription, or write e.g. 'Summarise:' / "
                     "'Translate to French:' for instruction-conditioned "
                     "transcription."),
        ]
    if family == STT_GRANITE:
        return [
            STTOptionSpec("task", "enum", "Task", "transcribe",
                help="ASR or speech translation (AST → English).",
                choices=_TASK_CHOICES),
            STTOptionSpec("prompt", "text", "Instruction", "",
                help="Optional instruction prompt for Granite-Speech."),
        ]
    if family == STT_PHI4MM:
        return [
            STTOptionSpec("prompt", "text", "Instruction",
                "Transcribe the audio.",
                help="Phi-4 is an instruction LLM — the prompt drives "
                     "behaviour. 'Transcribe the audio.' is the default "
                     "ASR prompt."),
            STTOptionSpec("temperature", "float", "Temperature", 0.0,
                help="0.0 = greedy.", min=0.0, max=1.0, step=0.05),
        ]
    if family == STT_QWEN_AUDIO:
        return [
            STTOptionSpec("prompt", "text", "Instruction",
                "Transcribe the audio.",
                help="Qwen2-Audio is instruction-tuned. Override to "
                     "translate, punctuate, summarise, etc."),
            STTOptionSpec("temperature", "float", "Temperature", 0.0,
                help="0.0 = greedy.", min=0.0, max=1.0, step=0.05),
            STTOptionSpec("top_p", "float", "Top-p", 0.9,
                help="Nucleus sampling. Only used when temperature > 0.",
                min=0.0, max=1.0, step=0.05),
        ]
    if family in {STT_WAV2VEC2, STT_MMS, STT_S2T, STT_SPEECHT5,
                   STT_VIBEVOICE, STT_GENERIC}:
        # CTC and pipeline-fallback families have no extra knobs —
        # language (universal) is enough.
        return []
    return []


def stt_capabilities(family: str) -> set[str]:
    """Capability flags advertised by a given family — drives the
    UI's universal-row visibility (e.g. dtype dropdown, language combo)."""
    base = {"dtype", "torch_compile", "attn_impl"}
    if family == STT_WHISPER:
        return base | {"multilingual", "task_translate", "initial_prompt",
                        "num_beams", "bf16", "chunk_length"}
    if family == STT_SEAMLESS:
        return base | {"multilingual", "task_translate", "num_beams", "bf16"}
    if family == STT_MMS:
        return base | {"multilingual"}
    if family == STT_MOONSHINE:
        return base | {"num_beams"}
    if family == STT_S2T:
        return base | {"multilingual", "num_beams"}
    if family == STT_SPEECHT5:
        return base
    if family == STT_WAV2VEC2:
        return base   # English-only or per-language repos
    if family == STT_QWEN_AUDIO:
        return base | {"multilingual", "bf16", "prompt"}
    if family == STT_VOXTRAL:
        return base | {"multilingual", "task_translate", "bf16", "prompt"}
    if family == STT_GRANITE:
        return base | {"multilingual", "task_translate", "bf16", "prompt"}
    if family == STT_PHI4MM:
        return base | {"multilingual", "bf16", "prompt"}
    if family == STT_VIBEVOICE:
        return base | {"multilingual", "bf16"}
    if family == STT_GENERIC:
        return base | {"chunk_length"}
    return set()


def tts_runtime_options(family: str) -> list[TTSOptionSpec]:
    if family == TTS_PARLER:
        return [
            TTSOptionSpec("style", "text", "Style Description",
                "A clear, calm voice with neutral pace.",
                help="Free-text description of the voice and delivery. "
                     "Parler-TTS conditions on this prompt — e.g. "
                     "'A male speaker with a slightly low-pitched voice, "
                     "speaking quickly in a small room.'"),
            TTSOptionSpec("temperature", "float", "Temperature", 1.0,
                help="Sampling temperature. Lower = more conservative.",
                min=0.1, max=1.5, step=0.05),
            TTSOptionSpec("max_new_tokens", "int", "Max Tokens", 2580,
                help="Cap on generated audio tokens. Raise if Parler "
                     "truncates mid-sentence on long inputs.",
                min=200, max=8192),
        ]
    if family == TTS_SPEECHT5:
        return [
            TTSOptionSpec("speaker_embedding", "str", "Speaker Embedding",
                "Matthijs/cmu-arctic-xvectors:7306",
                help="HF dataset id + ':' + row index for a speaker "
                     "x-vector. Examples: "
                     "Matthijs/cmu-arctic-xvectors:7306, ...:2271."),
        ]
    if family == TTS_XTTS:
        return [
            TTSOptionSpec("reference_audio", "str", "Reference WAV", "",
                help="Path to a 6-15s reference clip for voice cloning. "
                     "Required for XTTS."),
            TTSOptionSpec("language", "enum", "Language", "en",
                help="Synthesis language.",
                choices=[("en","English"),("es","Spanish"),("fr","French"),
                          ("de","German"),("it","Italian"),("pt","Portuguese"),
                          ("pl","Polish"),("tr","Turkish"),("ru","Russian"),
                          ("nl","Dutch"),("cs","Czech"),("ar","Arabic"),
                          ("zh-cn","Mandarin Chinese"),("hu","Hungarian"),
                          ("ko","Korean"),("ja","Japanese"),("hi","Hindi")]),
            TTSOptionSpec("temperature", "float", "Temperature", 0.75,
                help="Sampling temperature for the GPT decoder.",
                min=0.1, max=1.5, step=0.05),
            TTSOptionSpec("top_p", "float", "Top-p", 0.85,
                help="Nucleus sampling cutoff.",
                min=0.0, max=1.0, step=0.05),
            TTSOptionSpec("top_k", "int", "Top-k", 50,
                help="Top-k sampling cutoff.", min=0, max=200),
            TTSOptionSpec("repetition_penalty", "float", "Rep. Penalty", 10.0,
                help="XTTS-specific: raise to break out of audio loops "
                     "/ stutters. Default 10.0 (high vs. other LMs).",
                min=1.0, max=20.0, step=0.5),
            TTSOptionSpec("length_penalty", "float", "Length Penalty", 1.0,
                help="Encourage longer (>1) or shorter (<1) outputs.",
                min=0.1, max=3.0, step=0.1),
        ]
    if family == TTS_BARK:
        return [
            TTSOptionSpec("semantic_temperature", "float",
                "Semantic Temp.", 0.7,
                help="Temperature for the semantic (text→tokens) stage. "
                     "Lower = more deterministic phrasing.",
                min=0.1, max=1.5, step=0.05),
            TTSOptionSpec("coarse_temperature", "float",
                "Coarse Temp.", 0.7,
                help="Temperature for the coarse acoustic stage. "
                     "Affects prosody / cadence.",
                min=0.1, max=1.5, step=0.05),
            TTSOptionSpec("min_eos_p", "float", "Min EOS Prob.", 0.2,
                help="End-of-speech probability threshold. Lower if Bark "
                     "cuts off early; raise if it appends silence/babble.",
                min=0.01, max=0.95, step=0.01),
        ]
    if family == TTS_KOKORO:
        return [
            TTSOptionSpec("voice_blend", "str", "Voice Blend", "",
                help="Optional voice mix. Format: 'af_sarah:0.6,af_bella:0.4'. "
                     "Overrides the picked voice when set."),
        ]
    if family == TTS_VITS:
        return [
            TTSOptionSpec("noise_scale", "float", "Noise Scale", 0.667,
                help="Sampling variance — higher = more expressive but "
                     "less stable. VITS default 0.667.",
                min=0.0, max=1.5, step=0.05),
            TTSOptionSpec("noise_scale_duration", "float",
                "Duration Variance", 0.8,
                help="Variance of the duration predictor — controls "
                     "rhythm jitter. VITS default 0.8.",
                min=0.0, max=1.5, step=0.05),
            TTSOptionSpec("seed", "int", "Seed", -1,
                help="-1 = random. Set a fixed value for reproducible "
                     "renders.",
                min=-1, max=2_147_483_647),
        ]
    if family == TTS_QWEN_TTS:
        return [
            TTSOptionSpec("temperature", "float", "Temperature", 0.7,
                help="Sampling temperature.",
                min=0.1, max=1.5, step=0.05),
            TTSOptionSpec("top_p", "float", "Top-p", 0.9,
                help="Nucleus sampling cutoff.",
                min=0.0, max=1.0, step=0.05),
        ]
    if family == TTS_ORPHEUS:
        return [
            TTSOptionSpec("temperature", "float", "Temperature", 0.7,
                help="Sampling temperature for the LM-driven decoder.",
                min=0.1, max=1.5, step=0.05),
            TTSOptionSpec("top_p", "float", "Top-p", 0.9,
                help="Nucleus sampling cutoff.",
                min=0.0, max=1.0, step=0.05),
            TTSOptionSpec("emotion_tags", "str", "Emotion Tags", "",
                help="Optional tags injected into the text — e.g. "
                     "'<laugh>', '<sigh>'. Comma-separated."),
        ]
    if family == TTS_CSM:
        return [
            TTSOptionSpec("temperature", "float", "Temperature", 0.9,
                help="Sampling temperature.",
                min=0.1, max=1.5, step=0.05),
        ]
    if family == TTS_HIGGS:
        return [
            TTSOptionSpec("temperature", "float", "Temperature", 0.7,
                help="Sampling temperature.",
                min=0.1, max=1.5, step=0.05),
            TTSOptionSpec("reference_audio", "str", "Reference WAV", "",
                help="Optional reference clip for zero-shot voice "
                     "cloning. Higgs-Audio v2 supports this natively."),
        ]
    if family == TTS_VIBEVOICE:
        return [
            TTSOptionSpec("temperature", "float", "Temperature", 0.7,
                help="Sampling temperature.",
                min=0.1, max=1.5, step=0.05),
        ]
    # generic: no extra knobs.
    return []


def tts_capabilities(family: str) -> set[str]:
    base = {"speed", "torch_compile", "attn_impl"}
    if family == TTS_KOKORO:
        return base | {"stream", "multilingual"}
    if family == TTS_VITS:
        return base | {"seed"}  # per-language repo (multilingual via repo id)
    if family == TTS_SPEECHT5:
        return base
    if family == TTS_BARK:
        return {"torch_compile", "attn_impl"}   # Bark doesn't honour speed
    if family == TTS_PARLER:
        return base | {"style_prompt", "multilingual"}
    if family == TTS_XTTS:
        return base | {"voice_clone", "multilingual"}
    if family == TTS_QWEN_TTS:
        return base | {"multilingual"}
    if family == TTS_ORPHEUS:
        return base | {"emotion_tags"}
    if family == TTS_CSM:
        return base | {"multilingual"}
    if family == TTS_HIGGS:
        return base | {"voice_clone", "multilingual"}
    if family == TTS_VIBEVOICE:
        return base | {"multilingual"}
    if family == TTS_GENERIC:
        return {"torch_compile", "attn_impl"}
    return set()


def stt_family_label(family: str) -> str:
    """Human-readable family label for the UI status pill."""
    return {
        STT_WHISPER:    "Whisper · 99 langs · translate",
        STT_WAV2VEC2:   "Wav2Vec2 / HuBERT (CTC)",
        STT_MMS:        "MMS · 1107 langs (CTC + adapter)",
        STT_SEAMLESS:   "SeamlessM4T · multilingual · translate",
        STT_MOONSHINE:  "Moonshine · English",
        STT_S2T:        "Speech2Text · seq2seq",
        STT_SPEECHT5:   "SpeechT5 ASR",
        STT_QWEN_AUDIO: "Qwen2-Audio · multimodal",
        STT_VOXTRAL:    "Voxtral · multilingual · instruction-tuned",
        STT_GRANITE:    "Granite-Speech · multilingual · AST",
        STT_PHI4MM:     "Phi-4-Multimodal · instruction LLM",
        STT_VIBEVOICE:  "VibeVoice ASR · long-form",
        STT_GENERIC:    "Generic ASR (pipeline)",
    }.get(family, "")


# ── Static voice catalogs (UI uses these pre-load) ───────────────────

_KOKORO_VOICES_RAW: list[tuple[str, str, str, str]] = [
    # (voice_id, language, gender, display_name)
    ("af_alloy","American English","F","Alloy"),
    ("af_aoede","American English","F","Aoede"),
    ("af_bella","American English","F","Bella"),
    ("af_heart","American English","F","Heart"),
    ("af_jessica","American English","F","Jessica"),
    ("af_kore","American English","F","Kore"),
    ("af_nicole","American English","F","Nicole"),
    ("af_nova","American English","F","Nova"),
    ("af_river","American English","F","River"),
    ("af_sarah","American English","F","Sarah"),
    ("af_sky","American English","F","Sky"),
    ("am_adam","American English","M","Adam"),
    ("am_echo","American English","M","Echo"),
    ("am_eric","American English","M","Eric"),
    ("am_fenrir","American English","M","Fenrir"),
    ("am_liam","American English","M","Liam"),
    ("am_michael","American English","M","Michael"),
    ("am_onyx","American English","M","Onyx"),
    ("am_puck","American English","M","Puck"),
    ("am_santa","American English","M","Santa"),
    ("bf_alice","British English","F","Alice"),
    ("bf_emma","British English","F","Emma"),
    ("bf_isabella","British English","F","Isabella"),
    ("bf_lily","British English","F","Lily"),
    ("bm_daniel","British English","M","Daniel"),
    ("bm_fable","British English","M","Fable"),
    ("bm_george","British English","M","George"),
    ("bm_lewis","British English","M","Lewis"),
    ("ef_dora","Spanish","F","Dora"),
    ("em_alex","Spanish","M","Alex"),
    ("em_santa","Spanish","M","Santa"),
    ("ff_siwis","French","F","Siwis"),
    ("hf_alpha","Hindi","F","Alpha"),
    ("hf_beta","Hindi","F","Beta"),
    ("hm_omega","Hindi","M","Omega"),
    ("hm_psi","Hindi","M","Psi"),
    ("if_sara","Italian","F","Sara"),
    ("im_nicola","Italian","M","Nicola"),
    ("jf_alpha","Japanese","F","Alpha"),
    ("jf_gongitsune","Japanese","F","Gongitsune"),
    ("jf_nezumi","Japanese","F","Nezumi"),
    ("jf_tebukuro","Japanese","F","Tebukuro"),
    ("jm_kumo","Japanese","M","Kumo"),
    ("pf_dora","Brazilian Portuguese","F","Dora"),
    ("pm_alex","Brazilian Portuguese","M","Alex"),
    ("pm_santa","Brazilian Portuguese","M","Santa"),
    ("zf_xiaobei","Mandarin Chinese","F","Xiaobei"),
    ("zf_xiaoni","Mandarin Chinese","F","Xiaoni"),
    ("zf_xiaoxiao","Mandarin Chinese","F","Xiaoxiao"),
    ("zf_xiaoyi","Mandarin Chinese","F","Xiaoyi"),
    ("zm_yunjian","Mandarin Chinese","M","Yunjian"),
    ("zm_yunxi","Mandarin Chinese","M","Yunxi"),
    ("zm_yunxia","Mandarin Chinese","M","Yunxia"),
    ("zm_yunyang","Mandarin Chinese","M","Yunyang"),
]

_BARK_VOICES_RAW: list[tuple[str, str, str, str]] = [
    ("v2/en_speaker_0","English","M","Speaker 0"),
    ("v2/en_speaker_1","English","M","Speaker 1"),
    ("v2/en_speaker_3","English","M","Speaker 3"),
    ("v2/en_speaker_6","English","F","Speaker 6"),
    ("v2/en_speaker_9","English","F","Speaker 9"),
    ("v2/de_speaker_3","German","M","Speaker 3"),
    ("v2/es_speaker_0","Spanish","M","Speaker 0"),
    ("v2/fr_speaker_5","French","F","Speaker 5"),
    ("v2/hi_speaker_2","Hindi","F","Speaker 2"),
    ("v2/ja_speaker_0","Japanese","F","Speaker 0"),
    ("v2/zh_speaker_4","Chinese","F","Speaker 4"),
]

_PARLER_PRESETS_RAW: list[tuple[str, str, str, str]] = [
    ("calm-female","English","F","Calm Female"),
    ("calm-male","English","M","Calm Male"),
    ("excited","English","","Excited"),
    ("news","English","","News Anchor"),
    ("narrator","English","","Narrator"),
]

_SPEECHT5_DEFAULTS_RAW: list[tuple[str, str, str, str]] = [
    ("Matthijs/cmu-arctic-xvectors:7306","English","F","AWB-Female-7306"),
    ("Matthijs/cmu-arctic-xvectors:2271","English","M","SLT-Male-2271"),
    ("Matthijs/cmu-arctic-xvectors:1726","English","M","RMS-Male-1726"),
    ("Matthijs/cmu-arctic-xvectors:3457","English","F","CLB-Female-3457"),
]

_ORPHEUS_VOICES_RAW: list[tuple[str, str, str, str]] = [
    # Orpheus ships with named voices (tara, leah, jess, leo, dan, mia, …).
    ("tara",    "English", "F", "Tara"),
    ("leah",    "English", "F", "Leah"),
    ("jess",    "English", "F", "Jess"),
    ("mia",     "English", "F", "Mia"),
    ("leo",     "English", "M", "Leo"),
    ("dan",     "English", "M", "Dan"),
    ("zac",     "English", "M", "Zac"),
    ("zoe",     "English", "F", "Zoe"),
]

_CSM_VOICES_RAW: list[tuple[str, str, str, str]] = [
    # CSM ships a single base voice; users pass a reference clip for cloning.
    ("conversational", "English", "", "Conversational"),
]

_HIGGS_VOICES_RAW: list[tuple[str, str, str, str]] = [
    # Higgs-Audio v2 is zero-shot — voice is implied by reference audio.
    ("zero-shot", "Multi", "", "Zero-shot (use Reference)"),
]


def tts_voices_for_family(family: str) -> list[VoiceEntry]:
    """Static voice catalog for a family, available pre-load so the
    UI can populate the voice picker the moment the user picks/
    detects a model. For families whose voice list comes from the
    model itself (VITS — implicit single voice, generic pipeline),
    returns []. The post-load `backend.voices()` may extend this
    (e.g. SpeechT5 cached from user input)."""
    if family == TTS_KOKORO:
        return [VoiceEntry(*row) for row in _KOKORO_VOICES_RAW]
    if family == TTS_BARK:
        return [VoiceEntry(*row) for row in _BARK_VOICES_RAW]
    if family == TTS_PARLER:
        return [VoiceEntry(*row) for row in _PARLER_PRESETS_RAW]
    if family == TTS_SPEECHT5:
        return [VoiceEntry(*row) for row in _SPEECHT5_DEFAULTS_RAW]
    if family == TTS_ORPHEUS:
        return [VoiceEntry(*row) for row in _ORPHEUS_VOICES_RAW]
    if family == TTS_CSM:
        return [VoiceEntry(*row) for row in _CSM_VOICES_RAW]
    if family == TTS_HIGGS:
        return [VoiceEntry(*row) for row in _HIGGS_VOICES_RAW]
    return []


def tts_default_voice_for_family(family: str) -> str:
    voices = tts_voices_for_family(family)
    return voices[0].voice_id if voices else ""


def tts_family_label(family: str) -> str:
    return {
        TTS_KOKORO:    "Kokoro · 54 voices · 9 langs",
        TTS_VITS:      "VITS / MMS-TTS",
        TTS_SPEECHT5:  "SpeechT5 · speaker embeddings",
        TTS_BARK:      "Bark",
        TTS_PARLER:    "Parler · style-prompt",
        TTS_XTTS:      "XTTS · voice cloning",
        TTS_QWEN_TTS:  "Qwen3-TTS",
        TTS_ORPHEUS:   "Orpheus · Llama + SNAC · emotion tags",
        TTS_CSM:       "CSM · conversational",
        TTS_HIGGS:     "Higgs-Audio v2 · zero-shot cloning",
        TTS_VIBEVOICE: "VibeVoice · long-form · multi-speaker",
        TTS_GENERIC:   "Generic TTS (pipeline)",
    }.get(family, "")
