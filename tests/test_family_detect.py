"""Family detection: config.json blobs → STT/TTS family name."""
from __future__ import annotations

import unittest

from tests import _isolate  # noqa: F401  — sets VOXTYPE_DATA_DIR
from voxtype.backends import family_detect as fd


class STTFamilyFromConfig(unittest.TestCase):
    """Bare config.json `model_type` / `architectures` → STT family."""

    def _detect(self, cfg, repo=""):
        return fd._family_from_config(cfg, repo, stt=True)

    def test_whisper_by_model_type(self):
        self.assertEqual(
            self._detect({"model_type": "whisper"}, "openai/whisper-base"),
            fd.STT_WHISPER,
        )

    def test_whisper_by_repo_id(self):
        self.assertEqual(
            self._detect({}, "openai/whisper-large-v3-turbo"),
            fd.STT_WHISPER,
        )

    def test_wav2vec2_ctc(self):
        self.assertEqual(
            self._detect({"model_type": "wav2vec2"},
                          "facebook/wav2vec2-large-960h-lv60-self"),
            fd.STT_WAV2VEC2,
        )

    def test_hubert(self):
        self.assertEqual(
            self._detect({"model_type": "hubert"}),
            fd.STT_WAV2VEC2,
        )

    def test_mms_from_repo_id(self):
        # MMS is a wav2vec2-derivative; the family detector should pull
        # MMS specifically based on repo id substring.
        self.assertEqual(
            self._detect({"model_type": "wav2vec2"}, "facebook/mms-1b-all"),
            fd.STT_MMS,
        )

    def test_seamless(self):
        self.assertEqual(
            self._detect({"model_type": "seamless_m4t_v2"},
                          "facebook/seamless-m4t-v2-large"),
            fd.STT_SEAMLESS,
        )

    def test_moonshine(self):
        self.assertEqual(
            self._detect({"model_type": "moonshine"},
                          "UsefulSensors/moonshine-base"),
            fd.STT_MOONSHINE,
        )

    def test_speech_to_text(self):
        self.assertEqual(
            self._detect({"model_type": "speech_to_text"}),
            fd.STT_S2T,
        )

    def test_speecht5_asr_via_architectures(self):
        cfg = {
            "model_type": "speecht5",
            "architectures": ["SpeechT5ForSpeechToText"],
        }
        self.assertEqual(self._detect(cfg), fd.STT_SPEECHT5)

    def test_pipeline_tag_fallback(self):
        # No model_type, no architectures, just a pipeline_tag from
        # HF card metadata → generic.
        cfg = {"card": {"pipeline_tag": "automatic-speech-recognition"}}
        self.assertEqual(self._detect(cfg, "some/unknown-asr"), fd.STT_GENERIC)

    def test_unknown_returns_empty(self):
        self.assertEqual(self._detect({"model_type": "bert"}, "nlp/foo"), "")


class TTSFamilyFromConfig(unittest.TestCase):
    def _detect(self, cfg, repo=""):
        return fd._family_from_config(cfg, repo, stt=False)

    def test_kokoro_by_repo(self):
        self.assertEqual(
            self._detect({}, "hexgrad/Kokoro-82M"),
            fd.TTS_KOKORO,
        )

    def test_kokoro_local(self):
        # _kokoro_local marker emitted by _read_local_config when a
        # voices/ dir is present.
        self.assertEqual(
            self._detect({"_kokoro_local": True}, "/some/local/path"),
            fd.TTS_KOKORO,
        )

    def test_vits_by_model_type(self):
        self.assertEqual(
            self._detect({"model_type": "vits"}, "facebook/mms-tts-eng"),
            fd.TTS_VITS,
        )

    def test_mms_tts_by_repo(self):
        # facebook/mms-tts-* repos may use model_type=vits or not — both
        # paths should hit VITS.
        self.assertEqual(
            self._detect({}, "facebook/mms-tts-eng"),
            fd.TTS_VITS,
        )

    def test_speecht5_tts_via_architectures(self):
        cfg = {
            "model_type": "speecht5",
            "architectures": ["SpeechT5ForTextToSpeech"],
        }
        self.assertEqual(self._detect(cfg, "microsoft/speecht5_tts"),
                          fd.TTS_SPEECHT5)

    def test_bark(self):
        self.assertEqual(
            self._detect({"model_type": "bark"}, "suno/bark"),
            fd.TTS_BARK,
        )

    def test_parler(self):
        self.assertEqual(
            self._detect({}, "parler-tts/parler-tts-mini-v1"),
            fd.TTS_PARLER,
        )

    def test_generic_via_pipeline_tag(self):
        cfg = {"card": {"pipeline_tag": "text-to-speech"}}
        self.assertEqual(
            self._detect(cfg, "some/unknown-tts"),
            fd.TTS_GENERIC,
        )


class CapabilitiesAndOptions(unittest.TestCase):
    """Each family should advertise a sensible cap set + option list."""

    def test_whisper_caps(self):
        caps = fd.stt_capabilities(fd.STT_WHISPER)
        for needed in ("multilingual", "task_translate", "initial_prompt",
                        "num_beams", "dtype", "torch_compile"):
            self.assertIn(needed, caps, f"whisper should advertise {needed!r}")

    def test_wav2vec2_caps_minimal(self):
        # CTC family has no language/beams/prompt — only dtype + compile.
        caps = fd.stt_capabilities(fd.STT_WAV2VEC2)
        self.assertIn("dtype", caps)
        self.assertIn("torch_compile", caps)
        self.assertNotIn("task_translate", caps)
        self.assertNotIn("initial_prompt", caps)
        self.assertNotIn("multilingual", caps)
        self.assertNotIn("num_beams", caps)

    def test_mms_multilingual(self):
        caps = fd.stt_capabilities(fd.STT_MMS)
        self.assertIn("multilingual", caps)

    def test_kokoro_streams(self):
        self.assertIn("stream", fd.tts_capabilities(fd.TTS_KOKORO))
        self.assertIn("speed", fd.tts_capabilities(fd.TTS_KOKORO))

    def test_parler_has_style_prompt(self):
        caps = fd.tts_capabilities(fd.TTS_PARLER)
        self.assertIn("style_prompt", caps)

    def test_whisper_option_keys(self):
        keys = {o.key for o in fd.stt_runtime_options(fd.STT_WHISPER)}
        for needed in ("task", "num_beams", "temperature",
                        "repetition_penalty", "initial_prompt"):
            self.assertIn(needed, keys,
                           f"whisper opts missing {needed!r}")

    def test_seamless_option_keys(self):
        keys = {o.key for o in fd.stt_runtime_options(fd.STT_SEAMLESS)}
        for needed in ("task", "num_beams", "tgt_lang"):
            self.assertIn(needed, keys)

    def test_voxtral_option_keys(self):
        keys = {o.key for o in fd.stt_runtime_options(fd.STT_VOXTRAL)}
        for needed in ("task", "temperature", "prompt"):
            self.assertIn(needed, keys)

    def test_qwen_audio_option_keys(self):
        keys = {o.key for o in fd.stt_runtime_options(fd.STT_QWEN_AUDIO)}
        for needed in ("prompt", "temperature", "top_p"):
            self.assertIn(needed, keys)

    def test_wav2vec2_no_options(self):
        self.assertEqual(fd.stt_runtime_options(fd.STT_WAV2VEC2), [])

    def test_parler_options(self):
        keys = {o.key for o in fd.tts_runtime_options(fd.TTS_PARLER)}
        self.assertIn("style", keys)

    def test_xtts_options(self):
        keys = {o.key for o in fd.tts_runtime_options(fd.TTS_XTTS)}
        for needed in ("reference_audio", "language", "temperature",
                        "top_p", "top_k", "repetition_penalty",
                        "length_penalty"):
            self.assertIn(needed, keys, f"xtts missing {needed!r}")

    def test_vits_options(self):
        keys = {o.key for o in fd.tts_runtime_options(fd.TTS_VITS)}
        for needed in ("noise_scale", "noise_scale_duration", "seed"):
            self.assertIn(needed, keys)

    def test_kokoro_blend_option(self):
        keys = {o.key for o in fd.tts_runtime_options(fd.TTS_KOKORO)}
        self.assertIn("voice_blend", keys)

    def test_parler_extra_options(self):
        keys = {o.key for o in fd.tts_runtime_options(fd.TTS_PARLER)}
        for needed in ("style", "temperature", "max_new_tokens"):
            self.assertIn(needed, keys)

    def test_bark_split_temperatures(self):
        keys = {o.key for o in fd.tts_runtime_options(fd.TTS_BARK)}
        for needed in ("semantic_temperature", "coarse_temperature",
                        "min_eos_p"):
            self.assertIn(needed, keys, f"bark missing {needed!r}")
        # The single `temperature` knob is gone in favour of the split.
        self.assertNotIn("temperature", keys)

    def test_orpheus_options(self):
        keys = {o.key for o in fd.tts_runtime_options(fd.TTS_ORPHEUS)}
        for needed in ("temperature", "top_p", "emotion_tags"):
            self.assertIn(needed, keys)

    def test_speecht5_options(self):
        keys = {o.key for o in fd.tts_runtime_options(fd.TTS_SPEECHT5)}
        self.assertIn("speaker_embedding", keys)

    def test_attn_impl_capability_universal(self):
        for fam in (fd.STT_WHISPER, fd.STT_VOXTRAL, fd.STT_QWEN_AUDIO,
                     fd.TTS_KOKORO, fd.TTS_VITS, fd.TTS_BARK,
                     fd.TTS_ORPHEUS, fd.TTS_CSM, fd.TTS_HIGGS):
            if fam.startswith(("whisper", "voxtral", "qwen_audio")):
                caps = fd.stt_capabilities(fam)
            else:
                caps = fd.tts_capabilities(fam)
            self.assertIn("attn_impl", caps,
                           f"{fam!r} should advertise attn_impl")

    def test_family_labels_present(self):
        # Every detected family should have a human-readable label.
        for fam in (fd.STT_WHISPER, fd.STT_WAV2VEC2, fd.STT_MMS,
                     fd.STT_SEAMLESS, fd.STT_MOONSHINE,
                     fd.STT_VOXTRAL, fd.STT_GRANITE, fd.STT_PHI4MM,
                     fd.STT_QWEN_AUDIO, fd.STT_VIBEVOICE):
            self.assertTrue(fd.stt_family_label(fam),
                             f"missing STT label for {fam!r}")
        for fam in (fd.TTS_KOKORO, fd.TTS_VITS, fd.TTS_SPEECHT5,
                     fd.TTS_BARK, fd.TTS_PARLER, fd.TTS_ORPHEUS,
                     fd.TTS_CSM, fd.TTS_HIGGS, fd.TTS_VIBEVOICE):
            self.assertTrue(fd.tts_family_label(fam),
                             f"missing TTS label for {fam!r}")


class NewFamilyDetection(unittest.TestCase):
    """Repo-id heuristics for families added in the 2025 audit."""

    def test_voxtral(self):
        self.assertEqual(
            fd._stt_from_repo_id("mistralai/Voxtral-Mini-3B-2507"),
            fd.STT_VOXTRAL,
        )

    def test_granite_speech(self):
        self.assertEqual(
            fd._stt_from_repo_id("ibm-granite/granite-speech-3.3-8b"),
            fd.STT_GRANITE,
        )

    def test_phi4_multimodal(self):
        self.assertEqual(
            fd._stt_from_repo_id("microsoft/Phi-4-multimodal-instruct"),
            fd.STT_PHI4MM,
        )

    def test_vibevoice_stt(self):
        self.assertEqual(
            fd._stt_from_repo_id("microsoft/VibeVoice-7B-ASR"),
            fd.STT_VIBEVOICE,
        )

    def test_orpheus(self):
        self.assertEqual(
            fd._tts_from_repo_id("canopylabs/orpheus-3b-0.1-ft"),
            fd.TTS_ORPHEUS,
        )

    def test_csm(self):
        self.assertEqual(
            fd._tts_from_repo_id("sesame/csm-1b"),
            fd.TTS_CSM,
        )

    def test_higgs_audio(self):
        self.assertEqual(
            fd._tts_from_repo_id("bosonai/higgs-audio-v2-generation-3B-base"),
            fd.TTS_HIGGS,
        )

    def test_vibevoice_tts(self):
        self.assertEqual(
            fd._tts_from_repo_id("microsoft/VibeVoice-1.5B"),
            fd.TTS_VIBEVOICE,
        )

    def test_voxtral_via_config(self):
        cfg = {"model_type": "voxtral"}
        self.assertEqual(
            fd._family_from_config(cfg, "x/y", stt=True),
            fd.STT_VOXTRAL,
        )

    def test_csm_via_config(self):
        cfg = {"model_type": "csm"}
        self.assertEqual(
            fd._family_from_config(cfg, "sesame/csm-1b", stt=False),
            fd.TTS_CSM,
        )


class VerifyModelId(unittest.TestCase):
    """`verify_model_id` should distinguish local-vs-HF, valid-vs-not,
    and gated repos — all without burning real network calls."""

    def setUp(self):
        # Patch the internal HF GET so tests never touch the network.
        self._real = fd._hf_get_json
        self._responses: dict[str, tuple[int, dict | None, str]] = {}
        fd._hf_get_json = lambda url: self._responses.get(
            url, (0, None, "no stub"))

    def tearDown(self):
        fd._hf_get_json = self._real

    def _api(self, repo: str) -> str:
        return f"https://huggingface.co/api/models/{repo}"

    def _cfg(self, repo: str) -> str:
        return f"https://huggingface.co/{repo}/resolve/main/config.json"

    def test_empty(self):
        r = fd.verify_model_id("", stt=True)
        self.assertFalse(r.valid)
        self.assertEqual(r.source, "none")

    def test_not_a_repo_id(self):
        r = fd.verify_model_id("whisperbase", stt=True)
        self.assertFalse(r.valid)
        self.assertIn("HF repo id", r.error)

    def test_hf_repo_valid_with_family(self):
        repo = "openai/whisper-base"
        self._responses[self._api(repo)] = (200, {"pipeline_tag": "automatic-speech-recognition"}, "")
        self._responses[self._cfg(repo)] = (200, {"model_type": "whisper"}, "")
        r = fd.verify_model_id(repo, stt=True)
        self.assertTrue(r.valid)
        self.assertEqual(r.source, "hf")
        self.assertEqual(r.family, fd.STT_WHISPER)

    def test_hf_repo_404(self):
        repo = "nonexistent/repo-xyz"
        self._responses[self._api(repo)] = (404, None, "http 404")
        r = fd.verify_model_id(repo, stt=True)
        self.assertFalse(r.valid)
        self.assertEqual(r.source, "hf")
        self.assertIn("not found", r.error)

    def test_hf_repo_gated(self):
        repo = "meta-llama/Llama-3-8B"
        self._responses[self._api(repo)] = (401, None, "http 401")
        r = fd.verify_model_id(repo, stt=False)
        self.assertFalse(r.valid)
        self.assertTrue(r.gated)
        self.assertIn("gated", r.error)

    def test_hf_unreachable(self):
        repo = "openai/whisper-base"
        self._responses[self._api(repo)] = (0, None, "timed out")
        r = fd.verify_model_id(repo, stt=True)
        self.assertFalse(r.valid)
        self.assertIn("unreachable", r.error)

    def test_local_path_missing(self):
        r = fd.verify_model_id(r"C:\definitely\does\not\exist\model",
                                stt=True)
        self.assertFalse(r.valid)
        self.assertEqual(r.source, "local")

    def test_local_dir_with_config(self):
        import json as _json
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "config.json").write_text(
                _json.dumps({"model_type": "whisper"}), encoding="utf-8")
            r = fd.verify_model_id(td, stt=True)
        self.assertTrue(r.valid)
        self.assertEqual(r.source, "local")
        self.assertEqual(r.family, fd.STT_WHISPER)

    def test_local_dir_without_config(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            r = fd.verify_model_id(td, stt=True)
        self.assertFalse(r.valid)
        self.assertEqual(r.source, "local")
        self.assertIn("config.json", r.error)


if __name__ == "__main__":
    unittest.main()
