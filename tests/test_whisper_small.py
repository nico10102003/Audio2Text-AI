"""Tests for the OpenAI Whisper Small integration (openai/whisper-small).

Unit tests use fakes (no GPU, no model download).  Load tests of both models
run against the local NAVER checkpoint and the Hugging Face cache; they skip
when the Whisper Small weights are not present locally.  Real Spanish
inference runs only when AUDIO2TEXT_RUN_WHISPER_INFERENCE=1 (slow, optional).
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NAVER_MODEL_DIR = PROJECT_ROOT / "models" / "multilingual-distilwhisper-28k"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from audio2text import engine  # noqa: E402
from audio2text.engine import (  # noqa: E402
    MODEL_NAVER_ID,
    MODEL_WHISPER_SMALL_ID,
    WHISPER_SMALL_LANGUAGES,
    transcribe,
    transcribe_auto,
    transcribe_long,
)
from audio2text.ui_service import (  # noqa: E402
    MODEL_OPTIONS,
    SUPPORTED_LANGUAGES,
    languages_for_model,
    model_label,
)

# Whisper Small tokens confirmed against openai/whisper-small (2026-11-19):
# decoder_start_token_id 50258, <|es|> 50262, <|transcribe|> 50359,
# <|notimestamps|> 50363.
WHISPER_SMALL_PREFIX = [50258, 50262, 50359, 50363]


def whisper_small_cached() -> bool:
    """True when the openai/whisper-small weights are present in the HF cache."""
    snapshots = Path(os.path.expanduser("~/.cache/huggingface/hub")) / (
        "models--openai--whisper-small"
    ) / "snapshots"
    if not snapshots.is_dir():
        return False
    return any((snap / "model.safetensors").is_file() for snap in snapshots.glob("*"))


class FakeWhisperProcessor:
    """Mimics WhisperProcessor for the whisper-small fake path."""

    def get_decoder_prompt_ids(self, language=None, task=None, no_timestamps=True):
        assert language == "es", f"se pidio idioma incorrecto: {language!r}"
        return [(1, 50262), (2, 50359), (3, 50363)]

    def __call__(self, audio, sampling_rate, return_tensors):
        return SimpleNamespace(input_features=torch.zeros(1, 80, 3000))

    def batch_decode(self, ids, skip_special_tokens=True):
        return ["hola que tal estás"]


class FakeWhisperModel:
    """Mimics WhisperForConditionalGeneration.generate; records call kwargs."""

    SEQUENCES = [50258, 50262, 50359, 50363, 10, 11]

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.generation_config = SimpleNamespace(decoder_start_token_id=50258)
        self.config = SimpleNamespace(decoder_start_token_id=50258)

    def generate(self, input_features, **kwargs):
        self.calls.append(kwargs)
        return torch.tensor([self.SEQUENCES], dtype=torch.long)

    def to(self, device=None, dtype=None):
        return self

    def eval(self):
        return self


def fake_whisper_small() -> SimpleNamespace:
    return SimpleNamespace(
        language="es",
        device=torch.device("cpu"),
        dtype=torch.float32,
        model=FakeWhisperModel(),
        processor=FakeWhisperProcessor(),
        model_id=MODEL_WHISPER_SMALL_ID,
    )


class LanguageSelectionTest(unittest.TestCase):
    def test_nave_still_rejects_spanish(self) -> None:
        with self.assertRaisesRegex(ValueError, "catalan.*tamil.*thai"):
            engine.load_model(
                str(NAVER_MODEL_DIR), language="es", device="cpu", dtype="float32"
            )

    def test_whisper_small_rejects_non_spanish(self) -> None:
        for language in ("catalan", "tamil", "thai"):
            with self.assertRaisesRegex(ValueError, "Whisper Small solo incluye"):
                engine.load_model(
                    MODEL_WHISPER_SMALL_ID,
                    language=language,
                    device="cpu",
                    dtype="float32",
                    model_id=MODEL_WHISPER_SMALL_ID,
                )

    def test_languages_for_model(self) -> None:
        # NAVER keeps its three languages; Whisper Small is Spanish-only.
        self.assertEqual(SUPPORTED_LANGUAGES, ("catalan", "tamil", "thai"))
        self.assertEqual(languages_for_model(MODEL_NAVER_ID), ("catalan", "tamil", "thai"))
        self.assertEqual(languages_for_model(MODEL_WHISPER_SMALL_ID), ("es",))
        self.assertEqual(WHISPER_SMALL_LANGUAGES, ("es",))

    def test_model_options_and_labels(self) -> None:
        self.assertEqual([option_id for option_id, _ in MODEL_OPTIONS],
                         [MODEL_NAVER_ID, MODEL_WHISPER_SMALL_ID])
        self.assertIn("NAVER", model_label(MODEL_NAVER_ID))
        self.assertEqual(model_label(MODEL_WHISPER_SMALL_ID), "OpenAI Whisper Small")

    def test_bad_dtype_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "dtype no soportado"):
            engine.load_model(
                MODEL_WHISPER_SMALL_ID,
                language="es",
                device="cpu",
                dtype="float64",
                model_id=MODEL_WHISPER_SMALL_ID,
            )


class WhisperSmallEngineFakeTest(unittest.TestCase):
    """Routing tests: whisper-small generation must use language='es'."""

    def test_transcribe_uses_language_es(self) -> None:
        loaded = fake_whisper_small()
        result = transcribe(loaded, np.full(16_000, 0.1, dtype=np.float32))

        self.assertEqual(len(loaded.model.calls), 1)
        call = loaded.model.calls[0]
        self.assertEqual(call.get("language"), "es")
        self.assertEqual(call.get("task"), "transcribe")
        self.assertNotIn("decoder_input_ids", call)
        self.assertEqual(result.language, "es")
        self.assertEqual(result.text, "hola que tal estás")
        self.assertEqual(result.extra["decoder_prefix"], WHISPER_SMALL_PREFIX)

    def test_transcribe_auto_dispatch_short_and_long(self) -> None:
        # Short audio (<= 30 s): single-window path.
        loaded = fake_whisper_small()
        result = transcribe_auto(loaded, np.full(16_000, 0.1, dtype=np.float32))
        self.assertEqual(len(loaded.model.calls), 1)
        self.assertEqual(result.extra["chunking"]["strategy"], "single")
        self.assertEqual(result.language, "es")

        # Long audio (> 30 s): chunked path, each window keeps language="es".
        loaded = fake_whisper_small()
        result = transcribe_auto(loaded, np.full(int(45 * 16_000), 0.1, dtype=np.float32))
        self.assertGreaterEqual(len(loaded.model.calls), 2)
        for call in loaded.model.calls:
            self.assertEqual(call.get("language"), "es")
            self.assertEqual(call.get("task"), "transcribe")
            self.assertNotIn("decoder_input_ids", call)
        self.assertEqual(result.language, "es")
        self.assertGreaterEqual(len(result.segments), 2)

    def test_transcribe_long_all_silent_skips_model(self) -> None:
        # Silence detection must behave identically for the whisper-small path.
        loaded = fake_whisper_small()
        result = transcribe_long(loaded, np.zeros(5 * 16_000, dtype=np.float32))
        self.assertEqual(result.text, "")
        self.assertEqual(len(loaded.model.calls), 0)
        self.assertEqual(result.extra["chunking"]["num_silent_chunks"], 1)


@unittest.skipUnless(whisper_small_cached(), "Whisper Small no está en caché local")
class WhisperSmallLoadTest(unittest.TestCase):
    """Real loads: NAVER 28k (local) + openai/whisper-small (HF cache)."""

    def test_load_nave_model(self) -> None:
        loaded = engine.load_model(
            str(NAVER_MODEL_DIR), language="catalan", device="cpu", dtype="float32"
        )
        self.assertEqual(loaded.model_id, MODEL_NAVER_ID)
        self.assertEqual(loaded.language, "catalan")
        self.assertEqual(loaded.device.type, "cpu")

    def test_load_whisper_small(self) -> None:
        loaded = engine.load_model(
            MODEL_WHISPER_SMALL_ID,
            language="es",
            device="cpu",
            dtype="float32",
            model_id=MODEL_WHISPER_SMALL_ID,
        )
        self.assertEqual(loaded.model_id, MODEL_WHISPER_SMALL_ID)
        self.assertEqual(loaded.language, "es")
        self.assertEqual(loaded.device.type, "cpu")
        self.assertEqual(loaded.dtype, torch.float32)
        self.assertTrue(callable(getattr(loaded.model, "generate", None)))
        # _build_decoder_prefix works with the real processor/config.
        self.assertEqual(engine._build_decoder_prefix(loaded), WHISPER_SMALL_PREFIX)

    def test_load_both_models(self) -> None:
        nave = engine.load_model(
            str(NAVER_MODEL_DIR), language="catalan", device="cpu", dtype="float32"
        )
        whisper = engine.load_model(
            MODEL_WHISPER_SMALL_ID,
            language="es",
            device="cpu",
            dtype="float32",
            model_id=MODEL_WHISPER_SMALL_ID,
        )
        self.assertEqual(nave.model_id, MODEL_NAVER_ID)
        self.assertEqual(whisper.model_id, MODEL_WHISPER_SMALL_ID)
        self.assertNotEqual(nave.model_id, whisper.model_id)

    def test_load_whisper_small_cuda_fp16_if_available(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA no disponible en este equipo")
        loaded = engine.load_model(
            MODEL_WHISPER_SMALL_ID,
            language="es",
            device="cuda",
            dtype="float16",
            model_id=MODEL_WHISPER_SMALL_ID,
        )
        self.assertEqual(loaded.device.type, "cuda")
        self.assertEqual(loaded.dtype, torch.float16)


@unittest.skipUnless(
    os.environ.get("AUDIO2TEXT_RUN_WHISPER_INFERENCE") == "1",
    "define AUDIO2TEXT_RUN_WHISPER_INFERENCE=1 para la inferencia real en español",
)
class WhisperSmallRealInferenceTest(unittest.TestCase):
    """Real Spanish inference through the engine (slow, opt-in)."""

    def _load_whisper_small(self) -> object:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = "float16" if device == "cuda" else "float32"
        return engine.load_model(
            MODEL_WHISPER_SMALL_ID,
            language="es",
            device=device,
            dtype=dtype,
            model_id=MODEL_WHISPER_SMALL_ID,
        )

    def test_spanish_short_audio(self) -> None:
        sample = PROJECT_ROOT / "data" / "samples" / "es_es_sample.wav"
        if not sample.is_file():
            self.skipTest(f"no existe el audio español de prueba: {sample}")
        loaded = self._load_whisper_small()
        audio, sample_rate = engine.read_audio(sample)
        result = transcribe_auto(loaded, audio, sample_rate=sample_rate)
        print(
            f"\nWhisper Small español (corto): {result.latency_s:.2f} s -> "
            f"{result.text!r}"
        )
        self.assertTrue(result.text.strip(), "la transcripción en español está vacía")
        self.assertEqual(result.language, "es")
        self.assertGreaterEqual(len(result.segments), 1)

    def test_spanish_long_audio(self) -> None:
        sample = PROJECT_ROOT / "data" / "samples" / "es_es_long.wav"
        if not sample.is_file():
            self.skipTest(f"no existe el audio español largo de prueba: {sample}")
        loaded = self._load_whisper_small()
        audio, sample_rate = engine.read_audio(sample)
        result = transcribe_auto(loaded, audio, sample_rate=sample_rate)
        print(
            f"\nWhisper Small español (largo): {result.latency_s:.2f} s, "
            f"{len(result.segments)} segmento(s)"
        )
        self.assertTrue(result.text.strip(), "la transcripción larga está vacía")
        self.assertEqual(result.language, "es")
        self.assertGreaterEqual(len(result.segments), 2)


if __name__ == "__main__":
    unittest.main()
