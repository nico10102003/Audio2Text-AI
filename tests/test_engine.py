"""Unit tests for audio reading, dispatch and chunked orchestration.

These tests do not load the real model: a fake processor/model is enough to
check window timestamps, prefix reuse, overlap merging and the short path.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

from audio2text import engine  # noqa: E402
from audio2text.engine import (  # noqa: E402
    AudioReadError,
    SUPPORTED_EXTENSIONS,
    Transcription,
    read_audio,
    transcribe_auto,
    transcribe_long,
)


def fake_loaded(model=None, processor=None) -> SimpleNamespace:
    return SimpleNamespace(
        language="catalan",
        device=torch.device("cpu"),
        dtype=torch.float32,
        model=model,
        processor=processor,
    )


class FakeProcessor:
    def __init__(self, texts: list[str]) -> None:
        self.texts = list(texts)
        self.calls = 0

    def get_decoder_prompt_ids(self, language=None, task=None):
        return [(1, 50270), (2, 50359), (3, 50363)]

    def __call__(self, audio, sampling_rate, return_tensors):
        return SimpleNamespace(input_features=torch.zeros(1, 80, 3000))

    def batch_decode(self, ids, skip_special_tokens=True):
        self.calls += 1
        return [self.texts.pop(0)]


class FakeModel:
    def __init__(self, sequences: list[list[int]]) -> None:
        self.sequences = list(sequences)
        self.prefixes: list[list[int]] = []
        self.calls: list[dict] = []
        self.generation_config = SimpleNamespace(decoder_start_token_id=50258)

    def generate(self, input_features, **kwargs):
        self.calls.append(kwargs)
        self.prefixes.append(kwargs["decoder_input_ids"][0].tolist())
        return torch.tensor([self.sequences.pop(0)], dtype=torch.long)


class ChunkedTranscriptionTest(unittest.TestCase):
    def test_empty_audio_skips_model(self) -> None:
        loaded = fake_loaded()
        result = transcribe_long(loaded, np.zeros(0, dtype=np.float32))
        self.assertEqual(result.text, "")
        self.assertEqual(result.segments, [])
        self.assertEqual(result.extra["num_chunks"], 0)
        self.assertEqual(result.duration_audio_s, 0.0)

    def test_all_silent_audio_skips_model(self) -> None:
        loaded = fake_loaded()
        result = transcribe_long(loaded, np.zeros(5 * 16_000, dtype=np.float32))
        self.assertEqual(result.text, "")
        self.assertEqual(len(result.segments), 1)
        self.assertTrue(result.segments[0].silent)
        self.assertEqual(result.extra["chunking"]["num_silent_chunks"], 1)

    def test_two_windows_keep_prefix_and_merge_overlap(self) -> None:
        processor = FakeProcessor(
            ["hola mon això és una prova", "això és una prova de chunk"]
        )
        model = FakeModel(
            [
                [50258, 50270, 50359, 50363, 1, 2],
                [50258, 50270, 50359, 50363, 3, 4],
            ]
        )
        loaded = fake_loaded(model=model, processor=processor)
        audio = np.full(int(42.4 * 16_000), 0.1, dtype=np.float32)

        result = transcribe_long(loaded, audio, max_chunk_s=30.0, overlap_s=2.0)

        self.assertEqual(result.text, "hola mon això és una prova de chunk")
        self.assertEqual(len(result.segments), 2)
        self.assertEqual(model.prefixes, [[50258, 50270, 50359, 50363]] * 2)
        self.assertEqual(
            [(segment.start_s, segment.end_s) for segment in result.segments],
            [(0.0, 30.0), (28.0, 42.4)],
        )
        self.assertEqual(result.segments[0].text, "hola mon això és una prova")
        self.assertEqual(result.segments[1].text, "de chunk")
        self.assertEqual(result.extra["chunking"]["duplicate_words_removed"], 4)
        self.assertEqual(result.extra["num_tokens"], 12)

    def test_segment_token_ids_include_prefix(self) -> None:
        processor = FakeProcessor(["hola"])
        model = FakeModel([[50258, 50270, 50359, 50363, 9]])
        loaded = fake_loaded(model=model, processor=processor)
        result = transcribe_long(loaded, np.full(16_000, 0.1, dtype=np.float32))
        self.assertEqual(result.segments[0].token_ids, [50258, 50270, 50359, 50363, 9])
        self.assertEqual(result.extra["decoder_prefix"], [50258, 50270, 50359, 50363])


    def test_nave_does_not_request_word_alignment(self) -> None:
        # NAVER 28k keeps its behavior: no per-token timestamp requests.
        processor = FakeProcessor(["hola mon"])
        model = FakeModel([[50258, 50270, 50359, 50363, 1]])
        loaded = fake_loaded(model=model, processor=processor)
        result = transcribe_long(loaded, np.full(16_000, 0.1, dtype=np.float32))
        self.assertTrue(model.calls)
        for call in model.calls:
            self.assertNotIn("return_token_timestamps", call)

    def test_contiguous_pause_cut_is_not_deduplicated(self) -> None:
        processor = FakeProcessor(["hola mon", "hola mon"])
        model = FakeModel(
            [
                [50258, 50270, 50359, 50363, 1],
                [50258, 50270, 50359, 50363, 2],
            ]
        )
        loaded = fake_loaded(model=model, processor=processor)
        sample_rate = 16_000
        speech = np.full(int(20.7 * sample_rate), 0.1, dtype=np.float32)
        gap = np.zeros(sample_rate, dtype=np.float32)
        audio = np.concatenate([speech, gap, speech])

        result = transcribe_long(loaded, audio)

        self.assertEqual(result.text, "hola mon hola mon")
        self.assertEqual(result.extra["chunking"]["duplicate_words_removed"], 0)
        self.assertEqual(len(result.segments), 2)
        self.assertAlmostEqual(
            result.segments[1].start_s, result.segments[0].end_s, places=2
        )


class DispatchTest(unittest.TestCase):
    def _short_result(self) -> Transcription:
        return Transcription(
            text="hola",
            language="catalan",
            device="cpu",
            dtype="float32",
            duration_audio_s=1.0,
            latency_s=0.5,
            extra={
                "num_tokens": 5,
                "num_beams": 5,
                "decoder_prefix": [50258, 50270, 50359, 50363],
                "token_ids": [50258, 50270, 50359, 50363, 9],
            },
        )

    def test_short_audio_uses_single_window_path(self) -> None:
        audio = np.full(16_000, 0.1, dtype=np.float32)
        with mock.patch.object(engine, "transcribe", return_value=self._short_result()) as patched:
            result = transcribe_auto(fake_loaded(), audio)
        patched.assert_called_once()
        self.assertEqual(len(result.segments), 1)
        self.assertEqual(result.segments[0].start_s, 0.0)
        self.assertEqual(result.segments[0].end_s, 1.0)
        self.assertEqual(result.extra["chunking"]["strategy"], "single")

    def test_long_audio_uses_chunked_path(self) -> None:
        audio = np.full(int(31 * 16_000), 0.1, dtype=np.float32)
        sentinel = self._short_result()
        with mock.patch.object(engine, "transcribe_long", return_value=sentinel) as patched:
            result = transcribe_auto(fake_loaded(), audio)
        patched.assert_called_once()
        self.assertIs(result, sentinel)

    def test_no_chunk_flag_forces_single_window(self) -> None:
        audio = np.full(int(31 * 16_000), 0.1, dtype=np.float32)
        with mock.patch.object(engine, "transcribe", return_value=self._short_result()) as patched:
            transcribe_auto(fake_loaded(), audio, chunking=False)
        patched.assert_called_once()


class ReadAudioTest(unittest.TestCase):
    def test_missing_file(self) -> None:
        with self.assertRaises(AudioReadError):
            read_audio(PROJECT_ROOT / "data" / "samples" / "no_existe.wav")

    def test_invalid_contents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalido.wav"
            path.write_text("esto no es audio", encoding="utf-8")
            with self.assertRaises(AudioReadError):
                read_audio(path)

    def test_empty_wav(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vacio.wav"
            sf.write(str(path), np.zeros(0, dtype=np.float32), 16_000, subtype="FLOAT")
            audio, sample_rate = read_audio(path)
        self.assertEqual(audio.size, 0)
        self.assertEqual(sample_rate, 16_000)

    def test_non_finite_samples(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nan.wav"
            samples = np.zeros(160, dtype=np.float32)
            samples[10] = np.nan
            sf.write(str(path), samples, 16_000, subtype="FLOAT")
            with self.assertRaises(AudioReadError):
                read_audio(path)

    def test_supported_extensions_include_new_formats(self) -> None:
        self.assertIn(".wav", SUPPORTED_EXTENSIONS)
        self.assertIn(".flac", SUPPORTED_EXTENSIONS)
        for extension in (".mp3", ".m4a", ".aac", ".mp4", ".mov"):
            self.assertIn(extension, SUPPORTED_EXTENSIONS)

    def test_mp3_uses_ffmpeg_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clip.mp3"
            path.write_bytes(b"fake-mp3-bytes")

            def fake_run(command, **kwargs):
                sf.write(
                    command[-1],
                    np.zeros(16_000, dtype=np.float32),
                    16_000,
                    subtype="PCM_16",
                )
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with (
                mock.patch.object(engine, "_find_ffmpeg", return_value="ffmpeg.exe"),
                mock.patch.object(engine.subprocess, "run", side_effect=fake_run) as run,
            ):
                audio, sample_rate = read_audio(path)

            self.assertEqual(sample_rate, 16_000)
            self.assertEqual(audio.size, 16_000)
            self.assertEqual(run.call_count, 1)
            self.assertFalse(Path(run.call_args[0][0][-1]).exists())

    def test_ffmpeg_missing_raises_for_new_format(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clip.aac"
            path.write_bytes(b"fake-aac-bytes")
            with mock.patch.object(engine, "_find_ffmpeg", return_value=None):
                with self.assertRaises(AudioReadError) as context:
                    read_audio(path)
        self.assertIn("FFmpeg", str(context.exception))

    def test_ffmpeg_failure_cleans_temp_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clip.m4a"
            path.write_bytes(b"fake-m4a-bytes")
            seen: dict[str, Path] = {}

            def fake_run(command, **kwargs):
                seen["output"] = Path(command[-1])
                return SimpleNamespace(returncode=1, stdout="", stderr="simulated failure")

            with (
                mock.patch.object(engine, "_find_ffmpeg", return_value="ffmpeg.exe"),
                mock.patch.object(engine.subprocess, "run", side_effect=fake_run),
            ):
                with self.assertRaises(AudioReadError):
                    read_audio(path)

            self.assertFalse(seen["output"].exists())


if __name__ == "__main__":
    unittest.main()
