"""Unit tests for the UI helpers (no Streamlit runtime or model required)."""

from __future__ import annotations

import io
import json
import sys
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from audio2text.engine import (  # noqa: E402
    AudioReadError,
    Segment,
    Transcription,
)
from audio2text.ui_service import (  # noqa: E402
    AudioValidationError,
    TranscriptionConfig,
    UploadedAudio,
    add_record,
    default_dtype,
    download_filename,
    dtype_options,
    find_record,
    format_duration,
    format_file_size,
    format_timestamp,
    format_vram,
    friendly_error,
    history_label,
    load_uploaded_audio,
    make_record,
    record_to_json,
    record_to_payload,
    record_to_txt,
    segment_rows,
)


def audio_bytes(
    seconds: float = 1.0,
    *,
    sample_rate: int = 16_000,
    file_format: str = "WAV",
) -> bytes:
    buffer = io.BytesIO()
    samples = np.zeros(int(seconds * sample_rate), dtype=np.float32)
    sf.write(buffer, samples, sample_rate, format=file_format)
    return buffer.getvalue()


def uploaded_audio(
    *,
    filename: str = "clip.wav",
    duration_s: float = 2.0,
) -> UploadedAudio:
    return UploadedAudio(
        filename=filename,
        suffix=Path(filename).suffix.lower(),
        data=b"fake-audio",
        samples=np.zeros(int(duration_s * 16_000), dtype=np.float32),
        sample_rate=16_000,
        duration_s=duration_s,
    )


def sample_record() -> object:
    audio = uploaded_audio()
    result = Transcription(
        text="Hola món",
        language="catalan",
        device="cuda:0",
        dtype="float16",
        duration_audio_s=2.0,
        latency_s=0.75,
        peak_vram_mb=1315.0,
        extra={
            "num_tokens": 5,
            "num_beams": 5,
            "decoder_prefix": [50258, 50270, 50359, 50363],
            "num_chunks": 1,
            "chunking": {
                "strategy": "single",
                "num_chunks": 1,
                "num_silent_chunks": 0,
                "duplicate_words_removed": 0,
            },
        },
        segments=[
            Segment(
                index=0,
                start_s=0.0,
                end_s=2.0,
                text="Hola món",
                token_ids=[50258, 50270, 50359, 50363, 9],
                raw_text="Hola món",
                latency_s=0.75,
                peak_vram_mb=1315.0,
            )
        ],
    )
    config = TranscriptionConfig(
        language="catalan",
        device="cuda",
        dtype="float16",
        num_beams=5,
        model_dir="/models/distilwhisper",
    )
    return make_record(
        audio=audio,
        result=result,
        config=config,
        load_seconds=4.5,
        record_id="record-1",
        created_at=datetime(2026, 11, 18, 14, 32, 5),
    )


class UploadTest(unittest.TestCase):
    def test_valid_wav(self) -> None:
        audio = load_uploaded_audio("prueba.wav", audio_bytes(1.0))
        self.assertEqual(audio.filename, "prueba.wav")
        self.assertEqual(audio.format_label, "WAV")
        self.assertEqual(audio.mime_type, "audio/wav")
        self.assertEqual(audio.sample_rate, 16_000)
        self.assertAlmostEqual(audio.duration_s, 1.0, places=3)
        self.assertEqual(audio.samples.dtype, np.float32)

    def test_valid_flac_and_uppercase_extension(self) -> None:
        audio = load_uploaded_audio("PRUEBA.FLAC", audio_bytes(0.5, file_format="FLAC"))
        self.assertEqual(audio.suffix, ".flac")
        self.assertEqual(audio.format_label, "FLAC")
        self.assertEqual(audio.mime_type, "audio/flac")
        self.assertAlmostEqual(audio.duration_s, 0.5, places=3)

    def test_empty_bytes(self) -> None:
        with self.assertRaisesRegex(AudioValidationError, "vacío"):
            load_uploaded_audio("vacio.wav", b"")

    def test_unsupported_extension(self) -> None:
        with self.assertRaisesRegex(
            AudioValidationError, "WAV, FLAC, MP3, M4A, AAC, MP4 o MOV"
        ):
            load_uploaded_audio("audio.wma", b"data")

    def test_new_formats_accepted_when_decoding(self) -> None:
        samples = np.zeros(16_000, dtype=np.float32)
        expectations = {
            "song.mp3": ("MP3", "audio/mpeg"),
            "song.m4a": ("M4A", "audio/mp4"),
            "song.aac": ("AAC", "audio/aac"),
            "song.mp4": ("MP4", "audio/mp4"),
            "song.mov": ("MOV", "video/quicktime"),
        }
        for filename, (label, mime) in expectations.items():
            with mock.patch(
                "audio2text.ui_service.read_audio",
                return_value=(samples, 16_000),
            ):
                audio = load_uploaded_audio(filename, b"fake-audio")
            self.assertEqual(audio.filename, filename)
            self.assertEqual(audio.suffix, Path(filename).suffix.lower())
            self.assertEqual(audio.format_label, label)
            self.assertEqual(audio.mime_type, mime)
            self.assertAlmostEqual(audio.duration_s, 1.0, places=3)

    def test_missing_filename(self) -> None:
        with self.assertRaisesRegex(AudioValidationError, "nombre"):
            load_uploaded_audio("", b"data")

    def test_invalid_wav_content(self) -> None:
        with self.assertRaisesRegex(AudioValidationError, "decodificar"):
            load_uploaded_audio("roto.wav", b"esto no es audio")

    def test_empty_wav_is_rejected(self) -> None:
        with self.assertRaisesRegex(AudioValidationError, "muestras"):
            load_uploaded_audio("vacio.wav", audio_bytes(0.0))

    def test_upload_limit(self) -> None:
        with mock.patch("audio2text.ui_service.MAX_UPLOAD_BYTES", 10):
            with self.assertRaisesRegex(AudioValidationError, "límite"):
                load_uploaded_audio("grande.wav", b"x" * 11)

    def test_filename_path_is_reduced_to_basename(self) -> None:
        audio = load_uploaded_audio("../../clip.wav", audio_bytes(0.1))
        self.assertEqual(audio.filename, "clip.wav")


class FormattingTest(unittest.TestCase):
    def test_timestamps(self) -> None:
        self.assertEqual(format_timestamp(0), "00:00.000")
        self.assertEqual(format_timestamp(65.4321), "01:05.432")
        self.assertEqual(format_timestamp(3661.234), "01:01:01.234")
        self.assertEqual(format_timestamp(59.9999), "01:00.000")
        self.assertEqual(format_timestamp(-2), "00:00.000")

    def test_durations(self) -> None:
        self.assertEqual(format_duration(20.704), "20.70 s")
        self.assertEqual(format_duration(65), "01:05.000")

    def test_file_sizes_and_vram(self) -> None:
        self.assertEqual(format_file_size(0), "0 B")
        self.assertEqual(format_file_size(1536), "1.5 KB")
        self.assertEqual(format_file_size(5 * 1024**2), "5.0 MB")
        self.assertEqual(format_vram(1315.2), "1315 MiB")
        self.assertEqual(format_vram(None), "—")

    def test_dtype_compatibility(self) -> None:
        self.assertEqual(default_dtype("cuda"), "float16")
        self.assertEqual(default_dtype("cpu"), "float32")
        self.assertEqual(dtype_options("cuda"), ["float16", "float32"])
        self.assertEqual(dtype_options("cpu"), ["float32"])


class RecordTest(unittest.TestCase):
    def test_make_record_keeps_engine_values(self) -> None:
        record = sample_record()
        self.assertEqual(record.record_id, "record-1")
        self.assertEqual(record.device, "cuda:0")
        self.assertEqual(record.dtype, "float16")
        self.assertEqual(record.language, "catalan")
        self.assertEqual(record.decoder_prefix, [50258, 50270, 50359, 50363])
        self.assertEqual(record.num_tokens, 5)
        self.assertEqual(record.num_chunks, 1)
        self.assertEqual(record.segments[0].text, "Hola món")

    def test_payload_is_json_serializable_and_has_segments(self) -> None:
        record = sample_record()
        payload = record_to_payload(record)
        self.assertEqual(payload["text"], "Hola món")
        self.assertEqual(payload["segments"][0]["start_s"], 0.0)
        self.assertEqual(payload["segments"][0]["end_s"], 2.0)
        self.assertEqual(payload["num_beams"], 5)
        self.assertEqual(payload["audio_filename"], "clip.wav")

        parsed = json.loads(record_to_json(record))
        self.assertEqual(parsed["language"], "catalan")
        self.assertEqual(parsed["audio_format"], "WAV")
        self.assertEqual(parsed["segments"][0]["text"], "Hola món")

    def test_txt_contains_metadata_text_and_segments(self) -> None:
        text = record_to_txt(sample_record())
        self.assertIn("Audio2Text AI", text)
        self.assertIn("Archivo: clip.wav", text)
        self.assertIn("Idioma: Català", text)
        self.assertIn("Hola món", text)
        self.assertIn("[00:00.000 - 00:02.000] Hola món", text)

    def test_download_filename(self) -> None:
        record = sample_record()
        self.assertEqual(download_filename(record, "txt"), "clip.txt")
        self.assertEqual(download_filename(record, ".json"), "clip.json")

    def test_segment_rows(self) -> None:
        record = sample_record()
        rows = segment_rows(record)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["Segmento"], 1)
        self.assertEqual(rows[0]["Inicio"], "00:00.000")
        self.assertEqual(rows[0]["Fin"], "00:02.000")
        self.assertEqual(rows[0]["Texto"], "Hola món")

    def test_silent_segment_is_labelled(self) -> None:
        record = sample_record()
        record.segments.append(
            Segment(index=1, start_s=2.0, end_s=3.0, text="", silent=True)
        )
        rows = segment_rows(record)
        self.assertEqual(rows[1]["Texto"], "(silencio)")

    def test_history_label(self) -> None:
        label = history_label(sample_record())
        self.assertEqual(label, "14:32:05 · clip.wav · 2.00 s · Català")

    def test_history_is_bounded_and_newest_first(self) -> None:
        record = sample_record()
        history = [record]
        second = sample_record()
        second.record_id = "record-2"
        history = add_record(history, second, limit=2)
        self.assertEqual([item.record_id for item in history], ["record-2", "record-1"])

        third = sample_record()
        third.record_id = "record-3"
        history = add_record(history, third, limit=2)
        self.assertEqual([item.record_id for item in history], ["record-3", "record-2"])

    def test_find_record(self) -> None:
        record = sample_record()
        self.assertIs(find_record([record], "record-1"), record)
        self.assertIsNone(find_record([record], "missing"))
        self.assertIsNone(find_record([record], None))


class FriendlyErrorTest(unittest.TestCase):
    def test_validation_error_is_kept(self) -> None:
        self.assertEqual(
            friendly_error(AudioValidationError("Archivo inválido")),
            "Archivo inválido",
        )

    def test_audio_read_error(self) -> None:
        self.assertIn(
            "No se pudo leer el audio",
            friendly_error(AudioReadError("detalle técnico")),
        )

    def test_missing_model(self) -> None:
        self.assertIn(
            "No se encontró el modelo local",
            friendly_error(FileNotFoundError("missing")),
        )

    def test_oom(self) -> None:
        self.assertIn(
            "memoria suficiente",
            friendly_error(RuntimeError("CUDA out of memory")),
        )

    def test_cuda_unavailable(self) -> None:
        self.assertIn(
            "Selecciona CPU",
            friendly_error(RuntimeError("CUDA is not available")),
        )

    def test_generic_error(self) -> None:
        self.assertIn(
            "No se pudo completar",
            friendly_error(RuntimeError("fallo inesperado")),
        )


if __name__ == "__main__":
    unittest.main()
