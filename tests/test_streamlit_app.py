"""AppTest smoke tests for the Streamlit interface (no model inference)."""

from __future__ import annotations

import io
import sys
import unittest
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from audio2text.engine import Segment, Transcription  # noqa: E402
from audio2text.ui_service import (  # noqa: E402
    TranscriptionConfig,
    UploadedAudio,
    make_record,
)

APP_PATH = PROJECT_ROOT / "streamlit_app.py"


def wav_bytes(seconds: float = 0.2) -> bytes:
    buffer = io.BytesIO()
    sf.write(
        buffer,
        np.zeros(int(seconds * 16_000), dtype=np.float32),
        16_000,
        format="WAV",
    )
    return buffer.getvalue()


def seeded_record():
    audio = UploadedAudio(
        filename="historial.wav",
        suffix=".wav",
        data=b"fake",
        samples=np.zeros(32_000, dtype=np.float32),
        sample_rate=16_000,
        duration_s=2.0,
    )
    result = Transcription(
        text="Text del historial",
        language="catalan",
        device="cuda:0",
        dtype="float16",
        duration_audio_s=2.0,
        latency_s=0.5,
        peak_vram_mb=1315.0,
        extra={
            "num_tokens": 5,
            "num_chunks": 1,
            "decoder_prefix": [50258, 50270, 50359, 50363],
            "chunking": {"strategy": "single", "num_chunks": 1},
        },
        segments=[
            Segment(
                index=0,
                start_s=0.0,
                end_s=2.0,
                text="Text del historial",
                token_ids=[50258, 50270, 50359, 50363, 9],
            )
        ],
    )
    return make_record(
        audio=audio,
        result=result,
        config=TranscriptionConfig(
            language="catalan",
            device="cuda",
            dtype="float16",
            num_beams=5,
            model_dir="models/test",
        ),
        load_seconds=1.0,
        record_id="seeded",
        created_at=datetime(2026, 11, 18, 12, 0, 0),
    )


class StreamlitAppSmokeTest(unittest.TestCase):
    def test_app_renders_without_exceptions(self) -> None:
        at = AppTest.from_file(str(APP_PATH), default_timeout=30).run(timeout=30)
        self.assertFalse(at.exception)
        self.assertTrue(at.title)
        self.assertIn("Audio2Text AI", at.title[0].value)
        self.assertEqual(len(at.file_uploader), 1)
        self.assertEqual(at.button(key="transcribe").disabled, True)
        self.assertIn("Resultados", [item.value for item in at.subheader])

    def test_invalid_audio_shows_understandable_error(self) -> None:
        at = AppTest.from_file(str(APP_PATH), default_timeout=30).run(timeout=30)
        at.file_uploader[0].upload("roto.wav", b"esto no es audio").run(timeout=30)

        self.assertFalse(at.exception)
        self.assertTrue(at.error)
        self.assertIn("decodificar", at.error[0].value)
        self.assertEqual(at.button(key="transcribe").disabled, True)

    def test_valid_upload_shows_metadata_and_enables_button(self) -> None:
        at = AppTest.from_file(str(APP_PATH), default_timeout=30).run(timeout=30)
        at.file_uploader[0].upload("muestra.wav", wav_bytes()).run(timeout=30)

        self.assertFalse(at.exception)
        self.assertFalse(at.error)
        self.assertEqual(at.button(key="transcribe").disabled, False)
        metric_labels = [metric.label for metric in at.metric]
        self.assertIn("Formato", metric_labels)
        self.assertIn("Duración", metric_labels)
        self.assertIn("WAV", [metric.value for metric in at.metric])

    def test_seeded_history_renders_result_and_can_be_cleared(self) -> None:
        record = seeded_record()
        at = AppTest.from_file(str(APP_PATH), default_timeout=30)
        at.session_state["history"] = [record]
        at.session_state["active_record_id"] = record.record_id
        at.run(timeout=30)

        self.assertFalse(at.exception)
        self.assertTrue(at.code)
        self.assertEqual(at.code[0].value, "Text del historial")
        self.assertTrue(at.dataframe)
        self.assertEqual(len(at.download_button), 2)
        self.assertIn("historial.wav", at.selectbox(key="history_select").options[0])

        at.button(key="clear_history").click().run(timeout=30)
        self.assertFalse(at.exception)
        self.assertEqual(at.session_state["history"], [])
        self.assertIsNone(at.session_state["active_record_id"])

    def test_audio_section_has_upload_and_record_tabs(self) -> None:
        at = AppTest.from_file(str(APP_PATH), default_timeout=30).run(timeout=30)

        self.assertFalse(at.exception)
        self.assertIn("1. Audio", [item.value for item in at.subheader])
        self.assertEqual(len(at.file_uploader), 1)
        self.assertEqual(len(at.audio_input), 1)
        self.assertEqual(
            [tab.label for tab in at.tabs[:2]], ["Subir archivo", "Grabar audio"]
        )

    def test_audio_input_recording_enables_transcribe_and_shows_player(self) -> None:
        at = AppTest.from_file(str(APP_PATH), default_timeout=30).run(timeout=30)
        self.assertEqual(at.button(key="transcribe").disabled, True)

        at.audio_input[0].upload("grabacion.wav", wav_bytes(0.5)).run(timeout=30)

        self.assertFalse(at.exception)
        self.assertFalse(at.error)
        self.assertEqual(at.button(key="transcribe").disabled, False)
        self.assertTrue(at.get("audio"), "expected a player for the recording")
        captions = [item.value for item in at.caption]
        self.assertTrue(any("Grabación adjunta" in caption for caption in captions))

    def test_recording_invalid_audio_shows_error(self) -> None:
        at = AppTest.from_file(str(APP_PATH), default_timeout=30).run(timeout=30)
        at.audio_input[0].upload("roto.wav", b"esto no es audio").run(timeout=30)

        self.assertFalse(at.exception)
        self.assertTrue(at.error)
        self.assertIn("decodificar", at.error[0].value)
        self.assertEqual(at.button(key="transcribe").disabled, True)

    def test_recording_empty_audio_shows_error(self) -> None:
        at = AppTest.from_file(str(APP_PATH), default_timeout=30).run(timeout=30)
        at.audio_input[0].upload("vacio.wav", b"").run(timeout=30)

        self.assertFalse(at.exception)
        self.assertTrue(at.error)
        self.assertIn("vacío", at.error[0].value)
        self.assertEqual(at.button(key="transcribe").disabled, True)


if __name__ == "__main__":
    unittest.main()
