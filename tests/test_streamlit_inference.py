r"""Optional end-to-end Streamlit test with the real validated model.

Run explicitly when the local weights and a usable device are available:

    $env:AUDIO2TEXT_RUN_UI_INFERENCE = "1"
    .\.venv\Scripts\python.exe -m unittest tests.test_streamlit_inference -v

The test uploads the real Catalan sample through Streamlit's file uploader,
clicks the actual UI button and asserts the rendered result.  It is skipped in
the normal fast suite because it loads the model and runs inference.
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from audio2text.ui_service import record_to_payload  # noqa: E402

APP_PATH = PROJECT_ROOT / "streamlit_app.py"
SAMPLE_PATH = PROJECT_ROOT / "data" / "samples" / "ca_es_sample.wav"
LONG_SAMPLE_PATH = PROJECT_ROOT / "data" / "samples" / "ca_es_long.wav"
EVIDENCE_PATH = PROJECT_ROOT / "logs" / "streamlit_ui_inference.json"
LONG_EVIDENCE_PATH = PROJECT_ROOT / "logs" / "streamlit_ui_inference_long.json"


@unittest.skipUnless(
    os.environ.get("AUDIO2TEXT_RUN_UI_INFERENCE") == "1",
    "define AUDIO2TEXT_RUN_UI_INFERENCE=1 para ejecutar la inferencia real",
)
class StreamlitRealInferenceTest(unittest.TestCase):
    def test_long_audio_from_streamlit_ui(self) -> None:
        if not LONG_SAMPLE_PATH.is_file():
            self.skipTest(f"no existe el audio largo de prueba: {LONG_SAMPLE_PATH}")

        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(APP_PATH), default_timeout=300).run(timeout=120)
        self.assertFalse(at.exception)

        at.file_uploader[0].upload(
            LONG_SAMPLE_PATH.name, LONG_SAMPLE_PATH.read_bytes()
        ).run(timeout=120)
        self.assertFalse(at.exception)
        self.assertFalse(at.error)
        self.assertEqual(at.button(key="transcribe").disabled, False)

        at.button(key="transcribe").click().run(timeout=300)

        self.assertFalse(at.exception)
        self.assertFalse(at.error)
        self.assertTrue(at.code, "la transcripción larga no se renderizó")
        text = at.code[0].value
        self.assertTrue(text.strip(), "la transcripción larga está vacía")
        self.assertIn("cuina", text.lower())
        self.assertTrue(at.dataframe, "no se renderizaron los segmentos")
        self.assertGreaterEqual(len(at.dataframe[0].value), 2)

        record = at.session_state["history"][0]
        self.assertEqual(record.language, "catalan")
        self.assertGreaterEqual(len(record.segments), 2)
        self.assertEqual(record.segments[0].start_s, 0.0)
        self.assertAlmostEqual(record.segments[-1].end_s, 42.4, places=1)

        LONG_EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
        LONG_EVIDENCE_PATH.write_text(
            json.dumps(record_to_payload(record), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"\nStreamlit UI long inference OK: {record.latency_s:.2f} s, "
            f"{len(record.segments)} segment(s)"
        )

    def test_real_inference_from_streamlit_ui(self) -> None:
        if not SAMPLE_PATH.is_file():
            self.skipTest(f"no existe el audio de prueba: {SAMPLE_PATH}")

        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(APP_PATH), default_timeout=300).run(timeout=120)
        self.assertFalse(at.exception)

        at.file_uploader[0].upload(
            SAMPLE_PATH.name, SAMPLE_PATH.read_bytes()
        ).run(timeout=120)
        self.assertFalse(at.exception)
        self.assertFalse(at.error)
        self.assertEqual(at.button(key="transcribe").disabled, False)

        at.button(key="transcribe").click().run(timeout=300)

        self.assertFalse(at.exception)
        self.assertFalse(at.error)
        self.assertTrue(at.code, "la transcripción no se renderizó")
        text = at.code[0].value
        self.assertTrue(text.strip(), "la transcripción está vacía")
        self.assertIn("cuina", text.lower())
        self.assertTrue(at.dataframe, "no se renderizaron los segmentos")
        self.assertGreaterEqual(len(at.download_button), 2)

        history = at.session_state["history"]
        self.assertEqual(len(history), 1)
        record = history[0]
        self.assertEqual(record.language, "catalan")
        self.assertTrue(record.text.strip())
        self.assertEqual(record.segments[0].start_s, 0.0)

        EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
        EVIDENCE_PATH.write_text(
            json.dumps(record_to_payload(record), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nStreamlit UI inference OK: {record.latency_s:.2f} s")
        print(text)


if __name__ == "__main__":
    unittest.main()
