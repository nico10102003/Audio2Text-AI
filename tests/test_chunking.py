"""Unit tests for chunk planning, silence detection and overlap merging."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402

from audio2text.chunking import (  # noqa: E402
    find_silence_cuts,
    is_silent,
    merge_chunk_text,
    plan_windows,
    plan_windows_for_audio,
    rms,
)


class PlanWindowsTest(unittest.TestCase):
    def test_empty_audio_has_no_windows(self) -> None:
        self.assertEqual(plan_windows(0.0), [])

    def test_short_audio_single_window(self) -> None:
        windows = plan_windows(20.7)
        self.assertEqual(len(windows), 1)
        self.assertEqual((windows[0].start_s, windows[0].end_s), (0.0, 20.7))

    def test_exact_limit_is_single_window(self) -> None:
        windows = plan_windows(30.0)
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].duration_s, 30.0)

    def test_just_over_limit_has_two_overlapping_windows(self) -> None:
        windows = plan_windows(30.001)
        self.assertEqual(len(windows), 2)
        self.assertEqual(windows[0].end_s, 30.0)
        self.assertEqual(windows[1].start_s, 28.0)
        self.assertAlmostEqual(windows[1].end_s, 30.001)
        self.assertLess(windows[1].start_s, windows[0].end_s)

    def test_coverage_and_overlap(self) -> None:
        for duration in (42.4, 58.1, 60.0, 88.0, 123.456):
            windows = plan_windows(duration)
            self.assertEqual(windows[0].start_s, 0.0)
            self.assertAlmostEqual(windows[-1].end_s, duration, places=5)
            for previous, current in zip(windows, windows[1:]):
                self.assertLessEqual(current.start_s, previous.end_s)
                self.assertGreater(current.start_s, previous.start_s)
            for window in windows:
                self.assertGreater(window.duration_s, 0)
                self.assertLessEqual(window.duration_s, 30.0 + 1e-6)

    def test_sample_bounds(self) -> None:
        window = plan_windows(42.4)[0]
        self.assertEqual(window.sample_bounds(16_000), (0, 480_000))

    def test_invalid_configuration(self) -> None:
        with self.assertRaises(ValueError):
            plan_windows(10.0, max_chunk_s=0)
        with self.assertRaises(ValueError):
            plan_windows(10.0, overlap_s=-1)
        with self.assertRaises(ValueError):
            plan_windows(10.0, overlap_s=30)
        with self.assertRaises(ValueError):
            plan_windows(-1)


class SilenceTest(unittest.TestCase):
    def test_rms(self) -> None:
        self.assertEqual(rms(np.zeros(1600, dtype=np.float32)), 0.0)
        self.assertAlmostEqual(rms(np.ones(100, dtype=np.float32)), 1.0)

    def test_silence_detection(self) -> None:
        self.assertTrue(is_silent(np.zeros(1000, dtype=np.float32)))
        self.assertTrue(is_silent(np.full(1000, 1e-6, dtype=np.float32)))
        self.assertFalse(is_silent(np.full(1000, 0.05, dtype=np.float32)))

    def test_empty_fragment_is_silent(self) -> None:
        self.assertTrue(is_silent(np.zeros(0, dtype=np.float32)))


class SilenceCutPlanningTest(unittest.TestCase):
    def _paused_audio(self) -> np.ndarray:
        sample_rate = 16_000
        speech = np.full(int(20.7 * sample_rate), 0.1, dtype=np.float32)
        gap = np.zeros(int(1.0 * sample_rate), dtype=np.float32)
        return np.concatenate([speech, gap, speech])

    def test_finds_pause_center(self) -> None:
        audio = self._paused_audio()
        cuts = find_silence_cuts(audio, 16_000, threshold=1e-4, min_silence_s=0.4)
        self.assertEqual(len(cuts), 1)
        self.assertAlmostEqual(cuts[0], 21.2, delta=0.05)

    def test_cuts_at_pause_without_overlap(self) -> None:
        audio = self._paused_audio()
        windows = plan_windows_for_audio(audio, 16_000)
        self.assertEqual(len(windows), 2)
        self.assertAlmostEqual(windows[0].start_s, 0.0, places=2)
        self.assertAlmostEqual(windows[0].end_s, 21.2, delta=0.05)
        self.assertAlmostEqual(windows[1].start_s, windows[0].end_s, places=2)
        self.assertAlmostEqual(windows[1].end_s, 42.4, places=2)

    def test_continuous_audio_uses_overlap(self) -> None:
        audio = np.full(int(42.4 * 16_000), 0.1, dtype=np.float32)
        windows = plan_windows_for_audio(audio, 16_000)
        self.assertEqual(
            [(window.start_s, window.end_s) for window in windows],
            [(0.0, 30.0), (28.0, 42.4)],
        )

    def test_short_audio_is_single_window(self) -> None:
        audio = np.full(20 * 16_000, 0.1, dtype=np.float32)
        windows = plan_windows_for_audio(audio, 16_000)
        self.assertEqual(len(windows), 1)


class MergeChunkTextTest(unittest.TestCase):
    def test_no_overlap_concatenates(self) -> None:
        merged, removed = merge_chunk_text("hola mon", "una altra frase")
        self.assertEqual(merged, "hola mon una altra frase")
        self.assertEqual(removed, 0)

    def test_removes_repeated_prefix(self) -> None:
        merged, removed = merge_chunk_text(
            "avui fa un dia fantastic",
            "dia fantastic i assolellat",
        )
        self.assertEqual(merged, "avui fa un dia fantastic i assolellat")
        self.assertEqual(removed, 2)

    def test_ignores_case_and_punctuation(self) -> None:
        merged, removed = merge_chunk_text(
            "Això és una prova.",
            "una prova, de chunk",
        )
        self.assertEqual(merged, "Això és una prova. de chunk")
        self.assertEqual(removed, 2)

    def test_single_word_is_not_removed(self) -> None:
        merged, removed = merge_chunk_text("la casa", "casa nova")
        self.assertEqual(merged, "la casa casa nova")
        self.assertEqual(removed, 0)

    def test_empty_sides(self) -> None:
        self.assertEqual(merge_chunk_text("", "nou text"), ("nou text", 0))
        self.assertEqual(merge_chunk_text("text previ", ""), ("text previ", 0))

    def test_max_overlap_limit(self) -> None:
        merged, removed = merge_chunk_text(
            "x y a b", "a b c", max_overlap_words=2
        )
        self.assertEqual(removed, 2)
        self.assertEqual(merged, "x y a b c")


if __name__ == "__main__":
    unittest.main()
