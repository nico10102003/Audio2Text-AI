"""Dependency-free chunk planning and text merging for long audio.

Whisper only sees 30 seconds of audio per forward pass.  Long files are split
into overlapping windows; the text of the overlap is removed from the next
window before concatenation.  These functions are pure so they can be tested
without loading the model.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

import numpy as np

DEFAULT_MAX_CHUNK_S = 30.0
DEFAULT_OVERLAP_S = 2.0
DEFAULT_SILENCE_RMS = 1e-4
DEFAULT_MIN_SILENCE_S = 0.4
DEFAULT_MIN_CHUNK_S = 2.0
DEFAULT_MAX_MERGE_WORDS = 12
DEFAULT_MIN_MERGE_WORDS = 2


@dataclass(frozen=True)
class Window:
    """A slice of the source audio, in seconds."""

    index: int
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def sample_bounds(self, sample_rate: int) -> tuple[int, int]:
        start = int(round(self.start_s * sample_rate))
        end = int(round(self.end_s * sample_rate))
        return start, end


def plan_windows(
    duration_s: float,
    max_chunk_s: float = DEFAULT_MAX_CHUNK_S,
    overlap_s: float = DEFAULT_OVERLAP_S,
) -> list[Window]:
    """Split ``duration_s`` into windows of at most ``max_chunk_s``.

    Consecutive windows overlap by ``overlap_s`` (except when the whole audio
    fits in one window).  The union of the windows always covers
    ``[0, duration_s]`` exactly once in terms of timeline, with the overlap
    regions appearing in two windows.
    """
    if max_chunk_s <= 0:
        raise ValueError("max_chunk_s debe ser > 0")
    if overlap_s < 0:
        raise ValueError("overlap_s no puede ser negativo")
    if overlap_s >= max_chunk_s:
        raise ValueError("overlap_s debe ser menor que max_chunk_s")
    if duration_s < 0:
        raise ValueError("duration_s no puede ser negativo")
    if duration_s == 0:
        return []
    if duration_s <= max_chunk_s:
        return [Window(index=0, start_s=0.0, end_s=float(duration_s))]

    step = max_chunk_s - overlap_s
    windows: list[Window] = []
    start = 0.0
    while True:
        end = min(start + max_chunk_s, duration_s)
        windows.append(Window(index=len(windows), start_s=round(start, 6), end_s=round(end, 6)))
        if end >= duration_s:
            break
        start += step
    return windows


def find_silence_cuts(
    audio: np.ndarray,
    sample_rate: int,
    threshold: float = DEFAULT_SILENCE_RMS,
    min_silence_s: float = DEFAULT_MIN_SILENCE_S,
    frame_s: float = 0.02,
) -> list[float]:
    """Return the center (in seconds) of every silent run of ``min_silence_s``.

    The audio is scanned in 20 ms frames.  These cut points let long files be
    split at natural pauses instead of mid-word, avoiding both duplicated and
    missing speech at the boundary.
    """
    if threshold <= 0 or audio.size == 0 or sample_rate <= 0:
        return []

    frame = max(1, int(round(frame_s * sample_rate)))
    frame_count = audio.size // frame
    if frame_count == 0:
        return []

    frames = np.asarray(audio[: frame_count * frame], dtype=np.float64).reshape(
        frame_count, frame
    )
    frame_rms = np.sqrt(np.mean(frames * frames, axis=1))
    silent = frame_rms <= threshold
    min_frames = max(1, int(round(min_silence_s / frame_s)))

    cuts: list[float] = []
    index = 0
    while index < frame_count:
        if not silent[index]:
            index += 1
            continue
        run_end = index
        while run_end < frame_count and silent[run_end]:
            run_end += 1
        if run_end - index >= min_frames:
            center = (index + run_end) // 2
            cuts.append((center * frame + frame / 2.0) / sample_rate)
        index = run_end
    return cuts


def plan_windows_for_audio(
    audio: np.ndarray,
    sample_rate: int,
    max_chunk_s: float = DEFAULT_MAX_CHUNK_S,
    overlap_s: float = DEFAULT_OVERLAP_S,
    silence_rms: float = DEFAULT_SILENCE_RMS,
    min_silence_s: float = DEFAULT_MIN_SILENCE_S,
    min_chunk_s: float = DEFAULT_MIN_CHUNK_S,
) -> list[Window]:
    """Plan windows, preferring natural pauses as boundaries.

    If a silent run is found before the 30 s limit, the window ends at its
    center and the next one starts there (no overlap, nothing repeated).  When
    speech is continuous, the nominal 30 s boundary is used and the next window
    starts ``overlap_s`` earlier so text can be deduplicated.
    """
    if audio.size == 0:
        return []

    duration_s = len(audio) / sample_rate
    fixed = plan_windows(duration_s, max_chunk_s=max_chunk_s, overlap_s=overlap_s)
    if len(fixed) <= 1:
        return fixed

    cuts = find_silence_cuts(
        audio,
        sample_rate,
        threshold=silence_rms,
        min_silence_s=min_silence_s,
    )

    windows: list[Window] = []
    start = 0.0
    while start < duration_s - 1e-9:
        nominal_end = min(start + max_chunk_s, duration_s)
        if nominal_end >= duration_s - 1e-9:
            windows.append(
                Window(index=len(windows), start_s=round(start, 6), end_s=round(duration_s, 6))
            )
            break

        candidates = [
            cut for cut in cuts if start + min_chunk_s <= cut <= nominal_end
        ]
        if candidates:
            end = max(candidates)
            windows.append(
                Window(index=len(windows), start_s=round(start, 6), end_s=round(end, 6))
            )
            start = end
        else:
            windows.append(
                Window(
                    index=len(windows),
                    start_s=round(start, 6),
                    end_s=round(nominal_end, 6),
                )
            )
            start = max(nominal_end - overlap_s, start + 1e-3)
    return windows


def rms(audio: np.ndarray) -> float:
    """Root mean square of a mono float signal (0.0 for empty input)."""
    if audio.size == 0:
        return 0.0
    values = np.asarray(audio, dtype=np.float64)
    return float(np.sqrt(np.mean(values * values)))


def is_silent(audio: np.ndarray, threshold: float = DEFAULT_SILENCE_RMS) -> bool:
    """True when the fragment has no energy above ``threshold``.

    A threshold <= 0 disables silence detection (only exact zeros are silent).
    """
    if audio.size == 0:
        return True
    return rms(audio) <= max(threshold, 0.0)


def _normalize_word(word: str) -> str:
    """Lowercase and keep only alphanumeric characters for comparison."""
    word = unicodedata.normalize("NFKC", word).lower()
    return "".join(char for char in word if char.isalnum())


def merge_chunk_text(
    previous_text: str,
    new_text: str,
    max_overlap_words: int = DEFAULT_MAX_MERGE_WORDS,
    min_overlap_words: int = DEFAULT_MIN_MERGE_WORDS,
) -> tuple[str, int]:
    """Append ``new_text`` to ``previous_text`` removing the repeated prefix.

    The overlap between two audio windows makes the model transcribe the same
    words twice.  The longest suffix/prefix match (comparing words without
    case or punctuation) is removed from ``new_text``.

    Returns ``(merged_text, removed_words)``.  A match of a single word is not
    removed by default because it is usually a legitimate repetition.
    """
    previous_words = previous_text.split()
    new_words = new_text.split()
    if not previous_words:
        return new_text.strip(), 0
    if not new_words:
        return previous_text.strip(), 0

    limit = min(max_overlap_words, len(previous_words), len(new_words))
    best = 0
    for size in range(limit, min_overlap_words - 1, -1):
        previous_norm = [_normalize_word(word) for word in previous_words[-size:]]
        new_norm = [_normalize_word(word) for word in new_words[:size]]
        if previous_norm == new_norm and all(previous_norm):
            best = size
            break

    kept_words = new_words[best:]
    merged = " ".join(previous_words + kept_words).strip()
    return merged, best
