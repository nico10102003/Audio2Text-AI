"""Dependency-free CER/WER helpers for smoke-test evaluation."""

from __future__ import annotations

import re
import unicodedata


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = "".join(
        char if char.isalnum() or char.isspace() else " " for char in text
    )
    return re.sub(r"\s+", " ", text).strip()


def edit_distance(reference: str, hypothesis: str) -> int:
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for i, ref_char in enumerate(reference, start=1):
        current = [i]
        for j, hyp_char in enumerate(hypothesis, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            substitute = previous[j - 1] + (ref_char != hyp_char)
            current.append(min(insert, delete, substitute))
        previous = current
    return previous[-1]


def character_error_rate(reference: str, hypothesis: str) -> float:
    ref = normalize_text(reference).replace(" ", "")
    hyp = normalize_text(hypothesis).replace(" ", "")
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return edit_distance(ref_words, hyp_words) / len(ref_words)
