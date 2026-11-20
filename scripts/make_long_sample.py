"""Build a reproducible >30 s Catalan sample by repeating the short FLEURS clip."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=SAMPLES_DIR / "ca_es_sample.wav")
    parser.add_argument("--output", type=Path, default=SAMPLES_DIR / "ca_es_long.wav")
    parser.add_argument("--reference", type=Path, default=SAMPLES_DIR / "reference_ca.txt")
    parser.add_argument("--reference-out", type=Path, default=SAMPLES_DIR / "reference_ca_long.txt")
    parser.add_argument("--gap-seconds", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audio, sample_rate = sf.read(str(args.input), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    gap = np.zeros(int(round(args.gap_seconds * sample_rate)), dtype=np.float32)
    long_audio = np.concatenate([audio, gap, audio]).astype(np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.output), long_audio, sample_rate, subtype="FLOAT")
    print(
        f"Audio: {args.output} ({len(long_audio) / sample_rate:.2f} s, "
        f"origen {len(audio) / sample_rate:.2f} s x2 + {args.gap_seconds:.2f} s)"
    )

    reference = args.reference.read_text(encoding="utf-8").strip()
    args.reference_out.write_text(f"{reference} {reference}\n", encoding="utf-8")
    print(f"Referencia: {args.reference_out}")

    metadata_path = args.output.with_suffix(".json")
    metadata = {
        "source": str(args.input),
        "gap_seconds": args.gap_seconds,
        "duration_s": len(long_audio) / sample_rate,
        "sample_rate_hz": sample_rate,
        "reference": f"{reference} {reference}",
        "note": "Audio sintetico para validar chunking; no es una grabacion continua.",
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Metadatos: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
