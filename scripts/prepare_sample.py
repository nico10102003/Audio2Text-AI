"""Download one Catalan FLEURS sample without pulling the full 573 MB archive.

FLEURS stores each split as a large .tar.gz.  HTTP range requests let us fetch
only the first few megabytes, enough to decompress the first WAV entry.
"""

from __future__ import annotations

import io
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"
BASE_URL = "https://huggingface.co/datasets/google/fleurs/resolve/main/data/ca_es"
AUDIO_URL = f"{BASE_URL}/audio/dev.tar.gz"
TSV_URL = f"{BASE_URL}/dev.tsv"


def fetch(url: str, *, byte_range: str | None = None) -> bytes:
    headers = {"User-Agent": "Audio2Text-AI/0.1"}
    if byte_range:
        headers["Range"] = f"bytes={byte_range}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def extract_first_wav(archive_bytes: bytes) -> tuple[str, bytes]:
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r|gz") as tar:
        for member in tar:
            if member.isfile() and member.name.lower().endswith(".wav"):
                handle = tar.extractfile(member)
                if handle is None:
                    continue
                return Path(member.name).name, handle.read()
    raise RuntimeError("No WAV file found in the downloaded archive prefix")


def main() -> int:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    print("Descargando cabecera del archivo de audio (rango HTTP)...")
    archive = fetch(AUDIO_URL, byte_range="0-16777215")
    wav_name, wav_bytes = extract_first_wav(archive)
    wav_path = SAMPLES_DIR / "ca_es_sample.wav"
    wav_path.write_bytes(wav_bytes)
    print(f"Audio: {wav_path} ({len(wav_bytes)} bytes, origen {wav_name})")

    print("Descargando transcripciones dev.tsv...")
    tsv_path = SAMPLES_DIR / "dev.tsv"
    if not tsv_path.exists():
        tsv_path.write_bytes(fetch(TSV_URL))

    reference = None
    with tsv_path.open(encoding="utf-8") as handle:
        for line in handle:
            columns = line.rstrip("\n").split("\t")
            if len(columns) >= 3 and columns[1] == wav_name:
                reference = columns[2]
                break
    if reference is None:
        raise RuntimeError(f"No se encontro la transcripcion de {wav_name} en dev.tsv")

    reference_path = SAMPLES_DIR / "reference_ca.txt"
    reference_path.write_text(reference.strip() + "\n", encoding="utf-8")
    print(f"Referencia: {reference_path}")

    metadata = {
        "dataset": "google/fleurs",
        "config": "ca_es",
        "split": "dev",
        "source_audio": AUDIO_URL,
        "source_transcripts": TSV_URL,
        "archive_member": wav_name,
        "transcript": reference.strip(),
        "language": "ca",
        "sample_rate_hz": 16000,
        "license": "CC-BY-4.0 (FLEURS)",
    }
    metadata_path = SAMPLES_DIR / "sample_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Metadatos: {metadata_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
