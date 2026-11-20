"""Pure helpers shared by the Streamlit interface and its tests.

The interface must not contain model logic.  This module only prepares uploads,
serialises results and formats values for display; every model call still goes
through :mod:`audio2text.engine`.
"""

from __future__ import annotations

import json
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .engine import (
    MODEL_NAVER_ID,
    MODEL_WHISPER_SMALL_ID,
    SUPPORTED_EXPERTS,
    SUPPORTED_EXTENSIONS,
    WHISPER_SMALL_LANGUAGES,
    AudioReadError,
    Segment,
    Transcription,
    read_audio,
)

SUPPORTED_LANGUAGES = tuple(SUPPORTED_EXPERTS)
MODEL_OPTIONS = (
    (MODEL_NAVER_ID, "NAVER Multilingual DistilWhisper 28k"),
    (MODEL_WHISPER_SMALL_ID, "OpenAI Whisper Small"),
)
LANGUAGE_LABELS = {
    "catalan": "Català",
    "tamil": "Tamil",
    "thai": "Tailandès",
    "es": "Español",
}
DEVICE_LABELS = {
    "cuda": "CUDA (GPU)",
    "cpu": "CPU",
}
DTYPE_LABELS = {
    "float16": "FP16",
    "float32": "FP32",
}
MIME_TYPES = {
    ".wav": "audio/wav",
    ".flac": "audio/flac",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".mp4": "audio/mp4",
    ".mov": "video/quicktime",
}
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
MAX_HISTORY = 20


class AudioValidationError(ValueError):
    """Raised when an uploaded file is empty, unsupported or not decodable."""


@dataclass(frozen=True)
class UploadedAudio:
    """A validated audio upload ready for display and inference."""

    filename: str
    suffix: str
    data: bytes
    samples: np.ndarray
    sample_rate: int
    duration_s: float

    @property
    def format_label(self) -> str:
        return self.suffix.lstrip(".").upper()

    @property
    def mime_type(self) -> str:
        return MIME_TYPES.get(self.suffix, "application/octet-stream")


@dataclass(frozen=True)
class TranscriptionConfig:
    """Configuration that identifies a model load and a transcription run."""

    language: str
    device: str
    dtype: str
    num_beams: int
    model_dir: str
    model_id: str = MODEL_NAVER_ID


@dataclass
class TranscriptionRecord:
    """A completed transcription kept in the Streamlit session history."""

    record_id: str
    created_at: datetime
    filename: str
    file_format: str
    audio_duration_s: float
    language: str
    device: str
    dtype: str
    num_beams: int
    model_dir: str
    model_id: str
    text: str
    latency_s: float
    peak_vram_mb: float | None
    load_seconds: float
    num_tokens: int
    decoder_prefix: list[int]
    chunking: dict[str, Any]
    num_chunks: int
    segments: list[Segment]


def model_label(model_id: str) -> str:
    """Friendly display label for a model id (falls back to the id itself)."""
    for option_id, label in MODEL_OPTIONS:
        if option_id == model_id:
            return label
    return model_id


def languages_for_model(model_id: str) -> tuple[str, ...]:
    """Language options offered for a model: NAVER keeps its three; Whisper Small is Spanish."""
    if model_id == MODEL_WHISPER_SMALL_ID:
        return tuple(WHISPER_SMALL_LANGUAGES)
    return SUPPORTED_LANGUAGES


def language_label(language: str) -> str:
    return LANGUAGE_LABELS.get(language, language)


def device_label(device: str) -> str:
    return DEVICE_LABELS.get(device, device)


def dtype_label(dtype: str) -> str:
    return DTYPE_LABELS.get(dtype, dtype)


def default_dtype(device: str) -> str:
    """FP16 is the validated CUDA default; CPU uses FP32."""
    return "float16" if device == "cuda" else "float32"


def dtype_options(device: str) -> list[str]:
    """Only expose precisions that are compatible with the selected device."""
    return ["float16", "float32"] if device == "cuda" else ["float32"]


def format_duration(seconds: float) -> str:
    """Compact duration for cards and history labels."""
    value = max(0.0, float(seconds))
    if value < 60:
        return f"{value:.2f} s"
    return format_timestamp(value)


def format_timestamp(seconds: float) -> str:
    """Format a segment boundary as ``MM:SS.mmm`` or ``HH:MM:SS.mmm``.

    These are window boundaries from the engine.
    """
    total_ms = max(0, int(round(float(seconds) * 1000)))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    return f"{minutes:02d}:{secs:02d}.{millis:03d}"


def format_file_size(size_bytes: int) -> str:
    value = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def format_vram(peak_vram_mb: float | None) -> str:
    return "—" if peak_vram_mb is None else f"{peak_vram_mb:.0f} MiB"


def config_label(record: TranscriptionRecord) -> str:
    return (
        f"{language_label(record.language)} · {device_label(record.device)} · "
        f"{dtype_label(record.dtype)} · beam {record.num_beams}"
    )


def load_uploaded_audio(filename: str, data: bytes) -> UploadedAudio:
    """Validate and decode an uploaded audio file using the engine's reader.

    ``read_audio`` takes a path, so the bytes are written to a temporary file
    with the original suffix.  The temporary file is deleted immediately after
    decoding; the resulting samples stay in memory for the current session.
    """
    safe_name = Path(filename or "").name
    if not safe_name:
        raise AudioValidationError("El archivo no tiene nombre.")
    if not data:
        raise AudioValidationError("El archivo está vacío.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise AudioValidationError(
            f"El archivo supera el límite de {MAX_UPLOAD_BYTES // (1024**2)} MB."
        )

    suffix = Path(safe_name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise AudioValidationError(
            "Formato no soportado. Sube un archivo "
            "WAV, FLAC, MP3, M4A, AAC, MP4 o MOV."
        )

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="audio2text_upload_", suffix=suffix, delete=False
        ) as temporary:
            temporary.write(data)
            temporary_path = Path(temporary.name)

        try:
            samples, sample_rate = read_audio(temporary_path)
        except AudioReadError as exc:
            raise AudioValidationError(
                "No se pudo decodificar el audio. Comprueba que el archivo "
                "WAV, FLAC, MP3, M4A, AAC, MP4 o MOV sea válido."
            ) from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    if samples.size == 0:
        raise AudioValidationError("El archivo no contiene muestras de audio.")
    if sample_rate <= 0:
        raise AudioValidationError("La frecuencia de muestreo del audio no es válida.")

    duration_s = float(len(samples)) / float(sample_rate)
    return UploadedAudio(
        filename=safe_name,
        suffix=suffix,
        data=data,
        samples=samples,
        sample_rate=sample_rate,
        duration_s=duration_s,
    )


def make_record(
    *,
    audio: UploadedAudio,
    result: Transcription,
    config: TranscriptionConfig,
    load_seconds: float,
    record_id: str | None = None,
    created_at: datetime | None = None,
) -> TranscriptionRecord:
    """Build the session record from an engine result without altering it."""
    extra = result.extra or {}
    chunking = dict(extra.get("chunking") or {})
    segments = list(result.segments)
    return TranscriptionRecord(
        record_id=record_id or uuid.uuid4().hex[:12],
        created_at=created_at or datetime.now().astimezone(),
        filename=audio.filename,
        file_format=audio.format_label,
        audio_duration_s=float(result.duration_audio_s),
        language=result.language or config.language,
        device=str(result.device),
        dtype=str(result.dtype).replace("torch.", ""),
        num_beams=int(config.num_beams),
        model_dir=config.model_dir,
        model_id=config.model_id,
        text=result.text,
        latency_s=float(result.latency_s),
        peak_vram_mb=result.peak_vram_mb,
        load_seconds=float(load_seconds),
        num_tokens=int(extra.get("num_tokens") or 0),
        decoder_prefix=[int(token) for token in (extra.get("decoder_prefix") or [])],
        chunking=chunking,
        num_chunks=int(extra.get("num_chunks") or len(segments)),
        segments=segments,
    )


def add_record(
    history: list[TranscriptionRecord],
    record: TranscriptionRecord,
    *,
    limit: int = MAX_HISTORY,
) -> list[TranscriptionRecord]:
    """Put the newest record first and keep the session history bounded."""
    remaining = [item for item in history if item.record_id != record.record_id]
    return [record, *remaining][:limit]


def find_record(
    history: list[TranscriptionRecord], record_id: str | None
) -> TranscriptionRecord | None:
    if record_id is None:
        return None
    for record in history:
        if record.record_id == record_id:
            return record
    return None


def history_label(record: TranscriptionRecord) -> str:
    return (
        f"{record.created_at.strftime('%H:%M:%S')} · {record.filename} · "
        f"{format_duration(record.audio_duration_s)} · {language_label(record.language)}"
    )


def segment_rows(record: TranscriptionRecord) -> list[dict[str, Any]]:
    """Rows for ``st.dataframe``; timestamps are window-level estimates."""
    rows: list[dict[str, Any]] = []
    for segment in record.segments:
        text = segment.text.strip()
        if not text and segment.silent:
            text = "(silencio)"
        rows.append(
            {
                "Segmento": segment.index + 1,
                "Inicio": format_timestamp(segment.start_s),
                "Fin": format_timestamp(segment.end_s),
                "Duración": f"{max(0.0, segment.end_s - segment.start_s):.2f} s",
                "Texto": text,
            }
        )
    return rows


def record_to_payload(record: TranscriptionRecord) -> dict[str, Any]:
    """JSON-ready payload mirroring the CLI result schema plus UI metadata."""
    return {
        "app": "Audio2Text AI",
        "created_at": record.created_at.isoformat(timespec="seconds"),
        "text": record.text,
        "segments": [
            {
                "index": segment.index,
                "start_s": round(segment.start_s, 3),
                "end_s": round(segment.end_s, 3),
                "duration_s": round(segment.end_s - segment.start_s, 3),
                "text": segment.text,
                "raw_text": segment.raw_text,
                "silent": segment.silent,
                "latency_s": round(segment.latency_s, 3),
                "peak_vram_mb": segment.peak_vram_mb,
                "token_ids": segment.token_ids,
            }
            for segment in record.segments
        ],
        "chunking": record.chunking,
        "num_chunks": record.num_chunks,
        "language": record.language,
        "device": record.device,
        "dtype": record.dtype,
        "audio_duration_s": round(record.audio_duration_s, 3),
        "audio_filename": record.filename,
        "audio_format": record.file_format,
        "load_seconds": round(record.load_seconds, 3),
        "latency_s": round(record.latency_s, 3),
        "peak_vram_mb": record.peak_vram_mb,
        "num_tokens": record.num_tokens,
        "num_beams": record.num_beams,
        "decoder_prefix": record.decoder_prefix,
        "model_id": record.model_id,
        "model_dir": record.model_dir,
    }


def record_to_json(record: TranscriptionRecord) -> str:
    return json.dumps(
        record_to_payload(record), ensure_ascii=False, indent=2, default=str
    )


def record_to_txt(record: TranscriptionRecord) -> str:
    """Human-readable download with metadata and segment boundaries."""
    lines = [
        "Audio2Text AI",
        f"Archivo: {record.filename}",
        f"Formato: {record.file_format}",
        f"Duración: {format_duration(record.audio_duration_s)}",
        f"Idioma: {language_label(record.language)}",
        f"Dispositivo: {record.device} · {dtype_label(record.dtype)}",
        f"Beam search: {record.num_beams}",
        f"Inferencia: {record.latency_s:.2f} s",
        f"Creado: {record.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Modelo: {record.model_id}",
        "",
        "TRANSCRIPCIÓN",
        "=" * 13,
        record.text or "(sin voz detectada)",
    ]
    if record.segments:
        lines.extend(["", "SEGMENTOS (estimaciones por ventana)", "=" * 37])
        for segment in record.segments:
            text = segment.text.strip()
            if not text and segment.silent:
                text = "(silencio)"
            lines.append(
                f"[{format_timestamp(segment.start_s)} - "
                f"{format_timestamp(segment.end_s)}] {text}"
            )
    return "\n".join(lines) + "\n"


def download_filename(record: TranscriptionRecord, extension: str) -> str:
    stem = Path(record.filename).stem or "transcripcion"
    return f"{stem}.{extension.lstrip('.')}"


def friendly_error(exc: BaseException) -> str:
    """Translate engine/runtime errors into actionable UI messages."""
    if isinstance(exc, AudioValidationError):
        return str(exc)
    if isinstance(exc, AudioReadError):
        return (
            "No se pudo leer el audio. Comprueba que el archivo sea válido; si es "
            "MP3/M4A/AAC/MP4/MOV verifica también que FFmpeg esté instalado y en el PATH."
        )
    if isinstance(exc, FileNotFoundError):
        return (
            "No se encontró el modelo local. Verifica que los pesos estén en "
            "models/multilingual-distilwhisper-28k."
        )

    message = str(exc).strip()
    lower = message.lower()
    if isinstance(exc, MemoryError) or "out of memory" in lower:
        return (
            "La GPU no tiene memoria suficiente. Prueba con FP16, libera el modelo "
            "desde la barra lateral o usa CPU."
        )
    if "cuda" in lower and (
        "not available" in lower or "no cuda" in lower or "not compiled" in lower
    ):
        return "CUDA no está disponible en este equipo. Selecciona CPU."
    if isinstance(exc, ValueError) and message:
        return f"Configuración no válida: {message}"
    if message:
        return f"No se pudo completar la transcripción: {message}"
    return "No se pudo completar la transcripción por un error inesperado."
