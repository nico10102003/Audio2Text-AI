"""Minimal inference engine built on the official DistilWhisper classes."""

from __future__ import annotations

import gc
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from .chunking import (
    DEFAULT_MAX_CHUNK_S,
    DEFAULT_MIN_SILENCE_S,
    DEFAULT_OVERLAP_S,
    DEFAULT_SILENCE_RMS,
    is_silent,
    merge_chunk_text,
    plan_windows_for_audio,
)
from .compat import import_distilwhisper

MODEL_NAVER_ID = "naver/multilingual-distilwhisper-28k"
MODEL_ID = MODEL_NAVER_ID  # backwards-compatible alias
MODEL_WHISPER_SMALL_ID = "openai/whisper-small"
SUPPORTED_EXPERTS = ("catalan", "tamil", "thai")
WHISPER_SMALL_LANGUAGES = ("es",)
TARGET_SAMPLE_RATE = 16_000
SOUNDFILE_EXTENSIONS = (".wav", ".flac")
FFMPEG_EXTENSIONS = (".mp3", ".m4a", ".aac", ".mp4", ".mov")
SUPPORTED_EXTENSIONS = SOUNDFILE_EXTENSIONS + FFMPEG_EXTENSIONS

_GENERATOR_CLASS = None


def _get_generator_class():
    """Add GenerationMixin to the upstream class without editing it.

    transformers >= 4.50 removed GenerationMixin from PreTrainedModel.  The
    upstream class calls ``super().generate(...)``, so GenerationMixin must be
    placed *after* the upstream class in the MRO.
    """
    global _GENERATOR_CLASS
    if _GENERATOR_CLASS is not None:
        return _GENERATOR_CLASS

    from transformers.generation.utils import GenerationMixin

    DistilWhisper = import_distilwhisper()
    base = DistilWhisper.DistilWhisperForConditionalGeneration

    class DistilWhisperGenerator(base, GenerationMixin):
        @classmethod
        def _supports_default_dynamic_cache(cls) -> bool:
            # The upstream decoder consumes legacy tuples of past_key_values.
            return False

    _GENERATOR_CLASS = DistilWhisperGenerator
    return _GENERATOR_CLASS


@dataclass
class LoadedModel:
    processor: Any
    model: Any
    language: str
    device: torch.device
    dtype: torch.dtype
    load_seconds: float
    model_id: str = MODEL_NAVER_ID


def is_whisper_small_model(loaded: LoadedModel) -> bool:
    """True when the loaded checkpoint is openai/whisper-small."""
    return getattr(loaded, "model_id", MODEL_NAVER_ID) == MODEL_WHISPER_SMALL_ID


@dataclass
class Segment:
    """One transcribed audio window.

    ``start_s``/``end_s`` are window boundaries (segment-level estimates).
    ``raw_text`` keeps the model output before overlap deduplication;
    ``text`` is what was appended to the full transcript.
    """

    index: int
    start_s: float
    end_s: float
    text: str
    token_ids: list[int] = field(default_factory=list)
    raw_text: str = ""
    latency_s: float = 0.0
    silent: bool = False
    peak_vram_mb: float | None = None


@dataclass
class Transcription:
    text: str
    language: str
    device: str
    dtype: str
    duration_audio_s: float
    latency_s: float
    peak_vram_mb: float | None = None
    extra: dict = field(default_factory=dict)
    segments: list[Segment] = field(default_factory=list)


def _resolve_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Se pidio CUDA pero torch.cuda.is_available() es False. "
            "Revisa la instalacion de PyTorch CUDA."
        )
    return torch.device(device)


def _resolve_dtype(dtype: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return mapping[dtype]
    except KeyError as exc:
        raise ValueError(f"dtype no soportado: {dtype!r}; usa {sorted(mapping)}") from exc


def _load_whisper_small(
    language: str,
    device: str = "cuda",
    dtype: str = "float32",
) -> LoadedModel:
    """Load the official `openai/whisper-small` checkpoint for Spanish (`es`)."""
    if language not in WHISPER_SMALL_LANGUAGES:
        raise ValueError(
            f"Whisper Small solo incluye {WHISPER_SMALL_LANGUAGES}; se pidio {language!r}"
        )

    torch_device = _resolve_device(device)
    torch_dtype = _resolve_dtype(dtype)

    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    started = time.perf_counter()
    processor = WhisperProcessor.from_pretrained(MODEL_WHISPER_SMALL_ID)

    load_kwargs: dict[str, Any] = {}
    if torch_dtype != torch.float32:
        load_kwargs["dtype"] = torch_dtype

    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_WHISPER_SMALL_ID, **load_kwargs
    )
    model.to(device=torch_device, dtype=torch_dtype)
    model.eval()
    load_seconds = time.perf_counter() - started

    return LoadedModel(
        processor=processor,
        model=model,
        language=language,
        device=torch_device,
        dtype=torch_dtype,
        load_seconds=load_seconds,
        model_id=MODEL_WHISPER_SMALL_ID,
    )


def load_model(
    model_dir: str | Path,
    language: str = "catalan",
    device: str = "cuda",
    dtype: str = "float32",
    model_id: str = MODEL_NAVER_ID,
) -> LoadedModel:
    """Load the base checkpoint, the requested language expert and the processor.

    ``model_id`` selects the model family: the NAVER 28k DistilWhisper
    (default) or ``openai/whisper-small`` (Spanish, ``language="es"``).
    """
    if model_id == MODEL_WHISPER_SMALL_ID:
        return _load_whisper_small(language=language, device=device, dtype=dtype)

    if language not in SUPPORTED_EXPERTS:
        raise ValueError(f"El checkpoint 28k solo incluye {SUPPORTED_EXPERTS}; se pidio {language!r}")

    torch_device = _resolve_device(device)
    torch_dtype = _resolve_dtype(dtype)
    model_path = Path(model_dir).resolve()
    if not (model_path / "model.safetensors").is_file():
        raise FileNotFoundError(f"No se encontro model.safetensors en {model_path}")

    model_class = _get_generator_class()

    from transformers import GenerationConfig, WhisperProcessor

    started = time.perf_counter()
    processor = WhisperProcessor.from_pretrained(MODEL_WHISPER_SMALL_ID)

    load_kwargs: dict[str, Any] = {}
    if torch_dtype != torch.float32:
        load_kwargs["dtype"] = torch_dtype

    model = model_class.from_pretrained(str(model_path), **load_kwargs)
    model.load_experts(language, expert_local_dir=str(model_path))
    model.set_language_task(language=language, task="transcribe")

    # Modern transformers no longer auto-creates generation_config for custom
    # classes, so load the one shipped with the checkpoint explicitly.
    if model.generation_config is None:
        model.generation_config = GenerationConfig.from_pretrained(str(model_path))

    # Keep the legacy config field in sync for API compatibility, but note that
    # transformers >= 4.50 no longer reads forced_decoder_ids during generate().
    # The prefix is therefore applied explicitly with decoder_input_ids (see
    # _build_decoder_prefix / transcribe).
    forced_ids = processor.get_decoder_prompt_ids(language=language, task="transcribe")
    model.config.forced_decoder_ids = forced_ids
    model.generation_config.forced_decoder_ids = forced_ids

    model.to(device=torch_device, dtype=torch_dtype)
    model.eval()
    load_seconds = time.perf_counter() - started

    return LoadedModel(
        processor=processor,
        model=model,
        language=language,
        device=torch_device,
        dtype=torch_dtype,
        load_seconds=load_seconds,
        model_id=MODEL_NAVER_ID,
    )


def _build_decoder_prefix(loaded: LoadedModel, task: str = "transcribe") -> list[int]:
    """Build the forced Whisper prefix: <|startoftranscript|><|lang|><|task|><|notimestamps|>.

    ``transformers >= 4.50`` ignores ``generation_config.forced_decoder_ids``, so
    the prefix must be fed to ``generate`` as an explicit ``decoder_input_ids``.
    """
    forced_ids = loaded.processor.get_decoder_prompt_ids(
        language=loaded.language, task=task
    )
    start_id = loaded.model.generation_config.decoder_start_token_id
    if start_id is None:
        start_id = loaded.model.config.decoder_start_token_id
    if start_id is None:
        raise RuntimeError("El checkpoint no define decoder_start_token_id")

    positions = [int(pos) for pos, _ in forced_ids]
    expected = list(range(1, len(forced_ids) + 1))
    if positions != expected:
        raise RuntimeError(f"Prefijo de decoder no contiguo: {positions}")

    return [int(start_id)] + [int(tok) for _, tok in forced_ids]


def _generate(
    loaded: LoadedModel,
    input_features: torch.Tensor,
    *,
    max_new_tokens: int,
    num_beams: int,
) -> torch.Tensor:
    """Run generation with the decoder contract of the loaded model family.

    NAVER 28k: validated forced prefix as ``decoder_input_ids``.
    ``openai/whisper-small``: the modern ``WhisperForConditionalGeneration``
    ``language``/``task`` flags (``language="es"`` for Spanish).  transformers
    >= 4.57 rejects ``max_new_tokens`` when prefix + new tokens exceed
    ``max_target_positions`` (448), so the budget is reduced by the prefix
    length; the total generation length stays at ``max_new_tokens``.
    """
    prefix = _build_decoder_prefix(loaded)
    if is_whisper_small_model(loaded):
        new_tokens = max(1, max_new_tokens - len(prefix))
        return loaded.model.generate(
            input_features,
            language=loaded.language,
            task="transcribe",
            num_beams=num_beams,
            max_new_tokens=new_tokens,
        )
    decoder_input_ids = torch.tensor([prefix], dtype=torch.long, device=loaded.device)
    return loaded.model.generate(
        input_features,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        decoder_input_ids=decoder_input_ids,
    )


class AudioReadError(RuntimeError):
    """Raised when an audio file cannot be read or contains invalid samples."""


def _find_ffmpeg() -> str | None:
    """Locate the FFmpeg executable (PATH first, then common install paths)."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
    for candidate in (
        Path(program_files) / "ffmpeg" / "bin" / "ffmpeg.exe",
        Path("C:/ffmpeg/bin/ffmpeg.exe"),
        Path("C:/ffmpeg/ffmpeg.exe"),
    ):
        if candidate.is_file():
            return str(candidate)
    # Winget keeps installed packages under %LOCALAPPDATA%\Microsoft\WinGet\Packages.
    local_app_data = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    winget_packages = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
    if winget_packages.is_dir():
        for candidate in winget_packages.glob("Gyan.FFmpeg*/**/ffmpeg.exe"):
            return str(candidate)
    return None


def _convert_with_ffmpeg(path: Path) -> Path:
    """Convert MP3/M4A/AAC/MP4/MOV to a temporary 16 kHz mono WAV with FFmpeg.

    Returns the temporary WAV path; the caller deletes it after decoding.
    """
    ffmpeg = _find_ffmpeg()
    if ffmpeg is None:
        raise AudioReadError(
            "Formato sin decodificador instalado: se requiere FFmpeg para "
            f"{path.suffix.lower().upper()} (MP3, M4A, AAC, MP4, MOV). Instala "
            "FFmpeg y verifica que `ffmpeg` esté en el PATH."
        )

    descriptor, temp_name = tempfile.mkstemp(prefix="audio2text_ffmpeg_", suffix=".wav")
    os.close(descriptor)
    temp_path = Path(temp_name)
    try:
        command = [
            ffmpeg,
            "-y",
            "-i",
            str(path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(TARGET_SAMPLE_RATE),
            "-f",
            "wav",
            str(temp_path),
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=600,
            )
        except subprocess.TimeoutExpired as exc:
            raise AudioReadError(f"FFmpeg tardó demasiado convirtiendo {path}.") from exc
        except OSError as exc:
            raise AudioReadError(f"No se pudo ejecutar FFmpeg para {path}: {exc}") from exc

        if result.returncode != 0:
            lines = (result.stderr or result.stdout or "").strip().splitlines()
            detail = lines[-1] if lines else ""
            raise AudioReadError(
                f"Error de FFmpeg al decodificar {path}: {detail or 'sin salida de error'}"
            )
        if not temp_path.is_file() or temp_path.stat().st_size == 0:
            raise AudioReadError(f"FFmpeg no produjo audio a partir de {path}.")
        return temp_path
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def read_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Read an audio file as mono float32.

    WAV/FLAC are decoded directly by ``soundfile``; MP3, M4A, AAC, MP4 and
    MOV are converted with FFmpeg to a temporary 16 kHz mono WAV first.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise AudioReadError(f"No existe el archivo de audio: {file_path}")

    read_path = file_path
    converted: Path | None = None
    if file_path.suffix.lower() in FFMPEG_EXTENSIONS:
        converted = _convert_with_ffmpeg(file_path)
        read_path = converted

    try:
        try:
            audio, sample_rate = sf.read(str(read_path), dtype="float32", always_2d=False)
        except Exception as exc:  # soundfile raises different types per backend
            raise AudioReadError(
                f"No se pudo leer {file_path}. Formatos soportados: WAV/FLAC "
                f"(soundfile) y MP3/M4A/AAC/MP4/MOV (FFmpeg). Detalle: {exc}"
            ) from exc
    finally:
        if converted is not None:
            converted.unlink(missing_ok=True)

    if sample_rate <= 0:
        raise AudioReadError(f"Frecuencia de muestreo invalida en {file_path}: {sample_rate}")

    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return np.zeros(0, dtype=np.float32), int(sample_rate)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if not np.isfinite(audio).all():
        raise AudioReadError(f"El audio contiene NaN o infinitos: {file_path}")

    if sample_rate != TARGET_SAMPLE_RATE:
        audio = _resample_linear(audio, sample_rate, TARGET_SAMPLE_RATE)
        sample_rate = TARGET_SAMPLE_RATE
    return np.ascontiguousarray(audio, dtype=np.float32), sample_rate


def _resample_linear(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Small dependency-free linear resampler (adequate for a smoke test)."""
    if src_rate == dst_rate or audio.size == 0:
        return audio
    duration = audio.shape[0] / float(src_rate)
    target_length = int(round(duration * dst_rate))
    src_positions = np.arange(audio.shape[0], dtype=np.float64)
    dst_positions = np.linspace(0.0, audio.shape[0] - 1, target_length, dtype=np.float64)
    return np.interp(dst_positions, src_positions, audio).astype(np.float32)


@torch.inference_mode()
def transcribe(
    loaded: LoadedModel,
    audio: np.ndarray,
    sample_rate: int = TARGET_SAMPLE_RATE,
    max_new_tokens: int = 448,
    num_beams: int = 5,
) -> Transcription:
    """Run beam-search generation with the forced prefix and the loaded expert."""
    if sample_rate != TARGET_SAMPLE_RATE:
        audio = _resample_linear(audio, sample_rate, TARGET_SAMPLE_RATE)

    inputs = loaded.processor(
        audio,
        sampling_rate=TARGET_SAMPLE_RATE,
        return_tensors="pt",
    )
    input_features = inputs.input_features.to(device=loaded.device, dtype=loaded.dtype)
    prefix = _build_decoder_prefix(loaded)

    if loaded.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(loaded.device)
        torch.cuda.synchronize(loaded.device)

    started = time.perf_counter()
    predicted_ids = _generate(
        loaded,
        input_features,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
    )
    if loaded.device.type == "cuda":
        torch.cuda.synchronize(loaded.device)
    latency = time.perf_counter() - started

    token_ids = predicted_ids[0].tolist()
    text = loaded.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    peak_vram_mb = None
    if loaded.device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated(loaded.device) / (1024**2)

    duration_s = len(audio) / TARGET_SAMPLE_RATE

    return Transcription(
        text=text.strip(),
        language=loaded.language,
        device=str(loaded.device),
        dtype=str(loaded.dtype).replace("torch.", ""),
        duration_audio_s=duration_s,
        latency_s=latency,
        peak_vram_mb=peak_vram_mb,
        extra={
            "num_tokens": len(token_ids),
            "num_beams": num_beams,
            "decoder_prefix": prefix,
            "token_ids": token_ids,
        },
    )


def _empty_transcription(loaded: LoadedModel, num_beams: int) -> Transcription:
    """Result for an audio file with no samples; the model is never called."""
    return Transcription(
        text="",
        language=loaded.language,
        device=str(loaded.device),
        dtype=str(loaded.dtype).replace("torch.", ""),
        duration_audio_s=0.0,
        latency_s=0.0,
        peak_vram_mb=None,
        extra={
            "num_tokens": 0,
            "num_beams": num_beams,
            "decoder_prefix": [],
            "token_ids": [],
            "num_chunks": 0,
            "chunking": {
                "strategy": "single",
                "num_chunks": 0,
                "num_silent_chunks": 0,
                "duplicate_words_removed": 0,
            },
        },
        segments=[],
    )


@torch.inference_mode()
def transcribe_long(
    loaded: LoadedModel,
    audio: np.ndarray,
    sample_rate: int = TARGET_SAMPLE_RATE,
    max_new_tokens: int = 448,
    num_beams: int = 5,
    max_chunk_s: float = DEFAULT_MAX_CHUNK_S,
    overlap_s: float = DEFAULT_OVERLAP_S,
    silence_rms: float = DEFAULT_SILENCE_RMS,
    min_silence_s: float = DEFAULT_MIN_SILENCE_S,
) -> Transcription:
    """Transcribe audio of any length with silence-aware 30 s windows.

    Windows end at a natural pause when one is available before the 30 s limit;
    otherwise they use the nominal boundary and the next window starts
    ``overlap_s`` earlier.  Every non-silent window keeps the validated
    four-token decoder prefix and the same generation settings as the short
    path.  Overlap text is removed only when the windows really overlap, and
    each segment gets window-level timestamps.
    """
    if sample_rate != TARGET_SAMPLE_RATE:
        audio = _resample_linear(audio, sample_rate, TARGET_SAMPLE_RATE)
        sample_rate = TARGET_SAMPLE_RATE

    if audio.size == 0:
        return _empty_transcription(loaded, num_beams)

    total_duration_s = len(audio) / TARGET_SAMPLE_RATE
    windows = plan_windows_for_audio(
        audio,
        TARGET_SAMPLE_RATE,
        max_chunk_s=max_chunk_s,
        overlap_s=overlap_s,
        silence_rms=silence_rms,
        min_silence_s=min_silence_s,
    )
    if not windows:
        return _empty_transcription(loaded, num_beams)

    prefix: list[int] | None = None
    segments: list[Segment] = []
    full_text = ""
    total_tokens = 0
    silent_chunks = 0
    duplicate_words = 0
    peak_vram_mb: float | None = None
    previous_end_s: float | None = None

    started = time.perf_counter()
    for window in windows:
        start_sample, end_sample = window.sample_bounds(TARGET_SAMPLE_RATE)
        chunk = audio[start_sample:end_sample]
        chunk_started = time.perf_counter()

        if is_silent(chunk, threshold=silence_rms):
            silent_chunks += 1
            previous_end_s = window.end_s
            segments.append(
                Segment(
                    index=window.index,
                    start_s=window.start_s,
                    end_s=window.end_s,
                    text="",
                    token_ids=[],
                    raw_text="",
                    latency_s=0.0,
                    silent=True,
                    peak_vram_mb=None,
                )
            )
            continue

        if prefix is None:
            prefix = _build_decoder_prefix(loaded)

        if loaded.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(loaded.device)
            torch.cuda.synchronize(loaded.device)

        inputs = loaded.processor(
            chunk,
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(
            device=loaded.device, dtype=loaded.dtype
        )
        predicted_ids = _generate(
            loaded,
            input_features,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
        if loaded.device.type == "cuda":
            torch.cuda.synchronize(loaded.device)

        chunk_latency = time.perf_counter() - chunk_started
        chunk_peak = None
        if loaded.device.type == "cuda":
            chunk_peak = torch.cuda.max_memory_allocated(loaded.device) / (1024**2)
            peak_vram_mb = chunk_peak if peak_vram_mb is None else max(peak_vram_mb, chunk_peak)

        token_ids = [int(token) for token in predicted_ids[0].tolist()]
        raw_text = loaded.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0].strip()

        # Release per-chunk GPU/CPU references before the next window.
        del inputs, input_features, predicted_ids
        gc.collect()
        if loaded.device.type == "cuda":
            torch.cuda.empty_cache()

        merged_text, removed = (
            merge_chunk_text(full_text, raw_text)
            if previous_end_s is not None and window.start_s < previous_end_s - 1e-6
            else (
                (f"{full_text} {raw_text}".strip() if full_text else raw_text),
                0,
            )
        )
        segment_text = " ".join(raw_text.split()[removed:]).strip()
        full_text = merged_text
        previous_end_s = window.end_s
        total_tokens += len(token_ids)
        duplicate_words += removed

        segments.append(
            Segment(
                index=window.index,
                start_s=window.start_s,
                end_s=window.end_s,
                text=segment_text,
                token_ids=token_ids,
                raw_text=raw_text,
                latency_s=chunk_latency,
                silent=False,
                peak_vram_mb=chunk_peak,
            )
        )

    latency = time.perf_counter() - started
    return Transcription(
        text=full_text.strip(),
        language=loaded.language,
        device=str(loaded.device),
        dtype=str(loaded.dtype).replace("torch.", ""),
        duration_audio_s=total_duration_s,
        latency_s=latency,
        peak_vram_mb=peak_vram_mb,
        segments=segments,
        extra={
            "num_tokens": total_tokens,
            "num_beams": num_beams,
            "decoder_prefix": prefix or [],
            "token_ids": None,
            "num_chunks": len(windows),
            "chunking": {
                "strategy": "silence-aware-overlap",
                "max_chunk_s": max_chunk_s,
                "overlap_s": overlap_s,
                "silence_rms": silence_rms,
                "min_silence_s": min_silence_s,
                "num_chunks": len(windows),
                "num_silent_chunks": silent_chunks,
                "duplicate_words_removed": duplicate_words,
            },
        },
    )


def transcribe_auto(
    loaded: LoadedModel,
    audio: np.ndarray,
    sample_rate: int = TARGET_SAMPLE_RATE,
    max_new_tokens: int = 448,
    num_beams: int = 5,
    max_chunk_s: float = DEFAULT_MAX_CHUNK_S,
    overlap_s: float = DEFAULT_OVERLAP_S,
    silence_rms: float = DEFAULT_SILENCE_RMS,
    min_silence_s: float = DEFAULT_MIN_SILENCE_S,
    chunking: bool = True,
) -> Transcription:
    """Use the validated single-window path for <= 30 s, chunking above it."""
    if sample_rate != TARGET_SAMPLE_RATE:
        audio = _resample_linear(audio, sample_rate, TARGET_SAMPLE_RATE)
        sample_rate = TARGET_SAMPLE_RATE

    duration_s = len(audio) / TARGET_SAMPLE_RATE
    if duration_s == 0:
        return _empty_transcription(loaded, num_beams)

    if not chunking or duration_s <= max_chunk_s:
        result = transcribe(
            loaded,
            audio,
            sample_rate=TARGET_SAMPLE_RATE,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
        result.segments = [
            Segment(
                index=0,
                start_s=0.0,
                end_s=result.duration_audio_s,
                text=result.text,
                token_ids=result.extra.get("token_ids") or [],
                raw_text=result.text,
                latency_s=result.latency_s,
                silent=not result.text.strip(),
                peak_vram_mb=result.peak_vram_mb,
            )
        ]
        result.extra["num_chunks"] = 1
        result.extra["chunking"] = {
            "strategy": "single",
            "max_chunk_s": max_chunk_s,
            "overlap_s": 0.0,
            "num_chunks": 1,
            "num_silent_chunks": int(not result.text.strip()),
            "duplicate_words_removed": 0,
        }
        return result

    return transcribe_long(
        loaded,
        audio,
        sample_rate=TARGET_SAMPLE_RATE,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        max_chunk_s=max_chunk_s,
        overlap_s=overlap_s,
        silence_rms=silence_rms,
        min_silence_s=min_silence_s,
    )
