"""Run a single-file transcription with the official DistilWhisper code."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402

from audio2text.engine import (  # noqa: E402
    MODEL_NAVER_ID,
    MODEL_WHISPER_SMALL_ID,
    AudioReadError,
    load_model,
    read_audio,
    transcribe_auto,
)
from audio2text.metrics import character_error_rate, word_error_rate  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, default=PROJECT_ROOT / "data" / "samples" / "ca_es_sample.wav")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models" / "multilingual-distilwhisper-28k")
    parser.add_argument(
        "--model-id",
        choices=[MODEL_NAVER_ID, MODEL_WHISPER_SMALL_ID],
        default=MODEL_NAVER_ID,
        help="NAVER 28k (idiomas catalán/tamil/tailandés) o OpenAI Whisper Small (español).",
    )
    parser.add_argument("--language", choices=["catalan", "tamil", "thai", "es"], default=None)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default=None,
        help="Por defecto: float16 en CUDA, float32 en CPU",
    )
    parser.add_argument("--max-new-tokens", type=int, default=448)
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--max-chunk-seconds", type=float, default=30.0)
    parser.add_argument("--chunk-overlap", type=float, default=2.0)
    parser.add_argument("--silence-rms", type=float, default=1e-4)
    parser.add_argument(
        "--min-silence-seconds",
        type=float,
        default=0.4,
        help="Duracion minima de silencio para cortar un fragmento",
    )
    parser.add_argument("--no-chunk", action="store_true", help="Forzar una sola ventana (trunca a 30 s)")
    parser.add_argument("--reference", type=Path, help="Transcripcion de referencia para calcular CER/WER")
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    # Whisper Small solo soporta español; el default del NAVER sigue siendo catalán.
    language = args.language or ("es" if args.model_id == MODEL_WHISPER_SMALL_ID else "catalan")
    dtype = args.dtype or ("float16" if args.device == "cuda" else "float32")

    print(f"Python      : {sys.version.split()[0]}")
    print(f"Torch       : {torch.__version__} (CUDA {torch.version.cuda})")
    print(f"Device      : {args.device} / {dtype}")
    print(f"Audio       : {args.audio}")
    model_dir = Path(MODEL_WHISPER_SMALL_ID) if args.model_id == MODEL_WHISPER_SMALL_ID else args.model_dir
    print(f"Model       : {model_dir} ({args.model_id})")
    print(f"Language    : {language}")
    print(
        "Chunking    : "
        + (
            "desactivado"
            if args.no_chunk
            else f"max {args.max_chunk_seconds:.1f} s / solape {args.chunk_overlap:.1f} s"
        )
    )

    try:
        audio, sample_rate = read_audio(args.audio)
    except AudioReadError as exc:
        print(f"Error de audio: {exc}", file=sys.stderr)
        return 2
    print(f"Audio shape : {audio.shape} @ {sample_rate} Hz ({len(audio) / sample_rate:.2f} s)")

    started = time.perf_counter()
    loaded = load_model(
        model_dir=model_dir,
        language=language,
        device=args.device,
        dtype=dtype,
        model_id=args.model_id,
    )
    print(f"Load time   : {loaded.load_seconds:.2f} s")
    if loaded.device.type == "cuda":
        free, total = torch.cuda.mem_get_info(loaded.device)
        print(
            "VRAM        : "
            f"{(total - free) / 1024**2:.0f} MiB usados / {total / 1024**2:.0f} MiB totales"
        )

    result = transcribe_auto(
        loaded,
        audio,
        sample_rate=sample_rate,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        max_chunk_s=args.max_chunk_seconds,
        overlap_s=args.chunk_overlap,
        silence_rms=args.silence_rms,
        min_silence_s=args.min_silence_seconds,
        chunking=not args.no_chunk,
    )
    total_seconds = time.perf_counter() - started

    print("\n--- TRANSCRIPCION ---")
    print(result.text)
    print("--- FIN ---\n")
    print(f"Latencia    : {result.latency_s:.2f} s")
    print(f"Total       : {total_seconds:.2f} s")
    if result.peak_vram_mb is not None:
        print(f"VRAM pico   : {result.peak_vram_mb:.0f} MiB")
    print(f"Tokens      : {result.extra['num_tokens']}")
    print(f"Segmentos   : {len(result.segments)}")
    for segment in result.segments:
        label = "silencio" if segment.silent else segment.text
        print(
            f"  [{segment.index}] {segment.start_s:6.2f} - {segment.end_s:6.2f} s "
            f"({segment.end_s - segment.start_s:5.2f} s): {label}"
        )
    chunking = result.extra.get("chunking") or {}
    if chunking.get("duplicate_words_removed"):
        print(f"Solape      : {chunking['duplicate_words_removed']} palabras duplicadas eliminadas")

    cer = wer = None
    if args.reference:
        reference = args.reference.read_text(encoding="utf-8").strip()
        cer = character_error_rate(reference, result.text)
        wer = word_error_rate(reference, result.text)
        print(f"Referencia  : {reference}")
        print(f"CER         : {cer:.4f}")
        print(f"WER         : {wer:.4f}")

    payload = {
        "text": result.text,
        "segments": [
            {
                "index": segment.index,
                "start_s": round(segment.start_s, 3),
                "end_s": round(segment.end_s, 3),
                "duration_s": round(segment.end_s - segment.start_s, 3),
                "text": segment.text,
                "raw_text": segment.raw_text,
                "silent": segment.silent,
                "latency_s": segment.latency_s,
                "peak_vram_mb": segment.peak_vram_mb,
                "token_ids": segment.token_ids,
            }
            for segment in result.segments
        ],
        "chunking": result.extra.get("chunking"),
        "num_chunks": result.extra.get("num_chunks", 0),
        "language": result.language,
        "device": result.device,
        "dtype": result.dtype,
        "audio_duration_s": result.duration_audio_s,
        "load_seconds": loaded.load_seconds,
        "latency_s": result.latency_s,
        "total_seconds": total_seconds,
        "peak_vram_mb": result.peak_vram_mb,
        "num_tokens": result.extra["num_tokens"],
        "num_beams": args.num_beams,
        "decoder_prefix": result.extra["decoder_prefix"],
        "token_ids": result.extra["token_ids"],
        "cer": cer,
        "wer": wer,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "model_dir": str(args.model_dir),
        "audio": str(args.audio),
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"JSON        : {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
