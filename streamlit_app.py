"""Local Streamlit interface for the validated Audio2Text AI engine."""

from __future__ import annotations

import gc
import logging
import sys
import time
from pathlib import Path

import streamlit as st
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from audio2text.engine import (  # noqa: E402
    MODEL_NAVER_ID,
    MODEL_WHISPER_SMALL_ID,
    load_model,
    transcribe_auto,
)
from audio2text.ui_service import (  # noqa: E402
    MODEL_OPTIONS,
    AudioValidationError,
    TranscriptionConfig,
    add_record,
    config_label,
    default_dtype,
    device_label,
    download_filename,
    dtype_label,
    dtype_options,
    find_record,
    format_duration,
    format_file_size,
    format_vram,
    friendly_error,
    history_label,
    language_label,
    languages_for_model,
    load_uploaded_audio,
    make_record,
    model_label,
    record_to_json,
    record_to_txt,
    segment_rows,
)

MODEL_ID = MODEL_NAVER_ID
MODEL_DIR = PROJECT_ROOT / "models" / "multilingual-distilwhisper-28k"
DEFAULT_BEAM = 5
LOGGER = logging.getLogger(__name__)

st.set_page_config(
    page_title="Audio2Text AI",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _release_cached_model(loaded) -> None:
    """Release GPU/CPU references when Streamlit evicts a cached model."""
    for attribute in ("model", "processor"):
        try:
            delattr(loaded, attribute)
        except AttributeError:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@st.cache_resource(show_spinner=False, max_entries=1, on_release=_release_cached_model)
def get_model(model_dir: str, language: str, device: str, dtype: str, model_id: str):
    """Load and cache one model configuration at a time.

    ``max_entries=1`` avoids keeping two ~1 GB checkpoints in the 4 GB GPU.
    The engine still owns all model loading logic.
    """
    return load_model(
        model_dir=Path(model_dir),
        language=language,
        device=device,
        dtype=dtype,
        model_id=model_id,
    )


def _init_state() -> None:
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("active_record_id", None)
    st.session_state.setdefault("language", "catalan")
    st.session_state.setdefault("model", MODEL_NAVER_ID)


def model_dir_for(model_id: str) -> str:
    """Directory (local path or Hugging Face id) that identifies a model."""
    return MODEL_WHISPER_SMALL_ID if model_id == MODEL_WHISPER_SMALL_ID else str(MODEL_DIR)


def _record_for_history_label(label: str):
    for record in st.session_state.history:
        if history_label(record) == label:
            return record
    return None


def _sync_active_record() -> None:
    record = _record_for_history_label(st.session_state.get("history_select", ""))
    if record is not None:
        st.session_state.active_record_id = record.record_id


def _render_sidebar() -> tuple[str, str, str, int, str]:
    cuda_available = torch.cuda.is_available()

    if not cuda_available:
        st.session_state.device = "cpu"
    else:
        st.session_state.setdefault("device", "cuda")

    with st.sidebar:
        st.markdown("## Configuración")
        st.caption("El modelo se mantiene en caché entre interacciones.")

        model_id = st.selectbox(
            "Modelo",
            options=[option_id for option_id, _label in MODEL_OPTIONS],
            format_func=model_label,
            key="model",
            help=(
                "NAVER: expertos para catalán, tamil y tailandés (modelo académico "
                "principal). Whisper Small: español (language='es')."
            ),
        )

        languages = languages_for_model(model_id)
        if model_id == MODEL_WHISPER_SMALL_ID:
            # Whisper Small transcribe en español: selección automática.
            st.session_state["language"] = "es"
        elif st.session_state.get("language") not in languages:
            st.session_state["language"] = languages[0]

        language = st.selectbox(
            "Idioma",
            options=languages,
            format_func=language_label,
            key="language",
            help="Whisper Small solo soporta español; el 28k de NAVER, catalán, tamil y tailandés.",
        )

        device = st.radio(
            "Dispositivo",
            options=["cuda", "cpu"] if cuda_available else ["cpu"],
            format_func=device_label,
            key="device",
            horizontal=True,
            help="CUDA es la opción recomendada si hay una GPU NVIDIA disponible.",
        )

        allowed_dtypes = dtype_options(device)
        if st.session_state.get("dtype") not in allowed_dtypes:
            st.session_state.dtype = allowed_dtypes[0]
        dtype = st.selectbox(
            "Precisión",
            options=allowed_dtypes,
            format_func=dtype_label,
            key="dtype",
            help="FP16 reduce la VRAM en CUDA. En CPU se usa FP32.",
        )

        num_beams = st.slider(
            "Beam search",
            min_value=1,
            max_value=10,
            value=DEFAULT_BEAM,
            step=1,
            key="beam",
            help="Valor recomendado: 5. Un valor mayor puede mejorar la calidad a "
            "cambio de más latencia.",
        )

        st.divider()
        st.caption("Entorno de ejecución")
        if cuda_available:
            try:
                gpu_name = torch.cuda.get_device_name(0)
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                st.markdown(f"**CUDA** · {gpu_name}")
                st.caption(
                    f"{free_bytes / 1024**2:.0f} MiB libres de "
                    f"{total_bytes / 1024**2:.0f} MiB"
                )
            except Exception:
                st.markdown("**CUDA** · disponible")
        else:
            st.markdown("**CPU** · CUDA no disponible")

        st.caption(f"Modelo: {model_label(model_id)}")
        if st.button(
            "Liberar modelo de memoria",
            key="release_model",
            width="stretch",
            help="Vacía la caché del modelo y libera la VRAM.",
        ):
            get_model.clear()
            gc.collect()
            if cuda_available:
                torch.cuda.empty_cache()
            st.session_state["_model_released"] = True
            st.rerun()

        if st.session_state.pop("_model_released", False):
            st.success("Modelo liberado de la caché.")

        history = st.session_state.history
        st.divider()
        st.caption(
            f"{len(history)} transcripción(es) en esta sesión"
            if history
            else "Historial vacío"
        )

    return language, device, dtype, num_beams, model_id


def _get_uploaded_audio(uploaded):
    """Reuse the decoded upload across reruns and invalidate it on change."""
    file_id = getattr(uploaded, "file_id", "") or ""
    cache_key = f"{file_id}|{uploaded.name}|{uploaded.size}"
    cached = st.session_state.get("_audio_cache")
    if cached and cached[0] == cache_key:
        return cached[1]

    data = uploaded.getvalue()
    audio = load_uploaded_audio(uploaded.name, data)
    st.session_state["_audio_cache"] = (cache_key, audio)
    return audio


def _load_tab_audio(recording):
    """Validate a file upload or a microphone recording and decode it in memory.

    Returns the ``UploadedAudio`` for transcription, or ``None`` when the
    payload is invalid (an error is shown); the temporary decode file of
    ``load_uploaded_audio`` is deleted right after decoding.
    """
    try:
        return _get_uploaded_audio(recording)
    except Exception as exc:
        st.session_state.pop("_audio_cache", None)
        st.error(friendly_error(exc))
        return None


def _render_header() -> None:
    st.title("Audio2Text AI")
    st.caption(
        "Transcripción local: NAVER Multilingual DistilWhisper 28k (catalán, tamil, "
        "tailandés) y OpenAI Whisper Small (español) · audio corto y largo"
    )


def _render_upload() -> object | None:
    """Two entry points for audio: upload a file or record with the microphone.

    Both produce the same ``UploadedAudio`` and feed the same transcription
    flow.
    """
    audio = None
    with st.container(border=True):
        st.subheader("1. Audio")
        tab_upload, tab_record = st.tabs(["Subir archivo", "Grabar audio"])

        with tab_upload:
            uploaded = st.file_uploader(
                "Sube un archivo WAV, FLAC, MP3, M4A, AAC, MP4 o MOV",
                type=["wav", "flac", "mp3", "m4a", "aac", "mp4", "mov"],
                accept_multiple_files=False,
                key="audio_upload",
                help="Tamaño máximo: 200 MB.",
            )
            if uploaded is None:
                st.session_state.pop("_audio_cache", None)
                st.caption(
                    "Formatos admitidos: WAV, FLAC, MP3, M4A, AAC, MP4 y MOV. "
                    "El audio se procesa localmente."
                )
            else:
                audio = _load_tab_audio(uploaded)
                if audio is not None:
                    info_columns = st.columns([2.2, 1, 1])
                    info_columns[0].markdown(f"**{audio.filename}**")
                    info_columns[0].caption("Archivo cargado")
                    info_columns[1].metric("Formato", audio.format_label)
                    info_columns[2].metric(
                        "Duración", format_duration(audio.duration_s)
                    )
                    st.audio(
                        audio.data,
                        format=audio.mime_type,
                        alt=f"Audio cargado: {audio.filename}",
                    )
                    st.caption(
                        f"{audio.sample_rate:,} Hz · mono para inferencia · "
                        f"{format_file_size(len(audio.data))}"
                    )

        with tab_record:
            recorded = st.audio_input(
                "Grabar audio",
                sample_rate=16000,
                help=(
                    "Usa el micrófono del equipo; la grabación se envía al "
                    "mismo motor que procesa los archivos subidos."
                ),
                key="audio_recorder",
            )
            if recorded is not None:
                audio = _load_tab_audio(recorded)
                if audio is not None:
                    st.audio(
                        audio.data,
                        format=audio.mime_type,
                        alt="Grabación realizada",
                    )
                    st.caption(
                        f"Grabación adjunta · {format_duration(audio.duration_s)} · "
                        f"{audio.format_label}"
                    )
            else:
                st.caption(
                    "Presiona «Start recording» para grabar; luego «Stop "
                    "recording» para adjuntar el audio a la transcripción."
                )
    return audio


def _render_result(record) -> None:
    with st.container(border=True):
        st.subheader("Resultados")
        st.caption(config_label(record))

        metric_columns = st.columns(4)
        metric_columns[0].metric(
            "Duración del audio", format_duration(record.audio_duration_s)
        )
        metric_columns[1].metric("Inferencia", f"{record.latency_s:.2f} s")
        metric_columns[2].metric("Segmentos", str(len(record.segments)))
        metric_columns[3].metric("VRAM pico", format_vram(record.peak_vram_mb))

        st.markdown("#### Texto completo")
        if record.text:
            st.code(
                record.text,
                language=None,
                wrap_lines=True,
                height=260,
                width="stretch",
            )
            st.caption("El bloque incluye un botón para copiar el texto completo.")
        else:
            st.warning(
                "El modelo no devolvió texto. Puede ser un fragmento silencioso o "
                "audio sin voz reconocible."
            )

        download_columns = st.columns(2)
        with download_columns[0]:
            st.download_button(
                "Descargar TXT",
                data=record_to_txt(record),
                file_name=download_filename(record, "txt"),
                mime="text/plain; charset=utf-8",
                key=f"download_txt_{record.record_id}",
                width="stretch",
                on_click="ignore",
            )
        with download_columns[1]:
            st.download_button(
                "Descargar JSON",
                data=record_to_json(record),
                file_name=download_filename(record, "json"),
                mime="application/json",
                key=f"download_json_{record.record_id}",
                width="stretch",
                on_click="ignore",
            )

        st.markdown("#### Segmentos")
        rows = segment_rows(record)
        if rows:
            st.dataframe(
                rows,
                hide_index=True,
                width="stretch",
                column_config={
                    "Segmento": st.column_config.NumberColumn(
                        "Segmento", width="small"
                    ),
                    "Inicio": st.column_config.TextColumn("Inicio", width="small"),
                    "Fin": st.column_config.TextColumn("Fin", width="small"),
                    "Duración": st.column_config.TextColumn(
                        "Duración", width="small"
                    ),
                    "Texto": st.column_config.TextColumn("Texto", width="large"),
                },
            )
            st.caption(
                "Los timestamps son límites de ventana: estimaciones por segmento, "
                "no timestamps por palabra."
            )
        else:
            st.caption("No hay segmentos para mostrar.")

        with st.expander("Detalles técnicos"):
            st.markdown(f"- **Modelo:** `{record.model_id}`")
            st.markdown(
                f"- **Archivo:** `{record.filename}` · {record.file_format} · "
                f"{format_duration(record.audio_duration_s)}"
            )
            st.markdown(f"- **Carga del modelo:** {record.load_seconds:.2f} s")
            st.markdown(f"- **Tokens generados:** {record.num_tokens}")
            st.markdown(
                f"- **Ventanas procesadas:** {record.num_chunks} "
                f"({record.chunking.get('strategy', 'single')})"
            )
            if record.chunking.get("duplicate_words_removed"):
                st.markdown(
                    "- **Palabras duplicadas eliminadas en solapes:** "
                    f"{record.chunking['duplicate_words_removed']}"
                )
            st.markdown(
                f"- **Prefijo del decoder:** `{record.decoder_prefix}`"
            )
            if record.chunking:
                st.json(record.chunking)


def _render_history() -> None:
    with st.container(border=True):
        st.subheader("Historial de la sesión")
        history = st.session_state.history
        if not history:
            st.caption(
                "Las transcripciones completadas aparecerán aquí. El historial "
                "se conserva solo mientras dure esta sesión."
            )
            return

        valid_labels = [history_label(record) for record in history]
        current = st.session_state.get("history_select")
        if current not in valid_labels:
            active = find_record(
                history, st.session_state.get("active_record_id")
            )
            st.session_state.history_select = (
                history_label(active) if active is not None else valid_labels[0]
            )

        st.selectbox(
            "Selecciona una transcripción",
            options=valid_labels,
            key="history_select",
            on_change=_sync_active_record,
            label_visibility="collapsed",
        )
        st.caption(
            f"{len(history)} resultado(s) · se conserva solo en esta sesión · "
            "máximo 20"
        )
        if st.button("Limpiar historial", key="clear_history"):
            st.session_state.history = []
            st.session_state.active_record_id = None
            st.rerun()


def _sync_history_from_state() -> None:
    """Keep the active result valid before rendering results and history."""
    history = st.session_state.history
    valid_ids = [record.record_id for record in history]
    active_id = st.session_state.get("active_record_id")
    if active_id not in valid_ids:
        active_id = valid_ids[0] if valid_ids else None
        st.session_state.active_record_id = active_id

    valid_labels = [history_label(record) for record in history]
    if valid_labels and st.session_state.get("history_select") not in valid_labels:
        active = find_record(history, active_id)
        st.session_state.history_select = (
            history_label(active) if active is not None else valid_labels[0]
        )


_init_state()
language, device, dtype, num_beams, model_id = _render_sidebar()

_render_header()
_sync_history_from_state()

active_record = find_record(
    st.session_state.history, st.session_state.get("active_record_id")
)
audio = _render_upload()

st.subheader("2. Transcripción")
st.caption(
    f"Configuración activa: **{language_label(language)}** · "
    f"**{device_label(device)}** · **{dtype_label(dtype)}** · "
    f"beam **{num_beams}**"
)

transcribe_clicked = st.button(
    "Transcribir audio",
    type="primary",
    width="stretch",
    disabled=audio is None,
    key="transcribe",
)

if transcribe_clicked and audio is not None:
    progress = st.progress(0, text="Preparando la transcripción…")
    status = st.status("Preparando la transcripción", expanded=True)
    try:
        status.write(f"Audio: `{audio.filename}` · {format_duration(audio.duration_s)}")
        status.write(
            f"Configuración: {language_label(language)} · {device_label(device)} · "
            f"{dtype_label(dtype)} · beam {num_beams}"
        )
        progress.progress(15, text="Cargando el modelo…")

        config = TranscriptionConfig(
            language=language,
            device=device,
            dtype=dtype,
            num_beams=num_beams,
            model_dir=model_dir_for(model_id),
            model_id=model_id,
        )
        model = get_model(model_dir_for(model_id), language, device, dtype, model_id)
        status.write(
            f"Modelo cargado en {model.load_seconds:.2f} s "
            "(se reutiliza si la configuración no cambia)."
        )

        progress.progress(45, text="Ejecutando la inferencia…")
        started = time.perf_counter()
        result = transcribe_auto(
            model,
            audio.samples,
            sample_rate=audio.sample_rate,
            num_beams=num_beams,
        )
        elapsed = time.perf_counter() - started
        progress.progress(100, text="Transcripción completada")

        record = make_record(
            audio=audio,
            result=result,
            config=config,
            load_seconds=model.load_seconds,
        )
        st.session_state.history = add_record(st.session_state.history, record)
        st.session_state.active_record_id = record.record_id
        st.session_state.history_select = history_label(record)
        active_record = record

        status.update(
            label="Transcripción completada",
            state="complete",
            expanded=False,
        )
        status.write(f"Inferencia terminada en {elapsed:.2f} s.")
        st.toast("Transcripción completada.")
    except Exception as exc:
        LOGGER.exception("Error durante la transcripción desde Streamlit")
        status.update(
            label="No se pudo completar la transcripción",
            state="error",
            expanded=True,
        )
        st.error(friendly_error(exc))
    finally:
        progress.empty()

if active_record is not None:
    _render_result(active_record)
else:
    with st.container(border=True):
        st.subheader("Resultados")
        st.info(
            "Sube un audio, revisa la configuración y pulsa «Transcribir audio». "
            "Los resultados aparecerán aquí."
        )

_render_history()
