# Audio2Text AI — Etapas 1–5: inferencia, audio largo, interfaz Streamlit y entrada de audio

Proyecto académico **Audio2Text AI**: ejecutar el modelo oficial
[`naver/multilingual-distilwhisper-28k`](https://huggingface.co/naver/multilingual-distilwhisper-28k)
en local (RTX 2050, 4 GB), verificar una transcripción mínima en catalán, soportar
audios de más de 30 segundos mediante fragmentación con timestamps por segmento
y ofrecer una interfaz web local con Streamlit con dos formas de proporcionar
audio: subir un archivo o grabarlo directamente desde el micrófono. También se
incorporó [`openai/whisper-small`](https://huggingface.co/openai/whisper-small)
como modelo alternativo para transcribir en español.

El proyecto **no entrena ni modifica pesos**. Usa el código y los pesos oficiales, con
pequeños *shims* de compatibilidad en tiempo de ejecución para que el repositorio
(escrito para `transformers 4.30.1`) funcione con `transformers 4.57.3` y Python 3.13.
La interfaz reutiliza el mismo motor de inferencia; no reimplementa el modelo.

## Resultado

| Dispositivo | Precisión | Búsqueda | Latencia | VRAM pico | CER | WER |
|---|---|---|---|---|---|---|
| CUDA (RTX 2050) | fp32 | beam 5 | 9.66 s | 2592 MiB | 2.86 % | 9.80 % |
| CUDA (RTX 2050) | fp16 | beam 5 | 7.72 s | 1315 MiB | 2.86 % | 9.80 % |
| CPU | fp32 | beam 5 | 28.78 s | — | 2.86 % | 9.80 % |
| CUDA (RTX 2050) | fp32 | greedy | 6.92 s | 1603 MiB | 2.86 % | 9.80 % |

Audio: `data/samples/ca_es_sample.wav` (FLEURS `ca_es`, 20.70 s, 16 kHz mono).
Las cuatro ejecuciones producen **exactamente los mismos 109 tokens** que la
verificación de referencia con `transformers 4.30.1` (`logs/legacy_transformers430_results.json`).

Transcripción:

> Avui en dia, la cuina japanesa és fàcil de trobar per tot arreu de l'arxipèleg i ofereix una gran varietat de plats condimentats de forma senzilla.Els sabors preferits als japanesos solen ser els dels cacauets, els bitxos, el sucre, sobretot el sucre, coco javanès i diverses espècies aromàtiques.

La diferencia con la referencia de FLEURS es principalmente ortográfica
(`japanesa`/`javanesa`, `arxipèleg`/`arxipèlag`, `japanesos`/`javanesos`)
y de puntuación, no de contenido.

## Interfaz Streamlit

`streamlit_app.py` es la capa de presentación. Toda la lógica de modelo sigue en
`src/audio2text/engine.py` y `src/audio2text/chunking.py`; `ui_service.py` solo valida
subidas, forma resultados y prepara descargas.

Funcionalidades implementadas:

- Carga de archivos **WAV, FLAC, MP3, M4A, AAC, MP4 y MOV** (MP3/M4A/AAC/MP4/MOV se
  decodifican con FFmpeg) con validación de extensión, tamaño, contenido vacío y decodificación.
- Dos formas de proporcionar audio: **subir un archivo** (WAV, FLAC, MP3, M4A, AAC,
  MP4 y MOV; MP3/M4A/AAC/MP4/MOV se decodifican con FFmpeg) o **grabar audio** con
  `st.audio_input` desde el micrófono del equipo. Ambos llevan al mismo proceso de
  transcripción.
- Nombre, formato, duración, frecuencia de muestreo y reproductor de audio.
- **Modelo** (**NAVER Multilingual DistilWhisper 28k** / **OpenAI Whisper Small**).
- Idioma (**catalán, tamil, tailandés** para NAVER; **español** para Whisper Small),
  dispositivo (**CUDA/CPU**), precisión (**FP16/FP32** según compatibilidad) y beam
  search configurable (5 por defecto). Al seleccionar Whisper Small, el idioma se
  configura automáticamente en «Español».
- Indicadores visibles de dispositivo, GPU/VRAM y configuración usada.
- Progreso por fases: carga del modelo e inferencia. No se simula progreso por chunk.
- Texto completo con botón de copia, duración, latencia, segmentos con inicio/fin y
  descargas **TXT/JSON**.
- Historial de la sesión (máximo 20), selección de resultados anteriores y opción de
  limpiarlo.
- Modelo en caché con `st.cache_resource(max_entries=1)`, clave
  `(ruta, idioma, dispositivo, precisión)`, y liberación explícita de VRAM al cambiar
  de configuración o pulsar «Liberar modelo de memoria».
- Tema nativo en `.streamlit/config.toml`, sin CSS dependiente de clases internas.

La inferencia real se verificó a través del `file_uploader` y del botón de Streamlit
con `AppTest`:

| Flujo UI | Audio | Resultado | Latencia | Segmentos |
|---|---|---|---|---|
| Corto | `ca_es_sample.wav` (20.70 s) | mismo texto y 109 tokens que la etapa 1 | 6.35–8.18 s (con modelo en caché) | 1 (0.00–20.70 s) |
| Largo | `ca_es_long.wav` (42.40 s) | mismo texto que la etapa 2 | 15.33–18.57 s | 2 (0.00–21.01 s, 21.01–42.40 s) |

Evidencia: `logs/streamlit_ui_inference.json` y
`logs/streamlit_ui_inference_long.json`. Ambos usan CUDA FP16, beam 5 y conservan el
prefijo `[50258, 50270, 50359, 50363]`.

## Modelos: NAVER Multilingual DistilWhisper 28k y OpenAI Whisper Small

La barra lateral de Streamlit tiene un selector de modelo. La diferenciación:

| Modelo | Rol | Idiomas | Pesos | Notas |
|---|---|---|---|---|
| **NAVER Multilingual DistilWhisper 28k** (predeterminado) | Modelo académico principal | catalán, tamil, tailandés | `models/multilingual-distilwhisper-28k/` (local) | Clase custom del repo NAVER + experto por idioma (`expert.<lang>.bin`); prefijo forzado validado `[50258, 50270, 50359, 50363]`. |
| **OpenAI Whisper Small** | Alternativo para español | español (`es`) | `openai/whisper-small` (Hugging Face) | Usa `WhisperForConditionalGeneration` + `WhisperProcessor`; se carga con `language="es"`. |

- **NAVER 28k** sigue siendo el modelo académico predeterminado y sus resultados
  token a token no cambian. El checkpoint no incluye experto de español, por lo que
  el idioma debe ser uno de los tres (catalán/tamil/tailandés).
- **Whisper Small** transcribe en español vía `language="es"`. Al seleccionarlo en
  la UI, el idioma se setea a «Español»; al volver a NAVER, el idioma vuelve a uno
  de sus tres idiomas disponibles. La lista de idiomas del modelo NAVER no se
  modifica.
- **Chunking compartido:** `transcribe_auto` mantiene la ruta corta validada y
  fragmenta audios > 30 s para ambos modelos.
- **Memoria:** cada modelo se mantiene en su entrada de
  `st.cache_resource(max_entries=1)`; al cambiar de modelo se libera la entrada
  anterior para no acumular checkpoints en 4 GB.

Verificación real (CUDA fp16, RTX 2050, beam 5):

| Audio | Modelo | Latencia | VRAM pico |
|---|---|---|---|
| `es_es_sample.wav` (8.40 s, FLEURS es_419) | Whisper Small | 2.54 s | 786 MiB |
| `es_es_long.wav` (36.60 s) | Whisper Small | 15.84 s (2 segmentos) | — |

Texto corto (Whisper Small, español):

> Milenios atrés, un hombre llamado Aristarco comentó que el sistema solar giraba en torno al sol.

## Audio largo (chunking)

Whisper solo procesa 30 s por pasada. `transcribe_auto` mantiene la ruta corta
validada y, por encima de 30 s, fragmenta el audio:

1. Busca pausas reales (RMS < `--silence-rms` durante ≥ `--min-silence-seconds`,
   0.4 s por defecto) y corta en el centro de la última pausa anterior a los 30 s.
   Así el fragmento termina en silencio y el siguiente empieza ahí: no hay
   solapamiento ni texto duplicado.
2. Si no hay pausas, usa ventanas de 30 s con 2 s de solapamiento. El texto repetido
   al inicio del fragmento siguiente se elimina comparando palabras normalizadas
   (mínimo 2, máximo 12).
3. Cada fragmento conserva el prefijo explícito
   `[50258, 50270, 50359, 50363]` y los mismos parámetros de generación.
4. Los segmentos se liberan de memoria uno a uno (`gc.collect()` +
   `torch.cuda.empty_cache()`).

Resultados con un audio sintético de 42.40 s (el clip FLEURS repetido dos veces con
1 s de silencio, `data/samples/ca_es_long.wav`):

| Configuración | Segmentos | Latencia | VRAM pico | CER | WER |
|---|---|---|---|---|---|
| CUDA fp16, beam 5, silencio 0.4 s | 2 (0.00–21.01 s, 21.01–42.40 s) | 16.26 s | 1315 MiB | 4.49 % | 12.75 % |

El texto contiene las dos repeticiones sin pérdida ni duplicación **en la unión** (el
corte cae en la pausa). La segunda repetición omite `arreu de l'` por variación del
modelo, no por el chunking. Los timestamps son **estimaciones por segmento** (límites de
ventana).

La inferencia corta se revalidó tras el cambio: mismo texto y **mismos 109 tokens**
que la etapa 1 (`logs/inference_stage2_short_fp16.json`).

## Entrada de audio: subida y grabación

La sección «1. Audio» de la interfaz ofrece dos formas de proporcionar audio a
la transcripción:

1. **Subir un archivo**: elige un archivo desde el equipo. Formatos admitidos:
   WAV, FLAC, MP3, M4A, AAC, MP4 y MOV (los MP3/M4A/AAC/MP4/MOV se decodifican
   con FFmpeg). Validad: tamaño (≤ 200 MB), extensión, contenido vacío y
   decodificación.
2. **Grabar audio**: en la pestaña «Grabar audio», `st.audio_input` ofrece los
   controles de grabación que el navegador permita (inicio/fin de la grabación);
   al terminar, un reproductor permite escuchar el audio.

Ambas opciones producen el mismo audio procesado y llevan al mismo flujo de
transcripción (`transcribe`/`transcribe_auto` según la duración). La grabación
se convierte a WAV compatible con el motor, usando un archivo temporal que se
elimina tras decodificarlo, y el motor decodifica/procesa el audio en memoria.

### Requisitos

- Windows 11 (probado) o Linux.
- Python 3.13 (probado con 3.13.15).
- GPU NVIDIA con ≥ 2.6 GB libres para fp32, o ≥ 1.4 GB para fp16. También funciona en CPU.
- `uv` (recomendado) o `pip`.
- Streamlit 1.67.0 (incluido en `requirements.txt`).
- Los pesos del modelo NAVER en `models/multilingual-distilwhisper-28k/`.
- Whisper Small: pesos de `openai/whisper-small` (en caché de HF, o descargados bajo demanda).
- Navegador web moderno para la interfaz local.
- GNU Make (opcional, solo para los atajos; comprobado con 4.4.1 en este equipo).

## Instalación

```powershell
uv venv .venv --python 3.13
uv pip install --python .venv\Scripts\python.exe torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

En CPU, sustituir el primer `uv pip install` por la build CPU de PyTorch.
`requirements.txt` instala también `streamlit==1.67.0` y sus dependencias.

### Pesos

El checkpoint local contiene:

```
models/multilingual-distilwhisper-28k/
├── config.json
├── generation_config.json
├── model.safetensors          # 966,997,096 bytes
└── expert.catalan.bin         # 458,178,391 bytes
```

Para descargarlos de nuevo:

```powershell
.\.venv\Scripts\python.exe -c "from huggingface_hub import hf_hub_download as d; [d('naver/multilingual-distilwhisper-28k', f, local_dir='models/multilingual-distilwhisper-28k') for f in ['config.json','generation_config.json','model.safetensors','expert.catalan.bin']]"
```

Verificación de integridad (SHA-256):

```
BD07406AB1DD8061E7187A50C11E04CE1650C1EA4A2C4D6CF1EBD6B31C20C5E1  model.safetensors
DC7192D0B60A9960169CBC5C2F283B028B857D20307E822B4E30E293E4E88FE8  expert.catalan.bin
```

### Audio de prueba

```powershell
.\.venv\Scripts\python.exe scripts\prepare_sample.py
```

Descarga solo el primer WAV de FLEURS `ca_es` mediante petición HTTP por rangos
(sin bajar el archivo completo de 573 MB) y escribe la transcripción de referencia.

Para las pruebas de audio largo:

```powershell
.\.venv\Scripts\python.exe scripts\make_long_sample.py
```

Genera `data/samples/ca_es_long.wav` (42.40 s: clip x2 + 1 s de silencio) y su
referencia. Es un audio sintético para validar el chunking, no una grabación continua.

Audio en español para Whisper Small (muestras del FLEURS es_419, disponibles en el
repo): `data/samples/es_es_sample.wav` (8.40 s) y `data/samples/es_es_long.wav`
(36.60 s).

## Uso

### Interfaz web local

```powershell
.\.venv\Scripts\streamlit.exe run streamlit_app.py
```

También se puede lanzar con:

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

La app abre en `http://localhost:8501`. El límite de subida es 200 MB (configurado en
`.streamlit/config.toml`). La primera transcripción carga el modelo; las siguientes
interacciones con la misma configuración lo reutilizan. Si se cambia de idioma,
dispositivo o precisión, se libera la entrada anterior y se carga la nueva.

**Dos opciones de entrada de audio** (pestañas de la sección «1. Audio»):

1. **Subir un archivo**: elige un archivo desde el equipo. Formatos admitidos:
   WAV, FLAC, MP3, M4A, AAC, MP4 y MOV.
2. **Grabar audio**: en la pestaña «Grabar audio», el reproductor de `st.audio_input`
   ofrece los controles de grabación que permita el navegador; después de terminar
   la grabación aparece un reproductor para escucharla.

Ambas opciones llevan al mismo proceso de transcripción: el audio se procesa
localmente con el modelo seleccionado (idioma, dispositivo, precisión y beam
search configurables) y se muestra el resultado con texto, métricas y segmentos.

Para iniciar la app sin escribir comandos, haz **doble clic en `run.bat`** desde el
Explorador de Windows. El archivo comprueba que existe `.venv\Scripts\streamlit.exe`,
cambia al directorio del proyecto y abre Streamlit en el navegador. La ventana de
consola debe permanecer abierta mientras se usa la app.

### Atajos con GNU Make

En este equipo se comprobó **GNU Make 4.4.1** (`C:\ProgramData\chocolatey\bin\make.exe`).
Los atajos se pueden ejecutar desde PowerShell, `cmd` o Git Bash; el `Makefile` fija
`cmd.exe` como shell para que las rutas de `.venv` y las variables de entorno sean
consistentes en Windows. Todos los comandos usan exclusivamente los ejecutables de
`.venv`.

| Comando | Acción |
|---|---|
| `make help` | Muestra todos los comandos disponibles. Es el objetivo por defecto. |
| `make run` | Inicia Streamlit con `streamlit_app.py` en `http://localhost:8501`. |
| `make test` | Ejecuta `unittest discover -s tests -v`. |
| `make test-real` | Activa `AUDIO2TEXT_RUN_UI_INFERENCE=1` y ejecuta la inferencia real desde Streamlit. |
| `make check` | Ejecuta `compileall` sobre `streamlit_app.py`, `src`, `scripts` y `tests`. |
| `make clean` | Elimina solo `__pycache__`, `.pyc`/`.pyo` y cachés de herramientas. |

Notas:

- `make test-real` carga el modelo y tarda aproximadamente 40 s; además reescribe
  `logs/streamlit_ui_inference.json` y `logs/streamlit_ui_inference_long.json`.
- `make check` no compila `.venv`, `models`, `vendor`, `logs` ni `data`; solo el código
  fuente del proyecto.
- `make clean` **no borra** logs, audios de prueba, pesos del modelo, resultados JSON ni
  documentación. La lógica segura está en `scripts/clean_caches.py`.

### Línea de comandos

```powershell
.\.venv\Scripts\python.exe scripts\run_inference.py `
  --audio data\samples\ca_es_long.wav `
  --reference data\samples\reference_ca_long.txt `
  --json-out logs\inference_long_fp16.json
```

Por defecto usa **fp16 + beam 5 en CUDA** y fp32 en CPU. Opciones principales:
`--device {cuda,cpu}`, `--dtype {float32,float16,bfloat16}`, `--num-beams`,
`--max-new-tokens`, `--language {catalan,tamil,thai}`, `--max-chunk-seconds`,
`--chunk-overlap`, `--silence-rms`, `--min-silence-seconds` y `--no-chunk`.

El JSON incluye `text`, `segments[]` (inicio/fin, texto, tokens, latencia, silencio),
`chunking`, métricas y `token_ids` en la ruta corta.

## Estructura

```
Makefile                 # atajos Windows: run, test, test-real, check y clean
run.bat                  # inicio de Streamlit con doble clic, sin GNU Make
streamlit_app.py         # interfaz: subida/grabación de audio, configuración, progreso, resultados
.streamlit/config.toml   # tema nativo y límite de subida
src/audio2text/
├── chunking.py          # ventanas, cortes por silencio, unión sin duplicados
├── compat.py            # shims: stub de datasets, cached_file, atención 4.30
├── legacy_attention.py  # WhisperAttention con el contrato de transformers 4.30
├── engine.py            # carga, audio, transcribe / transcribe_long / transcribe_auto
├── metrics.py           # CER/WER sin dependencias externas
└── ui_service.py        # validación de subidas, serialización TXT/JSON, historial
scripts/
├── clean_caches.py      # limpieza segura usada por make clean
├── prepare_sample.py    # descarga reproducible del audio FLEURS corto
├── make_long_sample.py  # audio sintético de 42.4 s para probar chunking
└── run_inference.py     # CLI de inferencia + evaluación
tests/
├── test_chunking.py     # ventanas, límites, solape, silencio
├── test_engine.py       # audio vacío/silencioso, errores, dispatch, segmentos
├── test_ui_service.py   # validación, formatos, payload, historial
├── test_whisper_small.py # routing de Whisper Small (español), carga real
├── test_streamlit_app.py       # AppTest: subida, grabación, errores, historial
└── test_streamlit_inference.py # inferencia real opcional con AppTest
vendor/multilingual-distilwhisper/   # repo oficial sin modificar
logs/                    # resultados JSON y logs de las ejecuciones
docs/INFORME.md          # informe técnico
```

## Pruebas

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

90 pruebas: 86 rápidas y 4 de inferencia real omitidas por defecto (`Ran 90 tests
... OK (skipped=4)`). Las rápidas no cargan el modelo ni usan GPU y cubren audios
cortos, largos, límites de 30 s, cobertura de ventanas, solapamiento, deduplicación,
fragmentos silenciosos, audio vacío, errores de lectura, validación de subidas,
serialización TXT/JSON, historial, renderizado de la interfaz y grabación desde el
micrófono (conversión de la grabación a WAV, envío al motor y manejo de grabaciones
inválidas o vacías). Las pruebas de `test_whisper_small.py` añaden routing del modelo
Whisper Small, validación de idioma (español vs. expertos NAVER), carga real de ambos
modelos (saltan si Whisper Small no está en caché local) e inferencia real en español.

Prueba de inferencia real a través de la UI (requiere pesos y GPU/CPU disponible):

```powershell
$env:AUDIO2TEXT_RUN_UI_INFERENCE = "1"
.\.venv\Scripts\python.exe -m unittest tests.test_streamlit_inference -v
```

La prueba sube `ca_es_sample.wav` y `ca_es_long.wav` mediante el `file_uploader` de
Streamlit, pulsa el botón real y comprueba texto, segmentos, descargas e historial.
Escribe la evidencia en `logs/streamlit_ui_inference*.json`.

Pruebas reales de inferencia en español con Whisper Small (opcional, salte si faltan
los audios):

```powershell
$env:AUDIO2TEXT_RUN_WHISPER_INFERENCE = "1"
.\.venv\Scripts\python.exe -m unittest tests.test_whisper_small.WhisperSmallRealInferenceTest -v
```

Transcribe `es_es_sample.wav` (8.40 s) y `es_es_long.wav` (36.60 s, 2 segmentos) con
`language="es"` y comprueba que la salida es texto en español.

Equivalentes con GNU Make: `make test` y `make test-real`.

## Compatibilidad con `transformers 4.57.3`

El repo oficial fue escrito para `transformers 4.30.1`, que no tiene wheels para
Python 3.13. `src/audio2text/compat.py` aplica tres correcciones en tiempo de
ejecución, sin tocar `vendor/`:

1. **`datasets`**: el módulo se importa pero no se usa en inferencia; se instala un
   stub si no está disponible.
2. **`cached_file`**: se filtran `resume_download`/`proxies`, eliminados de
   `huggingface_hub`.
3. **`WhisperAttention`**: se restaura la implementación 4.30 (retorno de 3 tuplas y
   caché KV legacy) porque las versiones modernas devuelven 2 tuplas.

Además, `engine.py`:

- mezcla `GenerationMixin` en la clase upstream (transformers ≥ 4.50 lo quitó de
  `PreTrainedModel`) y desactiva el KV cache dinámico;
- carga `generation_config.json` explícitamente;
- construye el prefijo forzado
  `<|startoftranscript|><|ca|><|transcribe|><|notimestamps|>`
  (ids `[50258, 50270, 50359, 50363]`) y lo pasa como `decoder_input_ids`, porque
  `transformers` ≥ 4.50 ya no lee `generation_config.forced_decoder_ids`;
- en `_generate()`, el path de `openai/whisper-small` usa `generate(language="es",
  task="transcribe")`; transformers ≥ 4.57 rechaza prefijo + `max_new_tokens` >
  `max_target_positions` (448), por lo que el presupuesto se reduce por la longitud
  del prefijo (448 → 444) sin cambiar la longitud total.

Sin ese prefijo explícito, la generación moderna omite `<|transcribe|>` y
`<|notimestamps|>` y produce texto incorrecto. Con él, la salida es idéntica token a
token a la de `transformers 4.30.1`.

## Limitaciones conocidas

- **Cortes sin pausa**: si el audio es continuo y no hay silencios, se usan ventanas
  solapadas y la deduplicación es heurística (comparación de palabras). Si el modelo
  termina antes de lo esperado dentro de una ventana (p. ej. por una pausa larga
  interna), la región posterior puede perderse; en ese caso conviene reducir
  `--max-chunk-seconds` o ajustar `--min-silence-seconds`.
- **Silencio**: la detección por RMS es absoluta. Audio muy bajo o con ruido de fondo
  por encima de `--silence-rms` puede no detectarse como silencio.
- **Idiomas**: el checkpoint 28k solo incluye expertos para catalán, tamil y
  tailandés; no soporta español. Español está disponible con Whisper Small
  (`language="es"`); el idioma del modelo NAVER no se modifica.
- **Aviso benigno**: `The attention mask is not set ... pad token is same as eos token`.
  El modelo no usa máscara en evaluación y los resultados coinciden con la referencia;
  se deja visible por transparencia.
- **Whisper Small**: disponible en caché de Hugging Face o descargado bajo demanda;
  sin ejecución de inferencia real se ejecuta en la suite rápida (salta si no está
  en caché).
- **Formato de audio**: WAV/FLAC vía `soundfile`; MP3, M4A, AAC, MP4 y MOV vía `FFmpeg` (se busca `ffmpeg` en el PATH, `C:\ffmpeg\bin\ffmpeg.exe` y `%ProgramFiles%\ffmpeg\bin\ffmpeg.exe`). No se requiere FFmpeg para WAV/FLAC.
- **Historial de la interfaz**: vive solo en la sesión de Streamlit (máximo 20
  resultados). No se guarda el audio en el historial y se pierde al cerrar o recargar
  la sesión.
- **Caché del modelo**: se mantiene una sola configuración cargada por modelo para no
  agotar los 4 GB de VRAM. Cambiar de modelo, idioma, dispositivo o precisión provoca
  una recarga; el botón «Liberar modelo de memoria» permite vaciarla manualmente.
- **Progreso**: es por fases (carga e inferencia), no un porcentaje real por fragmento.
  El motor no expone callbacks de progreso por chunk.
- **CPU**: FP16 no se ofrece en CPU; se usa FP32. La inferencia en CPU es
  considerablemente más lenta (28.78 s para el clip de 20.70 s).
- **Uso local**: la interfaz no tiene autenticación ni multiusuario; está pensada para
  ejecutarse en `localhost`.

## Capturas pendientes

Aún no se han añadido capturas al repositorio. Pendiente de incluir:

- Pantalla inicial con carga de audio y configuración.
- Resultado con texto, métricas, segmentos y descargas.
- Historial de la sesión y selector de resultados.
- Indicadores de CUDA/CPU, FP16/FP32 y VRAM.

Las capturas deben generarse desde la app real, no con datos simulados.

## Documentación

El informe técnico completo (entorno, metodología, errores encontrados, resultados y
trabajo pendiente) está en [`docs/INFORME.md`](docs/INFORME.md).
