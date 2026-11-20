# Informe técnico — Etapas 1–3: inferencia, audio largo e interfaz Streamlit

**Proyecto:** Audio2Text AI
**Fecha:** 18 de noviembre de 2026
**Alcance:** ejecutar el modelo oficial `naver/multilingual-distilwhisper-28k` en local,
verificar una transcripción mínima en catalán, documentar la viabilidad sobre una GPU
de 4 GB, añadir audio largo con timestamps por segmento y ofrecer una interfaz web
local con Streamlit con dos opciones de entrada de audio (subida de archivo o grabación
desde el micrófono). Sin entrenamiento y sin modificación de pesos.

---

## 1. Resumen ejecutivo

Se consiguió ejecutar el modelo oficial multilingual DistilWhisper 28k en la RTX 2050
(4 GB) y en CPU, con resultados **token a token idénticos** a la ejecución de referencia
con `transformers 4.30.1` (la versión para la que fue escrito el repositorio). Después
se añadió fragmentación para audio de más de 30 s con timestamps por segmento.

- CUDA fp32, beam 5: **9.66 s**, **2592 MiB** de VRAM pico, CER 2.86 %, WER 9.80 %.
- CUDA fp16, beam 5: **7.72 s**, **1315 MiB**, mismo texto.
- CPU fp32, beam 5: **28.78 s**, mismo texto.
- Audio de 42.40 s en CUDA fp16: 2 segmentos, **16.26 s**, **1315 MiB**, CER 4.49 %,
  WER 12.75 %.
- El checkpoint (1.33 GB) cabe holgadamente en 4 GB, incluso en fp32 con beam search.
- La interfaz Streamlit reutiliza el motor sin duplicar lógica: audio corto (6.35–8.18 s
  con caché) y largo (15.33–18.57 s, 2 segmentos) verificados a través de la UI.
- El principal obstáculo no fue el hardware sino la incompatibilidad del código oficial
  (escrito para `transformers 4.30.1`) con `transformers 4.57.3`; se resolvió con
  *shims* en tiempo de ejecución, manteniendo `vendor/` intacto.
- Etapa 4: se incorporó `openai/whisper-small` como alternativo para transcribir en
  español; el modelo académico principal de NAVER (28k) no se alteró.

Conclusión: la inferencia corta, el chunking y la interfaz local son viables en el
equipo objetivo, con resultados idénticos a las etapas anteriores.

---

## 2. Objetivo y criterios de aceptación

Objetivo: comprobar que el modelo oficial funciona en local con audio catalán real, sin
entrenar ni alterar los pesos, y añadir transcripción de audio de más de 30 s con
timestamps por segmento, midiendo latencia, memoria y calidad.

Criterios aplicados:

1. Usar el checkpoint y el código oficiales (repositorio de NAVER).
2. Usar el experto de lenguaje correcto (catalán) y audio en catalán, porque el
   checkpoint 28k solo cubre catalán, tamil y tailandés (no español).
3. No asumir que algo funciona: toda dependencia, parche y resultado se probó.
4. No instalar dependencias innecesarias ni tocar instalaciones globales de Python.
5. Si no cabe en 4 GB, evaluar CPU o alternativas de memoria sin modificar el modelo.
6. Mantener intacta la ruta corta ya validada (mismos texto y tokens) y el prefijo
   explícito de cuatro tokens en cada fragmento.
7. Evitar pérdida o duplicación de texto en las uniones y no presentar los timestamps
   por segmento como si fueran precisión por palabra.

---

## 3. Modelo y datos

### 3.1 Modelo

- Repositorio: `naver/multilingual-distilwhisper-28k`.
- Arquitectura: DistilWhisper (encoder Whisper destilado + decoder con *gating* por
  lenguaje y adaptadores por idioma).
- Pesos base: `model.safetensors` (966 997 096 bytes).
- Experto catalán: `expert.catalan.bin` (458 178 391 bytes).
- Total local: ~1.33 GB.
- SHA-256:
  - `model.safetensors`: `BD07406AB1DD8061E7187A50C11E04CE1650C1EA4A2C4D6CF1EBD6B31C20C5E1`
  - `expert.catalan.bin`: `DC7192D0B60A9960169CBC5C2F283B028B857D20307E822B4E30E293E4E88FE8`
- Procesador: `openai/whisper-small` (tokenizer + extractor de mel).

Los pesos se cargaron y se compararon tensor a tensor contra el archivo original con
`torch.equal`, con resultado `True` para todos: la inferencia no altera el checkpoint.
El experto se cargó con `weights_only=True` (168 tensores).

### 3.2 Audio

- Dataset: Google FLEURS, configuración `ca_es`, split `dev`.
- Archivo: `data/samples/ca_es_sample.wav`.
- Duración: 20.70 s; 16 kHz; mono; float32.
- Referencia: transcripción oficial de FLEURS (`data/samples/reference_ca.txt`).
- Descarga reproducible con `scripts/prepare_sample.py` mediante petición HTTP por
  rangos, sin bajar el archivo completo de 573 MB.

---

## 4. Entorno

### 4.1 Equipo

- Windows 11.
- GPU NVIDIA RTX 2050, 4 GB, compute capability 8.6.
- Driver 610.78; CUDA UMD 13.3.
- `ffmpeg` instalado vía winget (Gyan.FFmpeg 9.0.2); WAV/FLAC se decodifican con `soundfile` y MP3/M4A/AAC/MP4/MOV se convierten a WAV 16 kHz mono con FFmpeg (`_find_ffmpeg`: PATH, `C:\ffmpeg\bin\ffmpeg.exe`, `%ProgramFiles%\ffmpeg\bin\ffmpeg.exe` y `%LOCALAPPDATA%\Microsoft\WinGet\Packages`).

### 4.2 Entorno principal (`.venv`)

| Componente | Versión |
|---|---|
| Python | 3.13.15 |
| PyTorch | 2.10.0+cu128 |
| transformers | 4.57.3 |
| tokenizers | 0.22.2 |
| huggingface-hub | 0.36.2 |
| safetensors | 0.8.0 |
| numpy | 2.5.5 |
| soundfile | 0.13.1 |

Creado con `uv`, aislado del Python global (3.14 con torch CPU) y de Anaconda
(torch 2.12.0+cpu). El índice usado para PyTorch fue
`https://download.pytorch.org/whl/cu128`.

### 4.3 Entorno de verificación (`.venv-legacy`)

Para demostrar que los resultados modernos son correctos se creó un entorno aparte con
las versiones originales:

| Componente | Versión |
|---|---|
| Python | 3.10.22 |
| PyTorch | 2.5.1+cpu |
| transformers | 4.30.1 |
| tokenizers | 0.13.3 |
| huggingface-hub | 0.16.4 |
| numpy | 1.26.4 |

Este entorno solo se usó como oráculo de comparación; no forma parte del runtime del
proyecto.

---

## 5. Arquitectura de la solución

```
src/audio2text/
├── chunking.py          # ventanas, cortes por silencio, unión sin duplicados
├── compat.py            # shims de importación y de atención
├── legacy_attention.py  # WhisperAttention con contrato 4.30
├── engine.py            # carga, audio, transcribe / transcribe_long / transcribe_auto
├── metrics.py           # CER/WER sin dependencias
└── ui_service.py        # validación de subidas, serialización TXT/JSON, historial
scripts/
├── prepare_sample.py    # audio FLEURS corto
├── make_long_sample.py  # audio sintético de 42.4 s
└── run_inference.py
tests/
├── test_chunking.py
├── test_engine.py
└── test_streamlit_app.py
vendor/multilingual-distilwhisper/   # repo oficial, sin modificar
```

Decisión de diseño: **no editar el código oficial**. Todas las adaptaciones se aplican
en tiempo de ejecución desde `src/audio2text/`, de modo que el *vendor* se pueda
actualizar o auditar contra el original.

---

## 6. Compatibilidad: problemas y soluciones

El repositorio oficial declara `transformers==4.30.1`, que no tiene wheels para Python
3.13. Se optó por Python 3.13 + transformers 4.57.3 y se corrigieron las diferencias.

### 6.1 Import de `datasets`

`DistilWhisper.py` importa `datasets` a nivel de módulo aunque la inferencia no lo usa.
Instalarlo arrastra pyarrow/pandas y cientos de MB. Solución: stub de `datasets` en
`sys.modules` si no está instalado, insertado después de importar `transformers` para no
romper su detección de submódulos.

### 6.2 `cached_file` con argumentos eliminados

`load_experts` pasa `resume_download` y `proxies` a `transformers.utils.hub.cached_file`;
`huggingface_hub` moderno ya no los acepta. Solución: envolver `cached_file` y filtrar
esos kwargs.

### 6.3 Contrato de `WhisperAttention`

El código de 4.30 espera que la atención devuelva
`(attn_output, attn_weights, past_key_value)` y concatena ella misma la caché KV.
Transformers ≥ 4.36 devuelve 2 tuplas y usa `Cache` objects. Solución: restaurar la
implementación 4.30 en `legacy_attention.py` e inyectarla como
`DistilWhisper.WhisperAttention`. Los módulos se construyen después de la inyección, por
lo que todos usan la clase legacy.

### 6.4 `GenerationMixin` fuera de `PreTrainedModel`

Transformers ≥ 4.50 eliminó `GenerationMixin` de `PreTrainedModel`; la clase oficial
llama a `super().generate(...)`. Solución: crear en `engine.py` una subclase
`class DistilWhisperGenerator(base, GenerationMixin)` y sobrescribir
`_supports_default_dynamic_cache()` para devolver `False`, porque el decoder espera
tuplas KV legacy.

### 6.5 `generation_config` ausente

`from_pretrained` sobre una clase custom no crea `generation_config`. Solución: cargarlo
explícitamente con `GenerationConfig.from_pretrained(model_path)`.

### 6.6 Prefijo forzado ignorado (el problema principal)

El código oficial fija `generation_config.forced_decoder_ids` y confía en que
`generate` los aplique. **Transformers ≥ 4.50 ya no lee ese campo**: se verificó que no
aparece ninguna referencia en `generation/utils.py` ni en `configuration_utils.py`.

Consecuencia observada con el código moderno sin corregir:

- greedy generaba `<|startoftranscript|><|ca|>` y solo 37 tokens basura;
- beam 5 generaba 108 tokens con texto casi correcto pero **no idéntico** al de 4.30.1.

La causa era la ausencia de `<|transcribe|>` (50359) y `<|notimestamps|>` (50363) al
inicio del decoder.

Solución: construir el prefijo completo y pasarlo como `decoder_input_ids`:

```python
prefix = [50258, 50270, 50359, 50363]   # start, ca, transcribe, notimestamps
model.generate(input_features, decoder_input_ids=torch.tensor([prefix]), ...)
```

Con este cambio, greedy y beam 5 en 4.57.3 producen **exactamente** la misma secuencia
de 109 tokens que 4.30.1. El prefijo queda registrado en `Transcription.extra` y en el
JSON de salida.

### 6.7 Experto en fp16

`load_experts` crea módulos nuevos en fp32 después de cargar el checkpoint. Por eso se
llama a `model.to(device=..., dtype=...)` **después** de `load_experts`; de lo contrario
parte del modelo queda en fp32 y la inferencia fp16 falla o mezcla precisiones.

### 6.8 Aviso de máscara de atención

Durante `generate` aparece:

```
The attention mask is not set and cannot be inferred from input because pad token
is same as eos token.
```

Es benigno en este caso: el `forward` del modelo solo usa `attention_mask` para
SpecAugment en entrenamiento, no en evaluación. Se mantiene visible y documentado en
lugar de silenciarlo.

---

## 7. Metodología de verificación

1. **Integridad de pesos:** comparación tensor a tensor contra `model.safetensors`
   (`torch.equal` → `True`).
2. **Oráculo legacy:** ejecución con `transformers 4.30.1` en `.venv-legacy` (CPU) para
   obtener la secuencia de tokens de referencia; greedy y beam 5 coinciden entre sí
   (109 tokens).
3. **Comparación exacta:** las ejecuciones modernas se comparan por lista de ids, no
   solo por texto; se exige igualdad token a token.
4. **Repetición:** CUDA fp32, CUDA fp16, CPU fp32 y CUDA greedy con el mismo audio y
   prefijo.
5. **Métricas:** CER y WER calculados con `metrics.py` (normalización: NFKC,
   minúsculas, puntuación → espacio; CER sin espacios).

---

## 8. Resultados

Audio de 20.70 s. `Latencia` es solo `generate`; `Total` incluye carga del modelo,
lectura de audio y generación. VRAM pico medida con `torch.cuda.max_memory_allocated`.

| Dispositivo | Precisión | Búsqueda | Carga | Latencia | Total | VRAM pico | Tokens | CER | WER |
|---|---|---|---|---|---|---|---|---|---|
| CUDA | fp32 | beam 5 | 5.16 s | 9.66 s | 20.15 s | 2592 MiB | 109 | 2.86 % | 9.80 % |
| CUDA | fp16 | beam 5 | 3.97 s | 7.72 s | 17.83 s | 1315 MiB | 109 | 2.86 % | 9.80 % |
| CPU | fp32 | beam 5 | 2.81 s | 28.78 s | 37.61 s | — | 109 | 2.86 % | 9.80 % |
| CUDA | fp32 | greedy | 4.21 s | 6.92 s | 17.70 s | 1603 MiB | 109 | 2.86 % | 9.80 % |

Observaciones:

- Las cuatro salidas son la **misma secuencia de 109 ids**, idéntica al oráculo 4.30.1.
- fp16 reduce VRAM a la mitad y es ~20 % más rápido que fp32 con beam 5 en esta GPU.
- Greedy es más rápido y usa menos memoria que beam 5, y en este audio produce el mismo
  resultado; aun así se deja beam 5 como valor por defecto por robustez.
- El modelo cabe en 4 GB con margen en todas las configuraciones probadas.

Texto obtenido:

> Avui en dia, la cuina japanesa és fàcil de trobar per tot arreu de l'arxipèleg i ofereix una gran varietat de plats condimentats de forma senzilla.Els sabors preferits als japanesos solen ser els dels cacauets, els bitxos, el sucre, sobretot el sucre, coco javanès i diverses espècies aromàtiques.

Referencia FLEURS:

> Avui en dia, la cuina javanesa és fàcil de trobar per tot arreu de l'arxipèlag i ofereix una gran varietat de plats condimentats de forma senzilla. Els sabors preferits dels javanesos solen ser els dels cacauets, els bitxos, el sucre (sobretot el sucre de coco javanès) i diverses espècies aromàtiques.

Los errores son esencialmente ortográficos y de puntuación; el contenido se preserva.
El CER/WER bajo no implica una transcripción perfecta: el modelo omite paréntesis y
algunos artículos, y escribe `japanesa`/`japanesos` en lugar de `javanesa`/`javanesos`.

---

## 9. Etapa 2: fragmentación de audio largo

### 9.1 Estrategia

`transcribe` se mantiene intacta para audios de hasta 30 s. `transcribe_auto` decide:

- duración ≤ 30 s o `--no-chunk` → ruta corta validada;
- duración > 30 s → `transcribe_long`.

`transcribe_long` usa ventanas con conciencia de silencio:

1. **Detección de pausas:** se calcula el RMS en tramas de 20 ms; una secuencia de
   tramas por debajo de `silence_rms` (1e-4) durante al menos `min_silence_s` (0.4 s)
   se considera una pausa. Se propone como corte el centro de la pausa.
2. **Selección del corte:** para cada ventana se elige la pausa más tardía que quepa
   antes de los 30 s y que deje al menos 2 s de audio. La ventana termina en la pausa y
   la siguiente empieza exactamente ahí: no hay audio solapado, así que no hay texto que
   deduplicar ni contenido perdido.
3. **Sin pausas:** se usan ventanas de 30 s con 2 s de solapamiento. El texto nuevo se
   compara con el final del texto acumulado (palabras normalizadas, sin mayúsculas ni
   puntuación) y se elimina el prefijo repetido si coincide en 2–12 palabras. Si no hay
   coincidencia, se concatena sin borrar nada.
4. **Prefijo:** cada fragmento no silencioso recibe
   `decoder_input_ids = [[50258, 50270, 50359, 50363]]` y los mismos
   `max_new_tokens`/`num_beams`.
5. **Silencio:** un fragmento con RMS ≤ `silence_rms` no llama al modelo; se registra
   como segmento `silent`.
6. **Memoria:** tras cada fragmento se liberan features, ids y features de entrada, se
   ejecuta `gc.collect()` y `torch.cuda.empty_cache()`. La VRAM pico es el máximo de los
   fragmentos.
7. **Timestamps:** `start_s`/`end_s` son los límites de la ventana. Son estimaciones por
   segmento, **no** timestamps por palabra.
8. **Audio vacío o ilegible:** un array vacío devuelve una transcripción vacía sin cargar
   el modelo; `read_audio` convierte archivos inexistentes, formato inválido, sample rate
   inválido o muestras NaN/inf en `AudioReadError`.

### 9.2 Pruebas unitarias

32 pruebas con `unittest`, sin GPU ni checkpoint:

- planificación de ventanas: audio vacío, 20.7 s, 30.0 s exactos, 30.001 s, 42.4 s,
  58.1 s, 60 s, 88 s, 123.456 s; cobertura, solapamiento y límite de 30 s;
- detección de pausas y corte sin solapamiento;
- fusión de texto: sin solape, solape exacto, mayúsculas/puntuación, límite de palabras,
  palabra única no eliminada;
- silencio, fragmento vacío, audio completamente silencioso;
- audio vacío sin llamada al modelo;
- errores de lectura: archivo inexistente, contenido inválido, WAV vacío, NaN;
- dispatch corto/largo y `--no-chunk`;
- orquestación con procesador/modelo falsos: dos ventanas, prefijo en cada llamada,
  timestamps, deduplicación y tokens por segmento.

Comando:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

### 9.3 Resultados con audio de 42.40 s

Audio sintético `data/samples/ca_es_long.wav`: clip FLEURS de 20.7 s + 1 s de silencio
+ el mismo clip. Referencia: la transcripción FLEURS repetida dos veces.

| Configuración | Segmentos | Latencia | VRAM pico | CER | WER |
|---|---|---|---|---|---|
| CUDA fp16, beam 5, pausas 0.4 s | 2 (0.00–21.01 s, 21.01–42.40 s) | 16.26 s | 1315 MiB | 4.49 % | 12.75 % |

- La pausa de 1 s se detectó y se usó como frontera, sin solapamiento.
- El texto completo contiene las dos repeticiones, sin duplicación ni pérdida en la
  unión.
- La segunda repetición perdió `arreu de l'` en `per tot arreu de l'arxipèleg`
  (variación del modelo), pero no faltan frases completas.
- Tokens generados: 213 (109 + 104), cada segmento con su prefijo.
- El CER sube de 2.86 % a 4.49 % por esa omisión y por puntuación; sigue siendo un
  resultado aceptable para una prueba de integración.

### 9.4 Comparación con la etapa 1

Se repitió la inferencia corta con la configuración por defecto (CUDA fp16, beam 5):

- texto **idéntico** al de la etapa 1;
- **mismos 109 tokens**, comparados id a id;
- prefijo `[50258, 50270, 50359, 50363]`;
- VRAM pico 1315 MiB; latencia 8.88 s (la etapa 1 midió 7.72 s; diferencia de ruido
  de medición, mismo resultado).

Evidencia: `logs/inference_stage2_short_fp16.json` frente a
`logs/inference_cuda_fp16.json`.

### 9.5 Limitación del solapamiento sin pausas

Se ejecutó un experimento forzando `min_silence_s=999` para desactivar los cortes por
pausa. En ese caso el modelo terminó la primera ventana en la pausa interna de 20.7 s y
no transcribió los 8 s siguientes; la segunda ventana empezó en mitad de una frase
(`Fareix...`). El resultado tuvo CER 16.12 % y WER 26.47 %, con pérdida y texto raro en
la unión.

Conclusión: el solapamiento y la deduplicación no pueden recuperar audio que el modelo
decidió no transcribir. Por eso la estrategia por defecto corta en pausas; para audio
continuo sin pausas largas el solapamiento funciona (verificado con modelo falso en las
pruebas unitarias), pero si el modelo emite EOS antes de tiempo dentro de una ventana
hay que reducir `--max-chunk-seconds` o ajustar `--min-silence-seconds`.

Evidencia: `logs/inference_stage2_long_overlap_fp16.json`.

---

## 10. Limitaciones

1. **Uniones:** los cortes por pausa evitan pérdida/duplicación; en audio continuo la
   deduplicación es heurística y puede fallar si el modelo termina antes de tiempo o
   transcribe distinto el mismo audio en dos ventanas.
3. **Detección de silencio:** umbral RMS absoluto; audio muy bajo o con ruido por encima
   de `--silence-rms` puede no detectarse como pausa/silencio.
4. **Idiomas:** solo catalán, tamil y tailandés en el 28k. No hay experto de español
   en el checkpoint; español está disponible con Whisper Small (`language="es"`).
5. **Formato:** WAV/FLAC vía `soundfile`; MP3, M4A, AAC, MP4 y MOV vía `FFmpeg` (conversión a WAV 16 kHz mono).
6. **Fragmentación de frases:** el texto puede unir frases sin espacio
   (`senzilla.Els`), comportamiento del tokenizer/modelo, no del pipeline.
7. **Evaluación:** una muestra corta y otra sintética; CER/WER no son estadísticamente
   representativos.
8. **Rendimiento:** la latencia de CPU es ~3× la de GPU; para uso interactivo conviene
   CUDA fp16.
9. **Versión fijada:** los shims dependen de la API de `transformers 4.57.3`; una
   actualización mayor requiere revisar `compat.py` y `legacy_attention.py`.

---

## 11. Trabajo pendiente

1. **Evaluación con audio largo real** (no sintético) y varias muestras de FLEURS.
2. **Capturas y revisión visual** de la interfaz Streamlit en un navegador real;
   la funcionalidad ya está cubierta con `AppTest`.
4. **Empaquetado**: `pyproject.toml` y CI sin GPU.
5. **Decidir** si se conserva `.venv-legacy` como entorno de verificación o se elimina.

---

## 12. Reproducibilidad

```powershell
# Entorno
uv venv .venv --python 3.13
uv pip install --python .venv\Scripts\python.exe torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv\Scripts\python.exe -r requirements.txt

# Audios de prueba
.\.venv\Scripts\python.exe scripts\prepare_sample.py
.\.venv\Scripts\python.exe scripts\make_long_sample.py

# Pruebas unitarias
.\.venv\Scripts\python.exe -m unittest discover -s tests -v

# Interfaz local
.\.venv\Scripts\streamlit.exe run streamlit_app.py

# Inferencia real a través de la UI (opcional, requiere modelo)
$env:AUDIO2TEXT_RUN_UI_INFERENCE = "1"
.\.venv\Scripts\python.exe -m unittest tests.test_streamlit_inference -v

# Inferencia corta (fp16 por defecto en CUDA)
.\.venv\Scripts\python.exe scripts\run_inference.py `
  --reference data\samples\reference_ca.txt `
  --json-out logs\inference_stage2_short_fp16.json

# Audio largo
.\.venv\Scripts\python.exe scripts\run_inference.py `
  --audio data\samples\ca_es_long.wav `
  --reference data\samples\reference_ca_long.txt `
  --json-out logs\inference_stage2_long_fp16.json

# Inferencia con Whisper Small (español; --language es opcional)
.\.venv\Scripts\python.exe scripts\run_inference.py `
  --model-id openai/whisper-small `
  --audio data\samples\es_es_sample.wav `
  --json-out logs\inference_whisper_small_es_short.json

# Inferencia real con Whisper Small desde Streamlit (opcional, requiere pesos)
$env:AUDIO2TEXT_RUN_WHISPER_INFERENCE = "1"
.\.venv\Scripts\python.exe -m unittest tests.test_whisper_small.WhisperSmallRealInferenceTest -v
```

Evidencias guardadas en `logs/`:

- `inference_cuda_fp32.json`, `inference_cuda_fp16.json`, `inference_cpu_fp32.json`,
  `inference_cuda_fp32_greedy.json`: etapa 1 con `token_ids`.
- `inference_stage2_short_fp16.json`: comparación corta de la etapa 2.
- `inference_stage2_long_fp16.json`: audio de 42.4 s con segmentos y timestamps.
- `inference_stage2_long_overlap_fp16.json`: experimento sin cortes por pausa.
- `legacy_transformers430_results.json`: oráculo con transformers 4.30.1.
- `legacy_verify.py`, `legacy_verify_out.txt`: verificación legacy.
- `streamlit_ui_inference.json`: transcripción corta real desde Streamlit.
- `streamlit_ui_inference_long.json`: transcripción larga real desde Streamlit.
- `*.log`: salidas completas de cada ejecución.

---

## 13. Conclusiones

- El modelo oficial funciona en la RTX 2050 de 4 GB y en CPU, sin entrenamiento ni
  cambios de pesos.
- La calidad en catalán es buena para una muestra de 20.7 s: CER 2.86 %, WER 9.80 %,
  con errores fundamentalmente ortográficos.
- El chunking por pausas permite transcribir audio de más de 30 s conservando el orden
  temporal y sin duplicar la unión; el audio sintético de 42.4 s se transcribió
  completo en 16.26 s con 1315 MiB.
- Los timestamps son estimaciones por segmento; no se presentan como precisión por
  palabra.
- El problema real de la etapa 1 fue de compatibilidad de API, no de hardware. Los
  *shims* documentados permiten usar el código oficial con un stack moderno y
  verificable, y la igualdad token a token con `transformers 4.30.1` sigue intacta.
- Etapa 4: se incorporó `openai/whisper-small` como alternativo para transcribir en
  español; el modelo académico principal de NAVER (28k) no se alteró.
- La interfaz Streamlit mantiene el motor separado, reutiliza el modelo con
  `st.cache_resource(max_entries=1)` y ofrece carga, configuración, progreso,
  resultados, descargas e historial de sesión.
- La etapa 3 queda verificada funcionalmente con `AppTest`, incluida la inferencia real
  corta y larga. Quedan pendientes las capturas visuales y la evaluación con más audio
  real.
- Entrada de audio: la interfaz permite dos opciones — subida de archivo o grabación
  desde el micrófono (`st.audio_input`) — que llevan al mismo proceso de transcripción.

---

## 14. Etapa 3: interfaz Streamlit

### 14.1 Arquitectura

- `streamlit_app.py`: composición de la UI, estado de sesión, caché del modelo,
  progreso y renderizado.
- `src/audio2text/ui_service.py`: helpers puros sin Streamlit para validar subidas,
  formatear timestamps, construir registros y serializar TXT/JSON.
- `src/audio2text/engine.py` y `chunking.py`: no se modificaron. La UI llama a
  `load_model` y `transcribe_auto`; no contiene lógica de inferencia.
- `.streamlit/config.toml`: tema nativo, límite de subida de 200 MB y sin CSS frágil.

### 14.2 Gestión del modelo y memoria

- `@st.cache_resource(show_spinner=False, max_entries=1, on_release=...)` con clave
  `(ruta del modelo, idioma, dispositivo, precisión, modelo)`.
- Solo se conserva una configuración cargada por modelo para no acumular checkpoints
  en 4 GB.
- Al desalojar la entrada se eliminan las referencias, se ejecuta `gc.collect()` y
  `torch.cuda.empty_cache()`.
- El botón «Liberar modelo de memoria» vacía la caché explícitamente.
- Cambiar beam search no recarga el modelo; cambiar idioma/dispositivo/precisión o
  el modelo (NAVER ⇄ Whisper Small) sí.

### 14.3 Flujo de usuario

1. Subida WAV, FLAC, MP3, M4A, AAC, MP4 y MOV; se validan extensión, tamaño, bytes vacíos y decodificación con
   `read_audio` (archivo temporal con el sufijo original).
2. Metadatos, reproductor y configuración visible.
3. Grabación desde el micrófono: `st.audio_input` ofrece los controles que permite el
   navegador; al terminar, el reproducer muestra el audio adjunto (duration y formato).
4. Botón de transcripción con progreso por fases: carga e inferencia. Ambos flujos
   (archivo subido y grabación) usan el mismo motor con `transcribe`/`transcribe_auto`.
5. Resultado con texto copiable, métricas, segmentos con inicio/fin (estimaciones por
   ventana), descargas TXT/JSON e historial de la sesión.
6. Errores traducidos a mensajes accionables (audio inválido, grabación vacía, modelo
   ausente, OOM, CUDA no disponible, configuración no válida).

### 14.4 Verificación

- Suite rápida: **90 pruebas, 86 ejecutadas y 4 omitidas** (inferencia real).
  Incluye pruebas de motor/chunking, helpers de UI, `AppTest` (subida, grabación,
  errores) y suites de Whisper Small (routing, carga real).
- Inferencia real desde Streamlit:
  - `ca_es_sample.wav`: texto y 109 tokens idénticos a la etapa 1; 6.35–8.18 s con
    el modelo en caché; VRAM pico 1315 MiB.
  - `ca_es_long.wav`: 15.33–18.57 s, 2 segmentos (0.00–21.01 s y 21.01–42.40 s),
    213 tokens, texto idéntico a la etapa 2.
- Servidor real comprobado con `streamlit run` y `/_stcore/health` = `ok`.
- Evidencias: `logs/streamlit_ui_inference.json` y
  `logs/streamlit_ui_inference_long.json`.

### 14.5 Limitaciones de la interfaz

- El historial es de sesión (máximo 20) y no persiste al recargar o cerrar.
- No se guarda el audio en el historial; solo el resultado y sus metadatos.
- El progreso es por fases, no por chunk.
- No hay autenticación ni multiusuario: está pensada para uso local.
- En CPU no se ofrece FP16; la latencia es mucho mayor.
- Los timestamps siguen siendo estimaciones por ventana.

---

## 16. Incorporación de OpenAI Whisper Small (español)

### 16.1 Motivación y alcance

El checkpoint 28k de NAVER no incluye experto de español (criterio de aceptación de
la etapa 2: "no español"). Para cubrir ese hueco, se incorporó
`openai/whisper-small` como modelo alternativo, sin reemplazar ni modificar el
modelo académico principal (NAVER 28k). Spanish (es) es soportado por Whisper
Small mediante `language="es"`; el selector de la interfaz permite elegir entre ambos.

### 16.2 Cambios en `src/audio2text/engine.py`

- Nuevas constantes: `MODEL_NAVER_ID` (antiguo `MODEL_ID`),
  `MODEL_WHISPER_SMALL_ID = "openai/whisper-small"` y
  `WHISPER_SMALL_LANGUAGES = ("es",)`.
- `_load_whisper_small()`: carga `WhisperProcessor` +
  `WhisperForConditionalGeneration` desde `openai/whisper-small` con los mismos
  `dtype`/`device` validados; sin entrenamiento ni alteración de pesos.
- `load_model(..., model_id=MODEL_NAVER_ID)`: para `openai/whisper-small` valida
  `language="es"` antes de cargar; el resto del flujo (`transcribe_auto`,
  `transcribe_long`, chunking) es compartido y sin cambios.
- `_generate()`:
  - NAVER 28k: prefijo forzado validado + `decoder_input_ids` (vendido);
  - `openai/whisper-small`: `generate(language="es", task="transcribe", num_beams,
    max_new_tokens)`. Transformers 4.57.3 rechaza prefijo + `max_new_tokens >
    max_target_positions` (448), así que el presupuesto se reduce por la longitud
    del prefijo (448 → 444); la longitud total generada no cambia.

### 16.3 Cambios en la interfaz

- Selector «Modelo» en la barra lateral: NAVER Multilingual DistilWhisper 28k
  (predeterminado) y OpenAI Whisper Small.
- `languages_for_model()`: NAVER → `("catalán", "tamil", "tailandés")` (lista
  inalterada); Whisper Small → `("es",)`. Al seleccionar Whisper Small, el idioma
  se setea automáticamente a «Español»; al volver a NAVER, el idioma vuelve a uno
  de los tres y se renormaliza.
- `TranscriptionConfig`/`TranscriptionRecord` incluyen `model_id`;
  `record_to_payload`/`record_to_txt` incluyen el modelo usado.
- `get_model` (`st.cache_resource`, `max_entries=1`): clave incluye `model_id`;
  cambiar de modelo libera la entrada anterior para no acumular checkpoints.
- CLI `scripts/run_inference.py`: nuevo `--model-id` (opciones NAVER/Whisper
  Small) y resolución de idioma por defecto (`es` si Whisper Small sin `--language`).

### 16.4 Verificación real (CUDA fp16, RTX 2050, beam 5)

Audio español descargado del FLEURS (`google/fleurs`, configuración `es_419`,
`data/samples/`):

- `es_es_sample.wav` (8.40 s): transcripción corta en 2.54 s → texto:
  > Milenios atrés, un hombre llamado Aristarco comentó que el sistema solar giraba en torno al sol.
- `es_es_long.wav` (36.60 s): inferencia larga en 15.84 s, 2 segmentos.

Ambos ejecutan con `language="es"` y el pipeline compartido (dispatch corto/largo,
silencio, dedup). Las pruebas reales viven en
`tests/test_whisper_small.py::WhisperSmallRealInferenceTest`
(`AUDIO2TEXT_RUN_WHISPER_INFERENCE=1`); `WhisperSmallLoadTest` carga ambos modelos
y salta si Whisper Small no está en caché local (su descarga no se hace en la suite
rápida).

### 16.5 Memoria

Whisper Small fp16 (786 MiB reportados mediante `peak_vram_mb` en la inferencia
corta, RTX 2050) es sustancialmente más pequeño que el NAVER fp16 (~1315 MiB
reportados), dejando margen en el equipo de 4 GB para cargar ambos y cambiar de
configuración sin acumular checkpoint.

---

## 15. Referencias

- NAVER, *multilingual-distilwhisper*: <https://github.com/naver/multilingual-distilwhisper>
- Checkpoint: <https://huggingface.co/naver/multilingual-distilwhisper-28k>
- FLEURS: <https://huggingface.co/datasets/google/fleurs>
- Transformers, *Generation strategies*: <https://huggingface.co/docs/transformers/main/en/generation_strategies>
- PyTorch CUDA wheels: <https://download.pytorch.org/whl/cu128>
