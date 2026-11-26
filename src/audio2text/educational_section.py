"""Sección «Arquitectura Transformer Encoder–Decoder» de Audio2Text AI.

Presentación exclusiva de la arquitectura del modelo (Encoder, Decoder,
Self-Attention, Cross-Attention y Q/K/V), verificada contra el código del
repositorio: `vendor/multilingual-distilwhisper/model/DistilWhisper.py`,
`models/multilingual-distilwhisper-28k/config.json` y
`src/audio2text/legacy_attention.py`.
"""

from __future__ import annotations

from html import escape

import streamlit as st

SECTION_TITLE = "Arquitectura Transformer Encoder–Decoder"

# Constantes verificadas en `models/multilingual-distilwhisper-28k/config.json`.
D_MODEL = 768
NUM_MEL_BINS = 80
MAX_SOURCE_POSITIONS = 1500
MAX_TARGET_POSITIONS = 448
VOCAB_SIZE = 51865
NUM_ENCODER_LAYERS = 12
NUM_DECODER_LAYERS = 12
NUM_ATTENTION_HEADS = 12
HEAD_DIM = D_MODEL // NUM_ATTENTION_HEADS  # 64

# Tonos diferenciados por componente (definidos en ui_styles.py).
ENCODER_FILL = "var(--a2t-tone-encoder)"
DECODER_FILL = "var(--a2t-tone-decoder)"
Q_FILL = "var(--a2t-tone-q)"
K_FILL = "var(--a2t-tone-k)"
V_FILL = "var(--a2t-tone-v)"
PROCESS_FILL = "var(--a2t-tone-process)"


# --------------------------------------------------------------------------
# Helpers: visual blocks
# --------------------------------------------------------------------------


def _svg(svg: str) -> None:
    """Render an inline SVG diagram inside the page flow."""
    st.html(svg)


def _svg_start(width: int, height: int) -> list[str]:
    """Open an SVG block with shared styles and an arrow marker."""
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        'style="display:block;font-family:&quot;Segoe UI&quot;,Roboto,sans-serif">',
        "<style>"
        ".node { stroke-width:2; rx:10; text-anchor:middle; font-size:13px; "
        "font-weight:600; fill:white; }"
        ".line { fill:none; stroke:var(--a2t-muted); stroke-width:2; "
        "marker-end:url(#arr); }"
        "</style>",
        '<defs><marker id="arr" markerWidth="10" markerHeight="7" viewBox="0 0 10 7" '
        'refX="8" refY="3.5" orient="auto">'
        '<path d="M1,0 L8,3.5 L1,7" fill="var(--a2t-muted)"/></marker></defs>',
    ]


def _box(x: float, y: float, w: float, h: float, label: str, fill: str) -> str:
    """Solid colored box with a centered label (up to two lines)."""
    lines = label.split("\n")
    parts = [
        f'<rect class="node" x="{x}" y="{y}" width="{w}" height="{h}" '
        f'style="fill:{fill}"/>'
    ]
    step = min(16, (h - 20) / max(len(lines), 1))
    for i, line in enumerate(lines):
        cy = y + h / 2 + (i - (len(lines) - 1) / 2) * step
        parts.append(
            f'<text x="{x + w / 2}" y="{cy}" text-anchor="middle" '
            f'fill="white" font-size="13" font-weight="600">{escape(line)}</text>'
        )
    return "".join(parts)


def _line(x1: float, y1: float, x2: float, y2: float) -> str:
    """Arrow between two points."""
    return f'<line class="line" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"/>'


def _card(title: str, body: str = "") -> None:
    """Simple titled card with a caption."""
    with st.container(border=True):
        st.markdown(f"**{escape(title)}**")
        st.caption(escape(body))


def _topic(title: str) -> None:
    st.subheader(title, divider=False)


# --------------------------------------------------------------------------
# SVG diagram builders
# --------------------------------------------------------------------------


def _pipeline_svg(steps: list[tuple[str, str]], width: int, box_h: int = 80) -> str:
    """Horizontal flow: boxes connected by arrows."""
    n = len(steps)
    step_w = (width - 40) // n
    box_w = 110
    parts = _svg_start(width, box_h + 30)
    for idx, (label, fill) in enumerate(steps):
        x = 20 + idx * step_w
        parts.append(_box(x, 12, box_w, box_h, label, fill))
        if idx < n - 1:
            arrow_x = x + box_w + 6
            parts.append(
                _line(
                    arrow_x,
                    box_h / 2 + 12,
                    arrow_x + max(step_w - 126, 40),
                    box_h / 2 + 12,
                )
            )
    parts.append("</svg>")
    return "".join(parts)


def _vstack_svg(items: list[tuple[str, str]], width: int = 300) -> str:
    """Vertical flow diagram: boxes connected by arrows, each (label, fill)."""
    pad, gap = 14, 18
    bw = width - 2 * pad
    heights = [68 + 14 * (label.count("\n") + 1) for label, _ in items]
    height = sum(heights) + gap * (len(items) - 1) + 2 * pad
    parts = _svg_start(width, height)
    y = pad
    for idx, (label, fill) in enumerate(items):
        if idx > 0:
            parts.append(_line(width / 2, y, width / 2, y + 6))
            y += 6
        parts.append(_box(pad, y, bw, heights[idx], label, fill))
        y += heights[idx] + gap
    parts.append("</svg>")
    return "".join(parts)


def _qkv_svg() -> str:
    """Horizontal diagram: Encoder → K, V; Decoder → Q; attention chain."""
    boxes = [
        (20, ENCODER_FILL),     # Encoder
        (240, K_FILL),          # K
        (370, Q_FILL),          # Q
        (500, PROCESS_FILL),    # QKᵀ
        (640, V_FILL),          # V
        (780, PROCESS_FILL),    # Softmax / √64
        (1010, PROCESS_FILL),   # pesos × V
        (1200, DECODER_FILL),   # contexto → out_proj → Decoder
    ]
    labels = [
        "encoder_hidden_states\n(1, ≈1500, 768)",
        "K = k_proj(encoder_hidden_states)",
        "Q = q_proj(estados del Decoder)",
        "QKᵀ",
        "V = v_proj(encoder_hidden_states)",
        "Softmax(QKᵀ/√64)",
        "pesos × V",
        "contexto → out_proj → Decoder",
    ]
    widths = [200, 110, 110, 120, 110, 180, 150, 160]
    parts = _svg_start(1400, 90)
    for (x, fill), label, w in zip(boxes, labels, widths):
        parts.append(_box(x, 10, w, 70, label, fill))

    def center(i):
        return boxes[i][0] + widths[i] / 2

    parts.append(_line(220, 85, 240, 85))                 # Encoder → K
    parts.append(_line(center(1), 55, 500, 55))           # K → QKᵀ
    parts.append(_line(center(2), 45, 500, 45))           # Q → QKᵀ
    parts.append(_line(center(3), 62, 640, 62))           # QKᵀ → V
    parts.append(_line(center(4), 55, 780, 55))           # V → Softmax
    parts.append(_line(center(5), 55, 1010, 55))          # Softmax → pesos × V
    parts.append(_line(center(6), 55, 1200, 55))          # pesos × V → Decoder
    parts.append("</svg>")
    return "".join(parts)


# --------------------------------------------------------------------------
# A. Visión general
# --------------------------------------------------------------------------


def _section_a() -> None:
    _topic("A. Visión general")
    st.caption(
        "Flujo completo: audio entrante, extracción de características acústicas "
        "(log-Mel) y cómo interactúan el Encoder y el Decoder."
    )
    _svg(
        _pipeline_svg(
            [
                ("Audio de entrada", PROCESS_FILL),
                ("Extracción de características acústicas (log-Mel)", PROCESS_FILL),
                (f"Encoder ({NUM_ENCODER_LAYERS} capas)", ENCODER_FILL),
                ("Representaciones acústicas", ENCODER_FILL),
                ("Cross-Attention", PROCESS_FILL),
                (f"Decoder ({NUM_DECODER_LAYERS} capas)", DECODER_FILL),
                ("Texto", PROCESS_FILL),
            ],
            width=1280,
        )
    )
    st.caption(
        "El Encoder procesa el audio una vez; el Decoder consulta las "
        "representaciones acústicas en cada paso mediante Cross-Attention y "
        "genera tokens de texto uno a uno."
    )


# --------------------------------------------------------------------------
# B. Encoder
# --------------------------------------------------------------------------


def _section_b() -> None:
    _topic("B. Encoder — representación acústica")
    _card(
        "¿Qué recibe?",
        f"El espectrograma log-Mel de audio: tensor `input_features`, forma "
        f"`(1, {NUM_MEL_BINS}, T)` — `num_mel_bins = {NUM_MEL_BINS}`.",
    )
    _card(
        "¿Qué procesa?",
        f"Stem conv: `conv1 = Conv1d({NUM_MEL_BINS} → {D_MODEL}, k=3, pad=1)` + "
        f"`conv2 = Conv1d({D_MODEL} → {D_MODEL}, k=3, stride 2, pad=1)`, ambos con "
        f"activación GELU. Posición: `embed_pos = nn.Embedding("
        f"{MAX_SOURCE_POSITIONS}, {D_MODEL})`. {NUM_ENCODER_LAYERS} capas pre-norm.",
    )
    _card(
        "¿Qué produce?",
        f"`encoder_hidden_states`: secuencia acústica, forma "
        f"`(1, ≈1500, {D_MODEL})`.",
    )
    _svg(
        _vstack_svg(
            [
                (f"input_features\nlog-Mel (1, {NUM_MEL_BINS}, T)", PROCESS_FILL),
                ("Stem conv1 + conv2 (+ GELU)", PROCESS_FILL),
                (f"embed_pos nn.Embedding({MAX_SOURCE_POSITIONS}, {D_MODEL})", PROCESS_FILL),
                (
                    f"{NUM_ENCODER_LAYERS} × Encoder layer (pre-norm, self-attn sin máscara)",
                    ENCODER_FILL,
                ),
                (
                    f"layer_norm → encoder_hidden_states\n(1, ≈1500, {D_MODEL})",
                    ENCODER_FILL,
                ),
            ]
        )
    )


# --------------------------------------------------------------------------
# C. Decoder
# --------------------------------------------------------------------------


def _section_c() -> None:
    _topic("C. Decoder — generación autoregresiva")
    _card(
        "¿Qué recibe?",
        "Prefijo forzado `decoder_input_ids` (startoftranscript, <|ca|>, "
        "<|transcribe|>, <|notimestamps|>) seguido de los tokens que se van "
        "generando (autoregresivo, hasta 448).",
    )
    _card(
        "¿Qué hace por capa?",
        f"Token → `embed_tokens` + posición; en cada una de las "
        f"{NUM_DECODER_LAYERS} capas (pre-norm): Self-Attention causal → "
        "Cross-Attention sobre `encoder_hidden_states` → FFN.",
    )
    _card(
        "¿Qué produce?",
        f"`proj_out` ({D_MODEL} → {VOCAB_SIZE}) → logits; el siguiente token se elige "
        "por beam search (`num_beams = 5`).",
    )
    _svg(
        _vstack_svg(
            [
                ("decoder_input_ids\nprefijo forzado + tokens generados", PROCESS_FILL),
                (f"embed_tokens nn.Embedding({VOCAB_SIZE}, {D_MODEL}) + posición", PROCESS_FILL),
                (
                    f"{NUM_DECODER_LAYERS} × Decoder layer (pre-norm): self-attn causal → cross-attn → FFN",
                    DECODER_FILL,
                ),
                ("layer_norm → estados del Decoder", DECODER_FILL),
                (f"proj_out: {D_MODEL} → {VOCAB_SIZE}", DECODER_FILL),
                ("Token siguiente (beam search)", PROCESS_FILL),
            ],
            width=360,
        )
    )


# --------------------------------------------------------------------------
# D. Self-Attention
# --------------------------------------------------------------------------


def _section_d() -> None:
    _topic("D. Self-Attention — relacionar posiciones de una misma secuencia")
    st.markdown("#### Encoder: sin máscara")
    _card(
        "¿Qué hace?",
        "Cada frame acústico puede atenderse a cualquier otro frame (incluyéndose "
        "a sí mismo). No hay preferencia temporal.",
    )
    st.markdown("#### Decoder: máscara triangular (causal)")
    _card(
        "¿Qué hace?",
        "El token en la posición `t` solo puede atender a sí mismo y a los "
        "anteriores; implementado con máscara triangular y caché de K/V "
        "(`past_key_value`).",
    )


# --------------------------------------------------------------------------
# E. Cross-Attention
# --------------------------------------------------------------------------


def _section_e() -> None:
    _topic("E. Cross-Attention — el Decoder consulta el Encoder")
    st.caption(
        "El Decoder busca en el audio codificado lo que cada token necesita. "
        "Flujo verificado en `legacy_attention.py`."
    )
    _svg(
        _pipeline_svg(
            [
                (f"Encoder — encoder_hidden_states\n(1, ≈1500, {D_MODEL})", ENCODER_FILL),
                ("K = k_proj(encoder_hidden_states)", K_FILL),
                ("V = v_proj(encoder_hidden_states)", V_FILL),
                ("Q = q_proj(estado actual del Decoder)", Q_FILL),
                ("QKᵀ", PROCESS_FILL),
                ("Softmax(QKᵀ/√64) → pesos", PROCESS_FILL),
                ("pesos × V → contexto → out_proj", DECODER_FILL),
            ],
            width=1300,
            box_h=100,
        )
    )


# --------------------------------------------------------------------------
# F. Q, K y V — apartado más importante
# --------------------------------------------------------------------------


def _section_f() -> None:
    _topic("F. Tensores Q, K y V")
    _card(
        "Q (Query)",
        "Se genera a partir de los estados del Decoder: `q_proj` sobre el estado "
        "actual autoregresivo. Representa qué información necesita consultar.",
    )
    _card(
        "K (Key)",
        "Se obtiene de `encoder_hidden_states` (`k_proj`). Permite calcular qué "
        "posiciones del audio son relevantes para Q.",
    )
    _card(
        "V (Value)",
        "También de `encoder_hidden_states` (`v_proj`). Contiene la información "
        "que se combina para construir el contexto.",
    )

    st.markdown("#### Fórmula")
    st.latex(
        r"\text{Attention}(Q,K,V) = "
        r"\operatorname{softmax}\left(\frac{QK^{T}}{\sqrt{d_k}}\right)V"
    )
    st.latex(
        r"\text{contexto} = "
        r"\operatorname{softmax}\left(\frac{QK^{T}}{\sqrt{" + str(HEAD_DIM) + r"}}\right) V \;\in\; R^{T_{dec} \times 768}"
    )

    st.markdown("#### Pasos")
    steps = [
        "El Decoder genera Q a partir de su estado actual.",
        "El Encoder proporciona las representaciones usadas para generar K y V.",
        "Se comparan Q y K mediante el producto escalar QKᵀ (puntuaciones).",
        f"Se escala por 1/√dₖ y se aplica softmax → pesos de atención (dₖ = {HEAD_DIM}).",
        "Los pesos se usan para combinar V (weighted sum → contexto).",
        "El contexto resultante ayuda al Decoder a generar el siguiente token.",
    ]
    for i, step in enumerate(steps, start=1):
        _card(f"Paso {i}", step)

    _svg(_qkv_svg())

    st.markdown("#### Analogía")
    st.caption(
        "- **Q (Query):** la pregunta que hace el lector: «¿cuál es el tema de este capítulo?».\n"
        "- **K (Key):** índice o título de cada libro: dice «aquí hay información sobre X, Y, Z». "
        "Se compara con Q mediante el producto escalar.\n"
        "- **V (Value):** el contenido escrito dentro del libro. El decodificador combina "
        "(suma ponderada) los V de los libros relevantes."
    )

    st.markdown("#### Proyecciones")
    _card(
        "Q, K y V son proyecciones numéricas aprendidas",
        "Matrices lineales `nn.Linear(768, 768)` sobre los estados ocultos; "
        "no son palabras ni fragmentos literales del audio.",
    )

    with st.expander("Código verificado de la atención (LegacyWhisperAttention)"):
        st.code(
            """is_cross_attention = key_value_states is not None
query_states = self.q_proj(hidden_states) * self.scaling          # Q
key_states = self._shape(self.k_proj(key_value_states), -1, bsz)  # K
value_states = self._shape(self.v_proj(key_value_states), -1, bsz)# V
attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))# QK^T
attn_weights = nn.functional.softmax(attn_weights, dim=-1)
attn_output = torch.bmm(attn_probs, value_states)                 # pesos @ V""",
            language="python",
            wrap_lines=True,
        )


# --------------------------------------------------------------------------
# Render
# --------------------------------------------------------------------------


def render_educational_section() -> None:
    """Render the compact, architecture-focused section below the main tool."""
    st.markdown("---")
    st.header(SECTION_TITLE, divider=False)
    st.caption(
        "Sección exclusiva sobre la arquitectura del motor: Encoder, Decoder, "
        "Self-Attention, Cross-Attention y Q, K, V."
    )
    _section_a()
    _section_b()
    _section_c()
    _section_d()
    _section_e()
    _section_f()
