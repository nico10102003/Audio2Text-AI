"""Design system for the Audio2Text AI Streamlit interface.

Paleta original de Audio2Text AI (restaurada): indigo (#4F46E5, color
principal y acentos), blanco (fondos principales), gris muy claro (#F8FAFC,
fondos secundarios), gris oscuro (#0F172A, textos), bordes #E2E8F0 y fondos
de código #F1F5F9. Misma paleta que `.streamlit/config.toml`. Injected once
per session; all rules are optional and reversible.
"""

from __future__ import annotations

import streamlit as st

STYLE_TAG = """
<style>
/* --- Paleta original ----------------------------------------------------- */
:root {
  --a2t-primary:    #4F46E5;   /* indigo: color principal (paleta original) */
  --a2t-secondary:  #0F172A;   /* gris oscuro: elementos secundarios */
  --a2t-accent:     #4F46E5;   /* indigo: acentos (mismo color que el principal) */
  --a2t-bg:         #FFFFFF;   /* blanco: fondos principales */
  --a2t-surface:    #F8FAFC;   /* gris muy claro: fondos secundarios */
  --a2t-text:       #0F172A;   /* gris oscuro: textos */
  --a2t-muted:      #64748B;
  --a2t-border:     #E2E8F0;
  --a2t-success:    #16A34A;
  --a2t-warning:    #D97706;
  --a2t-danger:     #DC2626;

  /* Tonos diferenciados para la sección educativa. */
  --a2t-tone-encoder: #0891B2;
  --a2t-tone-decoder: #4F46E5;
  --a2t-tone-q:       #7C3AED;
  --a2t-tone-k:       #15803D;
  --a2t-tone-v:       #2563EB;
  --a2t-tone-process: #06B8A4;
}

/* --- Tipografía ---------------------------------------------------------- */
body, .stApp {
  font-family: "Inter", "Segoe UI Variable", system-ui, -apple-system,
               "Segoe UI", Roboto, sans-serif;
}
h1, h2, h3, h4 { letter-spacing: -0.01em; margin-top: 0; }
.stMarkdown h1 { color: var(--a2t-text); font-weight: 700; }
.stMarkdown h2 { color: var(--a2t-text); font-weight: 600; }
.stMarkdown h3 { color: var(--a2t-text); font-weight: 600; }
.stMarkdown h4 { color: var(--a2t-text); font-weight: 600; }

/* --- Cabecera ------------------------------------------------------------ */
.a2t-subtitle { font-size: 1.5rem; font-weight: 600; color: var(--a2t-text); }
.stMarkdown h2.a2t-subtitle { color: var(--a2t-text); }
.a2t-lede { font-size: 1rem; color: var(--a2t-muted); margin-bottom: 20px; }
.a2t-icon { display: inline-block; vertical-align: middle; margin-right: 8px; }
.a2t-icon svg { display: block; }

/* --- Ondas de sonido (ilustración tecnológica) --------------------------- */
@keyframes a2t-wave {
  to { transform: translateX(-50%); }
}
.a2t-wave {
  display: block;
  filter: drop-shadow(0 1px 3px rgba(15, 23, 42, 0.16));
}
@media (prefers-reduced-motion: reduce) {
  .a2t-wave { animation: none; }
}

/* --- Tarjetas ------------------------------------------------------------ */
.a2t-card {
  background: var(--a2t-bg);
  border: 1px solid var(--a2t-border);
  border-radius: 14px;
  padding: 22px;
  box-shadow: 0 2px 10px rgba(15, 23, 42, 0.05);
  transition: transform 0.16s ease, box-shadow 0.16s ease;
}
.a2t-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
}
.a2t-card-title {
  font-weight: 600;
  color: var(--a2t-primary);
  font-size: 1.05rem;
  margin: 0 0 10px;
}

/* --- Botones y entradas ---------------------------------------------------- */
.stButton button, .stDownloadButton button {
  border-radius: 9px;
  font-weight: 500;
  transition: transform 0.14s ease, box-shadow 0.14s ease;
}
.stButton button:hover {
  transform: translateY(-1px);
  box-shadow: 0 6px 16px rgba(15, 23, 42, 0.14);
}
button:focus-visible, textarea:focus-visible, input:focus-visible,
[role="combobox"]:focus-visible {
  outline: 3px solid var(--a2t-accent);
  outline-offset: 2px;
}

/* --- Tarjetas de métricas -------------------------------------------------- */
[data-testid="stMetric"] {
  border: 1px solid var(--a2t-border);
  border-radius: 12px;
  background: var(--a2t-bg);
  box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
}

/* --- Fondos secundarios (gris muy claro) ---------------------------------- */
.a2t-surface { background: var(--a2t-surface); }
.a2t-text-secondary { color: var(--a2t-muted); }
[data-testid="stTabs"] {
  background: var(--a2t-surface);
  border-radius: 10px;
  padding: 8px 12px;
}

/* --- Adaptabilidad a pantallas más pequeñas -------------------------------- */
@media (max-width: 768px) {
  h1 { font-size: 2.1rem; }
  .a2t-subtitle { font-size: 1.25rem; }
}
</style>
"""


def inject_styles() -> None:
    """Inject the app-level CSS layer once per session (idempotent)."""
    if st.session_state.get("_a2t_styles_injected"):
        return
    st.session_state["_a2t_styles_injected"] = True
    st.html(STYLE_TAG)
