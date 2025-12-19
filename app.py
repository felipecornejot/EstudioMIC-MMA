# app.py
import os
import re
import textwrap
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

try:
    import altair as alt
    ALTAIR_OK = True
except Exception:
    ALTAIR_OK = False

# =========================
# Configuración general
# =========================
st.set_page_config(
    page_title="Ranking y métricas MIC (MCA)",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

TITLE = "Ranking y métricas asociadas a mecanismos de información al consumidor"
SUBTITLE = "Consultoría Sustrend para la Subsecretaría del Medio Ambiente | ID: 608897-205-COT25"
FOOTER_TEXT = (
    "P9 consolida variables base (P2/P5), resultados MCA (P8) "
    "y metadatos de trazabilidad para auditoría y dashboard."
)

DEFAULT_DATASET_CSV = "P9_Dataset_Trazable_MIC.csv"
DEFAULT_DATASET_XLSX = "P9_Dataset_Trazable_MIC.xlsx"
MEMBRETE_FILENAME = "membrete (1).png"

# =========================
# Paleta institucional (FIJA)
# =========================
COLOR_BLUE_DARK = "#005EA8"
COLOR_BLUE = "#039BE5"
COLOR_GREEN = "#7CB342"
COLOR_ORANGE = "#EF6C00"
COLOR_TEXT = "#4A4A4A"
COLOR_TEXT_SOFT = "#6B6B6B"
COLOR_BG = "#FFFFFF"

# =========================
# CSS global (forzar claro)
# =========================
st.markdown(
    f"""
<style>
html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"] {{
    background-color: {COLOR_BG} !important;
    color: {COLOR_TEXT} !important;
}}

[data-testid="stSidebar"] {{
    background-color: #FFFFFF !important;
}}

h1, h2, h3 {{
    color: {COLOR_BLUE_DARK};
}}

.main-title {{
    font-size: 34px;
    font-weight: 800;
    margin-bottom: 4px;
    color: {COLOR_BLUE_DARK};
}}

.main-subtitle {{
    font-size: 14px;
    color: {COLOR_TEXT_SOFT};
    margin-bottom: 16px;
}}

.card {{
    background: #FFFFFF;
    border: 1px solid rgba(0,0,0,0.08);
    border-radius: 14px;
    padding: 14px 16px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.06);
}}

.card-title {{
    font-size: 13px;
    font-weight: 700;
    color: {COLOR_TEXT_SOFT};
    margin-bottom: 6px;
}}

.small-note {{
    font-size: 12px;
    color: {COLOR_TEXT_SOFT};
}}

.membrete-wrap {{
    display: inline-block;
    padding: 10px 12px;
    border-radius: 14px;
    background: #fff;
    border: 1px solid rgba(0,0,0,0.08);
    box-shadow: 0 10px 30px rgba(0,0,0,0.12);
}}

.hr-soft {{
    border: 0;
    border-top: 1px solid rgba(0,0,0,0.08);
    margin: 10px 0 16px 0;
}}
</style>
""",
    unsafe_allow_html=True,
)

# =========================
# Utilidades (SIN CAMBIOS)
# =========================
def _normalize_colname(c: str) -> str:
    return re.sub(r"\s+", "_", str(c).strip().lower())

def _find_col(df: pd.DataFrame, candidates):
    norm_map = {_normalize_colname(c): c for c in df.columns}
    for cand in candidates:
        key = _normalize_colname(cand)
        if key in norm_map:
            return norm_map[key]
    return None

def _to_bool01(series: pd.Series) -> pd.Series:
    if series is None:
        return None
    s = series.copy()
    if pd.api.types.is_bool_dtype(s):
        return s.astype(int)
    if pd.api.types.is_numeric_dtype(s):
        return s.where(s.isna(), np.where(s.astype(float) > 0, 1, 0)).astype("float")
    s2 = s.astype(str).str.strip().str.lower()
    yes = {"si", "sí", "yes", "true", "1"}
    no = {"no", "false", "0"}
    out = pd.Series(np.nan, index=s.index, dtype="float")
    out[s2.isin(yes)] = 1
    out[s2.isin(no)] = 0
    return out

def _wrap_label(label: str, width: int = 26, max_lines: int = 3) -> str:
    wrapped = textwrap.wrap(str(label), width=width)
    return "\n".join(wrapped[:max_lines])

# =========================
# Sidebar – carga de datos
# =========================
st.sidebar.markdown("### Datos (P9)")
uploaded = st.sidebar.file_uploader("Cargar P9 (CSV o XLSX)", type=["csv", "xlsx", "xls"])

if uploaded:
    df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
else:
    if os.path.exists(DEFAULT_DATASET_CSV):
        df = pd.read_csv(DEFAULT_DATASET_CSV)
    elif os.path.exists(DEFAULT_DATASET_XLSX):
        df = pd.read_excel(DEFAULT_DATASET_XLSX)
    else:
        st.error("No se encontró dataset P9.")
        st.stop()

# =========================
# Header
# =========================
st.markdown(f"<div class='main-title'>{TITLE}</div>", unsafe_allow_html=True)
st.markdown(f"<div class='main-subtitle'>{SUBTITLE}</div>", unsafe_allow_html=True)
st.markdown("<div class='hr-soft'></div>", unsafe_allow_html=True)

# =========================
# Ranking (extracto visual)
# =========================
if ALTAIR_OK and "Score_total_adj" in df.columns and "mic_name_official" in df.columns:
    top = (
        df.dropna(subset=["Score_total_adj"])
        .sort_values("Score_total_adj", ascending=False)
        .head(15)
        .copy()
    )

    top["label"] = top["mic_name_official"].apply(lambda x: _wrap_label(x, 30, 3))

    chart = (
        alt.Chart(top)
        .mark_bar(color=COLOR_GREEN)
        .encode(
            y=alt.Y(
                "label:N",
                sort="-x",
                axis=alt.Axis(labelFontSize=9, title=None),
            ),
            x=alt.X(
                "Score_total_adj:Q",
                title="Score MCA",
                axis=alt.Axis(labelColor=COLOR_TEXT, titleColor=COLOR_TEXT),
            ),
            tooltip=[
                alt.Tooltip("mic_name_official:N", title="MIC"),
                alt.Tooltip("Score_total_adj:Q", title="Score", format=".2f"),
            ],
        )
        .properties(height=450)
        .configure_view(strokeOpacity=0)
        .configure_axis(grid=False)
        .configure(background=COLOR_BG)
    )

    st.altair_chart(chart, use_container_width=True)

# =========================
# Membrete al final
# =========================
st.write("")
left, _, _ = st.columns([1.2, 0.8, 2])
with left:
    if os.path.exists(MEMBRETE_FILENAME):
        st.markdown("<div class='membrete-wrap'>", unsafe_allow_html=True)
        st.image(MEMBRETE_FILENAME, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# =========================
# Footer técnico
# =========================
st.sidebar.markdown("---")
st.sidebar.caption(f"Última carga: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
