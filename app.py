# app.py
# Streamlit Visualizador P9 – Dataset Trazable MIC (MMA-ready)
# - Ranking MCA (P8) + Dataset trazable (P9)
# - Filtros por país, IPC, tipología, institucionalidad, enforcement, trazabilidad
# - Gráficos y tabla exportable
#
# Ejecutar:
#   streamlit run app.py

from __future__ import annotations

import io
import textwrap
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# -----------------------------
# Config general
# -----------------------------
st.set_page_config(
    page_title="MIC | Visualizador P9 (Dataset Trazable)",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

ACCENT = "#00BFA6"  # menta/cian (evitar naranja en textos, según preferencia)
BG = "#F6F8FB"
CARD = "#FFFFFF"
TEXT = "#0F172A"
MUTED = "#475569"
BORDER = "rgba(15, 23, 42, 0.08)"

st.markdown(
    f"""
<style>
/* --- Base --- */
html, body, [class*="css"]  {{
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
  color: {TEXT};
}}
.stApp {{
  background: radial-gradient(1200px 600px at 10% 0%, rgba(0,191,166,0.12), transparent 60%),
              radial-gradient(900px 500px at 90% 0%, rgba(0,136,207,0.10), transparent 55%),
              {BG};
}}
/* --- Sidebar --- */
section[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, rgba(255,255,255,0.92), rgba(255,255,255,0.92));
  border-right: 1px solid {BORDER};
}}
/* --- Cards --- */
.card {{
  background: {CARD};
  border: 1px solid {BORDER};
  border-radius: 18px;
  padding: 14px 16px;
  box-shadow: 0 10px 25px rgba(2, 6, 23, 0.06);
}}
.kpi {{
  display: flex; flex-direction: column; gap: 4px;
}}
.kpi .label {{
  font-size: 12px; color: {MUTED}; letter-spacing: 0.2px;
}}
.kpi .value {{
  font-size: 22px; font-weight: 700; line-height: 1.1;
}}
.badge {{
  display: inline-block;
  font-size: 12px;
  padding: 6px 10px;
  border-radius: 999px;
  border: 1px solid {BORDER};
  background: rgba(255,255,255,0.75);
}}
.badge-accent {{
  border-color: rgba(0,191,166,0.35);
  background: rgba(0,191,166,0.10);
}}
.small {{
  font-size: 12px; color: {MUTED};
}}
/* --- Streamlit widgets tweaks --- */
div[data-testid="stMetric"] {{
  background: {CARD};
  border: 1px solid {BORDER};
  border-radius: 18px;
  padding: 12px 14px;
}}
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------
# Helpers
# -----------------------------
@st.cache_data(show_spinner=False)
def load_dataset(file_bytes: Optional[bytes], filename: Optional[str]) -> pd.DataFrame:
    """
    Carga CSV o XLSX. Si no se entrega archivo, intenta paths por defecto.
    """
    if file_bytes and filename:
        if filename.lower().endswith(".csv"):
            return pd.read_csv(io.BytesIO(file_bytes), encoding="utf-8")
        if filename.lower().endswith(".xlsx"):
            return pd.read_excel(io.BytesIO(file_bytes))
        raise ValueError("Formato no soportado. Use .csv o .xlsx")

    # fallback: archivos locales esperados
    try:
        return pd.read_csv("P9_Dataset_Trazable_MIC.csv", encoding="utf-8")
    except Exception:
        return pd.read_excel("P9_Dataset_Trazable_MIC.xlsx")


def normalize_strings(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
            df.loc[df[c].isin(["nan", "NaN", "None", "ND", ""]), c] = np.nan
    return df


def safe_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def to_csv_download(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# -----------------------------
# Load
# -----------------------------
st.sidebar.markdown("### Dataset")
uploaded = st.sidebar.file_uploader(
    "Carga P9 (CSV o XLSX)",
    type=["csv", "xlsx"],
    help="Si no cargas, la app intentará leer P9_Dataset_Trazable_MIC.csv o .xlsx en la carpeta.",
)

file_bytes = uploaded.getvalue() if uploaded else None
filename = uploaded.name if uploaded else None

try:
    df = load_dataset(file_bytes, filename)
except Exception as e:
    st.error(f"No se pudo cargar el dataset. Detalle: {e}")
    st.stop()

# Normalizaciones típicas
df = normalize_strings(
    df,
    cols=[
        "country", "pais", "iso2", "iso3",
        "mic_id", "id_mic",
        "mic_name", "nombre_mic", "mic_official_name",
        "mic_type", "tipologia", "obligation_level",
        "institution_lead", "institution", "authority",
        "ipc_category", "ipc_principal",
        "source_id", "source_url", "notes_evidence",
        "enforcement_level", "transparency_level",
    ],
)

# Columnas clave (según tu P9 actual)
COL_MIC_ID = safe_col(df, ["mic_id", "id_mic"])
COL_NAME = safe_col(df, ["mic_name", "nombre_mic", "mic_official_name"])
COL_COUNTRY = safe_col(df, ["country", "pais"])
COL_ISO2 = safe_col(df, ["iso2"])
COL_IPC = safe_col(df, ["ipc_category", "ipc_principal"])
COL_TYPE = safe_col(df, ["mic_type", "tipologia"])
COL_OBL = safe_col(df, ["obligation_level"])
COL_ENF = safe_col(df, ["enforcement_level"])
COL_REG = safe_col(df, ["public_registry"])
COL_TRANS = safe_col(df, ["transparency_level"])
COL_DIM_ENV = safe_col(df, ["dim_env", "dim_ambiental"])
COL_DIM_SOC = safe_col(df, ["dim_soc", "dim_social"])
COL_DIM_ECO = safe_col(df, ["dim_eco", "dim_economica"])
COL_IPC_COV = safe_col(df, ["ipc_coverage_count"])
COL_SCORE = safe_col(df, ["score_total_adj", "score_total", "mca_score_total"])
COL_RANK = safe_col(df, ["rank_global", "rank", "ranking"])

# Validación mínima
required_any = [COL_MIC_ID, COL_NAME, COL_COUNTRY, COL_SCORE, COL_RANK]
if any(x is None for x in required_any):
    st.warning(
        "El dataset cargado no trae todas las columnas esperadas para ranking. "
        "Se intentará visualizar lo disponible, pero faltan campos clave."
    )

# Fuerza numéricos
for c in [COL_SCORE, COL_RANK, COL_DIM_ENV, COL_DIM_SOC, COL_DIM_ECO, COL_IPC_COV]:
    if c and c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### Filtros")

def multiselect_filter(label: str, col: Optional[str]):
    if not col or col not in df.columns:
        return None
    vals = sorted([v for v in df[col].dropna().unique().tolist() if str(v).strip() != ""])
    if not vals:
        return None
    return st.sidebar.multiselect(label, vals, default=vals)

sel_countries = multiselect_filter("País", COL_COUNTRY)
sel_ipc = multiselect_filter("Categoría IPC", COL_IPC)
sel_type = multiselect_filter("Tipología MIC", COL_TYPE)
sel_obl = multiselect_filter("Obligatoriedad", COL_OBL)
sel_enf = multiselect_filter("Enforcement", COL_ENF)
sel_trans = multiselect_filter("Transparencia", COL_TRANS)

top_n = st.sidebar.slider("Top N ranking", 5, 50, 15, step=1)

score_min, score_max = None, None
if COL_SCORE and COL_SCORE in df.columns and df[COL_SCORE].notna().any():
    lo = float(np.nanmin(df[COL_SCORE]))
    hi = float(np.nanmax(df[COL_SCORE]))
    score_min, score_max = st.sidebar.slider("Rango de score (MCA)", lo, hi, (lo, hi))

# -----------------------------
# Apply filters
# -----------------------------
dff = df.copy()

def apply_filter(d: pd.DataFrame, col: Optional[str], sel):
    if col and sel is not None:
        return d[d[col].isin(sel)]
    return d

dff = apply_filter(dff, COL_COUNTRY, sel_countries)
dff = apply_filter(dff, COL_IPC, sel_ipc)
dff = apply_filter(dff, COL_TYPE, sel_type)
dff = apply_filter(dff, COL_OBL, sel_obl)
dff = apply_filter(dff, COL_ENF, sel_enf)
dff = apply_filter(dff, COL_TRANS, sel_trans)

if COL_SCORE and score_min is not None and score_max is not None:
    dff = dff[(dff[COL_SCORE].fillna(-np.inf) >= score_min) & (dff[COL_SCORE].fillna(np.inf) <= score_max)]

# -----------------------------
# Header
# -----------------------------
left, right = st.columns([0.72, 0.28], gap="large")

with left:
    st.markdown("## Visualizador P9 — Dataset Trazable MIC")
    st.markdown(
        "<span class='badge badge-accent'>P9</span> "
        "<span class='badge'>Dataset trazable</span> "
        "<span class='badge'>P8: ranking MCA</span> "
        "<span class='badge'>P7: criterios y escalas</span>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='small'>Vista analítica para auditoría, dashboard y lectura comparada por país e IPC.</div>",
        unsafe_allow_html=True,
    )

with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='kpi'><div class='label'>Registros filtrados</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='value'>{len(dff):,}</div></div>", unsafe_allow_html=True)
    st.markdown("<div class='small'>Aplicando filtros del panel izquierdo.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# KPIs row
# -----------------------------
k1, k2, k3, k4 = st.columns(4, gap="large")

def kpi_card(container, label, value, note=""):
    container.markdown("<div class='card'>", unsafe_allow_html=True)
    container.markdown(f"<div class='kpi'><div class='label'>{label}</div>", unsafe_allow_html=True)
    container.markdown(f"<div class='value'>{value}</div></div>", unsafe_allow_html=True)
    if note:
        container.markdown(f"<div class='small'>{note}</div>", unsafe_allow_html=True)
    container.markdown("</div>", unsafe_allow_html=True)

if COL_COUNTRY:
    kpi_card(k1, "Países", f"{dff[COL_COUNTRY].nunique():,}", "Cobertura del subconjunto filtrado.")
else:
    kpi_card(k1, "Países", "ND")

if COL_IPC:
    kpi_card(k2, "Categorías IPC presentes", f"{dff[COL_IPC].nunique():,}", "Canasta IPC (INE) en el subconjunto.")
else:
    kpi_card(k2, "Categorías IPC presentes", "ND")

if COL_SCORE and dff[COL_SCORE].notna().any():
    kpi_card(k3, "Score MCA (promedio)", f"{dff[COL_SCORE].mean():.2f}", "Puntaje ponderado (P8).")
else:
    kpi_card(k3, "Score MCA (promedio)", "ND")

if COL_RANK and dff[COL_RANK].notna().any():
    kpi_card(k4, "Mejor ranking (mín.)", f"{int(np.nanmin(dff[COL_RANK])):,}", "Menor = mejor posición.")
else:
    kpi_card(k4, "Mejor ranking (mín.)", "ND")

st.markdown("---")

# -----------------------------
# Charts
# -----------------------------
c1, c2 = st.columns([0.58, 0.42], gap="large")

# Ranking (Top N)
with c1:
    st.markdown("### Ranking MCA (Top N)")
    if COL_SCORE and COL_RANK and COL_NAME and COL_MIC_ID and dff[COL_SCORE].notna().any():
        top = (
            dff.dropna(subset=[COL_SCORE])
            .sort_values(by=[COL_RANK, COL_SCORE], ascending=[True, False])
            .head(top_n)
            .copy()
        )
        top["label"] = top[COL_MIC_ID].astype(str) + " — " + top[COL_NAME].astype(str)

        fig = px.bar(
            top.sort_values(COL_SCORE, ascending=True),
            x=COL_SCORE,
            y="label",
            orientation="h",
            hover_data=[COL_COUNTRY] if COL_COUNTRY else None,
            title="",
        )
        fig.update_layout(
            height=520,
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No hay columnas suficientes para graficar ranking (requiere score + rank + nombre + id).")

# Distribuciones (obligatoriedad/enforcement) + IPC coverage
with c2:
    st.markdown("### Perfil normativo y de fiscalización")
    tabs = st.tabs(["Obligatoriedad", "Enforcement", "Cobertura IPC", "Triple dimensión"])

    with tabs[0]:
        if COL_OBL and dff[COL_OBL].notna().any():
            s = dff[COL_OBL].value_counts(dropna=True).reset_index()
            s.columns = ["obligation_level", "count"]
            fig = px.pie(s, names="obligation_level", values="count", title="")
            fig.update_layout(
                height=360,
                margin=dict(l=10, r=10, t=20, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ND: no se identificó 'obligation_level' para el subconjunto filtrado.")

    with tabs[1]:
        if COL_ENF and dff[COL_ENF].notna().any():
            s = dff[COL_ENF].value_counts(dropna=True).reset_index()
            s.columns = ["enforcement_level", "count"]
            fig = px.bar(s, x="enforcement_level", y="count", title="")
            fig.update_layout(
                height=360,
                margin=dict(l=10, r=10, t=20, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                xaxis_title="",
                yaxis_title="Nº MIC",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ND: no se identificó 'enforcement_level' para el subconjunto filtrado.")

    with tabs[2]:
        if COL_IPC_COV and dff[COL_IPC_COV].notna().any():
            fig = px.histogram(dff, x=COL_IPC_COV, nbins=10, title="")
            fig.update_layout(
                height=360,
                margin=dict(l=10, r=10, t=20, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Nº de categorías IPC cubiertas (count)",
                yaxis_title="Nº MIC",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ND: no se identificó 'ipc_coverage_count' para el subconjunto filtrado.")

    with tabs[3]:
        # Triple dimensión: contar cuántos MIC cubren cada dimensión (0/1)
        dims = []
        for col, label in [(COL_DIM_ENV, "Ambiental"), (COL_DIM_SOC, "Social"), (COL_DIM_ECO, "Económica")]:
            if col and col in dff.columns and dff[col].notna().any():
                dims.append((label, int((dff[col] >= 1).sum())))
        if dims:
            dd = pd.DataFrame(dims, columns=["dimensión", "MIC_con_cobertura"])
            fig = px.bar(dd, x="dimensión", y="MIC_con_cobertura", title="")
            fig.update_layout(
                height=360,
                margin=dict(l=10, r=10, t=20, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                xaxis_title="",
                yaxis_title="Nº MIC",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ND: no se identificaron columnas dim_env/dim_soc/dim_eco en el subconjunto filtrado.")

st.markdown("---")

# -----------------------------
# Explorer + MIC profile
# -----------------------------
st.markdown("### Explorador de MIC (tabla + ficha)")

left, right = st.columns([0.62, 0.38], gap="large")

with left:
    # Columnas a mostrar (mínimo MMA-ready)
    show_cols = [c for c in [
        COL_RANK, COL_SCORE,
        COL_MIC_ID, COL_NAME,
        COL_COUNTRY, COL_ISO2,
        COL_IPC, COL_TYPE, COL_OBL,
        COL_ENF, COL_REG, COL_TRANS,
        COL_DIM_ENV, COL_DIM_SOC, COL_DIM_ECO,
        "source_id", "source_url"
    ] if c and c in dff.columns]

    # Si hay columnas alternativas de fuentes
    if "source_id" not in show_cols:
        alt = safe_col(dff, ["source_id", "SRC_ID", "Source_ID"])
        if alt and alt in dff.columns:
            show_cols.append(alt)

    st.dataframe(
        dff[show_cols].sort_values(by=[COL_RANK] if COL_RANK else None, ascending=True),
        use_container_width=True,
        height=420,
    )

    # Descarga subconjunto
    st.download_button(
        "Descargar subconjunto filtrado (CSV)",
        data=to_csv_download(dff),
        file_name="P9_filtrado.csv",
        mime="text/csv",
        use_container_width=True,
    )

with right:
    # Selector de MIC para ficha
    if COL_MIC_ID and COL_NAME and len(dff) > 0:
        dff_sel = dff.copy()
        dff_sel["__label__"] = dff_sel[COL_MIC_ID].astype(str) + " — " + dff_sel[COL_NAME].astype(str)
        pick = st.selectbox("Seleccionar MIC", dff_sel["__label__"].tolist())
        row = dff_sel[dff_sel["__label__"] == pick].iloc[0]

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"**{row[COL_NAME]}**")
        st.markdown(f"<span class='badge badge-accent'>{row[COL_MIC_ID]}</span>", unsafe_allow_html=True)

        def line(k: str, v):
            if v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() == "":
                vv = "ND"
            else:
                vv = str(v)
            st.markdown(f"**{k}:** {vv}")

        if COL_COUNTRY: line("País", row[COL_COUNTRY])
        if COL_IPC: line("Categoría IPC", row[COL_IPC])
        if COL_TYPE: line("Tipología", row[COL_TYPE])
        if COL_OBL: line("Obligatoriedad", row[COL_OBL])
        if COL_ENF: line("Enforcement", row[COL_ENF])
        if COL_REG: line("Registro público", row[COL_REG])
        if COL_TRANS: line("Transparencia", row[COL_TRANS])
        if COL_SCORE: line("Score MCA", row[COL_SCORE])
        if COL_RANK: line("Ranking", row[COL_RANK])

        # Triple dimensión
        dims_txt = []
        if COL_DIM_ENV: dims_txt.append(f"Ambiental={int(row[COL_DIM_ENV]) if pd.notna(row[COL_DIM_ENV]) else 'ND'}")
        if COL_DIM_SOC: dims_txt.append(f"Social={int(row[COL_DIM_SOC]) if pd.notna(row[COL_DIM_SOC]) else 'ND'}")
        if COL_DIM_ECO: dims_txt.append(f"Económica={int(row[COL_DIM_ECO]) if pd.notna(row[COL_DIM_ECO]) else 'ND'}")
        if dims_txt:
            st.markdown("**Triple dimensión:** " + " | ".join(dims_txt))

        # Fuentes
        src_id = row.get("source_id", None)
        src_url = row.get("source_url", None)
        if src_id is not None:
            line("Source ID", src_id)
        if src_url is not None and str(src_url).strip() not in ["", "ND", "nan", "None"]:
            st.markdown(f"**Fuente (URL):** `{src_url}`")

        # Notas (si existen)
        for cand in ["notes_evidence", "evidence_notes", "notas"]:
            if cand in dff.columns:
                val = row.get(cand, None)
                if val is not None and str(val).strip() not in ["", "ND", "nan", "None"]:
                    st.markdown("**Notas de evidencia:**")
                    st.write(textwrap.fill(str(val), width=85))
                    break

        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("ND: no se puede construir ficha sin columnas MIC_ID + MIC_NAME.")


# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown(
    "<div class='small'>P9 consolida variables base (P2/P5), resultados MCA (P8) y metadatos de trazabilidad para auditoría y dashboard.</div>",
    unsafe_allow_html=True,
)
