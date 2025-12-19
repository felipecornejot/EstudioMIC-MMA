# app.py
# Streamlit Visualizador P9 – Dataset Trazable MIC (MMA-ready)
# Fixes:
# - Header (título/subtítulo) estilo MMA
# - Membrete fijo esquina inferior izquierda (membrete (1).png)
# - Ranking Top N robusto (fallback score/rank si faltan columnas)
# - Explorador + ficha robusta (fallback MIC_ID/MIC_NAME si faltan)
#
# Ejecutar:
#   streamlit run app.py

from __future__ import annotations

import base64
import io
import os
import textwrap
from typing import Optional, List

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# -----------------------------
# Config general
# -----------------------------
st.set_page_config(
    page_title="MIC | Ranking y métricas (P9)",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

ACCENT = "#00BFA6"  # menta/cian (evitar naranja en textos)
BG = "#F6F8FB"
CARD = "#FFFFFF"
TEXT = "#0F172A"
MUTED = "#475569"
BORDER = "rgba(15, 23, 42, 0.08)"

# -----------------------------
# CSS (estilo similar referencia)
# -----------------------------
st.markdown(
    f"""
<style>
/* --- Base --- */
html, body, [class*="css"] {{
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

/* --- Header (título/subtítulo) --- */
.hdr-title {{
  font-size: 56px;
  font-weight: 800;
  line-height: 1.02;
  margin: 6px 0 6px 0;
  color: #0B4FA2; /* azul sobrio (no naranja) */
}}
.hdr-sub {{
  font-size: 16px;
  margin: 0 0 14px 0;
  color: #6B8FBF;
}}
.hdr-rule {{
  height: 2px;
  width: 100%;
  background: rgba(11,79,162,0.25);
  margin: 6px 0 16px 0;
  border-radius: 999px;
}}

/* --- Top gov card --- */
.gov-card {{
  background: {CARD};
  border: 1px solid {BORDER};
  border-radius: 18px;
  padding: 16px 18px;
  box-shadow: 0 10px 25px rgba(2, 6, 23, 0.06);
}}
.gov-title {{
  font-weight: 700;
  font-size: 20px;
  margin: 0;
}}
.gov-sub {{
  font-weight: 500;
  font-size: 16px;
  color: {MUTED};
  margin: 4px 0 0 0;
}}

/* --- Footer membrete fixed --- */
.membrete-fixed {{
  position: fixed;
  left: 14px;
  bottom: 14px;
  z-index: 9999;
  opacity: 0.95;
}}
.membrete-fixed img {{
  height: 42px;
  border-radius: 8px;
  box-shadow: 0 10px 25px rgba(2, 6, 23, 0.18);
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
    """Carga CSV o XLSX. Si no se entrega archivo, intenta paths por defecto."""
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


def normalize_strings(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
            df.loc[df[c].isin(["nan", "NaN", "None", "ND", ""]), c] = np.nan
    return df


def safe_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def to_csv_download(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def b64_image(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea/renombra columnas canónicas para evitar errores:
    MIC_ID, MIC_NAME, COUNTRY, ISO2, IPC_CATEGORY, MIC_TYPE, OBLIGATION_LEVEL,
    ENFORCEMENT_LEVEL, PUBLIC_REGISTRY, TRANSPARENCY_LEVEL,
    DIM_ENV, DIM_SOC, DIM_ECO, IPC_COVERAGE_COUNT,
    SOURCE_ID, SOURCE_URL, NOTES_EVIDENCE,
    MCA_SCORE, MCA_RANK
    """
    df = df.copy()

    # 1) Normaliza nombres a lower para matching, pero preserva columnas existentes
    lower_map = {c: c.strip() for c in df.columns}
    df.rename(columns=lower_map, inplace=True)

    # 2) Encuentra equivalentes
    col_mic_id = safe_col(df, ["mic_id", "id_mic", "MIC_ID", "Mic_ID", "micid"])
    col_name = safe_col(df, ["mic_name", "nombre_mic", "mic_official_name", "MIC_NAME", "Mic_Name", "name"])
    col_country = safe_col(df, ["country", "pais", "COUNTRY", "País"])
    col_iso2 = safe_col(df, ["iso2", "ISO2"])
    col_ipc = safe_col(df, ["ipc_category", "ipc_principal", "IPC", "ipc"])
    col_type = safe_col(df, ["mic_type", "tipologia", "tipología", "type"])
    col_obl = safe_col(df, ["obligation_level", "obligatoriedad", "obligation"])
    col_enf = safe_col(df, ["enforcement_level", "fiscalizacion", "enforcement"])
    col_reg = safe_col(df, ["public_registry", "registro_publico", "registro", "registry"])
    col_trans = safe_col(df, ["transparency_level", "transparencia", "transparency"])
    col_dim_env = safe_col(df, ["dim_env", "dim_ambiental", "ambiental"])
    col_dim_soc = safe_col(df, ["dim_soc", "dim_social", "social"])
    col_dim_eco = safe_col(df, ["dim_eco", "dim_economica", "economica", "económica"])
    col_ipc_cov = safe_col(df, ["ipc_coverage_count", "ipc_cov", "ipc_count", "ipc_cobertura"])
    col_src_id = safe_col(df, ["source_id", "SRC_ID", "Source_ID"])
    col_src_url = safe_col(df, ["source_url", "SRC_URL", "Source_URL", "url"])
    col_notes = safe_col(df, ["notes_evidence", "evidence_notes", "notas", "notes"])

    # 3) Crea columnas canónicas (sin borrar originales)
    if col_mic_id and "MIC_ID" not in df.columns:
        df["MIC_ID"] = df[col_mic_id]
    elif "MIC_ID" not in df.columns:
        df["MIC_ID"] = np.nan

    if col_name and "MIC_NAME" not in df.columns:
        df["MIC_NAME"] = df[col_name]
    elif "MIC_NAME" not in df.columns:
        df["MIC_NAME"] = np.nan

    if col_country and "COUNTRY" not in df.columns:
        df["COUNTRY"] = df[col_country]
    elif "COUNTRY" not in df.columns:
        df["COUNTRY"] = np.nan

    if col_iso2 and "ISO2" not in df.columns:
        df["ISO2"] = df[col_iso2]
    elif "ISO2" not in df.columns:
        df["ISO2"] = np.nan

    if col_ipc and "IPC_CATEGORY" not in df.columns:
        df["IPC_CATEGORY"] = df[col_ipc]
    elif "IPC_CATEGORY" not in df.columns:
        df["IPC_CATEGORY"] = np.nan

    if col_type and "MIC_TYPE" not in df.columns:
        df["MIC_TYPE"] = df[col_type]
    elif "MIC_TYPE" not in df.columns:
        df["MIC_TYPE"] = np.nan

    if col_obl and "OBLIGATION_LEVEL" not in df.columns:
        df["OBLIGATION_LEVEL"] = df[col_obl]
    elif "OBLIGATION_LEVEL" not in df.columns:
        df["OBLIGATION_LEVEL"] = np.nan

    if col_enf and "ENFORCEMENT_LEVEL" not in df.columns:
        df["ENFORCEMENT_LEVEL"] = df[col_enf]
    elif "ENFORCEMENT_LEVEL" not in df.columns:
        df["ENFORCEMENT_LEVEL"] = np.nan

    if col_reg and "PUBLIC_REGISTRY" not in df.columns:
        df["PUBLIC_REGISTRY"] = df[col_reg]
    elif "PUBLIC_REGISTRY" not in df.columns:
        df["PUBLIC_REGISTRY"] = np.nan

    if col_trans and "TRANSPARENCY_LEVEL" not in df.columns:
        df["TRANSPARENCY_LEVEL"] = df[col_trans]
    elif "TRANSPARENCY_LEVEL" not in df.columns:
        df["TRANSPARENCY_LEVEL"] = np.nan

    if col_dim_env and "DIM_ENV" not in df.columns:
        df["DIM_ENV"] = df[col_dim_env]
    elif "DIM_ENV" not in df.columns:
        df["DIM_ENV"] = np.nan

    if col_dim_soc and "DIM_SOC" not in df.columns:
        df["DIM_SOC"] = df[col_dim_soc]
    elif "DIM_SOC" not in df.columns:
        df["DIM_SOC"] = np.nan

    if col_dim_eco and "DIM_ECO" not in df.columns:
        df["DIM_ECO"] = df[col_dim_eco]
    elif "DIM_ECO" not in df.columns:
        df["DIM_ECO"] = np.nan

    if col_ipc_cov and "IPC_COVERAGE_COUNT" not in df.columns:
        df["IPC_COVERAGE_COUNT"] = df[col_ipc_cov]
    elif "IPC_COVERAGE_COUNT" not in df.columns:
        df["IPC_COVERAGE_COUNT"] = np.nan

    if col_src_id and "SOURCE_ID" not in df.columns:
        df["SOURCE_ID"] = df[col_src_id]
    elif "SOURCE_ID" not in df.columns:
        df["SOURCE_ID"] = np.nan

    if col_src_url and "SOURCE_URL" not in df.columns:
        df["SOURCE_URL"] = df[col_src_url]
    elif "SOURCE_URL" not in df.columns:
        df["SOURCE_URL"] = np.nan

    if col_notes and "NOTES_EVIDENCE" not in df.columns:
        df["NOTES_EVIDENCE"] = df[col_notes]
    elif "NOTES_EVIDENCE" not in df.columns:
        df["NOTES_EVIDENCE"] = np.nan

    # Limpieza strings típicos
    df = normalize_strings(df, ["MIC_ID", "MIC_NAME", "COUNTRY", "ISO2", "IPC_CATEGORY", "MIC_TYPE",
                                "OBLIGATION_LEVEL", "ENFORCEMENT_LEVEL", "PUBLIC_REGISTRY",
                                "TRANSPARENCY_LEVEL", "SOURCE_ID", "SOURCE_URL"])

    return df


def ensure_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """
    Garantiza columnas MCA_SCORE (numérica) y MCA_RANK (numérica).
    Si faltan, construye fallback usando información disponible.
    """
    df = df.copy()

    # 1) score candidates
    score_col = safe_col(df, ["score_total_adj", "score_total", "mca_score_total", "MCA_SCORE", "score", "puntaje"])
    rank_col = safe_col(df, ["rank_global", "rank", "ranking", "MCA_RANK"])

    # 2) crea MCA_SCORE
    if "MCA_SCORE" not in df.columns:
        if score_col and score_col in df.columns:
            df["MCA_SCORE"] = pd.to_numeric(df[score_col], errors="coerce")
        else:
            # Fallback: suma/mean de columnas numéricas con patrones razonables
            numeric_cols = []
            for c in df.columns:
                cl = c.lower()
                if cl in ["mca_rank", "rank", "ranking", "rank_global"]:
                    continue
                if ("_score" in cl) or cl.startswith("crit_") or cl.startswith("criterion_") or cl.startswith("c_"):
                    if pd.api.types.is_numeric_dtype(df[c]) or df[c].dtype == object:
                        numeric_cols.append(c)

            # Si no hay criterios, fallback mínimo con triple dimensión + cobertura/enforcement/transparencia si son numéricos
            if not numeric_cols:
                fallback = []
                for c in ["DIM_ENV", "DIM_SOC", "DIM_ECO", "IPC_COVERAGE_COUNT"]:
                    if c in df.columns:
                        fallback.append(c)
                numeric_cols = fallback

            # Convertir candidatos a numérico y promediar (robusto)
            temp = df[numeric_cols].apply(pd.to_numeric, errors="coerce") if numeric_cols else pd.DataFrame(index=df.index)
            if temp.shape[1] >= 1:
                df["MCA_SCORE"] = temp.mean(axis=1)
            else:
                df["MCA_SCORE"] = np.nan

    # 3) crea MCA_RANK
    if "MCA_RANK" not in df.columns:
        if rank_col and rank_col in df.columns:
            df["MCA_RANK"] = pd.to_numeric(df[rank_col], errors="coerce")
        else:
            # Rank derivado del score (1 = mejor)
            s = pd.to_numeric(df["MCA_SCORE"], errors="coerce")
            df["MCA_RANK"] = s.rank(method="dense", ascending=False)

    # 4) asegurar tipos
    df["MCA_SCORE"] = pd.to_numeric(df["MCA_SCORE"], errors="coerce")
    df["MCA_RANK"] = pd.to_numeric(df["MCA_RANK"], errors="coerce")

    return df


# -----------------------------
# Sidebar: carga dataset
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
    df_raw = load_dataset(file_bytes, filename)
except Exception as e:
    st.error(f"No se pudo cargar el dataset. Detalle: {e}")
    st.stop()

# Standardiza + ranking robusto
df = standardize_columns(df_raw)
df = ensure_ranking(df)

# Fuerza numéricos para dimensiones típicas
for c in ["DIM_ENV", "DIM_SOC", "DIM_ECO", "IPC_COVERAGE_COUNT", "MCA_SCORE", "MCA_RANK"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")


# -----------------------------
# Inserta membrete fijo (bottom-left)
# -----------------------------
membrete_path = "membrete (1).png"
membrete_b64 = b64_image(membrete_path)
if membrete_b64:
    st.markdown(
        f"""
<div class="membrete-fixed">
  <img src="data:image/png;base64,{membrete_b64}" alt="Membrete Gobierno de Chile" />
</div>
""",
        unsafe_allow_html=True,
    )


# -----------------------------
# Header superior (gov card + título/subtítulo)
# -----------------------------
st.markdown(
    """
<div class="gov-card">
  <p class="gov-title">Gobierno de Chile</p>
  <p class="gov-sub">Ministerio del Medio Ambiente</p>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div style="padding-top: 10px;"></div>
<div class="hdr-title">Ranking y métricas asociadas a mecanismos de información al consumidor.</div>
<div class="hdr-rule"></div>
<div class="hdr-sub">Consultoría Sustrend para la Subsecretaría del Medio Ambiente | ID: 608897-205-COT25</div>
""",
    unsafe_allow_html=True,
)


# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### Filtros")

def multiselect_filter(label: str, col: str):
    vals = sorted([v for v in df[col].dropna().unique().tolist() if str(v).strip() != ""])
    if not vals:
        return None
    return st.sidebar.multiselect(label, vals, default=vals)

sel_countries = multiselect_filter("País", "COUNTRY") if "COUNTRY" in df.columns else None
sel_ipc = multiselect_filter("Categoría IPC", "IPC_CATEGORY") if "IPC_CATEGORY" in df.columns else None
sel_type = multiselect_filter("Tipología MIC", "MIC_TYPE") if "MIC_TYPE" in df.columns else None
sel_obl = multiselect_filter("Obligatoriedad", "OBLIGATION_LEVEL") if "OBLIGATION_LEVEL" in df.columns else None
sel_enf = multiselect_filter("Enforcement", "ENFORCEMENT_LEVEL") if "ENFORCEMENT_LEVEL" in df.columns else None
sel_trans = multiselect_filter("Transparencia", "TRANSPARENCY_LEVEL") if "TRANSPARENCY_LEVEL" in df.columns else None

top_n = st.sidebar.slider("Top N ranking", 5, 50, 15, step=1)

# Score slider (si existe)
score_min, score_max = None, None
if "MCA_SCORE" in df.columns and df["MCA_SCORE"].notna().any():
    lo = float(np.nanmin(df["MCA_SCORE"]))
    hi = float(np.nanmax(df["MCA_SCORE"]))
    score_min, score_max = st.sidebar.slider("Rango de score (MCA)", lo, hi, (lo, hi))


# -----------------------------
# Apply filters
# -----------------------------
dff = df.copy()

def apply_filter(d: pd.DataFrame, col: str, sel):
    if sel is None:
        return d
    return d[d[col].isin(sel)]

if sel_countries is not None:
    dff = apply_filter(dff, "COUNTRY", sel_countries)
if sel_ipc is not None:
    dff = apply_filter(dff, "IPC_CATEGORY", sel_ipc)
if sel_type is not None:
    dff = apply_filter(dff, "MIC_TYPE", sel_type)
if sel_obl is not None:
    dff = apply_filter(dff, "OBLIGATION_LEVEL", sel_obl)
if sel_enf is not None:
    dff = apply_filter(dff, "ENFORCEMENT_LEVEL", sel_enf)
if sel_trans is not None:
    dff = apply_filter(dff, "TRANSPARENCY_LEVEL", sel_trans)

if score_min is not None and score_max is not None:
    dff = dff[(dff["MCA_SCORE"].fillna(-np.inf) >= score_min) & (dff["MCA_SCORE"].fillna(np.inf) <= score_max)]


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

kpi_card(k1, "Registros filtrados", f"{len(dff):,}", "Aplicando filtros del panel izquierdo.")
kpi_card(k2, "Países", f"{dff['COUNTRY'].nunique():,}" if "COUNTRY" in dff.columns else "ND")
kpi_card(k3, "Categorías IPC presentes", f"{dff['IPC_CATEGORY'].nunique():,}" if "IPC_CATEGORY" in dff.columns else "ND")
kpi_card(k4, "Score MCA (promedio)", f"{dff['MCA_SCORE'].mean():.2f}" if dff["MCA_SCORE"].notna().any() else "ND")

st.markdown("---")


# -----------------------------
# Charts
# -----------------------------
c1, c2 = st.columns([0.58, 0.42], gap="large")

# Ranking (Top N) - SIEMPRE intenta con MCA_SCORE/MCA_RANK ya estandarizados
with c1:
    st.markdown("### Ranking MCA (Top N)")
    has_min_cols = (
        ("MIC_ID" in dff.columns) and
        ("MIC_NAME" in dff.columns) and
        dff["MIC_ID"].notna().any() and
        dff["MIC_NAME"].notna().any() and
        dff["MCA_SCORE"].notna().any()
    )

    if has_min_cols:
        top = (
            dff.dropna(subset=["MCA_SCORE"])
            .sort_values(by=["MCA_RANK", "MCA_SCORE"], ascending=[True, False])
            .head(top_n)
            .copy()
        )
        top["label"] = top["MIC_ID"].astype(str) + " — " + top["MIC_NAME"].astype(str)

        fig = px.bar(
            top.sort_values("MCA_SCORE", ascending=True),
            x="MCA_SCORE",
            y="label",
            orientation="h",
            hover_data=["COUNTRY"] if "COUNTRY" in top.columns else None,
            title="",
        )
        fig.update_layout(
            height=520,
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="Score MCA (ponderado)",
            yaxis_title="",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Mensaje útil y accionable, pero sin “ND” genérico
        st.info(
            "No se pudo construir el ranking con el subconjunto filtrado. "
            "Se requiere identificar MIC_ID, MIC_NAME y un score numérico (MCA_SCORE). "
            "El script intenta crear MCA_SCORE/MCA_RANK automáticamente; si aún falla, revisa nombres de columnas en P9."
        )

# Perfil normativo y de fiscalización
with c2:
    st.markdown("### Perfil normativo y de fiscalización")
    tabs = st.tabs(["Obligatoriedad", "Enforcement", "Cobertura IPC", "Triple dimensión"])

    with tabs[0]:
        if "OBLIGATION_LEVEL" in dff.columns and dff["OBLIGATION_LEVEL"].notna().any():
            s = dff["OBLIGATION_LEVEL"].value_counts(dropna=True).reset_index()
            s.columns = ["obligation_level", "count"]
            fig = px.pie(s, names="obligation_level", values="count", title="")
            fig.update_layout(
                height=340,
                margin=dict(l=10, r=10, t=20, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ND: no se identificó 'OBLIGATION_LEVEL' para el subconjunto filtrado.")

    with tabs[1]:
        if "ENFORCEMENT_LEVEL" in dff.columns and dff["ENFORCEMENT_LEVEL"].notna().any():
            s = dff["ENFORCEMENT_LEVEL"].value_counts(dropna=True).reset_index()
            s.columns = ["enforcement_level", "count"]
            fig = px.bar(s, x="enforcement_level", y="count", title="")
            fig.update_layout(
                height=340,
                margin=dict(l=10, r=10, t=20, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                xaxis_title="",
                yaxis_title="Nº MIC",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ND: no se identificó 'ENFORCEMENT_LEVEL' para el subconjunto filtrado.")

    with tabs[2]:
        if "IPC_COVERAGE_COUNT" in dff.columns and dff["IPC_COVERAGE_COUNT"].notna().any():
            fig = px.histogram(dff, x="IPC_COVERAGE_COUNT", nbins=10, title="")
            fig.update_layout(
                height=340,
                margin=dict(l=10, r=10, t=20, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Nº de categorías IPC cubiertas",
                yaxis_title="Nº MIC",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ND: no se identificó 'IPC_COVERAGE_COUNT' para el subconjunto filtrado.")

    with tabs[3]:
        dims = []
        for col, label in [("DIM_ENV", "Ambiental"), ("DIM_SOC", "Social"), ("DIM_ECO", "Económica")]:
            if col in dff.columns and dff[col].notna().any():
                dims.append((label, int((pd.to_numeric(dff[col], errors="coerce") >= 1).sum())))
        if dims:
            dd = pd.DataFrame(dims, columns=["dimensión", "MIC_con_cobertura"])
            fig = px.bar(dd, x="dimensión", y="MIC_con_cobertura", title="")
            fig.update_layout(
                height=340,
                margin=dict(l=10, r=10, t=20, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                xaxis_title="",
                yaxis_title="Nº MIC",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ND: no se identificaron columnas DIM_ENV/DIM_SOC/DIM_ECO en el subconjunto filtrado.")

st.markdown("---")


# -----------------------------
# Explorer + MIC profile
# -----------------------------
st.markdown("### Explorador de MIC (tabla + ficha)")

left, right = st.columns([0.62, 0.38], gap="large")

with left:
    # Columnas a mostrar (mínimo MMA-ready + trazabilidad)
    show_cols = [
        "MCA_RANK", "MCA_SCORE",
        "MIC_ID", "MIC_NAME",
        "COUNTRY", "ISO2",
        "IPC_CATEGORY", "MIC_TYPE",
        "OBLIGATION_LEVEL", "ENFORCEMENT_LEVEL",
        "PUBLIC_REGISTRY", "TRANSPARENCY_LEVEL",
        "DIM_ENV", "DIM_SOC", "DIM_ECO",
        "SOURCE_ID", "SOURCE_URL"
    ]
    show_cols = [c for c in show_cols if c in dff.columns]

    sort_cols = ["MCA_RANK"] if "MCA_RANK" in dff.columns else None
    dff_table = dff[show_cols].copy()
    if sort_cols:
        dff_table = dff_table.sort_values(by=sort_cols, ascending=True)

    st.dataframe(dff_table, use_container_width=True, height=420)

    st.download_button(
        "Descargar subconjunto filtrado (CSV)",
        data=to_csv_download(dff),
        file_name="P9_filtrado.csv",
        mime="text/csv",
        use_container_width=True,
    )

with right:
    # Ficha: asegurar MIC_ID + MIC_NAME
    can_profile = (
        ("MIC_ID" in dff.columns) and ("MIC_NAME" in dff.columns) and
        dff["MIC_ID"].notna().any() and dff["MIC_NAME"].notna().any()
    )

    if can_profile and len(dff) > 0:
        dff_sel = dff.copy()
        dff_sel["__label__"] = dff_sel["MIC_ID"].astype(str) + " — " + dff_sel["MIC_NAME"].astype(str)
        pick = st.selectbox("Seleccionar MIC", dff_sel["__label__"].tolist())
        row = dff_sel[dff_sel["__label__"] == pick].iloc[0]

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"**{row['MIC_NAME']}**")
        st.markdown(f"<span class='badge badge-accent'>{row['MIC_ID']}</span>", unsafe_allow_html=True)

        def line(k: str, v):
            if v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() == "":
                vv = "ND"
            else:
                vv = str(v)
            st.markdown(f"**{k}:** {vv}")

        if "COUNTRY" in dff.columns: line("País", row.get("COUNTRY", None))
        if "IPC_CATEGORY" in dff.columns: line("Categoría IPC", row.get("IPC_CATEGORY", None))
        if "MIC_TYPE" in dff.columns: line("Tipología", row.get("MIC_TYPE", None))
        if "OBLIGATION_LEVEL" in dff.columns: line("Obligatoriedad", row.get("OBLIGATION_LEVEL", None))
        if "ENFORCEMENT_LEVEL" in dff.columns: line("Enforcement", row.get("ENFORCEMENT_LEVEL", None))
        if "PUBLIC_REGISTRY" in dff.columns: line("Registro público", row.get("PUBLIC_REGISTRY", None))
        if "TRANSPARENCY_LEVEL" in dff.columns: line("Transparencia", row.get("TRANSPARENCY_LEVEL", None))

        line("Score MCA", row.get("MCA_SCORE", None))
        line("Ranking", row.get("MCA_RANK", None))

        dims_txt = []
        for col, label in [("DIM_ENV", "Ambiental"), ("DIM_SOC", "Social"), ("DIM_ECO", "Económica")]:
            if col in dff.columns:
                val = row.get(col, None)
                dims_txt.append(f"{label}={int(val) if pd.notna(val) else 'ND'}")
        if dims_txt:
            st.markdown("**Triple dimensión:** " + " | ".join(dims_txt))

        # Fuentes
        line("Source ID", row.get("SOURCE_ID", None))
        src_url = row.get("SOURCE_URL", None)
        if src_url is not None and str(src_url).strip() not in ["", "ND", "nan", "None"]:
            st.markdown(f"**Fuente (URL):** `{src_url}`")

        # Notas evidencia
        notes = row.get("NOTES_EVIDENCE", None)
        if notes is not None and str(notes).strip() not in ["", "ND", "nan", "None"]:
            st.markdown("**Notas de evidencia:**")
            st.write(textwrap.fill(str(notes), width=85))

        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info(
            "No se puede construir la ficha porque el dataset (o el subconjunto filtrado) no permite derivar MIC_ID y MIC_NAME. "
            "Revisa si existen columnas equivalentes (mic_id / mic_name / nombre_mic / mic_official_name)."
        )


# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown(
    "<div class='small'>P9 consolida variables base (P2/P5), resultados MCA (P8) y metadatos de trazabilidad para auditoría y dashboard.</div>",
    unsafe_allow_html=True,
)
