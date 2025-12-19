# app.py — Visualizador P9 (MMA-ready) | FIXED v3
# - Fondo blanco real
# - Membrete en cuerpo (abajo del texto P9...)
# - Ranking robusto + orden asc/desc
# - Labels envueltas (2–3 líneas) y únicas (evita error Categorical)
# - Triple dimensión SI/NO robusta
from __future__ import annotations

import base64
import io
import os
import textwrap
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# -----------------------------
# Config
# -----------------------------
st.set_page_config(
    page_title="MIC | Ranking y métricas (P9)",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Colores (acento menta/cian; NO usar naranja en textos)
ACCENT = "#00BFA6"
TEXT = "#0F172A"
MUTED = "#475569"
BORDER = "rgba(15, 23, 42, 0.10)"
BLUE_TITLE = "#0B4FA2"
BLUE_SUB = "#6B8FBF"


# -----------------------------
# CSS — Fondo BLANCO + cards
# -----------------------------
st.markdown(
    f"""
<style>
html, body, [class*="css"] {{
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
  color: {TEXT};
}}
/* Fondo blanco real */
.stApp {{
  background: #FFFFFF !important;
}}
/* Sidebar limpio */
section[data-testid="stSidebar"] {{
  background: #FFFFFF !important;
  border-right: 1px solid {BORDER};
}}
/* Cards */
.card {{
  background: #FFFFFF;
  border: 1px solid {BORDER};
  border-radius: 18px;
  padding: 14px 16px;
  box-shadow: 0 10px 25px rgba(2, 6, 23, 0.06);
}}
.kpi {{ display: flex; flex-direction: column; gap: 4px; }}
.kpi .label {{ font-size: 12px; color: {MUTED}; letter-spacing: 0.2px; }}
.kpi .value {{ font-size: 22px; font-weight: 700; line-height: 1.1; }}

.badge {{
  display: inline-block;
  font-size: 12px;
  padding: 6px 10px;
  border-radius: 999px;
  border: 1px solid {BORDER};
  background: rgba(255,255,255,0.95);
}}
.badge-accent {{
  border-color: rgba(0,191,166,0.35);
  background: rgba(0,191,166,0.10);
}}

.small {{ font-size: 12px; color: {MUTED}; }}

.hdr-title {{
  font-size: 52px;
  font-weight: 800;
  line-height: 1.04;
  margin: 6px 0 6px 0;
  color: {BLUE_TITLE};
}}
.hdr-sub {{
  font-size: 16px;
  margin: 0 0 14px 0;
  color: {BLUE_SUB};
}}
.hdr-rule {{
  height: 2px;
  width: 100%;
  background: rgba(11,79,162,0.25);
  margin: 6px 0 16px 0;
  border-radius: 999px;
}}

.gov-card {{
  background: #FFFFFF;
  border: 1px solid {BORDER};
  border-radius: 18px;
  padding: 16px 18px;
  box-shadow: 0 10px 25px rgba(2, 6, 23, 0.06);
}}
.gov-title {{ font-weight: 700; font-size: 20px; margin: 0; }}
.gov-sub {{ font-weight: 500; font-size: 16px; color: {MUTED}; margin: 4px 0 0 0; }}

/* Membrete en cuerpo (NO fixed) */
.membrete-body {{
  margin-top: 10px;
  display: flex;
  align-items: flex-end;
  gap: 10px;
}}
.membrete-body img {{
  height: 44px;
  border-radius: 10px;
  box-shadow: 0 10px 25px rgba(2, 6, 23, 0.18);
}}
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------
# Helpers
# -----------------------------
def b64_image(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def to_csv_download(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def safe_text(v) -> str:
    if v is None:
        return "ND"
    if isinstance(v, float) and np.isnan(v):
        return "ND"
    s = str(v).strip()
    return "ND" if s.lower() in ["", "nan", "none", "null"] else s


def coerce_num(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def to_bool01(series: pd.Series) -> pd.Series:
    """
    Convierte SI/NO (y variantes) a 1/0 robusto.
    """
    if series is None:
        return pd.Series([], dtype="float")
    s = series.copy()

    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0).clip(lower=0, upper=1)

    s = s.astype(str).str.strip().str.lower()

    yes = {"si", "sí", "s", "yes", "y", "true", "1", "ok", "x"}
    no = {"no", "n", "false", "0", "na", "nd", ""}

    out = pd.Series(np.nan, index=s.index, dtype="float")
    out[s.isin(yes)] = 1.0
    out[s.isin(no)] = 0.0

    fallback = pd.to_numeric(s, errors="coerce")
    out = out.fillna(fallback)

    return out.fillna(0).clip(lower=0, upper=1)


def wrap_label(text: str, width: int = 34, max_lines: int = 3) -> str:
    """
    Envuelve texto a 2–3 líneas con <br> para Plotly.
    """
    t = " ".join(str(text).split())
    lines = textwrap.wrap(t, width=width)
    lines = lines[:max_lines]
    if len(lines) < 1:
        return t
    if len(textwrap.wrap(t, width=width)) > max_lines:
        last = lines[-1]
        if len(last) > 1:
            last = last[:-1] + "…"
        lines[-1] = last
    return "<br>".join(lines)


def make_unique_labels(labels: List[str]) -> List[str]:
    """
    Garantiza unicidad (evita error: Categorical categories must be unique).
    Si hay repetidos, agrega sufijo invisible.
    """
    counts: Dict[str, int] = {}
    out: List[str] = []
    for lab in labels:
        k = lab
        if k not in counts:
            counts[k] = 0
            out.append(k)
        else:
            counts[k] += 1
            out.append(f"{k}  ({counts[k]})")  # sufijo corto
    return out


def get_id_name_cols(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """
    Devuelve (id_col, name_col) con fallback a alias.
    """
    id_candidates = ["mic_id", "MIC_ID"]
    name_candidates = ["mic_name_official", "MIC_NAME", "mic_name"]

    id_col = next((c for c in id_candidates if c in df.columns), None)
    name_col = next((c for c in name_candidates if c in df.columns), None)
    return id_col, name_col


@st.cache_data(show_spinner=False)
def load_dataset(file_bytes: Optional[bytes], filename: Optional[str]) -> pd.DataFrame:
    if file_bytes and filename:
        if filename.lower().endswith(".csv"):
            return pd.read_csv(io.BytesIO(file_bytes), encoding="utf-8")
        if filename.lower().endswith(".xlsx"):
            return pd.read_excel(io.BytesIO(file_bytes))
        raise ValueError("Formato no soportado. Use .csv o .xlsx")

    if os.path.exists("P9_Dataset_Trazable_MIC.csv"):
        return pd.read_csv("P9_Dataset_Trazable_MIC.csv", encoding="utf-8")
    if os.path.exists("P9_Dataset_Trazable_MIC.xlsx"):
        return pd.read_excel("P9_Dataset_Trazable_MIC.xlsx")
    raise FileNotFoundError("No se encontró P9_Dataset_Trazable_MIC.csv/.xlsx y no se cargó archivo.")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # Aliases mínimos
    if "mic_id" in df.columns and "MIC_ID" not in df.columns:
        df["MIC_ID"] = df["mic_id"]
    if "mic_name_official" in df.columns and "MIC_NAME" not in df.columns:
        df["MIC_NAME"] = df["mic_name_official"]

    # Aliases de ranking/score (compatibilidad con scripts antiguos)
    if "Score_total_adj" in df.columns and "score" not in df.columns:
        df["score"] = df["Score_total_adj"]
    if "Rank_global" in df.columns and "rank" not in df.columns:
        df["rank"] = df["Rank_global"]

    return df


def ensure_rank_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye score/rank si faltan o vienen vacíos.
    Prioriza Score_total_adj y Rank_global (P8).
    Fallbacks:
    - Score_total_min
    - suma ponderada (C1..C10 * W_C1..W_C10)
    - ranking por score
    """
    df = df.copy()

    if "Score_total_adj" not in df.columns:
        df["Score_total_adj"] = np.nan
    if "Score_total_min" not in df.columns:
        df["Score_total_min"] = np.nan

    # 1) si adj vacío pero min existe
    if df["Score_total_adj"].isna().all() and df["Score_total_min"].notna().any():
        df["Score_total_adj"] = pd.to_numeric(df["Score_total_min"], errors="coerce")

    # 2) si ambos vacíos, intenta reconstruir desde C y W
    has_c = all(f"C{i}" in df.columns for i in range(1, 11))
    has_w = all(f"W_C{i}" in df.columns for i in range(1, 11))
    if df["Score_total_adj"].isna().all() and has_c and has_w:
        ccols = [f"C{i}" for i in range(1, 11)]
        wcols = [f"W_C{i}" for i in range(1, 11)]
        temp_c = df[ccols].apply(pd.to_numeric, errors="coerce")
        temp_w = df[wcols].apply(pd.to_numeric, errors="coerce")
        df["Score_total_adj"] = (temp_c * temp_w).sum(axis=1)

    if "Rank_global" not in df.columns:
        df["Rank_global"] = np.nan

    # Si ranking vacío pero score existe, crea ranking dense
    if df["Rank_global"].isna().all() and df["Score_total_adj"].notna().any():
        df["Rank_global"] = df["Score_total_adj"].rank(method="dense", ascending=False)

    df["score"] = pd.to_numeric(df["Score_total_adj"], errors="coerce")
    df["rank"] = pd.to_numeric(df["Rank_global"], errors="coerce")
    return df


def multi_filter(dfx: pd.DataFrame, col: str, label: str):
    if col not in dfx.columns:
        return None
    vals = sorted([v for v in dfx[col].dropna().unique().tolist() if str(v).strip() != ""])
    if not vals:
        return None
    return st.sidebar.multiselect(label, vals, default=vals)


# -----------------------------
# Sidebar: dataset load
# -----------------------------
st.sidebar.markdown("### Dataset")
uploaded = st.sidebar.file_uploader("Carga P9 (CSV o XLSX)", type=["csv", "xlsx"])
file_bytes = uploaded.getvalue() if uploaded else None
filename = uploaded.name if uploaded else None

try:
    df = load_dataset(file_bytes, filename)
except Exception as e:
    st.error(f"No se pudo cargar el dataset. Detalle: {e}")
    st.stop()

df = normalize_columns(df)

# Numéricos relevantes
coerce_num(
    df,
    [
        "Score_total_adj",
        "Score_total_min",
        "Rank_global",
        "Rank_country",
        "ipc_coverage_count",
        "last_update_year",
        "longevity_years",
    ],
)

df = ensure_rank_score(df)

# -----------------------------
# Header
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

sel_country = multi_filter(df, "country_name", "País")
sel_group = multi_filter(df, "country_group", "Grupo de países")
sel_ipc = multi_filter(df, "ipc_category_primary_label", "Categoría IPC (principal)")
sel_type = multi_filter(df, "mic_type", "Tipología MIC")
sel_obl = multi_filter(df, "obligation_level", "Obligatoriedad")
sel_enf = multi_filter(df, "enforcement_level", "Enforcement")
sel_trans = multi_filter(df, "transparency_level", "Transparencia")
sel_registry = multi_filter(df, "public_registry", "Registro público")

top_n = st.sidebar.slider("Top N ranking", 5, 50, 15, step=1)

score_minmax = None
if "Score_total_adj" in df.columns and df["Score_total_adj"].notna().any():
    lo = float(np.nanmin(df["Score_total_adj"]))
    hi = float(np.nanmax(df["Score_total_adj"]))
    score_minmax = st.sidebar.slider("Rango de score (MCA)", lo, hi, (lo, hi))


# Apply filters
dff = df.copy()

def apply(col: str, sel):
    global dff
    if sel is None or col not in dff.columns:
        return
    dff = dff[dff[col].isin(sel)]

apply("country_name", sel_country)
apply("country_group", sel_group)
apply("ipc_category_primary_label", sel_ipc)
apply("mic_type", sel_type)
apply("obligation_level", sel_obl)
apply("enforcement_level", sel_enf)
apply("transparency_level", sel_trans)
apply("public_registry", sel_registry)

if score_minmax is not None:
    a, b = score_minmax
    dff = dff[(dff["Score_total_adj"].fillna(-np.inf) >= a) & (dff["Score_total_adj"].fillna(np.inf) <= b)]


# -----------------------------
# KPIs
# -----------------------------
k1, k2, k3, k4 = st.columns(4, gap="large")

def kpi(container, label, value, note=""):
    container.markdown("<div class='card'>", unsafe_allow_html=True)
    container.markdown(f"<div class='kpi'><div class='label'>{label}</div>", unsafe_allow_html=True)
    container.markdown(f"<div class='value'>{value}</div></div>", unsafe_allow_html=True)
    if note:
        container.markdown(f"<div class='small'>{note}</div>", unsafe_allow_html=True)
    container.markdown("</div>", unsafe_allow_html=True)

kpi(k1, "Registros filtrados", f"{len(dff):,}", "Aplicando filtros del panel izquierdo.")
kpi(k2, "Países", f"{dff['country_name'].nunique():,}" if "country_name" in dff.columns else "ND")
kpi(k3, "Categorías IPC (principal)", f"{dff['ipc_category_primary_label'].nunique():,}" if "ipc_category_primary_label" in dff.columns else "ND")
kpi(k4, "Score MCA (promedio)", f"{dff['Score_total_adj'].mean():.2f}" if dff["Score_total_adj"].notna().any() else "ND")

st.markdown("---")


# -----------------------------
# Ranking + perfil
# -----------------------------
c1, c2 = st.columns([0.58, 0.42], gap="large")

with c1:
    st.markdown("### Ranking MCA (Top N)")

    sort_mode = st.radio(
        "Orden de barras",
        ["Descendente (mejor score arriba)", "Ascendente (mejor score abajo)"],
        horizontal=True,
        index=0,
    )
    asc = sort_mode.startswith("Ascendente")

    id_col, name_col = get_id_name_cols(dff)

    # Condición robusta: id+name+score con datos (acepta alias)
    can_rank = (
        (id_col is not None) and (name_col is not None) and
        dff[id_col].notna().any() and dff[name_col].notna().any() and
        dff["Score_total_adj"].notna().any()
    )

    if can_rank:
        top = dff.dropna(subset=["Score_total_adj"]).copy()

        # orden base por score
        top = top.sort_values(by="Score_total_adj", ascending=asc).head(top_n)

        # labels envueltas (2–3 líneas)
        base_labels = top.apply(
            lambda r: wrap_label(f"{r[id_col]} — {r[name_col]}", width=38, max_lines=3),
            axis=1,
        ).tolist()

        # FIX: asegurar que categorías sean únicas (evita ValueError)
        unique_labels = make_unique_labels(base_labels)

        top["label_wrapped"] = unique_labels
        top["label_order"] = pd.Categorical(
            top["label_wrapped"],
            categories=top["label_wrapped"].tolist(),
            ordered=True
        )

        fig = px.bar(
            top,
            x="Score_total_adj",
            y="label_order",
            orientation="h",
            hover_data=[c for c in ["country_name", "country_group", "ipc_category_primary_label"] if c in top.columns],
            text=top["Score_total_adj"].round(2),
        )

        # Barras más visibles
        fig.update_traces(
            textposition="inside",
            insidetextanchor="start",
            cliponaxis=False,
            marker_line_width=0.0,
        )

        fig.update_layout(
            height=640,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            xaxis_title="Score MCA (ponderado)",
            yaxis_title="",
            yaxis=dict(tickfont=dict(size=12)),
        )

        # Eje X bien legible
        fig.update_xaxes(showgrid=True, gridcolor="rgba(15, 23, 42, 0.08)")
        fig.update_yaxes(showgrid=False)

        st.plotly_chart(fig, use_container_width=True)
    else:
        missing = []
        for need in [("id", id_col), ("name", name_col), ("Score_total_adj", "Score_total_adj")]:
            k, col = need
            if col is None or col not in dff.columns or not dff[col].notna().any():
                missing.append(k)
        st.info(
            "No se pudo graficar el ranking con el subconjunto filtrado. "
            f"Campos requeridos (con datos): {', '.join(missing) if missing else 'id, name, Score_total_adj'}."
        )

with c2:
    st.markdown("### Perfil normativo y de fiscalización")
    tabs = st.tabs(["Obligatoriedad", "Enforcement", "Cobertura IPC", "Triple dimensión"])

    with tabs[0]:
        if "obligation_level" in dff.columns and dff["obligation_level"].notna().any():
            s = dff["obligation_level"].value_counts(dropna=True).reset_index()
            s.columns = ["obligation_level", "count"]
            fig = px.pie(s, names="obligation_level", values="count", title="")
            fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="#FFFFFF")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No se identificó 'obligation_level' con datos en el subconjunto filtrado.")

    with tabs[1]:
        if "enforcement_level" in dff.columns and dff["enforcement_level"].notna().any():
            s = dff["enforcement_level"].value_counts(dropna=True).reset_index()
            s.columns = ["enforcement_level", "count"]
            fig = px.bar(s, x="enforcement_level", y="count", title="")
            fig.update_layout(
                height=340,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                xaxis_title="",
                yaxis_title="Nº MIC",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No se identificó 'enforcement_level' con datos en el subconjunto filtrado.")

    with tabs[2]:
        if "ipc_coverage_count" in dff.columns and dff["ipc_coverage_count"].notna().any():
            fig = px.histogram(dff, x="ipc_coverage_count", nbins=10, title="")
            fig.update_layout(
                height=340,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                xaxis_title="Nº de categorías IPC cubiertas",
                yaxis_title="Nº MIC",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No se identificó 'ipc_coverage_count' con datos en el subconjunto filtrado.")

    with tabs[3]:
        # dim_env/dim_soc/dim_eco vienen como SI/NO => convertir a 1/0
        dims = []
        for col, label in [("dim_env", "Ambiental"), ("dim_soc", "Social"), ("dim_eco", "Económica")]:
            if col in dff.columns and dff[col].notna().any():
                b = to_bool01(dff[col])
                dims.append((label, int((b >= 1).sum())))
        if dims:
            dd = pd.DataFrame(dims, columns=["dimensión", "MIC_con_cobertura"])
            fig = px.bar(dd, x="dimensión", y="MIC_con_cobertura", title="")
            fig.update_layout(
                height=340,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="#FFFFFF",
                plot_bgcolor="#FFFFFF",
                xaxis_title="",
                yaxis_title="Nº MIC",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No se identificaron dim_env/dim_soc/dim_eco con datos en el subconjunto filtrado.")

st.markdown("---")


# -----------------------------
# Explorador + ficha
# -----------------------------
st.markdown("### Explorador de MIC (tabla + ficha)")
left, right = st.columns([0.62, 0.38], gap="large")

with left:
    show_cols = [
        "Rank_global", "Score_total_adj",
        "mic_id", "mic_name_official", "mic_acronym",
        "country_name", "country_iso2", "country_group",
        "ipc_category_primary_label", "ipc_coverage_count",
        "mic_type", "mic_subtype", "channel",
        "obligation_level", "verification_model", "third_party_required", "accreditation_required",
        "public_registry", "enforcement_level", "sanctions_exist",
        "transparency_level", "comparability_support",
        "dim_env", "dim_soc", "dim_eco",
        "source_id_primary", "source_url_primary",
    ]
    # si vienen por alias, igual mostramos:
    if "mic_id" not in dff.columns and "MIC_ID" in dff.columns:
        show_cols = ["MIC_ID" if c == "mic_id" else c for c in show_cols]
    if "mic_name_official" not in dff.columns and "MIC_NAME" in dff.columns:
        show_cols = ["MIC_NAME" if c == "mic_name_official" else c for c in show_cols]

    show_cols = [c for c in show_cols if c in dff.columns]

    table = dff[show_cols].copy()
    if "Rank_global" in table.columns and table["Rank_global"].notna().any():
        table = table.sort_values(by=["Rank_global"], ascending=True)
    else:
        table = table.sort_values(by=["Score_total_adj"], ascending=False, na_position="last")

    st.dataframe(table, use_container_width=True, height=440)

    st.download_button(
        "Descargar subconjunto filtrado (CSV)",
        data=to_csv_download(dff),
        file_name="P9_filtrado.csv",
        mime="text/csv",
        use_container_width=True,
    )

with right:
    id_col, name_col = get_id_name_cols(dff)
    can_profile = (
        (id_col is not None) and (name_col is not None) and
        dff[id_col].notna().any() and dff[name_col].notna().any()
    )

    if can_profile and len(dff) > 0:
        dff_sel = dff.copy()
        dff_sel["__label__"] = dff_sel[id_col].astype(str) + " — " + dff_sel[name_col].astype(str)

        pick = st.selectbox("Seleccionar MIC", dff_sel["__label__"].tolist())
        row = dff_sel[dff_sel["__label__"] == pick].iloc[0]

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"**{safe_text(row.get(name_col))}**")
        st.markdown(f"<span class='badge badge-accent'>{safe_text(row.get(id_col))}</span>", unsafe_allow_html=True)

        def line(k: str, v):
            st.markdown(f"**{k}:** {safe_text(v)}")

        line("País", row.get("country_name"))
        line("Grupo", row.get("country_group"))
        line("Categoría IPC (principal)", row.get("ipc_category_primary_label"))
        line("Cobertura IPC (conteo)", row.get("ipc_coverage_count"))
        line("Tipo MIC", row.get("mic_type"))
        line("Subtipo", row.get("mic_subtype"))
        line("Canal", row.get("channel"))
        line("Obligatoriedad", row.get("obligation_level"))
        line("Modelo de verificación", row.get("verification_model"))
        line("Tercero requerido", row.get("third_party_required"))
        line("Acreditación requerida", row.get("accreditation_required"))
        line("Registro público", row.get("public_registry"))
        line("Enforcement", row.get("enforcement_level"))
        line("Sanciones existen", row.get("sanctions_exist"))
        line("Transparencia", row.get("transparency_level"))
        line("Soporte comparabilidad", row.get("comparability_support"))

        # Triple dimensión — mostrar SI/NO normalizado (aunque venga 'no')
        env = int(to_bool01(pd.Series([row.get("dim_env")]))[0]) if "dim_env" in dff.columns else 0
        soc = int(to_bool01(pd.Series([row.get("dim_soc")]))[0]) if "dim_soc" in dff.columns else 0
        eco = int(to_bool01(pd.Series([row.get("dim_eco")]))[0]) if "dim_eco" in dff.columns else 0

        st.markdown(
            "**Triple dimensión:** "
            + " | ".join(
                [
                    f"Ambiental={'sí' if env==1 else 'no'}",
                    f"Social={'sí' if soc==1 else 'no'}",
                    f"Económica={'sí' if eco==1 else 'no'}",
                ]
            )
        )

        line("Score MCA", row.get("Score_total_adj"))
        line("Ranking global", row.get("Rank_global"))
        line("Ranking país", row.get("Rank_country"))

        st.markdown("---")
        line("Source ID (primario)", row.get("source_id_primary"))
        src_url = safe_text(row.get("source_url_primary"))
        if src_url != "ND":
            st.markdown(f"**Source URL (primario):** `{src_url}`")

        notes = safe_text(row.get("evidence_quality_note"))
        if notes != "ND":
            st.markdown("**Nota calidad evidencia:**")
            st.write(notes)

        gaps = safe_text(row.get("gaps_text"))
        if gaps != "ND":
            st.markdown("**Brechas declaradas (gaps):**")
            st.write(gaps)

        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info(
            "No se puede construir ficha con el subconjunto filtrado. "
            "Se requieren columnas mic_id/MIC_ID + mic_name_official/MIC_NAME con datos."
        )

st.markdown("---")

# Footer texto + membrete en el CUERPO
st.markdown(
    "<div class='small'>P9 consolida variables base (P2/P5), resultados MCA (P8) y metadatos de trazabilidad para auditoría y dashboard.</div>",
    unsafe_allow_html=True,
)

membrete_b64 = b64_image("membrete (1).png")
if membrete_b64:
    st.markdown(
        f"""
<div class="membrete-body">
  <img src="data:image/png;base64,{membrete_b64}" alt="Membrete Gobierno de Chile" />
</div>
""",
        unsafe_allow_html=True,
    )
