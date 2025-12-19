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
# Config
# =========================
st.set_page_config(
    page_title="Ranking y métricas MIC (MCA)",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

TITLE = "Ranking y métricas asociadas a mecanismos de información al consumidor."
SUBTITLE = "Consultoría Sustrend para la Subsecretaría del Medio Ambiente | ID: 608897-205-COT25"
FOOTER_TEXT = "P9 consolida variables base (P2/P5), resultados MCA (P8) y metadatos de trazabilidad para auditoría y dashboard."

DEFAULT_DATASET_CSV = "P9_Dataset_Trazable_MIC.csv"
DEFAULT_DATASET_XLSX = "P9_Dataset_Trazable_MIC.xlsx"
MEMBRETE_FILENAME = "membrete (1).png"  # debe estar junto a app.py o en el mismo repo

# =========================
# Estilo (fondo blanco + cards)
# =========================
st.markdown(
    """
<style>
/* Fondo global blanco */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"]{
    background: #ffffff !important;
}
[data-testid="stHeader"] { background: rgba(255,255,255,0.9) !important; }
[data-testid="stSidebar"] { background: #ffffff !important; }

/* Tipografía y espaciados */
.main-title{
    font-size: 34px;
    font-weight: 800;
    line-height: 1.1;
    margin: 0 0 6px 0;
}
.main-subtitle{
    font-size: 14px;
    font-weight: 500;
    opacity: 0.75;
    margin: 0 0 16px 0;
}

/* Tarjetas */
.card{
    border: 1px solid rgba(0,0,0,0.08);
    border-radius: 14px;
    padding: 14px 16px;
    background: #fff;
    box-shadow: 0 10px 30px rgba(0,0,0,0.06);
}
.card-title{
    font-size: 13px;
    font-weight: 700;
    opacity: 0.72;
    margin-bottom: 6px;
}
.small-note{
    font-size: 12px;
    opacity: 0.7;
}

/* Bloque membrete con sombra */
.membrete-wrap{
    display: inline-block;
    padding: 10px 12px;
    border-radius: 14px;
    background: #fff;
    border: 1px solid rgba(0,0,0,0.08);
    box-shadow: 0 10px 30px rgba(0,0,0,0.10);
}

/* Separadores suaves */
.hr-soft{
    border: 0;
    border-top: 1px solid rgba(0,0,0,0.08);
    margin: 10px 0 16px 0;
}
</style>
""",
    unsafe_allow_html=True,
)

# =========================
# Utilidades
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
    yes = {"si", "sí", "sì", "yes", "y", "true", "1", "t"}
    no = {"no", "n", "false", "0", "f"}
    out = pd.Series(np.nan, index=s.index, dtype="float")
    out[s2.isin(yes)] = 1
    out[s2.isin(no)] = 0
    return out

def _wrap_label(label: str, width: int = 26, max_lines: int = 3) -> str:
    if label is None:
        return ""
    wrapped = textwrap.wrap(str(label), width=width)
    wrapped = wrapped[:max_lines]
    return "\n".join(wrapped)

def _safe_read_dataset(uploaded_file):
    if uploaded_file is None:
        if os.path.exists(DEFAULT_DATASET_CSV):
            return pd.read_csv(DEFAULT_DATASET_CSV)
        if os.path.exists(DEFAULT_DATASET_XLSX):
            return pd.read_excel(DEFAULT_DATASET_XLSX)
        return None

    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    return None

def _compute_ranking_fallback(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    score_col = _find_col(df, ["Score_total_adj", "score_total_adj", "score"])
    rank_col = _find_col(df, ["Rank_global", "rank_global", "rank"])
    if score_col and rank_col:
        return df

    c_cols = [c for c in df.columns if re.fullmatch(r"C\d+", str(c).strip(), flags=re.IGNORECASE)]
    w_cols = [c for c in df.columns if re.fullmatch(r"W_C\d+", str(c).strip(), flags=re.IGNORECASE)]

    def _num(s):
        m = re.search(r"(\d+)", str(s))
        return int(m.group(1)) if m else 999

    c_cols = sorted(c_cols, key=_num)
    w_cols = sorted(w_cols, key=_num)

    if len(c_cols) >= 3 and len(w_cols) >= 3:
        weights = []
        used_c = []
        for c in c_cols:
            k = _num(c)
            w_match = None
            for w in w_cols:
                if _num(w) == k:
                    w_match = w
                    break
            if w_match is not None:
                used_c.append(c)
                weights.append(w_match)

        if len(used_c) >= 3:
            C = df[used_c].apply(pd.to_numeric, errors="coerce")
            W = df[weights].apply(pd.to_numeric, errors="coerce")

            w_mean = W.mean(axis=0, skipna=True)
            w_mean = w_mean.replace([np.inf, -np.inf], np.nan).fillna(0)

            denom = w_mean.sum()
            if denom <= 0:
                df["Score_total_adj"] = C.mean(axis=1, skipna=True)
            else:
                df["Score_total_adj"] = (C * w_mean.values).sum(axis=1, skipna=True) / denom

            df["Rank_global"] = df["Score_total_adj"].rank(ascending=False, method="dense").astype(int)
            return df

    if len(c_cols) >= 3:
        C = df[c_cols].apply(pd.to_numeric, errors="coerce")
        df["Score_total_adj"] = C.mean(axis=1, skipna=True)
        df["Rank_global"] = df["Score_total_adj"].rank(ascending=False, method="dense").astype(int)
        return df

    score_min = _find_col(df, ["Score_total_min", "score_total_min"])
    if score_min:
        df["Score_total_adj"] = pd.to_numeric(df[score_min], errors="coerce")
        df["Rank_global"] = df["Score_total_adj"].rank(ascending=False, method="dense").astype(int)
        return df

    df["Score_total_adj"] = np.nan
    df["Rank_global"] = np.nan
    return df

def _ensure_core_columns(df: pd.DataFrame) -> dict:
    col_mic_id = _find_col(df, ["mic_id", "MIC_ID", "id", "micid"])
    col_name = _find_col(df, ["mic_name_official", "MIC_NAME", "mic_name", "name", "nombre"])
    col_country = _find_col(df, ["country_name", "pais", "country"])
    col_group = _find_col(df, ["country_group", "group"])
    col_type = _find_col(df, ["mic_type", "type"])
    col_owner_type = _find_col(df, ["mic_owner_type", "owner_type"])
    col_ipc_label = _find_col(df, ["ipc_category_primary_label", "ipc_primary_label"])
    col_score = _find_col(df, ["Score_total_adj", "score_total_adj", "score"])
    col_rank = _find_col(df, ["Rank_global", "rank_global", "rank"])
    col_dim_env = _find_col(df, ["dim_env", "DIM_ENV"])
    col_dim_soc = _find_col(df, ["dim_soc", "DIM_SOC"])
    col_dim_eco = _find_col(df, ["dim_eco", "DIM_ECO"])
    return {
        "mic_id": col_mic_id,
        "mic_name": col_name,
        "country": col_country,
        "group": col_group,
        "mic_type": col_type,
        "owner_type": col_owner_type,
        "ipc_label": col_ipc_label,
        "score": col_score,
        "rank": col_rank,
        "dim_env": col_dim_env,
        "dim_soc": col_dim_soc,
        "dim_eco": col_dim_eco,
    }

def _render_membrete_block():
    membrete_path_candidates = [
        MEMBRETE_FILENAME,
        os.path.join("assets", MEMBRETE_FILENAME),
        os.path.join("static", MEMBRETE_FILENAME),
        os.path.join("img", MEMBRETE_FILENAME),
    ]
    membrete_path = next((p for p in membrete_path_candidates if os.path.exists(p)), None)

    # alineado al cuerpo, abajo-izquierda, como bloque independiente con sombra
    left, _mid, _right = st.columns([1.2, 0.8, 2.0])
    with left:
        if membrete_path:
            st.markdown("<div class='membrete-wrap'>", unsafe_allow_html=True)
            st.image(membrete_path, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning(
                "No se encontró el archivo de membrete. "
                f"Asegura que '{MEMBRETE_FILENAME}' esté en el repo (junto a app.py o en /assets)."
            )

# =========================
# Sidebar: carga de datos
# =========================
st.sidebar.markdown("### Datos (P9)")
uploaded = st.sidebar.file_uploader("Cargar P9 (CSV o XLSX)", type=["csv", "xlsx", "xls"])

df_raw = _safe_read_dataset(uploaded)
if df_raw is None or df_raw.empty:
    st.error("No se encontró un dataset para visualizar. Carga el archivo P9 (CSV/XLSX) desde la barra lateral.")
    st.stop()

df_raw.columns = [str(c).strip() for c in df_raw.columns]
df = _compute_ranking_fallback(df_raw)
cols = _ensure_core_columns(df)

# =========================
# Header
# =========================
st.markdown(f"<div class='main-title'>{TITLE}</div>", unsafe_allow_html=True)
st.markdown(f"<div class='main-subtitle'>{SUBTITLE}</div>", unsafe_allow_html=True)
st.markdown("<div class='hr-soft'></div>", unsafe_allow_html=True)

# =========================
# Intro (sin membrete aquí)
# =========================
intro_col, _ = st.columns([1.6, 1.0])
with intro_col:
    st.markdown(
        f"""
<div class="card">
  <div class="card-title">Propósito del visualizador</div>
  <div class="small-note">
    Este tablero lee el dataset P9 trazable y permite explorar ranking (MCA), métricas, cobertura IPC y atributos normativos.
    {FOOTER_TEXT}
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

st.write("")

# =========================
# Filtros principales (sidebar)
# =========================
st.sidebar.markdown("---")
st.sidebar.markdown("### Filtros")

def _unique_sorted(series):
    if series is None:
        return []
    vals = series.dropna().astype(str).unique().tolist()
    return sorted(vals)

country_vals = _unique_sorted(df[cols["country"]]) if cols["country"] else []
group_vals = _unique_sorted(df[cols["group"]]) if cols["group"] else []
type_vals = _unique_sorted(df[cols["mic_type"]]) if cols["mic_type"] else []
owner_vals = _unique_sorted(df[cols["owner_type"]]) if cols["owner_type"] else []
ipc_vals = _unique_sorted(df[cols["ipc_label"]]) if cols["ipc_label"] else []

sel_group = st.sidebar.multiselect("Grupo país", group_vals, default=[])
sel_country = st.sidebar.multiselect("País", country_vals, default=[])
sel_type = st.sidebar.multiselect("Tipología MIC", type_vals, default=[])
sel_owner = st.sidebar.multiselect("Propiedad (público/privado/mixto)", owner_vals, default=[])
sel_ipc = st.sidebar.multiselect("IPC (categoría primaria)", ipc_vals, default=[])

# =========================
# Aplicar filtros
# =========================
df_f = df.copy()
if sel_group and cols["group"]:
    df_f = df_f[df_f[cols["group"]].astype(str).isin(sel_group)]
if sel_country and cols["country"]:
    df_f = df_f[df_f[cols["country"]].astype(str).isin(sel_country)]
if sel_type and cols["mic_type"]:
    df_f = df_f[df_f[cols["mic_type"]].astype(str).isin(sel_type)]
if sel_owner and cols["owner_type"]:
    df_f = df_f[df_f[cols["owner_type"]].astype(str).isin(sel_owner)]
if sel_ipc and cols["ipc_label"]:
    df_f = df_f[df_f[cols["ipc_label"]].astype(str).isin(sel_ipc)]

# =========================
# Tabs
# =========================
tab_rank, tab_metrics, tab_explorer = st.tabs(["📈 Ranking MCA", "🧩 Métricas y perfil", "🔎 Explorador MIC"])

# -------------------------
# TAB 1: Ranking
# -------------------------
with tab_rank:
    st.markdown("### Ranking MCA (Top N)")

    if cols["mic_id"] is None or cols["mic_name"] is None:
        st.error("No se puede construir ranking/explorador sin 'mic_id' y 'mic_name_official'.")
        st.stop()

    score_col = _find_col(df_f, ["Score_total_adj"])
    rank_col = _find_col(df_f, ["Rank_global"])

    if score_col is None or rank_col is None or df_f[score_col].dropna().empty:
        st.warning(
            "No hay información suficiente para calcular ranking (se requiere Score_total_adj y Rank_global "
            "o criterios C1..C10 con pesos). Se mostrará tabla básica."
        )
        st.dataframe(df_f.head(50), use_container_width=True)
        st.write("")
        # membrete AL FINAL del ranking, incluso si no hubo gráfico
        _render_membrete_block()
    else:
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            top_n = st.slider("Top N", min_value=5, max_value=50, value=15, step=1)
        with c2:
            sort_dir = st.radio("Orden barras", ["Descendente", "Ascendente"], horizontal=True)
        with c3:
            show_values = st.toggle("Mostrar valor en barra", value=True)

        df_plot = df_f.copy()
        df_plot[score_col] = pd.to_numeric(df_plot[score_col], errors="coerce")
        df_plot = df_plot.dropna(subset=[score_col])

        ascending = (sort_dir == "Ascendente")
        df_plot = df_plot.sort_values(score_col, ascending=ascending).head(top_n)

        # etiqueta única + wrap
        label_base = df_plot[cols["mic_name"]].astype(str)
        label = label_base + " · " + df_plot[cols["mic_id"]].astype(str)
        df_plot["label_wrapped"] = label.apply(lambda x: _wrap_label(x, width=28, max_lines=3))

        # evitar duplicados por cualquier motivo
        if df_plot["label_wrapped"].duplicated().any():
            counts = {}
            new = []
            for v in df_plot["label_wrapped"].tolist():
                counts[v] = counts.get(v, 0) + 1
                new.append(f"{v} ({counts[v]})" if counts[v] > 1 else v)
            df_plot["label_wrapped"] = new

        if not ALTAIR_OK:
            st.info("Altair no está disponible. Instala 'altair' para ver el gráfico.")
            st.dataframe(df_plot[[cols["mic_id"], cols["mic_name"], score_col, rank_col]], use_container_width=True)
        else:
            base = alt.Chart(df_plot).encode(
                y=alt.Y(
                    "label_wrapped:N",
                    sort=None,
                    axis=alt.Axis(labelFontSize=9, title=None),  # MÁS pequeño
                ),
                x=alt.X(f"{score_col}:Q", title="Score (ponderado)"),
                tooltip=[
                    alt.Tooltip(cols["mic_id"] + ":N", title="MIC ID"),
                    alt.Tooltip(cols["mic_name"] + ":N", title="Nombre"),
                    alt.Tooltip(f"{score_col}:Q", title="Score", format=".3f"),
                    alt.Tooltip(f"{rank_col}:Q", title="Rank"),
                ],
            )

            bars = base.mark_bar()
            if show_values:
                text = base.mark_text(
                    align="left",
                    baseline="middle",
                    dx=4,
                    fontSize=9,  # MÁS pequeño
                ).encode(
                    text=alt.Text(f"{score_col}:Q", format=".2f")
                )
                chart = (bars + text).properties(height=450)
            else:
                chart = bars.properties(height=450)

            st.altair_chart(chart, use_container_width=True)

        st.markdown("**Tabla ranking (subconjunto filtrado)**")
        st.dataframe(
            df_plot[[cols["mic_id"], cols["mic_name"], score_col, rank_col]],
            use_container_width=True,
            hide_index=True,
        )

        # === MEMBRETE AQUÍ: al final, DESPUÉS de la tabla de ranking ===
        st.write("")
        _render_membrete_block()

# -------------------------
# TAB 2: Métricas y perfil
# -------------------------
with tab_metrics:
    st.markdown("### Perfil normativo y de fiscalización (subconjunto filtrado)")

    d_env = cols["dim_env"]
    d_soc = cols["dim_soc"]
    d_eco = cols["dim_eco"]

    df_f["_dim_env01"] = _to_bool01(df_f[d_env]) if d_env else np.nan
    df_f["_dim_soc01"] = _to_bool01(df_f[d_soc]) if d_soc else np.nan
    df_f["_dim_eco01"] = _to_bool01(df_f[d_eco]) if d_eco else np.nan

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("MIC en vista", int(len(df_f)))
    with m2:
        st.metric("Países", int(df_f[cols["country"]].nunique()) if cols["country"] else 0)
    with m3:
        st.metric("IPC primarias", int(df_f[cols["ipc_label"]].nunique()) if cols["ipc_label"] else 0)
    with m4:
        st.metric("Tipos MIC", int(df_f[cols["mic_type"]].nunique()) if cols["mic_type"] else 0)

    st.write("")
    st.markdown("#### Triple dimensión")

    dim_sub = df_f[["_dim_env01", "_dim_soc01", "_dim_eco01"]].dropna(how="all")
    if dim_sub.empty:
        st.warning("No se identificaron dim_env/dim_soc/dim_eco con datos utilizables en el subconjunto filtrado.")
    else:
        env_rate = float(dim_sub["_dim_env01"].mean(skipna=True)) if dim_sub["_dim_env01"].notna().any() else np.nan
        soc_rate = float(dim_sub["_dim_soc01"].mean(skipna=True)) if dim_sub["_dim_soc01"].notna().any() else np.nan
        eco_rate = float(dim_sub["_dim_eco01"].mean(skipna=True)) if dim_sub["_dim_eco01"].notna().any() else np.nan

        c1, c2, c3 = st.columns(3)
        c1.metric("Ambiental (promedio)", "ND" if np.isnan(env_rate) else f"{env_rate:.2%}")
        c2.metric("Social (promedio)", "ND" if np.isnan(soc_rate) else f"{soc_rate:.2%}")
        c3.metric("Económica (promedio)", "ND" if np.isnan(eco_rate) else f"{eco_rate:.2%}")

        if ALTAIR_OK:
            dplot = pd.DataFrame({
                "Dimensión": ["Ambiental", "Social", "Económica"],
                "Cobertura": [env_rate, soc_rate, eco_rate]
            }).dropna()
            chart = alt.Chart(dplot).mark_bar().encode(
                x=alt.X("Dimensión:N", title=None),
                y=alt.Y("Cobertura:Q", title="Proporción (0–1)", axis=alt.Axis(format="%")),
                tooltip=[alt.Tooltip("Cobertura:Q", format=".2%")]
            ).properties(height=260)
            st.altair_chart(chart, use_container_width=True)

    st.write("")
    st.markdown("#### Vista de datos (subconjunto filtrado)")
    preview_cols = [c for c in [
        cols["mic_id"], cols["mic_name"], cols["country"], cols["group"], cols["mic_type"], cols["ipc_label"],
        "Score_total_adj", "Rank_global"
    ] if c and c in df_f.columns]
    st.dataframe(df_f[preview_cols].head(200), use_container_width=True, hide_index=True)

# -------------------------
# TAB 3: Explorador MIC
# -------------------------
with tab_explorer:
    st.markdown("### Explorador de MIC (tabla + ficha)")

    if cols["mic_id"] is None or cols["mic_name"] is None:
        st.error("No se puede construir ficha sin columnas equivalentes a MIC_ID + MIC_NAME.")
        st.stop()

    df_ex = df_f.copy()
    df_ex["_label"] = df_ex[cols["mic_name"]].astype(str) + " · " + df_ex[cols["mic_id"]].astype(str)

    q = st.text_input("Buscar (nombre o ID)", value="")
    if q.strip():
        mask = df_ex["_label"].str.contains(q, case=False, na=False)
        df_ex = df_ex[mask]

    table_cols = [c for c in [
        cols["mic_id"], cols["mic_name"], cols["country"], cols["mic_type"], cols["owner_type"],
        cols["ipc_label"], "Score_total_adj", "Rank_global", "status_active", "start_year", "last_verified_date"
    ] if c and c in df_ex.columns]

    st.dataframe(df_ex[table_cols].head(500), use_container_width=True, hide_index=True)

    st.write("")
    options = df_ex["_label"].dropna().unique().tolist()
    if not options:
        st.warning("No hay resultados con los filtros/búsqueda actuales.")
    else:
        sel = st.selectbox("Seleccionar MIC para ficha", options=options, index=0)
        row = df_ex[df_ex["_label"] == sel].head(1)
        if row.empty:
            st.warning("No se pudo recuperar el registro seleccionado.")
        else:
            r = row.iloc[0].to_dict()

            left, right = st.columns([1.2, 1.0])
            with left:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown(f"**{r.get(cols['mic_name'], 'ND')}**")
                st.markdown(f"- **MIC ID:** {r.get(cols['mic_id'], 'ND')}")
                if cols["country"]:
                    st.markdown(f"- **País:** {r.get(cols['country'], 'ND')}")
                if cols["group"]:
                    st.markdown(f"- **Grupo:** {r.get(cols['group'], 'ND')}")
                if cols["mic_type"]:
                    st.markdown(f"- **Tipología:** {r.get(cols['mic_type'], 'ND')}")
                if cols["owner_type"]:
                    st.markdown(f"- **Propiedad:** {r.get(cols['owner_type'], 'ND')}")
                if cols["ipc_label"]:
                    st.markdown(f"- **IPC primaria:** {r.get(cols['ipc_label'], 'ND')}")
                st.markdown("</div>", unsafe_allow_html=True)

            with right:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("**MCA**")
                st.markdown(f"- **Score_total_adj:** {r.get('Score_total_adj', 'ND')}")
                st.markdown(f"- **Rank_global:** {r.get('Rank_global', 'ND')}")
                st.markdown(f"- **Rank_country:** {r.get('Rank_country', 'ND')}")
                st.markdown("</div>", unsafe_allow_html=True)

            st.write("")
            st.markdown("#### Triple dimensión (MIC seleccionado)")
            env = _to_bool01(pd.Series([r.get(cols["dim_env"]) if cols["dim_env"] else np.nan])).iloc[0]
            soc = _to_bool01(pd.Series([r.get(cols["dim_soc"]) if cols["dim_soc"] else np.nan])).iloc[0]
            eco = _to_bool01(pd.Series([r.get(cols["dim_eco"]) if cols["dim_eco"] else np.nan])).iloc[0]

            c1, c2, c3 = st.columns(3)
            c1.metric("Ambiental", "Sí" if env == 1 else ("No" if env == 0 else "ND"))
            c2.metric("Social", "Sí" if soc == 1 else ("No" if soc == 0 else "ND"))
            c3.metric("Económica", "Sí" if eco == 1 else ("No" if eco == 0 else "ND"))

# =========================
# Footer técnico
# =========================
st.sidebar.markdown("---")
st.sidebar.caption(f"Última carga: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
