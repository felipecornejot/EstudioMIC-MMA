import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import re

# =========================
# Configuración de página
# =========================
st.set_page_config(
    page_title="Mapa de Replicabilidad - MMA",
    layout="wide",
    page_icon="🌍"
)

# =========================
# Helpers (paths / assets)
# =========================
def app_dir() -> Path:
    try:
        return Path(__file__).resolve().parent
    except Exception:
        return Path.cwd()

def find_asset(filename: str) -> Path | None:
    candidates = [
        app_dir() / filename,
        Path.cwd() / filename,
        app_dir() / "assets" / filename,
        Path.cwd() / "assets" / filename,
        app_dir() / "static" / filename,
        Path.cwd() / "static" / filename,
        app_dir() / "images" / filename,
        Path.cwd() / "images" / filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

def _clean_label(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    # Quita emoji inicial tipo "🔵 " o "🟢 "
    s = re.sub(r"^[\U0001F300-\U0001FAFF]\s*", "", s)
    return s

def _class_badge(label: str) -> str:
    raw = "" if label is None else str(label)
    clean = _clean_label(raw)

    # Colores
    color_map = {
        "Quick Win": "#27AE60",
        "Estratégico": "#0f69b4",
        "Táctico": "#F39C12",
        "Largo Plazo": "#E62639",
        "Largo plazo": "#E62639",
        "Largo-Plazo": "#E62639",
    }

    # Detecta por contenido (tolerante)
    dot = "#9AA5B1"
    for k, v in color_map.items():
        if k.lower() in clean.lower():
            dot = v
            break

    return (
        "<span style='display:inline-flex; align-items:center; gap:8px;'>"
        f"<span style='width:10px; height:10px; border-radius:999px; background:{dot}; "
        "display:inline-block; box-shadow: 0 0 0 1px rgba(15,105,180,0.20);'></span>"
        f"<span>{clean}</span>"
        "</span>"
    )

# =========================
# Estilo CSS (sobrio)
# =========================
st.markdown(
    """
    <style>
      /* --- Header superior transparente / oculto --- */
      header[data-testid="stHeader"] { background: transparent !important; box-shadow: none !important; }
      div[data-testid="stDecoration"] { background: transparent !important; }
      div[data-testid="stToolbar"] { background: transparent !important; box-shadow: none !important; }
      div[data-testid="stToolbar"] { visibility: hidden !important; height: 0 !important; }
      header[data-testid="stHeader"] { height: 0 !important; }

      .stApp, .main, .block-container { background-color: white !important; }
      .block-container { padding-top: 1rem !important; }

      h1, h2, h3, h4, h5, h6 {
        color: #0f69b4 !important;
        font-family: Arial, sans-serif !important;
        background-color: transparent !important;
      }

      /* Línea bajo el título más delgada */
      h1 {
        font-size: 32px !important;
        font-weight: 600 !important;
        border-bottom: 1px solid rgba(15,105,180,0.55) !important;
        padding-bottom: 10px !important;
        margin-bottom: 15px !important;
      }
      h2 { font-size: 22px !important; font-weight: 600 !important; margin-top: 20px !important; margin-bottom: 10px !important; }
      h3 { font-size: 18px !important; font-weight: 600 !important; }

      .stMarkdown, .stCaption, p, .stText { color: #0f69b4 !important; }

      /* Métricas en azul */
      div[data-testid="stMetricValue"] { color: #0f69b4 !important; }
      div[data-testid="stMetricLabel"] { color: #0f69b4 !important; font-weight: 600 !important; }
      div[data-testid="stMetricDelta"] { color: #0f69b4 !important; opacity: 0.85 !important; }

      /* Nota al pie */
      .footnote {
        margin-top: 10px;
        padding: 10px 12px;
        border: 1px solid rgba(15,105,180,0.18);
        border-radius: 8px;
        background: rgba(15,105,180,0.03);
        font-size: 12.5px;
        line-height: 1.45;
        color: #0f69b4;
        opacity: 0.95;
      }
      .footnote .ft-title { font-weight: 600; margin-bottom: 6px; }
      .footnote .ft-label { font-weight: 600; }

      /* Widgets sobrios */
      .stSelectbox label, .stMultiselect label {
        color: #0f69b4 !important;
        font-weight: 600 !important;
        font-size: 14px !important;
      }
      div[data-baseweb="select"] > div, div[data-baseweb="popover"], div[data-baseweb="input"] {
        background-color: transparent !important;
        border: 1px solid rgba(15,105,180,0.45) !important;
        border-radius: 6px !important;
      }
      div[data-baseweb="select"] span, div[data-baseweb="input"] input {
        color: #0f69b4 !important;
        background-color: transparent !important;
      }
      div[role="listbox"] li { color: #0f69b4 !important; background-color: white !important; }

      /* =========================
         Tabla HTML (control total)
         ========================= */
      .mma-table-wrap{
        border: 1px solid rgba(15,105,180,0.18);
        border-radius: 10px;
        overflow: auto;
        max-height: 340px;
        background: white;
      }
      .mma-table-wrap table{
        width: 100%;
        border-collapse: collapse;
        font-family: Arial, sans-serif;
        font-size: 13px;
        background: white;
      }
      .mma-table-wrap thead th{
        position: sticky;
        top: 0;
        z-index: 2;
        background: rgba(15,105,180,0.03);
        color: #0f69b4;
        font-weight: 600;
        text-align: left;
        padding: 10px 12px;
        border-bottom: 1px solid rgba(15,105,180,0.18);
        white-space: nowrap;
      }
      .mma-table-wrap tbody td{
        color: #0f69b4;
        background: white;
        padding: 8px 12px;
        border-bottom: 1px solid rgba(15,105,180,0.08);
        vertical-align: top;
      }
      .mma-table-wrap tbody tr:hover td{
        background: rgba(15,105,180,0.03);
      }
      .mma-table-wrap::-webkit-scrollbar-thumb{
        background: rgba(15,105,180,0.25);
        border-radius: 8px;
      }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# Data
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("P7 Mapa de Replicabilidad Chile - Tabla de resultados procesados.csv")

try:
    df = load_data()

    st.title("Mapa de Replicabilidad de Instrumentos Internacionales")
    st.caption("Consultoría Sustrend para la Subsecretaría del Medio Ambiente | ID: 608897-205-COT25")

    st.markdown("---")
    st.markdown(
        """
### Guía de Interpretación del Análisis

Este dashboard analiza la replicabilidad en Chile de instrumentos internacionales de gestión eficiente de recursos y economía circular.
La evaluación se basa en dos dimensiones:

- Impacto ambiental (1–5): potencial beneficio ambiental del instrumento si se implementa en Chile.
- Factibilidad en Chile (1–5): viabilidad de implementación considerando el contexto chileno actual.

Clasificación estratégica:
- 🟢 Quick Wins: alto impacto, factibilidad media-baja (implementación rápida).
- 🔵 Estratégicos: alto impacto, alta factibilidad (prioridad máxima).
- 🟡 Tácticos: bajo impacto, alta factibilidad (implementación sencilla).
        """
    )

    st.markdown("---")
    st.markdown("### Filtros de Análisis")
    st.markdown("Seleccione los países y clasificaciones que desea analizar.")

    col1, col2 = st.columns(2)
    with col1:
        filtro_pais = st.multiselect(
            "País de Origen",
            options=df["País Origen (P2)"].unique(),
            default=df["País Origen (P2)"].unique(),
        )
    with col2:
        filtro_clase = st.multiselect(
            "Clasificación Estratégica",
            options=df["Clasificación"].unique(),
            default=df["Clasificación"].unique(),
        )

    df_filtered = df[
        (df["País Origen (P2)"].isin(filtro_pais)) &
        (df["Clasificación"].isin(filtro_clase))
    ].copy()

    df_filtered["Size"] = 25
    df_filtered.loc[df_filtered["Clasificación"] == "🔵 Estratégico", "Size"] = 35
    df_filtered.loc[df_filtered["Clasificación"] == "🟢 Quick Win", "Size"] = 30

    st.markdown("---")
    st.markdown("### Análisis de Replicabilidad")

    fig = px.scatter(
        df_filtered,
        x="Score Factib. Chile",
        y="Score Impacto (1-5)",
        text="ID (P2)",
        color="Clasificación",
        hover_name="Instrumento (Nombre Original/Local)",
        hover_data={
            "País Origen (P2)": True,
            "Categoría (P2)": True,
            "KPI Principal Afectado (P5)": True,
            "Score Factib. Chile": ":.1f",
            "Score Impacto (1-5)": ":.1f"
        },
        size="Size",
        size_max=38,
        opacity=0.85,
        color_discrete_map={
            "🟢 Quick Win": "#27AE60",
            "🔵 Estratégico": "#0f69b4",
            "🟡 Táctico": "#F39C12",
            "🔴 Largo Plazo": "#E62639"
        },
        labels={
            "Score Factib. Chile": "Factibilidad en Chile (1-5)",
            "Score Impacto (1-5)": "Impacto Ambiental (1-5)"
        }
    )

    fig.add_vline(x=3, line_dash="dash", line_width=1.8, line_color="#eb3c46", opacity=0.8)
    fig.add_hline(y=3, line_dash="dash", line_width=1.8, line_color="#eb3c46", opacity=0.8)

    fig.update_traces(
        textposition="top center",
        marker=dict(line=dict(width=0.8, color="rgba(255,255,255,0.8)")),
        textfont=dict(size=9, family="Arial", color="#0f69b4"),
        hovertemplate="<b>%{hovertext}</b><br>" +
                      "Factibilidad: %{x:.1f} | Impacto: %{y:.1f}<br>" +
                      "País: %{customdata[0]}<br>" +
                      "Categoría: %{customdata[1]}<br>" +
                      "KPI: %{customdata[2]}<br>" +
                      "<extra></extra>"
    )

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(
            range=[0.5, 5.5],
            gridcolor="rgba(15, 105, 180, 0.08)",
            gridwidth=0.5,
            showline=True,
            linecolor="#0f69b4",
            linewidth=1.2,
            showgrid=True,
            zeroline=False,
            title_font=dict(size=13, family="Arial", color="#0f69b4"),
            tickfont=dict(size=11, family="Arial", color="#0f69b4"),
            tickmode="linear",
            tick0=1,
            dtick=1,
            ticks="outside",
            ticklen=4,
            tickcolor="#0f69b4",
            title_text="Factibilidad en Chile (1-5)"
        ),
        yaxis=dict(
            range=[0.5, 5.5],
            gridcolor="rgba(15, 105, 180, 0.08)",
            gridwidth=0.5,
            showline=True,
            linecolor="#0f69b4",
            linewidth=1.2,
            showgrid=True,
            zeroline=False,
            title_font=dict(size=13, family="Arial", color="#0f69b4"),
            tickfont=dict(size=11, family="Arial", color="#0f69b4"),
            tickmode="linear",
            tick0=1,
            dtick=1,
            ticks="outside",
            ticklen=4,
            tickcolor="#0f69b4",
            title_text="Impacto Ambiental (1-5)"
        ),
        legend=dict(
            title=dict(text="Clasificación", font=dict(size=12, family="Arial", color="#0f69b4")),
            font=dict(size=11, family="Arial", color="#0f69b4"),
            bordercolor="#0f69b4",
            borderwidth=0.8,
            bgcolor="white",
            x=1.02,
            xanchor="left",
            y=1,
            yanchor="top"
        ),
        hoverlabel=dict(
            bgcolor="white",
            font_size=11,
            font_family="Arial",
            font_color="#0f69b4",
            bordercolor="#0f69b4"
        ),
        margin=dict(l=60, r=60, t=40, b=60),
        height=550,
        showlegend=True
    )

    fig.add_shape(type="rect", x0=0.5, y0=3,   x1=3,   y1=5.5, fillcolor="rgba(39, 174, 96, 0.05)", line=dict(width=0), layer="below")
    fig.add_shape(type="rect", x0=3,   y0=3,   x1=5.5, y1=5.5, fillcolor="rgba(15, 105, 180, 0.05)", line=dict(width=0), layer="below")
    fig.add_shape(type="rect", x0=0.5, y0=0.5, x1=3,   y1=3,   fillcolor="rgba(243, 156, 18, 0.04)", line=dict(width=0), layer="below")

    for label in [
        dict(x=1.75, y=4.5,  text="QUICK WINS",   font=dict(size=10, family="Arial", color="#27AE60"), showarrow=False, bgcolor="white", bordercolor="#27AE60", borderwidth=0.5, borderpad=3),
        dict(x=4.25, y=4.5,  text="ESTRATÉGICOS", font=dict(size=10, family="Arial", color="#0f69b4"), showarrow=False, bgcolor="white", bordercolor="#0f69b4", borderwidth=0.5, borderpad=3),
        dict(x=4.25, y=1.75, text="TÁCTICOS",     font=dict(size=10, family="Arial", color="#F39C12"), showarrow=False, bgcolor="white", bordercolor="#F39C12", borderwidth=0.5, borderpad=3),
    ]:
        fig.add_annotation(**label)

    fig.add_annotation(
        x=3, y=5.4,
        text="Umbral Factibilidad",
        showarrow=False,
        font=dict(size=10, color="#eb3c46", family="Arial"),
        bgcolor="white",
        bordercolor="#eb3c46",
        borderwidth=0.8,
        borderpad=4,
        xanchor="center",
        yanchor="bottom"
    )
    fig.add_annotation(
        x=5.4, y=3,
        text="Umbral Impacto",
        showarrow=False,
        font=dict(size=10, color="#eb3c46", family="Arial"),
        bgcolor="white",
        bordercolor="#eb3c46",
        borderwidth=0.8,
        borderpad=4,
        xanchor="left",
        yanchor="middle"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
<div class="footnote">
  <div class="ft-title">Interpretación del gráfico</div>

  <div><span class="ft-label">Eje X:</span> Factibilidad de implementación en Chile (1 = baja, 5 = alta)</div>
  <div><span class="ft-label">Eje Y:</span> Impacto ambiental potencial (1 = bajo, 5 = alto)</div>

  <div style="margin-top:8px;"><span class="ft-label">Líneas de referencia (rojo):</span></div>
  <div>Línea vertical (3 en X): umbral mínimo de factibilidad para considerar implementación</div>
  <div>Línea horizontal (3 en Y): umbral mínimo de impacto ambiental para ser considerado relevante</div>

  <div style="margin-top:8px;"><span class="ft-label">Cuadrantes estratégicos:</span></div>
  <div>Superior derecho (🔵): Estratégicos, alta prioridad</div>
  <div>Superior izquierdo (🟢): Quick Wins, oportunidades rápidas</div>
  <div>Inferior derecho (🟡): Tácticos, implementación sencilla</div>
  <div>Inferior izquierdo: baja prioridad, revisar en el largo plazo</div>
</div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown("### Resumen de Clasificaciones")
    st.markdown("Este resumen muestra la cantidad de instrumentos por categoría estratégica según los filtros aplicados.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Quick Wins", int((df_filtered["Clasificación"] == "🟢 Quick Win").sum()))
    with col2:
        st.metric("Estratégicos", int((df_filtered["Clasificación"] == "🔵 Estratégico").sum()))
    with col3:
        st.metric("Tácticos", int((df_filtered["Clasificación"] == "🟡 Táctico").sum()))

    # =========================
    # Tabla (HTML con badge de clasificación)
    # =========================
    st.markdown("---")
    st.markdown("### Ficha Técnica de Instrumentos")
    st.markdown("Tabla con detalle de instrumentos evaluados (ordenada por impacto, descendente).")

    display_df = df_filtered[
        [
            "ID (P2)",
            "Instrumento (Nombre Original/Local)",
            "País Origen (P2)",
            "Score Factib. Chile",
            "Score Impacto (1-5)",
            "Clasificación",
        ]
    ].copy()

    display_df = display_df.sort_values("Score Impacto (1-5)", ascending=False)
    display_df = display_df.rename(columns={
        "ID (P2)": "ID",
        "Instrumento (Nombre Original/Local)": "Instrumento",
        "País Origen (P2)": "País",
        "Score Factib. Chile": "Factibilidad",
        "Score Impacto (1-5)": "Impacto",
        "Clasificación": "Clasificación",
    })

    # Badge: puntito + texto (HTML)
    display_df["Clasificación"] = display_df["Clasificación"].apply(_class_badge)

    table_html = display_df.to_html(index=False, escape=False)
    st.markdown(f"<div class='mma-table-wrap'>{table_html}</div>", unsafe_allow_html=True)

    # =========================
    # Pie de página con membrete
    # =========================
    st.markdown("---")
    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    membrete_path = find_asset("membrete.png")
    col1, col2, col3 = st.columns([3, 6, 3])
    with col1:
        if membrete_path is not None:
            st.image(str(membrete_path), width=120)
        else:
            st.markdown(
                """
<div style="font-size: 10px; color: #0f69b4; opacity: 0.65; line-height: 1.3;">
  <div style="font-weight: 600;">Gobierno de Chile</div>
  <div>Ministerio del Medio Ambiente</div>
  <div style="font-size: 9px;">Dashboard de Replicabilidad</div>
</div>
                """,
                unsafe_allow_html=True
            )

except Exception as e:
    st.error(f"Error cargando el Dashboard: {e}")
    st.warning("Verifica que el nombre del archivo CSV sea: P7 Mapa de Replicabilidad Chile - Tabla de resultados procesados.csv")
