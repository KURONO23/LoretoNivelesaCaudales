from __future__ import annotations

import base64
import html
import re
import sqlite3
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="AMARU",
    page_icon="🌊",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "Data"
OUTPUT_DIR = BASE_DIR / "outputs"
CACHE_DIR = BASE_DIR / "backend" / "cache"

GPKG_FILE = DATA_DIR / "estaciones_hidrometricas_loreto_latlong_COMID_actualizado.gpkg"

PRONOSTICO_CSV = OUTPUT_DIR / "pronostico_nivel_7dias_estaciones.csv"
DIAGNOSTICO_CSV = OUTPUT_DIR / "diagnostico_ajuste_continuidad.csv"
METRICAS_PARQUET = OUTPUT_DIR / "metricas_dwlt_estaciones.parquet"
OBS_PARQUET = CACHE_DIR / "observado_estaciones.parquet"

LOGO_FILE = BASE_DIR / "frontend" / "assets" / "logo_amaru.png"

DIAS_OBS_GRAFICO = 5
MAX_DIAS_OBS_GRAFICO = 7


# ============================================================
# ESTILOS
# ============================================================

st.markdown(
    """
    <style>
    /* ============================================================
       AJUSTES GENERALES DEL VISOR AMARU
       Estilo institucional compacto tipo dashboard técnico
    ============================================================ */

    .block-container {
        padding-top: 1.0rem !important;
        padding-bottom: 0.5rem !important;
        padding-left: 1.4rem !important;
        padding-right: 1.4rem !important;
        max-width: 100% !important;
    }

    section[data-testid="stSidebar"] {
        background: #f3f7fb;
        border-right: 1px solid #dbe4ef;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1.2rem;
    }

    /* ============================================================
       CABECERA
    ============================================================ */

    .header-box {
        display: flex;
        align-items: center;
        gap: 18px;
        margin-top: 0.1rem;
        margin-bottom: 0.8rem;
        padding: 0.15rem 0 0.25rem 0;
    }

    .header-logo {
        width: 105px;
        min-width: 105px;
        height: auto !important;
        object-fit: contain !important;
        display: block;
    }

    .header-text {
        display: flex;
        flex-direction: column;
        justify-content: center;
        min-width: 0;
    }

    .main-title {
        font-size: 2.35rem;
        font-weight: 900;
        color: #0f2f6b;
        margin-bottom: 0.05rem;
        line-height: 0.95;
        letter-spacing: 0.6px;
    }

    .subtitle {
        font-size: 1.02rem;
        color: #14532d;
        margin-bottom: 0rem;
        font-weight: 700;
        line-height: 1.15;
    }

    /* ============================================================
       TÍTULOS DE SECCIÓN
    ============================================================ */

    .section-title {
        font-size: 1.03rem;
        font-weight: 800;
        color: #0f172a;
        margin-top: 0.25rem;
        margin-bottom: 0.35rem;
        letter-spacing: 0.1px;
    }

    /* ============================================================
       MÉTRICAS / KPIs
    ============================================================ */

    .stMetric {
        background-color: #ffffff;
        border: 1px solid #dbe4ef;
        border-radius: 10px;
        padding: 10px 12px;
        box-shadow: 0px 1px 3px rgba(15, 23, 42, 0.08);
    }

    div[data-testid="stMetricLabel"] {
        font-size: 0.78rem;
        color: #334155;
        font-weight: 600;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0f172a;
    }

    div[data-testid="stMetricDelta"] {
        font-size: 0.78rem;
    }

    /* ============================================================
       SIDEBAR
    ============================================================ */

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #0f2f6b;
        font-weight: 800;
    }

    section[data-testid="stSidebar"] label {
        font-size: 0.82rem;
        font-weight: 700;
        color: #334155;
    }

    section[data-testid="stSidebar"] .stRadio,
    section[data-testid="stSidebar"] .stSelectbox {
        margin-bottom: 0.45rem;
    }

    section[data-testid="stSidebar"] hr {
        margin-top: 0.8rem;
        margin-bottom: 0.8rem;
        border-color: #cbd5e1;
    }

    section[data-testid="stSidebar"] p {
        font-size: 0.80rem;
        line-height: 1.25;
    }

    /* ============================================================
       MAPA Y GRÁFICOS
    ============================================================ */

    iframe {
        border-radius: 10px !important;
    }

    div[data-testid="stPlotlyChart"] {
        border-radius: 10px;
    }

    /* Captions más compactos */
    div[data-testid="stCaptionContainer"] {
        font-size: 0.78rem;
        color: #64748b;
        line-height: 1.25;
    }

    /* Reducir espacios verticales entre elementos */
    div[data-testid="stVerticalBlock"] {
        gap: 0.45rem;
    }

    /* Mejor ajuste de columnas */
    div[data-testid="column"] {
        padding-left: 0.25rem;
        padding-right: 0.25rem;
    }

    /* Oculta un poco el exceso visual de Streamlit */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: visible;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# UTILIDADES
# ============================================================

def normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def normalizar_texto(x) -> str:
    if pd.isna(x):
        return ""

    x = str(x).strip().upper()
    x = x.replace("Á", "A")
    x = x.replace("É", "E")
    x = x.replace("Í", "I")
    x = x.replace("Ó", "O")
    x = x.replace("Ú", "U")
    x = x.replace("Ñ", "N")
    x = re.sub(r"\s+", " ", x)

    return x


def formato_num(x, nd: int = 2) -> str:
    if x is None or pd.isna(x):
        return "Sin dato"

    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "Sin dato"


def clasificar_tendencia(delta) -> str:
    if delta is None or pd.isna(delta):
        return "Sin dato"

    if delta > 0.10:
        return "Ascendente"

    if delta < -0.10:
        return "Descendente"

    return "Estable"


def convertir_fechas(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()

    for col in [
        "fecha",
        "fecha_obs_ajuste",
        "fecha_emision_pronostico",
        "fecha_ejecucion_workflow",
    ]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "fecha" not in df.columns and "fecha_texto" in df.columns:
        df["fecha"] = pd.to_datetime(
            df["fecha_texto"],
            format="%d/%m/%Y",
            errors="coerce",
        )

    return df


def convertir_comid(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()

    if "comid" in df.columns:
        df["comid"] = pd.to_numeric(df["comid"], errors="coerce")

    return df


def convertir_numericos(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()

    for col in df.columns:
        if (
            col.startswith("nivel_")
            or col.startswith("prob_")
            or col.startswith("primer_")
            or col in [
                "offset_ajuste_m",
                "offset_aplicado",
                "dias_desde_obs_ajuste",
                "dias_desde_obs",
                "diferencia_control_m",
                "kge_2009",
                "rmse_m",
                "nse",
                "r_pearson",
            ]
        ):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ============================================================
# FUNCIONES PARA ÍCONOS DEL MAPA
# ============================================================

def tendencia_a_estado_mapa(tendencia) -> str:
    if tendencia is None or pd.isna(tendencia):
        return "sin_datos"

    t = normalizar_texto(tendencia)

    if "DESC" in t:
        return "descendiendo"

    if "ASC" in t or "SUB" in t:
        return "subiendo"

    if "ESTABLE" in t:
        return "estable"

    return "sin_datos"


def estilo_estado_mapa(estado: str) -> dict:
    estilos = {
        "descendiendo": {
            "bg": "#f59e0b",
            "label": "Descendiendo",
        },
        "estable": {
            "bg": "#16a34a",
            "label": "Estable",
        },
        "subiendo": {
            "bg": "#2563eb",
            "label": "Subiendo",
        },
        "sin_datos": {
            "bg": "#9ca3af",
            "label": "Sin datos",
        },
    }

    return estilos.get(estado, estilos["sin_datos"])


def construir_svg_estado(estado: str) -> str:
    if estado == "subiendo":
        return """
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none"
             xmlns="http://www.w3.org/2000/svg">
            <path d="M4 16 L9 11 L13 15 L20 8"
                  stroke="white" stroke-width="3.2"
                  stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M16 8 H20 V12"
                  stroke="white" stroke-width="3.2"
                  stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        """

    if estado == "descendiendo":
        return """
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none"
             xmlns="http://www.w3.org/2000/svg">
            <path d="M4 8 L9 13 L13 9 L20 16"
                  stroke="white" stroke-width="3.2"
                  stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M16 16 H20 V12"
                  stroke="white" stroke-width="3.2"
                  stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        """

    if estado == "estable":
        return """
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none"
             xmlns="http://www.w3.org/2000/svg">
            <path d="M5 12 H19"
                  stroke="white" stroke-width="3.8"
                  stroke-linecap="round"/>
        </svg>
        """

    return """
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none"
         xmlns="http://www.w3.org/2000/svg">
        <circle cx="12" cy="12" r="9"
                stroke="white" stroke-width="2.6"/>
        <path d="M9.5 9 C9.8 7.5 11 6.8 12.4 6.8 C14 6.8 15.2 7.8 15.2 9.3 C15.2 10.6 14.5 11.3 13.2 12.1 C12.4 12.6 12.1 13.1 12.1 14"
              stroke="white" stroke-width="2.4"
              stroke-linecap="round"/>
        <circle cx="12" cy="17" r="1.4" fill="white"/>
    </svg>
    """


def construir_icono_estacion(
    estado: str,
    seleccionado: bool = False,
) -> folium.DivIcon:
    estilo = estilo_estado_mapa(estado)

    bg = estilo["bg"]
    svg_icon = construir_svg_estado(estado)

    tam = 31 if seleccionado else 27
    borde = "#ef4444" if seleccionado else "#ffffff"
    grosor_borde = 3 if seleccionado else 2
    radio = 7

    html_icon = f"""
    <div style="
        width:{tam}px;
        height:{tam}px;
        background:{bg};
        border:{grosor_borde}px solid {borde};
        border-radius:{radio}px;
        display:flex;
        align-items:center;
        justify-content:center;
        box-shadow:0 2px 6px rgba(0,0,0,0.35);
        line-height:1;
    ">
        {svg_icon}
    </div>
    """

    return folium.DivIcon(
        html=html_icon,
        icon_size=(tam, tam),
        icon_anchor=(tam // 2, tam // 2),
    )


def construir_etiqueta_estacion(
    nombre: str,
    seleccionado: bool = False,
) -> folium.DivIcon:
    nombre_safe = html.escape(str(nombre))

    font_size = 11 if seleccionado else 9
    font_weight = 800 if seleccionado else 600
    color = "#111827" if seleccionado else "#374151"
    border_color = "rgba(239,68,68,0.45)" if seleccionado else "rgba(0,0,0,0.12)"
    fondo = "rgba(255,255,255,0.96)" if seleccionado else "rgba(255,255,255,0.88)"

    html_label = f"""
    <div style="
        display:inline-block;
        width:auto;
        min-width:auto;
        max-width:none;
        white-space:nowrap;
        font-size:{font_size}px;
        font-weight:{font_weight};
        color:{color};
        background:{fondo};
        padding:1px 5px;
        border-radius:4px;
        border:1px solid {border_color};
        box-shadow:0 1px 3px rgba(0,0,0,0.18);
        line-height:1.15;
        text-align:left;
        transform: translate(15px, -8px);
    ">
        {nombre_safe}
    </div>
    """

    return folium.DivIcon(
        html=html_label,
        icon_size=(1, 1),
        icon_anchor=(0, 0),
        class_name="station-label",
    )


# ============================================================
# CARGA DE ARCHIVOS
# ============================================================

@st.cache_data(show_spinner=False)
def cargar_estaciones_gpkg(gpkg_path: str) -> pd.DataFrame:
    path = Path(gpkg_path)

    if not path.exists():
        return pd.DataFrame()

    try:
        conn = sqlite3.connect(str(path))

        df = pd.read_sql_query(
            """
            SELECT
                estacion,
                COMID,
                tipo,
                departamento,
                provincia,
                distrito,
                cuenca,
                latitud,
                longitud
            FROM estaciones_latlong
            """,
            conn,
        )

        conn.close()

    except Exception as e:
        st.error(f"No se pudo leer el GPKG: {e}")
        return pd.DataFrame()

    df = normalizar_columnas(df)

    out = pd.DataFrame({
        "estacion": df["estacion"].astype(str).str.strip(),
        "estacion_norm": df["estacion"].apply(normalizar_texto),
        "comid": pd.to_numeric(df["comid"], errors="coerce"),
        "lat": pd.to_numeric(df["latitud"], errors="coerce"),
        "lon": pd.to_numeric(df["longitud"], errors="coerce"),
    })

    for col in ["tipo", "departamento", "provincia", "distrito", "cuenca"]:
        if col in df.columns:
            out[col] = df[col]

    out = out.dropna(subset=["estacion", "comid", "lat", "lon"]).copy()
    out = out[out["estacion"].str.strip() != ""].copy()

    if out.empty:
        return pd.DataFrame()

    out["comid"] = out["comid"].astype("int64")
    out = out.drop_duplicates(subset=["comid"], keep="first")
    out = out.sort_values("estacion").reset_index(drop=True)

    return out


def cargar_csv_sin_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
        df = normalizar_columnas(df)
        df = convertir_fechas(df)
        df = convertir_comid(df)
        df = convertir_numericos(df)
        return df

    except Exception as e:
        st.error(f"No se pudo leer CSV {path.name}: {e}")
        return pd.DataFrame()


def cargar_parquet_sin_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_parquet(path)
        df = normalizar_columnas(df)
        df = convertir_fechas(df)
        df = convertir_comid(df)
        df = convertir_numericos(df)
        return df

    except Exception as e:
        st.error(f"No se pudo leer Parquet {path.name}: {e}")
        return pd.DataFrame()


def cargar_historico_pronosticos() -> pd.DataFrame:
    archivos_parquet = sorted(OUTPUT_DIR.glob("historico_pronosticos_AH_*.parquet"))

    partes = []

    for path in archivos_parquet:
        try:
            tmp = pd.read_parquet(path)
            tmp = normalizar_columnas(tmp)
            tmp = convertir_fechas(tmp)
            tmp = convertir_comid(tmp)
            tmp = convertir_numericos(tmp)
            partes.append(tmp)
        except Exception as e:
            st.warning(f"No se pudo leer histórico {path.name}: {e}")

    if partes:
        hist = pd.concat(partes, ignore_index=True)
        hist = hist.drop_duplicates(
            subset=[
                "anio_hidrologico",
                "fecha_emision_pronostico",
                "estacion",
                "comid",
                "fecha",
            ],
            keep="last",
        )
        return hist

    archivos_csv = sorted(OUTPUT_DIR.glob("historico_pronosticos_AH_*.csv"))

    partes_csv = []

    for path in archivos_csv:
        try:
            tmp = pd.read_csv(path)
            tmp = normalizar_columnas(tmp)
            tmp = convertir_fechas(tmp)
            tmp = convertir_comid(tmp)
            tmp = convertir_numericos(tmp)
            partes_csv.append(tmp)
        except Exception as e:
            st.warning(f"No se pudo leer histórico {path.name}: {e}")

    if partes_csv:
        hist = pd.concat(partes_csv, ignore_index=True)
        hist = hist.drop_duplicates(
            subset=[
                "anio_hidrologico",
                "fecha_emision_pronostico",
                "estacion",
                "comid",
                "fecha",
            ],
            keep="last",
        )
        return hist

    return pd.DataFrame()


# ============================================================
# VALIDACIONES Y PROCESOS
# ============================================================

def validar_columnas_pronostico(pron: pd.DataFrame) -> None:
    requeridas = [
        "nivel_prom_ajustado_m",
        "nivel_min_ajustado_m",
        "nivel_max_ajustado_m",
        "ajuste_continuidad",
        "offset_ajuste_m",
        "nivel_obs_ajuste_m",
        "advertencia_ajuste",
    ]

    faltantes = [c for c in requeridas if c not in pron.columns]

    if faltantes:
        st.error(
            "El archivo de pronóstico no tiene las columnas necesarias. "
            "Ejecuta primero el workflow con el script 04 corregido."
        )
        st.write("Columnas faltantes:")
        st.write(faltantes)
        st.write("Columnas disponibles:")
        st.write(list(pron.columns))
        st.stop()


def validar_columnas_historico(hist: pd.DataFrame) -> bool:
    if hist.empty:
        return False

    requeridas = [
        "fecha_emision_pronostico",
        "anio_hidrologico",
        "estacion",
        "comid",
        "fecha",
        "nivel_prom_ajustado_m",
    ]

    faltantes = [c for c in requeridas if c not in hist.columns]

    if faltantes:
        st.warning(f"El histórico existe, pero le faltan columnas: {faltantes}")
        return False

    return True


def preparar_resumen_pronostico(pron: pd.DataFrame) -> pd.DataFrame:
    if pron.empty:
        return pd.DataFrame()

    resumen = (
        pron.groupby(["estacion", "comid"], dropna=False)
        .agg(
            fecha_inicio=("fecha", "min"),
            fecha_fin=("fecha", "max"),
            nivel_min_7dias=("nivel_min_ajustado_m", "min"),
            nivel_prom_7dias=("nivel_prom_ajustado_m", "mean"),
            nivel_max_7dias=("nivel_max_ajustado_m", "max"),
            nivel_inicio=("nivel_prom_ajustado_m", "first"),
            nivel_fin=("nivel_prom_ajustado_m", "last"),
            ajuste_continuidad=("ajuste_continuidad", "first"),
            offset_ajuste_m=("offset_ajuste_m", "first"),
            nivel_obs_ajuste_m=("nivel_obs_ajuste_m", "first"),
            advertencia_ajuste=("advertencia_ajuste", "first"),
        )
        .reset_index()
    )

    resumen["tendencia_7dias_m"] = resumen["nivel_fin"] - resumen["nivel_inicio"]
    resumen["tendencia"] = resumen["tendencia_7dias_m"].apply(clasificar_tendencia)

    return resumen


def observado_diario(obs_est: pd.DataFrame) -> pd.DataFrame:
    if obs_est.empty or "fecha" not in obs_est.columns or "nivel_m" not in obs_est.columns:
        return pd.DataFrame()

    obs_est = obs_est.dropna(subset=["fecha", "nivel_m"]).copy()
    obs_est = obs_est.sort_values("fecha")

    return (
        obs_est.groupby("fecha", dropna=False)
        .agg(nivel_m=("nivel_m", "mean"))
        .reset_index()
        .sort_values("fecha")
    )


def obtener_observado_reciente_para_grafico(
    obs_est: pd.DataFrame,
    fecha_inicio_pronostico,
    dias_obs: int = DIAS_OBS_GRAFICO,
    max_dias_antiguedad: int = MAX_DIAS_OBS_GRAFICO,
) -> tuple[pd.DataFrame, bool, str]:
    if obs_est.empty or "fecha" not in obs_est.columns or "nivel_m" not in obs_est.columns:
        return pd.DataFrame(), False, "Sin datos observados"

    if fecha_inicio_pronostico is None or pd.isna(fecha_inicio_pronostico):
        return pd.DataFrame(), False, "Sin fecha de inicio de pronóstico"

    obs_d = observado_diario(obs_est)

    obs_prev = obs_d[obs_d["fecha"] <= fecha_inicio_pronostico].copy()

    if obs_prev.empty:
        return pd.DataFrame(), False, "Sin observado previo al pronóstico"

    fecha_ult_obs = obs_prev["fecha"].max()
    dias_desde_obs = int((fecha_inicio_pronostico - fecha_ult_obs).days)

    if dias_desde_obs > max_dias_antiguedad:
        msg = f"Observado no graficado: último dato hace {dias_desde_obs} días"
        return pd.DataFrame(), False, msg

    fecha_min = fecha_inicio_pronostico - pd.Timedelta(days=dias_obs)

    obs_plot = obs_prev[
        (obs_prev["fecha"] >= fecha_min)
        & (obs_prev["fecha"] <= fecha_inicio_pronostico)
    ].copy()

    if obs_plot.empty:
        obs_plot = obs_prev.tail(1).copy()

    msg = f"Observado graficado: {len(obs_plot)} dato(s), último hace {dias_desde_obs} día(s)"

    return obs_plot, True, msg


def calcular_metricas_validacion(df_comp: pd.DataFrame) -> dict:
    if df_comp.empty:
        return {
            "n": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "bias": np.nan,
        }

    tmp = df_comp.dropna(subset=["nivel_m", "nivel_prom_ajustado_m"]).copy()

    if tmp.empty:
        return {
            "n": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "bias": np.nan,
        }

    error = tmp["nivel_prom_ajustado_m"] - tmp["nivel_m"]

    return {
        "n": len(tmp),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "bias": float(np.mean(error)),
    }


# ============================================================
# MAPA
# ============================================================

def crear_mapa_estaciones(
    estaciones: pd.DataFrame,
    estacion_sel: str,
    comid_sel,
    pron_resumen: pd.DataFrame,
) -> folium.Map:
    if estaciones.empty:
        mapa = folium.Map(
            location=[-5.0, -74.5],
            zoom_start=6,
            tiles="CartoDB positron",
            control_scale=True,
        )
    else:
        mapa = folium.Map(
            location=[estaciones["lat"].mean(), estaciones["lon"].mean()],
            zoom_start=6,
            tiles="CartoDB positron",
            control_scale=True,
        )

    if not estaciones.empty:
        for _, row in estaciones.iterrows():
            estacion = str(row.get("estacion", "SIN_NOMBRE"))
            comid = row.get("comid", None)
            lat = row.get("lat", None)
            lon = row.get("lon", None)

            if pd.isna(lat) or pd.isna(lon) or pd.isna(comid):
                continue

            comid_int = int(comid)
            seleccionado = comid_sel is not None and comid_int == int(comid_sel)

            info_extra = ""
            tendencia = "Sin dato"
            estado_mapa = "sin_datos"

            if not pron_resumen.empty and "comid" in pron_resumen.columns:
                tmp = pron_resumen[
                    pd.to_numeric(pron_resumen["comid"], errors="coerce") == float(comid_int)
                ]

                if not tmp.empty:
                    nivel_prom = tmp["nivel_prom_7dias"].iloc[0]
                    tendencia = tmp["tendencia"].iloc[0]
                    ajuste = tmp["ajuste_continuidad"].iloc[0]
                    offset = tmp["offset_ajuste_m"].iloc[0]
                    estado_mapa = tendencia_a_estado_mapa(tendencia)

                    info_extra = f"""
                    <br><b>Nivel prom. ajustado 7 días:</b> {formato_num(nivel_prom)} m
                    <br><b>Tendencia:</b> {tendencia}
                    <br><b>Ajuste:</b> {ajuste}
                    <br><b>Offset:</b> {formato_num(offset, 3)} m
                    """

            estilo = estilo_estado_mapa(estado_mapa)

            popup_html = f"""
            <b>{html.escape(estacion)}</b><br>
            <b>COMID:</b> {comid_int}
            <br><b>Estado:</b> {estilo["label"]}
            {info_extra}
            <br><br><i>Haz click para seleccionar esta estación.</i>
            """

            tooltip_val = f"{estacion}||{comid_int}"

            folium.Marker(
                location=[lat, lon],
                icon=construir_icono_estacion(
                    estado=estado_mapa,
                    seleccionado=seleccionado,
                ),
                popup=folium.Popup(popup_html, max_width=320),
                tooltip=tooltip_val,
            ).add_to(mapa)

            folium.Marker(
                location=[lat, lon],
                icon=construir_etiqueta_estacion(
                    nombre=estacion,
                    seleccionado=seleccionado,
                ),
                interactive=False,
            ).add_to(mapa)

    return mapa


# ============================================================
# GRÁFICOS
# ============================================================

def graficar_pronostico_actual(
    pron_est: pd.DataFrame,
    obs_est: pd.DataFrame,
) -> None:
    fecha_inicio_pron = None

    if not pron_est.empty and "fecha" in pron_est.columns:
        fecha_inicio_pron = pron_est["fecha"].min()

    obs_plot, mostrar_obs, mensaje_obs = obtener_observado_reciente_para_grafico(
        obs_est=obs_est,
        fecha_inicio_pronostico=fecha_inicio_pron,
    )

    pron_plot = pron_est.copy()

    if mostrar_obs and not obs_plot.empty:
        fecha_ult_obs = obs_plot["fecha"].max()
        pron_plot = pron_plot[pron_plot["fecha"] > fecha_ult_obs].copy()

    if mostrar_obs and not obs_plot.empty:
        st.caption(
            f"Última observación graficada: "
            f"{pd.to_datetime(obs_plot['fecha'].iloc[-1]).strftime('%d/%m/%Y')} | {mensaje_obs}"
        )
    else:
        st.caption(mensaje_obs)

    st.markdown(
        '<div class="section-title">Nivel observado + pronóstico ajustado de 7 días</div>',
        unsafe_allow_html=True,
    )

    fig = go.Figure()

    # ------------------------------------------------------------
    # 1. Observado reciente
    # ------------------------------------------------------------
    if mostrar_obs and not obs_plot.empty:
        fig.add_trace(
            go.Scatter(
                x=obs_plot["fecha"],
                y=obs_plot["nivel_m"],
                mode="lines+markers",
                name="Observado reciente",
                line=dict(width=3, color="black"),
                marker=dict(size=8, color="black"),
            )
        )

    # ------------------------------------------------------------
    # 2. Banda ajustada min-max
    # ------------------------------------------------------------
    if (
        not pron_plot.empty
        and "nivel_min_ajustado_m" in pron_plot.columns
        and "nivel_max_ajustado_m" in pron_plot.columns
    ):
        fig.add_trace(
            go.Scatter(
                x=pron_plot["fecha"],
                y=pron_plot["nivel_max_ajustado_m"],
                mode="lines",
                name="Máximo ajustado",
                line=dict(width=0, color="rgba(59, 130, 246, 0)"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=pron_plot["fecha"],
                y=pron_plot["nivel_min_ajustado_m"],
                mode="lines",
                name="Rango ajustado min–max",
                fill="tonexty",
                fillcolor="rgba(59, 130, 246, 0.18)",
                line=dict(width=0, color="rgba(59, 130, 246, 0)"),
                hoverinfo="skip",
            )
        )

    # ------------------------------------------------------------
    # 3. Línea punteada de continuidad observado-pronóstico
    # ------------------------------------------------------------
    # Esta línea no cambia los datos. Solo une visualmente el último
    # observado con el primer pronóstico ajustado.
    col_pron_union = None

    for c in [
        "nivel_prom_ajustado_m",
        "nivel_eta_eqm_ajustado_m",
        "nivel_eta_scal_ajustado_m",
        "nivel_gfs_ajustado_m",
        "nivel_wrf_ajustado_m",
    ]:
        if c in pron_plot.columns:
            col_pron_union = c
            break

    if (
        mostrar_obs
        and obs_plot is not None
        and pron_plot is not None
        and not obs_plot.empty
        and not pron_plot.empty
        and "fecha" in obs_plot.columns
        and "fecha" in pron_plot.columns
        and "nivel_m" in obs_plot.columns
        and col_pron_union is not None
    ):
        obs_tmp = obs_plot.dropna(subset=["fecha", "nivel_m"]).copy()
        pron_tmp = pron_plot.dropna(subset=["fecha", col_pron_union]).copy()

        if not obs_tmp.empty and not pron_tmp.empty:
            obs_tmp = obs_tmp.sort_values("fecha")
            pron_tmp = pron_tmp.sort_values("fecha")

            ultimo_obs = obs_tmp.iloc[-1]
            primer_pron = pron_tmp.iloc[0]

            fig.add_trace(
                go.Scatter(
                    x=[
                        ultimo_obs["fecha"],
                        primer_pron["fecha"],
                    ],
                    y=[
                        ultimo_obs["nivel_m"],
                        primer_pron[col_pron_union],
                    ],
                    mode="lines",
                    name="Continuidad observado-pronóstico",
                    line=dict(
                        width=2,
                        color="black",
                        dash="dot",
                    ),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    # ------------------------------------------------------------
    # 4. Modelos individuales ajustados
    # ------------------------------------------------------------
    modelos_ajustados = [
        ("nivel_eta_eqm_ajustado_m", "ETA_eqm ajustado", "dot"),
        ("nivel_eta_scal_ajustado_m", "ETA_scal ajustado", "dash"),
        ("nivel_gfs_ajustado_m", "GFS ajustado", "dashdot"),
        ("nivel_wrf_ajustado_m", "WRF ajustado", "longdash"),
    ]

    for col, nombre, estilo in modelos_ajustados:
        if not pron_plot.empty and col in pron_plot.columns:
            fig.add_trace(
                go.Scatter(
                    x=pron_plot["fecha"],
                    y=pron_plot[col],
                    mode="lines+markers",
                    name=nombre,
                    line=dict(width=1.5, dash=estilo),
                    marker=dict(size=5),
                    visible="legendonly",
                )
            )

    # ------------------------------------------------------------
    # 5. Media ajustada de modelos
    # ------------------------------------------------------------
    if not pron_plot.empty and "nivel_prom_ajustado_m" in pron_plot.columns:
        fig.add_trace(
            go.Scatter(
                x=pron_plot["fecha"],
                y=pron_plot["nivel_prom_ajustado_m"],
                mode="lines+markers",
                name="Media ajustada de modelos",
                line=dict(width=4, color="blue"),
                marker=dict(size=9, color="blue"),
            )
        )

    # ------------------------------------------------------------
    # 6. Línea vertical de referencia
    # ------------------------------------------------------------
    fecha_referencia = None
    texto_referencia = "Último observado"

    if mostrar_obs and not obs_plot.empty:
        fecha_referencia = pd.to_datetime(obs_plot["fecha"].iloc[-1])
    elif fecha_inicio_pron is not None and pd.notna(fecha_inicio_pron):
        fecha_referencia = pd.to_datetime(fecha_inicio_pron)
        texto_referencia = "Inicio pronóstico"

    if fecha_referencia is not None and pd.notna(fecha_referencia):
        fig.add_shape(
            type="line",
            x0=fecha_referencia,
            x1=fecha_referencia,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(color="gray", width=2, dash="dash"),
        )

        fig.add_annotation(
            x=fecha_referencia,
            y=1,
            xref="x",
            yref="paper",
            text=texto_referencia,
            showarrow=False,
            yanchor="bottom",
            font=dict(size=11, color="gray"),
        )

    fig.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Fecha",
        yaxis_title="Nivel del río (m)",
        legend_title="Serie",
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "La línea negra representa el nivel observado reciente. "
        "La línea punteada negra une visualmente el último observado con el primer pronóstico ajustado. "
        "La línea azul representa la media ajustada de los modelos. "
        "La banda azul clara representa el rango ajustado min–max entre modelos. "
        "Los modelos individuales ajustados pueden activarse desde la leyenda."
    )


def graficar_validacion_historica(
    hist_est: pd.DataFrame,
    obs_est: pd.DataFrame,
) -> None:
    if hist_est.empty:
        st.info("No hay histórico de pronósticos para esta estación.")
        return

    fechas_disp = (
        hist_est["fecha_emision_pronostico"]
        .dropna()
        .dt.normalize()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    if not fechas_disp:
        st.info("No hay fechas de emisión disponibles para esta estación.")
        return

    fechas_disp_date = [f.date() for f in fechas_disp]

    fecha_sel = st.selectbox(
        "Fecha de emisión del pronóstico histórico",
        fechas_disp_date,
        index=len(fechas_disp_date) - 1,
    )

    fecha_sel_ts = pd.to_datetime(fecha_sel)

    pron_hist = hist_est[
        hist_est["fecha_emision_pronostico"].dt.normalize() == fecha_sel_ts
    ].copy()

    pron_hist = pron_hist.sort_values("fecha").copy()

    if pron_hist.empty:
        st.warning("No hay pronóstico histórico para la fecha seleccionada.")
        return

    fecha_ini = fecha_sel_ts - pd.Timedelta(days=5)
    fecha_fin = pron_hist["fecha"].max()

    obs_d = observado_diario(obs_est)

    obs_plot = obs_d[
        (obs_d["fecha"] >= fecha_ini)
        & (obs_d["fecha"] <= fecha_fin)
    ].copy()

    obs_val = obs_d[
        (obs_d["fecha"] >= pron_hist["fecha"].min())
        & (obs_d["fecha"] <= fecha_fin)
    ].copy()

    comp = pron_hist.merge(
        obs_val,
        on="fecha",
        how="left",
    )

    metricas = calcular_metricas_validacion(comp)

    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Días comparados", str(metricas["n"]))
    k2.metric("MAE ajustado", f"{formato_num(metricas['mae'], 3)} m")
    k3.metric("RMSE ajustado", f"{formato_num(metricas['rmse'], 3)} m")
    k4.metric("BIAS ajustado", f"{formato_num(metricas['bias'], 3)} m")

    st.markdown(
        '<div class="section-title">Validación histórica: observado real vs pronóstico ajustado emitido</div>',
        unsafe_allow_html=True,
    )

    fig = go.Figure()

    if not obs_plot.empty:
        fig.add_trace(
            go.Scatter(
                x=obs_plot["fecha"],
                y=obs_plot["nivel_m"],
                mode="lines+markers",
                name="Observado real",
                line=dict(width=3, color="black"),
                marker=dict(size=8, color="black"),
            )
        )

    if (
        "nivel_min_ajustado_m" in pron_hist.columns
        and "nivel_max_ajustado_m" in pron_hist.columns
    ):
        fig.add_trace(
            go.Scatter(
                x=pron_hist["fecha"],
                y=pron_hist["nivel_max_ajustado_m"],
                mode="lines",
                name="Máximo ajustado emitido",
                line=dict(width=0, color="rgba(59, 130, 246, 0)"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=pron_hist["fecha"],
                y=pron_hist["nivel_min_ajustado_m"],
                mode="lines",
                name="Rango ajustado min–max emitido",
                fill="tonexty",
                fillcolor="rgba(59, 130, 246, 0.18)",
                line=dict(width=0, color="rgba(59, 130, 246, 0)"),
                hoverinfo="skip",
            )
        )

    modelos_ajustados = [
        ("nivel_eta_eqm_ajustado_m", "ETA_eqm ajustado emitido", "dot"),
        ("nivel_eta_scal_ajustado_m", "ETA_scal ajustado emitido", "dash"),
        ("nivel_gfs_ajustado_m", "GFS ajustado emitido", "dashdot"),
        ("nivel_wrf_ajustado_m", "WRF ajustado emitido", "longdash"),
    ]

    for col, nombre, estilo in modelos_ajustados:
        if col in pron_hist.columns:
            fig.add_trace(
                go.Scatter(
                    x=pron_hist["fecha"],
                    y=pron_hist[col],
                    mode="lines+markers",
                    name=nombre,
                    line=dict(width=1.5, dash=estilo),
                    marker=dict(size=5),
                    visible="legendonly",
                )
            )

    if "nivel_prom_ajustado_m" in pron_hist.columns:
        fig.add_trace(
            go.Scatter(
                x=pron_hist["fecha"],
                y=pron_hist["nivel_prom_ajustado_m"],
                mode="lines+markers",
                name="Media ajustada emitida",
                line=dict(width=4, color="blue"),
                marker=dict(size=9, color="blue"),
            )
        )

    fig.add_shape(
        type="line",
        x0=fecha_sel_ts,
        x1=fecha_sel_ts,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line=dict(color="gray", width=2, dash="dash"),
    )

    fig.add_annotation(
        x=fecha_sel_ts,
        y=1,
        xref="x",
        yref="paper",
        text="Fecha emisión",
        showarrow=False,
        yanchor="bottom",
        font=dict(size=11, color="gray"),
    )

    fig.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Fecha",
        yaxis_title="Nivel del río (m)",
        legend_title="Serie",
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "La línea negra representa el nivel observado real. "
        "La línea azul representa la media ajustada del pronóstico emitido. "
        "La banda azul clara representa el rango ajustado min–max entre modelos. "
        "Los modelos individuales ajustados pueden activarse desde la leyenda."
    )


# ============================================================
# CARGA PRINCIPAL
# ============================================================

estaciones = cargar_estaciones_gpkg(str(GPKG_FILE))
pron = cargar_csv_sin_cache(PRONOSTICO_CSV)
diagnostico = cargar_csv_sin_cache(DIAGNOSTICO_CSV)
metricas = cargar_parquet_sin_cache(METRICAS_PARQUET)
obs = cargar_parquet_sin_cache(OBS_PARQUET)
historico = cargar_historico_pronosticos()

if "nivel_m" in obs.columns:
    obs["nivel_m"] = pd.to_numeric(obs["nivel_m"], errors="coerce")


# ============================================================
# CABECERA CON LOGO
# ============================================================

if LOGO_FILE.exists():
    logo_base64 = base64.b64encode(LOGO_FILE.read_bytes()).decode("utf-8")

    st.markdown(
        f"""
        <div class="header-box">
            <div>
                <img src="data:image/png;base64,{logo_base64}" class="header-logo">
            </div>
            <div class="header-text">
                <div class="main-title">AMARU</div>
                <div class="subtitle">
                    Análisis, Monitoreo y Pronóstico de niveles de ríos Amazónicos peruanos
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <div class="header-box">
            <div class="header-text">
                <div class="main-title">AMARU</div>
                <div class="subtitle">
                    Análisis, Monitoreo y Pronóstico de niveles de ríos Amazónicos peruanos
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# VALIDACIÓN
# ============================================================

faltantes = []

if estaciones.empty:
    faltantes.append("Data/estaciones_hidrometricas_loreto_latlong_COMID_actualizado.gpkg")

if pron.empty:
    faltantes.append("outputs/pronostico_nivel_7dias_estaciones.csv")

if obs.empty:
    faltantes.append("backend/cache/observado_estaciones.parquet")

if diagnostico.empty:
    faltantes.append("outputs/diagnostico_ajuste_continuidad.csv")

if faltantes:
    st.error("Faltan archivos necesarios para el visor:")
    for f in faltantes:
        st.write(f"- `{f}`")
    st.stop()

validar_columnas_pronostico(pron)

historico_ok = validar_columnas_historico(historico)

pron_resumen = preparar_resumen_pronostico(pron)


# ============================================================
# ESTACIONES
# ============================================================

estaciones_disponibles = sorted(pron["estacion"].dropna().astype(str).unique().tolist())

if not estaciones_disponibles:
    st.error("No hay estaciones disponibles en el archivo de pronóstico.")
    st.stop()

tabla_estaciones_pron = (
    pron[["estacion", "comid"]]
    .dropna()
    .drop_duplicates(subset=["estacion"])
    .copy()
)

tabla_estaciones_pron["comid"] = pd.to_numeric(tabla_estaciones_pron["comid"], errors="coerce")
tabla_estaciones_pron = tabla_estaciones_pron.dropna(subset=["comid"]).copy()
tabla_estaciones_pron["comid"] = tabla_estaciones_pron["comid"].astype("int64")

estacion_a_comid = dict(zip(tabla_estaciones_pron["estacion"], tabla_estaciones_pron["comid"]))
comid_a_estacion = dict(zip(tabla_estaciones_pron["comid"], tabla_estaciones_pron["estacion"]))

if "estacion_sel" not in st.session_state:
    st.session_state.estacion_sel = estaciones_disponibles[0]

if "ultimo_click_comid" not in st.session_state:
    st.session_state.ultimo_click_comid = None

if st.session_state.estacion_sel not in estaciones_disponibles:
    st.session_state.estacion_sel = estaciones_disponibles[0]


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("Filtros")

    modo = st.radio(
        "Modo de visualización",
        [
            "Pronóstico actual",
            "Validación histórica",
        ],
    )

    index_actual = estaciones_disponibles.index(st.session_state.estacion_sel)

    estacion_sidebar = st.selectbox(
        "Estación",
        estaciones_disponibles,
        index=index_actual,
        key="selector_estacion_sidebar",
    )

    st.session_state.estacion_sel = estacion_sidebar

    st.markdown("---")
    st.subheader("Archivos cargados")

    checks = {
        "GPKG estaciones": not estaciones.empty,
        "pronóstico actual CSV": not pron.empty,
        "diagnóstico ajuste CSV": not diagnostico.empty,
        "métricas Parquet": not metricas.empty,
        "observado Parquet": not obs.empty,
        "histórico pronósticos AH": historico_ok,
    }

    for nombre, ok in checks.items():
        st.write(("✅ " if ok else "❌ ") + nombre)

    if modo == "Validación histórica" and not historico_ok:
        st.warning("Aún no existe histórico. Ejecuta primero el workflow con el script 04 actualizado.")

    st.caption("Puedes seleccionar estaciones desde el selector o haciendo click en el mapa.")


# ============================================================
# FILTRO POR ESTACIÓN
# ============================================================

estacion_sel = st.session_state.estacion_sel

pron_est = pron[pron["estacion"] == estacion_sel].copy()
pron_est = pron_est.sort_values("fecha").copy()

comid_sel = None

if estacion_sel in estacion_a_comid:
    comid_sel = estacion_a_comid[estacion_sel]
elif not pron_est.empty and "comid" in pron_est.columns:
    valores_comid = pron_est["comid"].dropna()
    if not valores_comid.empty:
        comid_sel = int(valores_comid.iloc[0])

obs_est = pd.DataFrame()
estacion_map_est = pd.DataFrame()
hist_est = pd.DataFrame()

if comid_sel is not None:
    obs_est = (
        obs[pd.to_numeric(obs["comid"], errors="coerce") == float(comid_sel)].copy()
        if not obs.empty and "comid" in obs.columns
        else pd.DataFrame()
    )

    estacion_map_est = (
        estaciones[estaciones["comid"] == int(comid_sel)].copy()
        if not estaciones.empty and "comid" in estaciones.columns
        else pd.DataFrame()
    )

    hist_est = (
        historico[pd.to_numeric(historico["comid"], errors="coerce") == float(comid_sel)].copy()
        if historico_ok and "comid" in historico.columns
        else pd.DataFrame()
    )


# ============================================================
# LAYOUT
# ============================================================

col_mapa, col_panel = st.columns([0.95, 2.05], gap="large")


# ============================================================
# MAPA
# ============================================================

with col_mapa:
    st.markdown('<div class="section-title">Mapa de estaciones</div>', unsafe_allow_html=True)

    mapa = crear_mapa_estaciones(
        estaciones=estaciones,
        estacion_sel=estacion_sel,
        comid_sel=comid_sel,
        pron_resumen=pron_resumen,
    )

    mapa_evento = st_folium(
        mapa,
        width=520,
        height=560,
        key="mapa_estaciones_click",
        returned_objects=["last_object_clicked_tooltip"],
    )

    tooltip_click = None

    if mapa_evento:
        tooltip_click = mapa_evento.get("last_object_clicked_tooltip")

    if tooltip_click and "||" in tooltip_click:
        _, comid_click_txt = tooltip_click.split("||", 1)

        try:
            comid_click = int(float(comid_click_txt))

            if (
                comid_click in comid_a_estacion
                and comid_click != st.session_state.ultimo_click_comid
            ):
                nueva_estacion = comid_a_estacion[comid_click]

                st.session_state.ultimo_click_comid = comid_click
                st.session_state.estacion_sel = nueva_estacion

                st.rerun()

        except Exception:
            pass



# ============================================================
# PANEL PRINCIPAL
# ============================================================

with col_panel:
    st.markdown(
        f'<div class="section-title">Estación: {estacion_sel}</div>',
        unsafe_allow_html=True,
    )

    if modo == "Pronóstico actual":
        nivel_obs_reciente = None

        obs_d = observado_diario(obs_est)

        if not obs_d.empty:
            nivel_obs_reciente = obs_d["nivel_m"].iloc[-1]

        nivel_inicio = None
        nivel_fin = None
        tendencia_val = None

        if not pron_est.empty:
            serie_prom = pron_est["nivel_prom_ajustado_m"].dropna()

            if not serie_prom.empty:
                nivel_inicio = serie_prom.iloc[0]
                nivel_fin = serie_prom.iloc[-1]
                tendencia_val = nivel_fin - nivel_inicio

        k1, k2, k3 = st.columns(3)

        k1.metric(
            "Nivel observado reciente",
            f"{formato_num(nivel_obs_reciente)} m",
        )

        k2.metric(
            "Nivel pronosticado ajustado",
            f"{formato_num(nivel_fin)} m",
        )

        k3.metric(
            "Tendencia esperada",
            clasificar_tendencia(tendencia_val),
            delta=f"{formato_num(tendencia_val)} m" if tendencia_val is not None else None,
        )

        graficar_pronostico_actual(
            pron_est=pron_est,
            obs_est=obs_est,
        )

    else:
        st.info(
            "Modo de validación histórica: selecciona una fecha de emisión para comparar "
            "el pronóstico guardado contra el nivel observado real."
        )

        if not historico_ok:
            st.warning("Todavía no existe archivo histórico. Ejecuta el workflow actualizado.")
        else:
            graficar_validacion_historica(
                hist_est=hist_est,
                obs_est=obs_est,
            )
