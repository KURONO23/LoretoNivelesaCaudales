from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import folium
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="Visor de Niveles Loreto",
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

# Observado para gráfico.
# Solo se grafica observado si el último dato está cerca del inicio del pronóstico.
DIAS_OBS_GRAFICO = 5
MAX_DIAS_OBS_GRAFICO = 7


# ============================================================
# ESTILOS
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0rem;
    }

    .subtitle {
        font-size: 0.95rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }

    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0f172a;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }

    .stMetric {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 14px;
        box-shadow: 0px 1px 3px rgba(15, 23, 42, 0.08);
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.35rem;
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


def clasificar_calidad(kge) -> str:
    if kge is None or pd.isna(kge):
        return "Sin evaluar"

    if kge >= 0.75:
        return "Buena"

    if kge >= 0.50:
        return "Moderada"

    return "Revisar"


def convertir_fechas(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()

    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    elif "fecha_texto" in df.columns:
        df["fecha"] = pd.to_datetime(
            df["fecha_texto"],
            format="%d/%m/%Y",
            errors="coerce",
        )

    if "fecha_obs_ajuste" in df.columns:
        df["fecha_obs_ajuste"] = pd.to_datetime(df["fecha_obs_ajuste"], errors="coerce")
    elif "fecha_obs_ajuste_texto" in df.columns:
        df["fecha_obs_ajuste"] = pd.to_datetime(
            df["fecha_obs_ajuste_texto"],
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


# ============================================================
# VALIDACIONES Y PROCESOS
# ============================================================

def validar_columnas_ajustadas(pron: pd.DataFrame) -> None:
    requeridas = [
        "nivel_prom_ajustado_m",
        "nivel_min_ajustado_m",
        "nivel_max_ajustado_m",
        "nivel_prom_m",
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


def obtener_observado_reciente_para_grafico(
    obs_est: pd.DataFrame,
    fecha_inicio_pronostico,
    dias_obs: int = DIAS_OBS_GRAFICO,
    max_dias_antiguedad: int = MAX_DIAS_OBS_GRAFICO,
) -> tuple[pd.DataFrame, bool, str]:
    """
    Devuelve observado reciente para graficar.

    Reglas:
    - Solo usa observado antes del inicio del pronóstico.
    - Si el último observado antes del pronóstico está muy antiguo, no grafica observado.
    - Si hay menos de 5 días disponibles dentro de la ventana, grafica solo lo disponible.
    """

    if obs_est.empty or "fecha" not in obs_est.columns or "nivel_m" not in obs_est.columns:
        return pd.DataFrame(), False, "Sin datos observados"

    if fecha_inicio_pronostico is None or pd.isna(fecha_inicio_pronostico):
        return pd.DataFrame(), False, "Sin fecha de inicio de pronóstico"

    obs_est = obs_est.dropna(subset=["fecha", "nivel_m"]).copy()
    obs_est = obs_est.sort_values("fecha")

    obs_diario = (
        obs_est.groupby("fecha", dropna=False)
        .agg(nivel_m=("nivel_m", "mean"))
        .reset_index()
        .sort_values("fecha")
    )

    obs_prev = obs_diario[obs_diario["fecha"] < fecha_inicio_pronostico].copy()

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
        & (obs_prev["fecha"] < fecha_inicio_pronostico)
    ].copy()

    if obs_plot.empty:
        obs_plot = obs_prev.tail(1).copy()

    msg = f"Observado graficado: {len(obs_plot)} dato(s), último hace {dias_desde_obs} día(s)"

    return obs_plot, True, msg


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
        return folium.Map(
            location=[-5.0, -74.5],
            zoom_start=6,
            tiles="CartoDB positron",
            control_scale=True,
        )

    mapa = folium.Map(
        location=[estaciones["lat"].mean(), estaciones["lon"].mean()],
        zoom_start=6,
        tiles="CartoDB positron",
        control_scale=True,
    )

    for _, row in estaciones.iterrows():
        estacion = str(row.get("estacion", "SIN_NOMBRE"))
        comid = row.get("comid", None)
        lat = row.get("lat", None)
        lon = row.get("lon", None)

        if pd.isna(lat) or pd.isna(lon) or pd.isna(comid):
            continue

        comid_int = int(comid)
        seleccionado = comid_sel is not None and comid_int == int(comid_sel)

        color = "red" if seleccionado else "blue"
        radio = 9 if seleccionado else 6

        info_extra = ""

        if not pron_resumen.empty and "comid" in pron_resumen.columns:
            tmp = pron_resumen[
                pd.to_numeric(pron_resumen["comid"], errors="coerce") == float(comid_int)
            ]

            if not tmp.empty:
                nivel_prom = tmp["nivel_prom_7dias"].iloc[0]
                tendencia = tmp["tendencia"].iloc[0]
                ajuste = tmp["ajuste_continuidad"].iloc[0]
                offset = tmp["offset_ajuste_m"].iloc[0]

                info_extra = f"""
                <br><b>Nivel prom. ajustado 7 días:</b> {formato_num(nivel_prom)} m
                <br><b>Tendencia:</b> {tendencia}
                <br><b>Ajuste:</b> {ajuste}
                <br><b>Offset:</b> {formato_num(offset, 3)} m
                """

        popup_html = f"""
        <b>{estacion}</b><br>
        <b>COMID:</b> {comid_int}
        {info_extra}
        <br><br><i>Haz click para seleccionar esta estación.</i>
        """

        tooltip_val = f"{estacion}||{comid_int}"

        folium.CircleMarker(
            location=[lat, lon],
            radius=radio,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=tooltip_val,
            color=color,
            fill=True,
            fill_opacity=0.85,
        ).add_to(mapa)

    return mapa


# ============================================================
# CARGA PRINCIPAL
# ============================================================

estaciones = cargar_estaciones_gpkg(str(GPKG_FILE))
pron = cargar_csv_sin_cache(PRONOSTICO_CSV)
diagnostico = cargar_csv_sin_cache(DIAGNOSTICO_CSV)
metricas = cargar_parquet_sin_cache(METRICAS_PARQUET)
obs = cargar_parquet_sin_cache(OBS_PARQUET)

if "nivel_m" in obs.columns:
    obs["nivel_m"] = pd.to_numeric(obs["nivel_m"], errors="coerce")


# ============================================================
# CABECERA
# ============================================================

st.markdown(
    '<div class="main-title">🌊 Visor de niveles y pronóstico hidrológico - Loreto</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">Sistema automático SONICS + HidroMet + DWLT ajustado por continuidad</div>',
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

validar_columnas_ajustadas(pron)

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
        "pronóstico ajustado CSV": not pron.empty,
        "diagnóstico ajuste CSV": not diagnostico.empty,
        "métricas Parquet": not metricas.empty,
        "observado Parquet": not obs.empty,
    }

    for nombre, ok in checks.items():
        st.write(("✅ " if ok else "❌ ") + nombre)

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

metricas_est = pd.DataFrame()
obs_est = pd.DataFrame()
estacion_map_est = pd.DataFrame()

if comid_sel is not None:
    metricas_est = (
        metricas[pd.to_numeric(metricas["comid"], errors="coerce") == float(comid_sel)].copy()
        if not metricas.empty and "comid" in metricas.columns
        else pd.DataFrame()
    )

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

fecha_inicio_pron = None

if not pron_est.empty and "fecha" in pron_est.columns:
    fecha_inicio_pron = pron_est["fecha"].min()

obs_plot, mostrar_obs, mensaje_obs = obtener_observado_reciente_para_grafico(
    obs_est=obs_est,
    fecha_inicio_pronostico=fecha_inicio_pron,
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
        height=650,
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

    st.markdown("**Leyenda**")
    st.markdown(
        """
        🔴 Estación seleccionada  
        🔵 Otras estaciones  
        """
    )

    if not estacion_map_est.empty:
        lat_sel = estacion_map_est["lat"].iloc[0]
        lon_sel = estacion_map_est["lon"].iloc[0]

        st.caption(
            f"Coordenadas: lat {formato_num(lat_sel, 4)}, lon {formato_num(lon_sel, 4)}"
        )


# ============================================================
# PANEL PRINCIPAL
# ============================================================

with col_panel:
    st.markdown(
        f'<div class="section-title">Estación: {estacion_sel}</div>',
        unsafe_allow_html=True,
    )

    nivel_obs_reciente = None
    fecha_obs_reciente = None

    if mostrar_obs and not obs_plot.empty:
        nivel_obs_reciente = obs_plot["nivel_m"].iloc[-1]
        fecha_obs_reciente = obs_plot["fecha"].iloc[-1]

    nivel_inicio = None
    nivel_fin = None
    tendencia_val = None

    if not pron_est.empty:
        serie_prom = pron_est["nivel_prom_ajustado_m"].dropna()

        if not serie_prom.empty:
            nivel_inicio = serie_prom.iloc[0]
            nivel_fin = serie_prom.iloc[-1]
            tendencia_val = nivel_fin - nivel_inicio

    kge = None
    rmse = None

    if not metricas_est.empty:
        if "kge_2009" in metricas_est.columns:
            kge = metricas_est["kge_2009"].iloc[0]

        if "rmse_m" in metricas_est.columns:
            rmse = metricas_est["rmse_m"].iloc[0]

    k1, k2, k3, k4 = st.columns(4)

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

    k4.metric(
        "Calidad DWLT / KGE",
        clasificar_calidad(kge),
        delta=f"KGE {formato_num(kge, 3)} | RMSE {formato_num(rmse, 3)} m",
    )

    if fecha_obs_reciente is not None:
        st.caption(
            f"Última observación graficada: {pd.to_datetime(fecha_obs_reciente).strftime('%d/%m/%Y')} | {mensaje_obs}"
        )
    else:
        st.caption(mensaje_obs)

    st.markdown(
        '<div class="section-title">Observado reciente + pronóstico original y ajustado de 7 días</div>',
        unsafe_allow_html=True,
    )

    fig = go.Figure()

    # --------------------------------------------------------
    # Observado reciente
    # --------------------------------------------------------

    if mostrar_obs and not obs_plot.empty:
        fig.add_trace(
            go.Scatter(
                x=obs_plot["fecha"],
                y=obs_plot["nivel_m"],
                mode="lines+markers",
                name="Observado reciente",
                line=dict(width=3, color="black"),
                marker=dict(size=8),
            )
        )

    # --------------------------------------------------------
    # Conexión observado-pronóstico ajustado
    # Solo si el observado es reciente.
    # --------------------------------------------------------

    if mostrar_obs and not obs_plot.empty and not pron_est.empty:
        ultimo_obs_fecha = obs_plot["fecha"].iloc[-1]
        ultimo_obs_nivel = obs_plot["nivel_m"].iloc[-1]

        primer_pron_fecha = pron_est["fecha"].iloc[0]
        primer_pron_nivel = pron_est["nivel_prom_ajustado_m"].iloc[0]

        fig.add_trace(
            go.Scatter(
                x=[ultimo_obs_fecha, primer_pron_fecha],
                y=[ultimo_obs_nivel, primer_pron_nivel],
                mode="lines",
                name="Conexión observado-pronóstico",
                line=dict(width=3, color="black"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # --------------------------------------------------------
    # Pronóstico original
    # --------------------------------------------------------

    if not pron_est.empty:
        fig.add_trace(
            go.Scatter(
                x=pron_est["fecha"],
                y=pron_est["nivel_prom_m"],
                mode="lines+markers",
                name="Pronóstico original",
                line=dict(width=2, dash="dash"),
                marker=dict(size=6),
            )
        )

    # --------------------------------------------------------
    # Pronóstico ajustado
    # --------------------------------------------------------

    if not pron_est.empty:
        fig.add_trace(
            go.Scatter(
                x=pron_est["fecha"],
                y=pron_est["nivel_prom_ajustado_m"],
                mode="lines+markers",
                name="Pronóstico ajustado",
                line=dict(width=3, color="blue"),
                marker=dict(size=8, color="blue"),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=pron_est["fecha"],
                y=pron_est["nivel_max_ajustado_m"],
                mode="lines",
                name="Máximo ajustado",
                line=dict(width=1, dash="dot"),
                visible="legendonly",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=pron_est["fecha"],
                y=pron_est["nivel_min_ajustado_m"],
                mode="lines",
                name="Mínimo ajustado",
                line=dict(width=1, dash="dot"),
                visible="legendonly",
            )
        )

    # --------------------------------------------------------
    # Línea vertical de inicio de pronóstico
    # --------------------------------------------------------

    if fecha_inicio_pron is not None and pd.notna(fecha_inicio_pron):
        fecha_corte = pd.to_datetime(fecha_inicio_pron)

        fig.add_shape(
            type="line",
            x0=fecha_corte,
            x1=fecha_corte,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(color="gray", width=2, dash="dash"),
        )

        fig.add_annotation(
            x=fecha_corte,
            y=1,
            xref="x",
            yref="paper",
            text="Inicio pronóstico",
            showarrow=False,
            yanchor="bottom",
            font=dict(size=11, color="gray"),
        )

    fig.update_layout(
        height=430,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Fecha",
        yaxis_title="Nivel (m)",
        legend_title="Serie",
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "El observado solo se grafica si es reciente. Si el último observado está muy antiguo, "
        "se muestra únicamente el pronóstico. El gráfico incluye pronóstico original y ajustado."
    )
