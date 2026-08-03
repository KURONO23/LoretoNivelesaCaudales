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
VALIDACION_CSV = OUTPUT_DIR / "validacion_dwlt_hasta_observado_disponible.csv"
METRICAS_PARQUET = OUTPUT_DIR / "metricas_dwlt_estaciones.parquet"
FORE_DWLT_PARQUET = OUTPUT_DIR / "fore_nivel_transformado.parquet"
OBS_PARQUET = CACHE_DIR / "observado_estaciones.parquet"


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

    .small-note {
        color: #64748b;
        font-size: 0.85rem;
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
# UTILIDADES GENERALES
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
    if pd.isna(x):
        return "Sin dato"

    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "Sin dato"


def clasificar_tendencia(delta: float | None) -> str:
    if delta is None or pd.isna(delta):
        return "Sin dato"

    if delta > 0.10:
        return "Ascendente"

    if delta < -0.10:
        return "Descendente"

    return "Estable"


def clasificar_calidad(kge) -> str:
    if pd.isna(kge):
        return "Sin evaluar"

    if kge >= 0.75:
        return "Buena"

    if kge >= 0.50:
        return "Moderada"

    return "Revisar"


# ============================================================
# LECTURA DIRECTA DEL GPKG
# ============================================================

def quote_sql_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


@st.cache_data(show_spinner=False)
def find_table_with_comid(gpkg_path: str) -> str:
    conn = sqlite3.connect(gpkg_path)

    try:
        tables_df = pd.read_sql_query(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            ORDER BY name
            """,
            conn,
        )

        tables = tables_df["name"].astype(str).tolist()

        excluded_prefixes = (
            "gpkg_",
            "sqlite_",
            "rtree_",
        )

        for table in tables:
            table_lower = table.lower()

            if table_lower.startswith(excluded_prefixes):
                continue

            info = conn.execute(
                f"PRAGMA table_info({quote_sql_identifier(table)})"
            ).fetchall()

            cols = [row[1] for row in info]
            cols_lower = [c.lower() for c in cols]

            if "comid" in cols_lower:
                return table

        raise ValueError("No se encontró una tabla con columna COMID en el GPKG.")

    finally:
        conn.close()


def detectar_columna_nombre(df: pd.DataFrame) -> str:
    posibles = [
        "estacion",
        "estacion_nombre",
        "estacion_catalogo",
        "estaciones_hidro",
        "nombre",
        "name",
    ]

    for col in posibles:
        if col in df.columns:
            return col

    raise ValueError(
        "No se encontró columna de nombre de estación en el GPKG. "
        f"Columnas disponibles: {list(df.columns)}"
    )


def detectar_columna_lat_lon(df: pd.DataFrame) -> tuple[str, str]:
    posibles_lat = [
        "lat",
        "latitude",
        "latitud",
        "y",
    ]

    posibles_lon = [
        "lon",
        "lng",
        "longitude",
        "longitud",
        "x",
    ]

    lat_col = None
    lon_col = None

    for col in posibles_lat:
        if col in df.columns:
            lat_col = col
            break

    for col in posibles_lon:
        if col in df.columns:
            lon_col = col
            break

    if lat_col and lon_col:
        return lat_col, lon_col

    raise ValueError(
        "No se encontraron columnas de latitud/longitud en el GPKG. "
        "El visor necesita columnas tipo lat/lon, latitud/longitud o x/y."
    )


@st.cache_data(show_spinner=False)
def cargar_estaciones_gpkg(gpkg_path: Path) -> pd.DataFrame:
    if not gpkg_path.exists():
        return pd.DataFrame()

    table = find_table_with_comid(str(gpkg_path))

    conn = sqlite3.connect(str(gpkg_path))

    try:
        query = f"SELECT * FROM {quote_sql_identifier(table)}"
        df = pd.read_sql_query(query, conn)

    finally:
        conn.close()

    df = normalizar_columnas(df)

    if "comid" not in df.columns:
        raise ValueError("El GPKG debe tener columna COMID.")

    nombre_col = detectar_columna_nombre(df)
    lat_col, lon_col = detectar_columna_lat_lon(df)

    out = pd.DataFrame({
        "estacion": df[nombre_col].astype(str).str.strip(),
        "estacion_norm": df[nombre_col].apply(normalizar_texto),
        "comid": pd.to_numeric(df["comid"], errors="coerce"),
        "lat": pd.to_numeric(df[lat_col], errors="coerce"),
        "lon": pd.to_numeric(df[lon_col], errors="coerce"),
    })

    out = out.dropna(subset=["estacion", "comid", "lat", "lon"]).copy()
    out = out[out["estacion"].str.strip() != ""].copy()

    out["comid"] = out["comid"].astype("int64")

    out = out.drop_duplicates(subset=["comid"], keep="first")
    out = out.sort_values("estacion").reset_index(drop=True)

    return out


# ============================================================
# CARGA DE ARCHIVOS DE RESULTADOS
# ============================================================

@st.cache_data(show_spinner=False)
def cargar_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    df = normalizar_columnas(df)

    return df


@st.cache_data(show_spinner=False)
def cargar_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(path)
    df = normalizar_columnas(df)

    return df


def preparar_fechas(df: pd.DataFrame) -> pd.DataFrame:
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

    return df


def preparar_comid(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()

    if "comid" in df.columns:
        df["comid"] = pd.to_numeric(df["comid"], errors="coerce")

    return df


def preparar_resumen_pronostico(pron: pd.DataFrame) -> pd.DataFrame:
    if pron.empty:
        return pd.DataFrame()

    requeridas = {"estacion", "comid", "fecha", "nivel_prom_m"}

    if not requeridas.issubset(set(pron.columns)):
        return pd.DataFrame()

    pron = pron.copy()

    for col in ["nivel_min_m", "nivel_prom_m", "nivel_max_m"]:
        if col in pron.columns:
            pron[col] = pd.to_numeric(pron[col], errors="coerce")

    resumen = (
        pron.groupby(["estacion", "comid"], dropna=False)
        .agg(
            fecha_inicio=("fecha", "min"),
            fecha_fin=("fecha", "max"),
            nivel_min_7dias=("nivel_min_m", "min"),
            nivel_prom_7dias=("nivel_prom_m", "mean"),
            nivel_max_7dias=("nivel_max_m", "max"),
            nivel_inicio=("nivel_prom_m", "first"),
            nivel_fin=("nivel_prom_m", "last"),
        )
        .reset_index()
    )

    resumen["tendencia_7dias_m"] = resumen["nivel_fin"] - resumen["nivel_inicio"]
    resumen["tendencia"] = resumen["tendencia_7dias_m"].apply(clasificar_tendencia)

    return resumen


def obtener_observado_5dias_previos(
    obs_est: pd.DataFrame,
    fecha_inicio_pronostico,
) -> pd.DataFrame:
    if obs_est.empty or "fecha" not in obs_est.columns:
        return pd.DataFrame()

    obs_est = obs_est.dropna(subset=["fecha"]).copy()
    obs_est = obs_est.sort_values("fecha")

    obs_diario = (
        obs_est.groupby("fecha", dropna=False)
        .agg(
            nivel_m=("nivel_m", "mean"),
        )
        .reset_index()
    )

    if fecha_inicio_pronostico is not None and pd.notna(fecha_inicio_pronostico):
        obs_diario = obs_diario[obs_diario["fecha"] < fecha_inicio_pronostico].copy()

    return obs_diario.tail(5).copy()


def obtener_pronostico_estacion(pron_est: pd.DataFrame) -> pd.DataFrame:
    if pron_est.empty:
        return pd.DataFrame()

    pron_est = pron_est.copy()

    for col in [
        "nivel_min_m",
        "nivel_prom_m",
        "nivel_max_m",
        "nivel_eta_eqm_m",
        "nivel_eta_scal_m",
        "nivel_gfs_m",
        "nivel_wrf_m",
    ]:
        if col in pron_est.columns:
            pron_est[col] = pd.to_numeric(pron_est[col], errors="coerce")

    return pron_est.sort_values("fecha").copy()


# ============================================================
# MAPA
# ============================================================

def crear_mapa_estaciones(
    estaciones: pd.DataFrame,
    estacion_sel: str,
    comid_sel,
    pron_resumen: pd.DataFrame,
) -> folium.Map:
    mapa = folium.Map(
        location=[-5.0, -74.5],
        zoom_start=6,
        tiles="CartoDB positron",
        control_scale=True,
    )

    if estaciones.empty:
        return mapa

    estaciones_plot = estaciones.dropna(subset=["lat", "lon"]).copy()

    for _, row in estaciones_plot.iterrows():
        estacion = str(row.get("estacion", "SIN_NOMBRE"))
        comid = row.get("comid", None)
        lat = row.get("lat", None)
        lon = row.get("lon", None)

        if pd.isna(lat) or pd.isna(lon) or pd.isna(comid):
            continue

        comid_int = int(comid)

        seleccionado = False

        if pd.notna(comid_sel):
            seleccionado = comid_int == int(comid_sel)
        elif estacion == estacion_sel:
            seleccionado = True

        color = "red" if seleccionado else "blue"
        radio = 9 if seleccionado else 6

        info_extra = ""

        if not pron_resumen.empty:
            tmp = pron_resumen[
                pd.to_numeric(pron_resumen["comid"], errors="coerce") == float(comid_int)
            ]

            if not tmp.empty:
                nivel_prom = tmp["nivel_prom_7dias"].iloc[0] if "nivel_prom_7dias" in tmp.columns else None
                tendencia = tmp["tendencia"].iloc[0] if "tendencia" in tmp.columns else None

                info_extra = f"""
                <br><b>Nivel prom. 7 días:</b> {formato_num(nivel_prom)} m
                <br><b>Tendencia:</b> {tendencia}
                """

        popup_html = f"""
        <b>{estacion}</b><br>
        <b>COMID:</b> {comid_int}
        {info_extra}
        <br><br><i>Haz click para seleccionar esta estación.</i>
        """

        folium.CircleMarker(
            location=[lat, lon],
            radius=radio,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{estacion}||{comid_int}",
            color=color,
            fill=True,
            fill_opacity=0.85,
        ).add_to(mapa)

    return mapa


# ============================================================
# CARGA
# ============================================================

estaciones = cargar_estaciones_gpkg(GPKG_FILE)
pron = cargar_csv(PRONOSTICO_CSV)
valid = cargar_csv(VALIDACION_CSV)
metricas = cargar_parquet(METRICAS_PARQUET)
fore_dwlt = cargar_parquet(FORE_DWLT_PARQUET)
obs = cargar_parquet(OBS_PARQUET)

for df_name in ["pron", "valid", "metricas", "fore_dwlt", "obs"]:
    df_tmp = globals()[df_name]
    df_tmp = preparar_fechas(df_tmp)
    df_tmp = preparar_comid(df_tmp)
    globals()[df_name] = df_tmp

if "nivel_m" in obs.columns:
    obs["nivel_m"] = pd.to_numeric(obs["nivel_m"], errors="coerce")

pron_resumen = preparar_resumen_pronostico(pron)


# ============================================================
# CABECERA
# ============================================================

st.markdown(
    '<div class="main-title">🌊 Visor de niveles y pronóstico hidrológico - Loreto</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">Sistema automático SONICS + HidroMet + DWLT</div>',
    unsafe_allow_html=True,
)


# ============================================================
# VALIDACIÓN DE ARCHIVOS
# ============================================================

faltantes = []

if estaciones.empty:
    faltantes.append("Data/estaciones_hidrometricas_loreto_latlong_COMID_actualizado.gpkg")

if pron.empty:
    faltantes.append("outputs/pronostico_nivel_7dias_estaciones.csv")

if obs.empty:
    faltantes.append("backend/cache/observado_estaciones.parquet")

if faltantes:
    st.error("Faltan archivos necesarios para el visor:")
    for f in faltantes:
        st.write(f"- `{f}`")
    st.stop()


# ============================================================
# RELACIÓN ESTACIÓN - COMID
# ============================================================

estaciones_disponibles = sorted(pron["estacion"].dropna().unique().tolist())

tabla_estaciones_pron = (
    pron[["estacion", "comid"]]
    .dropna()
    .drop_duplicates(subset=["estacion"])
    .copy()
)

tabla_estaciones_pron["comid"] = pd.to_numeric(
    tabla_estaciones_pron["comid"],
    errors="coerce",
)

tabla_estaciones_pron = tabla_estaciones_pron.dropna(subset=["comid"]).copy()
tabla_estaciones_pron["comid"] = tabla_estaciones_pron["comid"].astype("int64")

estacion_a_comid = dict(
    zip(tabla_estaciones_pron["estacion"], tabla_estaciones_pron["comid"])
)

comid_a_estacion = dict(
    zip(tabla_estaciones_pron["comid"], tabla_estaciones_pron["estacion"])
)

if "estacion_sel" not in st.session_state:
    st.session_state.estacion_sel = estaciones_disponibles[0]


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("Filtros")

    index_actual = 0

    if st.session_state.estacion_sel in estaciones_disponibles:
        index_actual = estaciones_disponibles.index(st.session_state.estacion_sel)

    estacion_sidebar = st.selectbox(
        "Estación",
        estaciones_disponibles,
        index=index_actual,
        key="selector_estacion_sidebar",
    )

    if estacion_sidebar != st.session_state.estacion_sel:
        st.session_state.estacion_sel = estacion_sidebar
        st.rerun()

    estacion_sel = st.session_state.estacion_sel

    mostrar_tablas = st.checkbox("Mostrar tablas completas", value=False)

    st.markdown("---")
    st.subheader("Archivos cargados")

    checks = {
        "GPKG estaciones": not estaciones.empty,
        "pronostico_nivel_7dias_estaciones.csv": not pron.empty,
        "validacion_dwlt_hasta_observado_disponible.csv": not valid.empty,
        "metricas_dwlt_estaciones.parquet": not metricas.empty,
        "fore_nivel_transformado.parquet": not fore_dwlt.empty,
        "observado_estaciones.parquet": not obs.empty,
    }

    for nombre, ok in checks.items():
        st.write(("✅ " if ok else "❌ ") + nombre)

    st.markdown("---")
    st.info("Los datos se actualizan automáticamente mediante GitHub Actions.")


# ============================================================
# FILTRO POR ESTACIÓN
# ============================================================

estacion_sel = st.session_state.estacion_sel

pron_est = pron[pron["estacion"] == estacion_sel].copy()

comid_sel = None

if estacion_sel in estacion_a_comid:
    comid_sel = estacion_a_comid[estacion_sel]
elif not pron_est.empty and "comid" in pron_est.columns:
    comid_sel = pron_est["comid"].dropna().iloc[0]

metricas_est = pd.DataFrame()
valid_est = pd.DataFrame()
obs_est = pd.DataFrame()
estacion_map_est = pd.DataFrame()

if comid_sel is not None:
    metricas_est = (
        metricas[metricas["comid"] == comid_sel].copy()
        if not metricas.empty and "comid" in metricas.columns
        else pd.DataFrame()
    )

    valid_est = (
        valid[valid["comid"] == comid_sel].copy()
        if not valid.empty and "comid" in valid.columns
        else pd.DataFrame()
    )

    obs_est = (
        obs[obs["comid"] == comid_sel].copy()
        if not obs.empty and "comid" in obs.columns
        else pd.DataFrame()
    )

    estacion_map_est = (
        estaciones[estaciones["comid"] == int(comid_sel)].copy()
        if not estaciones.empty and "comid" in estaciones.columns
        else pd.DataFrame()
    )

pron_est = obtener_pronostico_estacion(pron_est)

fecha_inicio_pron = None

if not pron_est.empty and "fecha" in pron_est.columns:
    fecha_inicio_pron = pron_est["fecha"].min()

obs_5dias = obtener_observado_5dias_previos(
    obs_est=obs_est,
    fecha_inicio_pronostico=fecha_inicio_pron,
)


# ============================================================
# LAYOUT PRINCIPAL
# ============================================================

col_mapa, col_panel = st.columns([0.95, 2.05], gap="large")


# ============================================================
# MAPA IZQUIERDA
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
        width=None,
        height=650,
        key="mapa_estaciones",
        returned_objects=["last_object_clicked_tooltip"],
    )

    tooltip_click = None

    if mapa_evento:
        tooltip_click = mapa_evento.get("last_object_clicked_tooltip")

    if tooltip_click and "||" in tooltip_click:
        _, comid_click = tooltip_click.split("||", 1)

        try:
            comid_click = int(float(comid_click))

            if comid_click in comid_a_estacion:
                nueva_estacion = comid_a_estacion[comid_click]

                if nueva_estacion != st.session_state.estacion_sel:
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
# PANEL DERECHA
# ============================================================

with col_panel:
    st.markdown(
        f'<div class="section-title">Estación: {estacion_sel}</div>',
        unsafe_allow_html=True,
    )

    nivel_obs_reciente = None
    fecha_obs_reciente = None

    if not obs_5dias.empty:
        nivel_obs_reciente = obs_5dias["nivel_m"].iloc[-1]
        fecha_obs_reciente = obs_5dias["fecha"].iloc[-1]

    nivel_inicio = None
    nivel_fin = None
    nivel_min = None
    nivel_max = None
    tendencia_val = None

    if not pron_est.empty:
        if "nivel_prom_m" in pron_est.columns:
            nivel_inicio = pron_est["nivel_prom_m"].iloc[0]
            nivel_fin = pron_est["nivel_prom_m"].iloc[-1]
            tendencia_val = nivel_fin - nivel_inicio

        if "nivel_min_m" in pron_est.columns:
            nivel_min = pron_est["nivel_min_m"].min()

        if "nivel_max_m" in pron_est.columns:
            nivel_max = pron_est["nivel_max_m"].max()

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
        help="Último nivel observado disponible antes del inicio del pronóstico.",
    )

    k2.metric(
        "Nivel pronosticado promedio",
        f"{formato_num(nivel_fin)} m",
        help="Nivel promedio pronosticado al final del horizonte.",
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
            f"Última observación usada: {pd.to_datetime(fecha_obs_reciente).strftime('%d/%m/%Y')}"
        )

    st.markdown(
        '<div class="section-title">Observado reciente + pronóstico de 7 días</div>',
        unsafe_allow_html=True,
    )

    fig = go.Figure()

    if not obs_5dias.empty:
        fig.add_trace(
            go.Scatter(
                x=obs_5dias["fecha"],
                y=obs_5dias["nivel_m"],
                mode="lines+markers",
                name="Observado 5 días previos",
                line=dict(width=3, color="black"),
                marker=dict(size=8),
            )
        )

    if not pron_est.empty:
        if "nivel_prom_m" in pron_est.columns:
            fig.add_trace(
                go.Scatter(
                    x=pron_est["fecha"],
                    y=pron_est["nivel_prom_m"],
                    mode="lines+markers",
                    name="Pronóstico medio",
                    line=dict(width=3),
                    marker=dict(size=8),
                )
            )

        if {"nivel_min_m", "nivel_max_m"}.issubset(pron_est.columns):
            fig.add_trace(
                go.Scatter(
                    x=pron_est["fecha"],
                    y=pron_est["nivel_max_m"],
                    mode="lines",
                    name="Máximo pronosticado",
                    line=dict(width=1, dash="dot"),
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=pron_est["fecha"],
                    y=pron_est["nivel_min_m"],
                    mode="lines",
                    name="Mínimo pronosticado",
                    line=dict(width=1, dash="dot"),
                    fill="tonexty",
                    fillcolor="rgba(37, 99, 235, 0.15)",
                )
            )

    if fecha_inicio_pron is not None and pd.notna(fecha_inicio_pron):
        fig.add_vline(
            x=fecha_inicio_pron,
            line_width=2,
            line_dash="dash",
            line_color="gray",
            annotation_text="Inicio pronóstico",
            annotation_position="top",
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
        "El gráfico muestra los 5 días observados previos al pronóstico y luego el pronóstico DWLT de 7 días."
    )

    col_tabla, col_resumen = st.columns([1.45, 1.0], gap="large")

    with col_tabla:
        st.markdown(
            '<div class="section-title">Pronóstico de niveles - 7 días</div>',
            unsafe_allow_html=True,
        )

        cols_pron = [
            "fecha",
            "nivel_prom_m",
            "nivel_min_m",
            "nivel_max_m",
            "nivel_eta_eqm_m",
            "nivel_eta_scal_m",
            "nivel_gfs_m",
            "nivel_wrf_m",
        ]

        cols_pron = [c for c in cols_pron if c in pron_est.columns]

        tabla_pron = pron_est[cols_pron].copy()

        for c in tabla_pron.columns:
            if c.startswith("nivel_"):
                tabla_pron[c] = pd.to_numeric(tabla_pron[c], errors="coerce").round(3)

        st.dataframe(
            tabla_pron,
            use_container_width=True,
            hide_index=True,
        )

    with col_resumen:
        st.markdown(
            '<div class="section-title">Resumen de pronóstico</div>',
            unsafe_allow_html=True,
        )

        st.write(f"**COMID:** {int(comid_sel) if comid_sel is not None else 'Sin dato'}")
        st.write(f"**Nivel mínimo 7 días:** {formato_num(nivel_min)} m")
        st.write(f"**Nivel máximo 7 días:** {formato_num(nivel_max)} m")
        st.write(f"**Cambio esperado:** {formato_num(tendencia_val)} m")
        st.write(f"**Tendencia:** {clasificar_tendencia(tendencia_val)}")
        st.write(f"**Calidad DWLT:** {clasificar_calidad(kge)}")

        st.info("Los niveles corresponden al datum local de cada estación hidrométrica.")

    st.markdown(
        '<div class="section-title">Validación reciente DWLT vs observado</div>',
        unsafe_allow_html=True,
    )

    if valid_est.empty:
        st.info("No hay validación reciente para esta estación.")
    else:
        valid_est = preparar_fechas(valid_est)
        valid_est = valid_est.sort_values("fecha").copy()

        fig_val = go.Figure()

        if "nivel_observado_m" in valid_est.columns:
            fig_val.add_trace(
                go.Scatter(
                    x=valid_est["fecha"],
                    y=valid_est["nivel_observado_m"],
                    mode="lines+markers",
                    name="Nivel observado",
                    line=dict(width=3),
                )
            )

        if "nivel_dwlt_m" in valid_est.columns:
            fig_val.add_trace(
                go.Scatter(
                    x=valid_est["fecha"],
                    y=valid_est["nivel_dwlt_m"],
                    mode="lines+markers",
                    name="Nivel DWLT",
                    line=dict(width=3),
                )
            )

        fig_val.update_layout(
            height=330,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title="Fecha",
            yaxis_title="Nivel (m)",
            legend_title="Serie",
            hovermode="x unified",
        )

        st.plotly_chart(fig_val, use_container_width=True)

        cols_valid = [
            "fecha",
            "nivel_observado_m",
            "nivel_dwlt_m",
            "error_m",
            "error_abs_m",
            "tendencia_obs",
            "tendencia_dwlt",
            "coincide_tendencia",
        ]

        cols_valid = [c for c in cols_valid if c in valid_est.columns]

        st.dataframe(
            valid_est[cols_valid],
            use_container_width=True,
            hide_index=True,
        )

    if mostrar_tablas:
        st.markdown(
            '<div class="section-title">Tablas completas</div>',
            unsafe_allow_html=True,
        )

        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "Pronóstico",
                "Validación",
                "Métricas",
                "Observado",
            ]
        )

        with tab1:
            st.dataframe(pron_est, use_container_width=True, hide_index=True)

        with tab2:
            st.dataframe(valid_est, use_container_width=True, hide_index=True)

        with tab3:
            st.dataframe(metricas_est, use_container_width=True, hide_index=True)

        with tab4:
            obs_show = obs_est.sort_values("fecha").tail(90).copy() if not obs_est.empty else pd.DataFrame()
            st.dataframe(obs_show, use_container_width=True, hide_index=True)

    st.markdown(
        '<div class="section-title">Descargas</div>',
        unsafe_allow_html=True,
    )

    d1, d2 = st.columns(2)

    with d1:
        st.download_button(
            label="Descargar pronóstico CSV",
            data=pron.to_csv(index=False, encoding="utf-8-sig"),
            file_name="pronostico_nivel_7dias_estaciones.csv",
            mime="text/csv",
        )

    with d2:
        if not valid.empty:
            st.download_button(
                label="Descargar validación CSV",
                data=valid.to_csv(index=False, encoding="utf-8-sig"),
                file_name="validacion_dwlt_hasta_observado_disponible.csv",
                mime="text/csv",
            )
