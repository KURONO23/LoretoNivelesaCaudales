from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="Visor DWLT Loreto",
    page_icon="🌊",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = BASE_DIR / "outputs"
CACHE_DIR = BASE_DIR / "backend" / "cache"

PRONOSTICO_CSV = OUTPUT_DIR / "pronostico_nivel_7dias_estaciones.csv"
VALIDACION_CSV = OUTPUT_DIR / "validacion_dwlt_hasta_observado_disponible.csv"
METRICAS_PARQUET = OUTPUT_DIR / "metricas_dwlt_estaciones.parquet"
FORE_DWLT_PARQUET = OUTPUT_DIR / "fore_nivel_transformado.parquet"
OBS_PARQUET = CACHE_DIR / "observado_estaciones.parquet"


# ============================================================
# FUNCIONES
# ============================================================

@st.cache_data(show_spinner=False)
def cargar_pronostico() -> pd.DataFrame:
    if not PRONOSTICO_CSV.exists():
        return pd.DataFrame()

    df = pd.read_csv(PRONOSTICO_CSV)
    df.columns = [str(c).strip().lower() for c in df.columns]

    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    elif "fecha_texto" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha_texto"], format="%d/%m/%Y", errors="coerce")

    if "comid" in df.columns:
        df["comid"] = pd.to_numeric(df["comid"], errors="coerce")

    return df


@st.cache_data(show_spinner=False)
def cargar_validacion() -> pd.DataFrame:
    if not VALIDACION_CSV.exists():
        return pd.DataFrame()

    df = pd.read_csv(VALIDACION_CSV)
    df.columns = [str(c).strip().lower() for c in df.columns]

    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    elif "fecha_texto" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha_texto"], format="%d/%m/%Y", errors="coerce")

    if "comid" in df.columns:
        df["comid"] = pd.to_numeric(df["comid"], errors="coerce")

    return df


@st.cache_data(show_spinner=False)
def cargar_metricas() -> pd.DataFrame:
    if not METRICAS_PARQUET.exists():
        return pd.DataFrame()

    df = pd.read_parquet(METRICAS_PARQUET)
    df.columns = [str(c).strip().lower() for c in df.columns]

    if "comid" in df.columns:
        df["comid"] = pd.to_numeric(df["comid"], errors="coerce")

    return df


@st.cache_data(show_spinner=False)
def cargar_fore_dwlt() -> pd.DataFrame:
    if not FORE_DWLT_PARQUET.exists():
        return pd.DataFrame()

    df = pd.read_parquet(FORE_DWLT_PARQUET)
    df.columns = [str(c).strip().lower() for c in df.columns]

    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")

    if "comid" in df.columns:
        df["comid"] = pd.to_numeric(df["comid"], errors="coerce")

    return df


@st.cache_data(show_spinner=False)
def cargar_observado() -> pd.DataFrame:
    if not OBS_PARQUET.exists():
        return pd.DataFrame()

    df = pd.read_parquet(OBS_PARQUET)
    df.columns = [str(c).strip().lower() for c in df.columns]

    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")

    if "comid" in df.columns:
        df["comid"] = pd.to_numeric(df["comid"], errors="coerce")

    return df


def formato_num(x, nd=2):
    if pd.isna(x):
        return "Sin dato"
    return f"{x:.{nd}f}"


def clasificar_color_calidad(kge):
    if pd.isna(kge):
        return "Sin evaluar"
    if kge >= 0.75:
        return "Buena"
    if kge >= 0.50:
        return "Moderada"
    return "Revisar"


# ============================================================
# CARGA
# ============================================================

pron = cargar_pronostico()
valid = cargar_validacion()
metricas = cargar_metricas()
fore_dwlt = cargar_fore_dwlt()
obs = cargar_observado()


# ============================================================
# CABECERA
# ============================================================

st.title("🌊 Visor de Pronóstico de Niveles - Loreto")
st.caption("Sistema automático SONICS + HidroMet + DWLT")

if pron.empty:
    st.error("No se encontró el archivo de pronóstico. Revisa outputs/pronostico_nivel_7dias_estaciones.csv")
    st.stop()

estaciones = sorted(pron["estacion"].dropna().unique().tolist())

with st.sidebar:
    st.header("Filtros")

    estacion_sel = st.selectbox(
        "Estación",
        estaciones,
        index=0,
    )

    mostrar_tablas = st.checkbox("Mostrar tablas completas", value=False)


pron_est = pron[pron["estacion"] == estacion_sel].copy()

comid_sel = None
if not pron_est.empty and "comid" in pron_est.columns:
    comid_sel = pron_est["comid"].dropna().iloc[0]

valid_est = pd.DataFrame()
metricas_est = pd.DataFrame()
fore_est = pd.DataFrame()
obs_est = pd.DataFrame()

if comid_sel is not None:
    valid_est = valid[valid["comid"] == comid_sel].copy() if not valid.empty else pd.DataFrame()
    metricas_est = metricas[metricas["comid"] == comid_sel].copy() if not metricas.empty else pd.DataFrame()
    fore_est = fore_dwlt[fore_dwlt["comid"] == comid_sel].copy() if not fore_dwlt.empty else pd.DataFrame()
    obs_est = obs[obs["comid"] == comid_sel].copy() if not obs.empty else pd.DataFrame()


# ============================================================
# RESUMEN
# ============================================================

st.subheader(f"Estación: {estacion_sel}")

c1, c2, c3, c4, c5 = st.columns(5)

nivel_inicio = pron_est["nivel_prom_m"].iloc[0] if "nivel_prom_m" in pron_est.columns and len(pron_est) else None
nivel_fin = pron_est["nivel_prom_m"].iloc[-1] if "nivel_prom_m" in pron_est.columns and len(pron_est) else None
nivel_max = pron_est["nivel_max_m"].max() if "nivel_max_m" in pron_est.columns else None
nivel_min = pron_est["nivel_min_m"].min() if "nivel_min_m" in pron_est.columns else None

tendencia = None
if nivel_inicio is not None and nivel_fin is not None:
    tendencia_val = nivel_fin - nivel_inicio
    if tendencia_val > 0.10:
        tendencia = "Ascendente"
    elif tendencia_val < -0.10:
        tendencia = "Descendente"
    else:
        tendencia = "Estable"
else:
    tendencia_val = None

kge = None
rmse = None

if not metricas_est.empty:
    if "kge_2009" in metricas_est.columns:
        kge = metricas_est["kge_2009"].iloc[0]
    if "rmse_m" in metricas_est.columns:
        rmse = metricas_est["rmse_m"].iloc[0]

c1.metric("COMID", int(comid_sel) if comid_sel is not None else "Sin dato")
c2.metric("Nivel mínimo 7 días", f"{formato_num(nivel_min)} m")
c3.metric("Nivel promedio inicial", f"{formato_num(nivel_inicio)} m")
c4.metric("Nivel promedio final", f"{formato_num(nivel_fin)} m")
c5.metric("Tendencia", tendencia or "Sin dato", delta=f"{formato_num(tendencia_val)} m" if tendencia_val is not None else None)

c6, c7, c8 = st.columns(3)
c6.metric("KGE histórico", formato_num(kge, 3))
c7.metric("RMSE histórico", f"{formato_num(rmse, 3)} m")
c8.metric("Calidad DWLT", clasificar_color_calidad(kge))


# ============================================================
# GRÁFICO PRONÓSTICO
# ============================================================

st.subheader("Pronóstico de nivel a 7 días")

if pron_est.empty:
    st.warning("No hay pronóstico para esta estación.")
else:
    cols_y = [
        c for c in ["nivel_min_m", "nivel_prom_m", "nivel_max_m"]
        if c in pron_est.columns
    ]

    if cols_y:
        pron_plot = pron_est[["fecha"] + cols_y].copy()
        pron_long = pron_plot.melt(
            id_vars="fecha",
            value_vars=cols_y,
            var_name="variable",
            value_name="nivel_m",
        )

        fig = px.line(
            pron_long,
            x="fecha",
            y="nivel_m",
            color="variable",
            markers=True,
            title=f"Pronóstico de niveles - {estacion_sel}",
        )

        fig.update_layout(
            xaxis_title="Fecha",
            yaxis_title="Nivel (m)",
            legend_title="Variable",
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No se encontraron columnas de nivel para graficar.")


# ============================================================
# VALIDACIÓN RECIENTE
# ============================================================

st.subheader("Validación reciente DWLT vs observado")

if valid_est.empty:
    st.info("No hay validación reciente para esta estación.")
else:
    cols_valid_graf = [
        c for c in ["nivel_observado_m", "nivel_dwlt_m"]
        if c in valid_est.columns
    ]

    if "fecha" in valid_est.columns and cols_valid_graf:
        valid_plot = valid_est[["fecha"] + cols_valid_graf].copy()
        valid_long = valid_plot.melt(
            id_vars="fecha",
            value_vars=cols_valid_graf,
            var_name="variable",
            value_name="nivel_m",
        )

        fig_val = px.line(
            valid_long,
            x="fecha",
            y="nivel_m",
            color="variable",
            markers=True,
            title=f"Validación reciente - {estacion_sel}",
        )

        fig_val.update_layout(
            xaxis_title="Fecha",
            yaxis_title="Nivel (m)",
            legend_title="Serie",
        )

        st.plotly_chart(fig_val, use_container_width=True)

    cols_tabla_valid = [
        "fecha",
        "nivel_observado_m",
        "nivel_dwlt_m",
        "error_m",
        "error_abs_m",
        "tendencia_obs",
        "tendencia_dwlt",
        "coincide_tendencia",
    ]

    cols_tabla_valid = [c for c in cols_tabla_valid if c in valid_est.columns]

    st.dataframe(
        valid_est[cols_tabla_valid].sort_values("fecha"),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# TABLAS
# ============================================================

if mostrar_tablas:
    st.subheader("Tabla de pronóstico")
    st.dataframe(pron_est, use_container_width=True, hide_index=True)

    if not metricas_est.empty:
        st.subheader("Métricas DWLT")
        st.dataframe(metricas_est, use_container_width=True, hide_index=True)

    if not obs_est.empty:
        st.subheader("Últimos observados")
        obs_show = obs_est.sort_values("fecha").tail(60)
        st.dataframe(obs_show, use_container_width=True, hide_index=True)


# ============================================================
# DESCARGAS
# ============================================================

st.subheader("Descargas")

col_d1, col_d2 = st.columns(2)

with col_d1:
    st.download_button(
        label="Descargar pronóstico CSV",
        data=pron.to_csv(index=False, encoding="utf-8-sig"),
        file_name="pronostico_nivel_7dias_estaciones.csv",
        mime="text/csv",
    )

with col_d2:
    if not valid.empty:
        st.download_button(
            label="Descargar validación CSV",
            data=valid.to_csv(index=False, encoding="utf-8-sig"),
            file_name="validacion_dwlt_hasta_observado_disponible.csv",
            mime="text/csv",
        )
