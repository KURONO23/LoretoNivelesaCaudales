from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = BASE_DIR / "outputs"
CACHE_DIR = BASE_DIR / "backend" / "cache"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FORE_DWLT = OUTPUT_DIR / "fore_nivel_transformado.parquet"
METRICAS = OUTPUT_DIR / "metricas_dwlt_estaciones.xlsx"
OBS_FILE = CACHE_DIR / "observado_estaciones.parquet"

OUT_XLSX = OUTPUT_DIR / "pronostico_nivel_7dias_estaciones.xlsx"
OUT_CSV = OUTPUT_DIR / "pronostico_nivel_7dias_estaciones.csv"

# Exportar solo los primeros 7 días disponibles.
DIAS_EXPORTAR = 7

# Máxima antigüedad permitida del observado usado para anclar el pronóstico.
MAX_DIAS_OBS_AJUSTE = 7

# Si el salto necesario es mayor a este valor, se ajusta igual, pero queda marcado.
UMBRAL_OFFSET_ADVERTENCIA_M = 1.50


# ============================================================
# UTILIDADES
# ============================================================

def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo requerido: {path}")


def normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def clasificar_tendencia(x):
    if pd.isna(x):
        return "Sin dato"

    if x > 0.10:
        return "Ascendente"

    if x < -0.10:
        return "Descendente"

    return "Estable"


def calidad_kge(kge):
    if pd.isna(kge):
        return "Sin evaluar"

    if kge >= 0.75:
        return "Buena"

    if kge >= 0.50:
        return "Moderada"

    return "Revisar"


def preparar_observado(obs_path: Path) -> pd.DataFrame:
    if not obs_path.exists():
        print(f"ADVERTENCIA: no existe observado para ajuste: {obs_path}")
        return pd.DataFrame()

    obs = pd.read_parquet(obs_path)
    obs = normalizar_columnas(obs)

    requeridas = {"fecha", "comid", "nivel_m"}

    if not requeridas.issubset(set(obs.columns)):
        print("ADVERTENCIA: observado_estaciones.parquet no tiene columnas fecha, comid, nivel_m.")
        return pd.DataFrame()

    obs["fecha"] = pd.to_datetime(obs["fecha"], errors="coerce")
    obs["comid"] = pd.to_numeric(obs["comid"], errors="coerce").astype("Int64")
    obs["nivel_m"] = pd.to_numeric(obs["nivel_m"], errors="coerce")

    obs = obs.dropna(subset=["fecha", "comid", "nivel_m"]).copy()
    obs = obs[obs["nivel_m"] > 0].copy()

    obs_diario = (
        obs.groupby(["comid", "fecha"], dropna=False)
        .agg(
            nivel_obs_diario_m=("nivel_m", "mean"),
            n_obs_dia=("nivel_m", "count"),
        )
        .reset_index()
    )

    obs_diario = obs_diario.sort_values(["comid", "fecha"]).reset_index(drop=True)

    return obs_diario


def aplicar_ajuste_continuidad(pron: pd.DataFrame, obs_diario: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica ajuste por continuidad estación por estación.

    offset = último observado antes del pronóstico - primer pronóstico DWLT

    Luego:
    nivel_*_ajustado_m = nivel_*_m + offset
    """

    pron = pron.copy()

    nivel_cols = [
        c for c in pron.columns
        if c.startswith("nivel_")
        and c.endswith("_m")
        and not c.endswith("_ajustado_m")
    ]

    for col in nivel_cols:
        pron[col] = pd.to_numeric(pron[col], errors="coerce")

    for col in nivel_cols:
        pron[col.replace("_m", "_ajustado_m")] = np.nan

    pron["ajuste_continuidad"] = "No"
    pron["offset_ajuste_m"] = np.nan
    pron["fecha_obs_ajuste"] = pd.NaT
    pron["nivel_obs_ajuste_m"] = np.nan
    pron["dias_desde_obs_ajuste"] = np.nan
    pron["advertencia_ajuste"] = ""

    if obs_diario.empty:
        pron["advertencia_ajuste"] = "Sin archivo observado para ajuste"

        for col in nivel_cols:
            pron[col.replace("_m", "_ajustado_m")] = pron[col]

        return pron

    partes = []

    for comid, g in pron.groupby("comid", dropna=False):
        g = g.sort_values("fecha").copy()

        if pd.isna(comid) or g.empty:
            partes.append(g)
            continue

        fecha_inicio = g["fecha"].min()

        obs_est = obs_diario[
            (obs_diario["comid"] == comid)
            & (obs_diario["fecha"] < fecha_inicio)
        ].copy()

        if obs_est.empty:
            for col in nivel_cols:
                g[col.replace("_m", "_ajustado_m")] = g[col]

            g["advertencia_ajuste"] = "Sin observado previo al inicio del pronóstico"
            partes.append(g)
            continue

        obs_ult = obs_est.sort_values("fecha").iloc[-1]

        fecha_obs = obs_ult["fecha"]
        nivel_obs = obs_ult["nivel_obs_diario_m"]

        dias_desde_obs = int((fecha_inicio - fecha_obs).days)

        if dias_desde_obs > MAX_DIAS_OBS_AJUSTE:
            for col in nivel_cols:
                g[col.replace("_m", "_ajustado_m")] = g[col]

            g["fecha_obs_ajuste"] = fecha_obs
            g["nivel_obs_ajuste_m"] = nivel_obs
            g["dias_desde_obs_ajuste"] = dias_desde_obs
            g["advertencia_ajuste"] = (
                f"Observado muy antiguo para ajuste: {dias_desde_obs} días"
            )

            partes.append(g)
            continue

        if "nivel_prom_m" not in g.columns:
            for col in nivel_cols:
                g[col.replace("_m", "_ajustado_m")] = g[col]

            g["advertencia_ajuste"] = "No existe nivel_prom_m para calcular offset"
            partes.append(g)
            continue

        primer_pron = pd.to_numeric(g["nivel_prom_m"], errors="coerce").dropna()

        if primer_pron.empty or pd.isna(nivel_obs):
            for col in nivel_cols:
                g[col.replace("_m", "_ajustado_m")] = g[col]

            g["advertencia_ajuste"] = "No se pudo calcular offset"
            partes.append(g)
            continue

        nivel_pron_ini = float(primer_pron.iloc[0])
        offset = float(nivel_obs) - nivel_pron_ini

        for col in nivel_cols:
            g[col.replace("_m", "_ajustado_m")] = g[col] + offset

        g["ajuste_continuidad"] = "Sí"
        g["offset_ajuste_m"] = offset
        g["fecha_obs_ajuste"] = fecha_obs
        g["nivel_obs_ajuste_m"] = nivel_obs
        g["dias_desde_obs_ajuste"] = dias_desde_obs

        if abs(offset) > UMBRAL_OFFSET_ADVERTENCIA_M:
            g["advertencia_ajuste"] = (
                f"Offset alto: {offset:.2f} m. Revisar estación."
            )
        else:
            g["advertencia_ajuste"] = "Ajuste aplicado"

        partes.append(g)

    out = pd.concat(partes, ignore_index=True)
    out = out.sort_values(["estacion", "comid", "fecha"]).reset_index(drop=True)

    return out


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 90)
    print("EXPORTANDO PRONÓSTICO DE NIVEL A 7 DÍAS - DWLT")
    print("CON AJUSTE DE CONTINUIDAD AL ÚLTIMO OBSERVADO")
    print("=" * 90)

    require_file(FORE_DWLT)

    print(f"Leyendo pronóstico DWLT: {FORE_DWLT}")

    df = pd.read_parquet(FORE_DWLT)
    df = normalizar_columnas(df)

    if "fecha" not in df.columns:
        raise ValueError("El archivo fore_nivel_transformado.parquet no tiene columna 'fecha'.")

    if "comid" not in df.columns:
        raise ValueError("El archivo fore_nivel_transformado.parquet no tiene columna 'comid'.")

    if "estacion" not in df.columns:
        df["estacion"] = "SIN_NOMBRE"

    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df["comid"] = pd.to_numeric(df["comid"], errors="coerce").astype("Int64")

    df = df.dropna(subset=["fecha", "comid"]).copy()

    # ========================================================
    # FILTRAR SOLO LOS PRIMEROS 7 DÍAS DISPONIBLES
    # ========================================================

    fechas_disponibles = sorted(df["fecha"].dropna().unique())

    if not fechas_disponibles:
        raise ValueError("No hay fechas válidas en el pronóstico DWLT.")

    fechas_exportar = fechas_disponibles[:DIAS_EXPORTAR]

    df = df[df["fecha"].isin(fechas_exportar)].copy()

    print(f"Fechas disponibles en forecast: {len(fechas_disponibles)}")
    print(f"Fechas exportadas: {len(fechas_exportar)}")
    print(f"Inicio: {pd.to_datetime(fechas_exportar[0]).date()}")
    print(f"Fin: {pd.to_datetime(fechas_exportar[-1]).date()}")

    df = df.sort_values(["estacion", "comid", "fecha"]).copy()

    # ========================================================
    # APLICAR AJUSTE DE CONTINUIDAD
    # ========================================================

    obs_diario = preparar_observado(OBS_FILE)

    df = aplicar_ajuste_continuidad(
        pron=df,
        obs_diario=obs_diario,
    )

    # ========================================================
    # TABLA DETALLADA DEL PRONÓSTICO
    # ========================================================

    cols_base = [
        "estacion",
        "comid",
        "fecha",

        # Bruto
        "nivel_eta_eqm_m",
        "nivel_eta_scal_m",
        "nivel_gfs_m",
        "nivel_wrf_m",
        "nivel_min_m",
        "nivel_p25_m",
        "nivel_prom_m",
        "nivel_p75_m",
        "nivel_max_m",

        # Ajustado
        "nivel_eta_eqm_ajustado_m",
        "nivel_eta_scal_ajustado_m",
        "nivel_gfs_ajustado_m",
        "nivel_wrf_ajustado_m",
        "nivel_min_ajustado_m",
        "nivel_p25_ajustado_m",
        "nivel_prom_ajustado_m",
        "nivel_p75_ajustado_m",
        "nivel_max_ajustado_m",

        # Control de ajuste
        "ajuste_continuidad",
        "offset_ajuste_m",
        "fecha_obs_ajuste",
        "nivel_obs_ajuste_m",
        "dias_desde_obs_ajuste",
        "advertencia_ajuste",
    ]

    cols = [c for c in cols_base if c in df.columns]
    pron = df[cols].copy()

    for col in pron.columns:
        if col.startswith("nivel_") or col in ["offset_ajuste_m", "nivel_obs_ajuste_m", "dias_desde_obs_ajuste"]:
            pron[col] = pd.to_numeric(pron[col], errors="coerce").round(3)

    pron["fecha_texto"] = pron["fecha"].dt.strftime("%d/%m/%Y")

    if "fecha_obs_ajuste" in pron.columns:
        pron["fecha_obs_ajuste_texto"] = pd.to_datetime(
            pron["fecha_obs_ajuste"],
            errors="coerce",
        ).dt.strftime("%d/%m/%Y")

    cols_final = [
        "estacion",
        "comid",
        "fecha",
        "fecha_texto",

        # Ajustado para operación
        "nivel_prom_ajustado_m",
        "nivel_min_ajustado_m",
        "nivel_max_ajustado_m",
        "nivel_p25_ajustado_m",
        "nivel_p75_ajustado_m",

        # Bruto DWLT
        "nivel_prom_m",
        "nivel_min_m",
        "nivel_max_m",
        "nivel_p25_m",
        "nivel_p75_m",

        # Modelos ajustados
        "nivel_eta_eqm_ajustado_m",
        "nivel_eta_scal_ajustado_m",
        "nivel_gfs_ajustado_m",
        "nivel_wrf_ajustado_m",

        # Modelos brutos
        "nivel_eta_eqm_m",
        "nivel_eta_scal_m",
        "nivel_gfs_m",
        "nivel_wrf_m",

        # Control de ajuste
        "ajuste_continuidad",
        "offset_ajuste_m",
        "fecha_obs_ajuste",
        "fecha_obs_ajuste_texto",
        "nivel_obs_ajuste_m",
        "dias_desde_obs_ajuste",
        "advertencia_ajuste",
    ]

    cols_final = [c for c in cols_final if c in pron.columns]
    pron = pron[cols_final].copy()

    # ========================================================
    # RESUMEN POR ESTACIÓN
    # ========================================================

    col_min = "nivel_min_ajustado_m" if "nivel_min_ajustado_m" in pron.columns else "nivel_min_m"
    col_prom = "nivel_prom_ajustado_m" if "nivel_prom_ajustado_m" in pron.columns else "nivel_prom_m"
    col_max = "nivel_max_ajustado_m" if "nivel_max_ajustado_m" in pron.columns else "nivel_max_m"

    resumen = (
        pron.groupby(["estacion", "comid"], dropna=False)
        .agg(
            fecha_inicio=("fecha", "min"),
            fecha_fin=("fecha", "max"),
            nivel_min_7dias=(col_min, "min"),
            nivel_prom_7dias=(col_prom, "mean"),
            nivel_max_7dias=(col_max, "max"),
            nivel_inicio=(col_prom, "first"),
            nivel_fin=(col_prom, "last"),
            ajuste_continuidad=("ajuste_continuidad", "first"),
            offset_ajuste_m=("offset_ajuste_m", "first"),
            fecha_obs_ajuste=("fecha_obs_ajuste", "first"),
            nivel_obs_ajuste_m=("nivel_obs_ajuste_m", "first"),
            dias_desde_obs_ajuste=("dias_desde_obs_ajuste", "first"),
            advertencia_ajuste=("advertencia_ajuste", "first"),
        )
        .reset_index()
    )

    resumen["fecha_inicio_texto"] = resumen["fecha_inicio"].dt.strftime("%d/%m/%Y")
    resumen["fecha_fin_texto"] = resumen["fecha_fin"].dt.strftime("%d/%m/%Y")
    resumen["fecha_obs_ajuste_texto"] = pd.to_datetime(
        resumen["fecha_obs_ajuste"],
        errors="coerce",
    ).dt.strftime("%d/%m/%Y")

    resumen["tendencia_7dias_m"] = resumen["nivel_fin"] - resumen["nivel_inicio"]
    resumen["tendencia"] = resumen["tendencia_7dias_m"].apply(clasificar_tendencia)

    for col in [
        "nivel_min_7dias",
        "nivel_prom_7dias",
        "nivel_max_7dias",
        "nivel_inicio",
        "nivel_fin",
        "tendencia_7dias_m",
        "offset_ajuste_m",
        "nivel_obs_ajuste_m",
        "dias_desde_obs_ajuste",
    ]:
        if col in resumen.columns:
            resumen[col] = pd.to_numeric(resumen[col], errors="coerce").round(3)

    # ========================================================
    # AGREGAR MÉTRICAS DWLT
    # ========================================================

    if METRICAS.exists():
        print(f"Leyendo métricas DWLT: {METRICAS}")

        met = pd.read_excel(METRICAS, sheet_name="metricas_dwlt")
        met = normalizar_columnas(met)

        if "comid" in met.columns:
            met["comid"] = pd.to_numeric(met["comid"], errors="coerce").astype("Int64")

            met_cols = [
                "comid",
                "n_validacion",
                "r_pearson",
                "nse",
                "kge_2009",
                "rmse_m",
                "metodo_dwlt",
            ]

            met_cols = [c for c in met_cols if c in met.columns]

            resumen = resumen.merge(
                met[met_cols].drop_duplicates(subset=["comid"], keep="first"),
                on="comid",
                how="left",
            )

            for col in ["r_pearson", "nse", "kge_2009", "rmse_m"]:
                if col in resumen.columns:
                    resumen[col] = pd.to_numeric(resumen[col], errors="coerce").round(3)

            if "kge_2009" in resumen.columns:
                resumen["calidad_dwlt"] = resumen["kge_2009"].apply(calidad_kge)
            else:
                resumen["calidad_dwlt"] = "Sin evaluar"

        else:
            print("ADVERTENCIA: metricas_dwlt_estaciones.xlsx no tiene columna comid.")
            resumen["calidad_dwlt"] = "Sin evaluar"

    else:
        print("ADVERTENCIA: no existe metricas_dwlt_estaciones.xlsx. Se exportará sin métricas.")
        resumen["calidad_dwlt"] = "Sin evaluar"

    # ========================================================
    # PIVOT
    # ========================================================

    pivot_prom = (
        pron.pivot_table(
            index=["estacion", "comid"],
            columns="fecha_texto",
            values=col_prom,
            aggfunc="mean",
        )
        .reset_index()
    )

    # ========================================================
    # ORDEN DE COLUMNAS DEL RESUMEN
    # ========================================================

    cols_resumen = [
        "estacion",
        "comid",
        "fecha_inicio",
        "fecha_fin",
        "fecha_inicio_texto",
        "fecha_fin_texto",
        "nivel_min_7dias",
        "nivel_prom_7dias",
        "nivel_max_7dias",
        "nivel_inicio",
        "nivel_fin",
        "tendencia_7dias_m",
        "tendencia",
        "ajuste_continuidad",
        "offset_ajuste_m",
        "fecha_obs_ajuste",
        "fecha_obs_ajuste_texto",
        "nivel_obs_ajuste_m",
        "dias_desde_obs_ajuste",
        "advertencia_ajuste",
        "n_validacion",
        "r_pearson",
        "nse",
        "kge_2009",
        "rmse_m",
        "metodo_dwlt",
        "calidad_dwlt",
    ]

    cols_resumen = [c for c in cols_resumen if c in resumen.columns]
    resumen = resumen[cols_resumen].copy()

    # ========================================================
    # EXPORTAR
    # ========================================================

    pron.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        resumen.to_excel(writer, sheet_name="resumen_estaciones", index=False)
        pron.to_excel(writer, sheet_name="pronostico_7dias", index=False)
        pivot_prom.to_excel(writer, sheet_name="pivot_nivel_prom_ajustado", index=False)

    print("\nArchivos generados:")
    print(f" - {OUT_XLSX}")
    print(f" - {OUT_CSV}")

    print("\nResumen por estación:")
    cols_print = [
        "estacion",
        "comid",
        "nivel_prom_7dias",
        "nivel_inicio",
        "nivel_fin",
        "tendencia_7dias_m",
        "tendencia",
        "ajuste_continuidad",
        "offset_ajuste_m",
        "fecha_obs_ajuste_texto",
        "kge_2009",
        "calidad_dwlt",
        "advertencia_ajuste",
    ]

    cols_print = [c for c in cols_print if c in resumen.columns]
    print(resumen[cols_print].to_string(index=False))

    print("\nProceso terminado correctamente.")


if __name__ == "__main__":
    main()
