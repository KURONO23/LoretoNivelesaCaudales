from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = BASE_DIR / "backend" / "cache"

HIST_DWLT_FILE = OUTPUT_DIR / "hist_nivel_transformado.parquet"
FORE_DWLT_FILE = OUTPUT_DIR / "fore_nivel_transformado.parquet"

OBS_FILE_CACHE = CACHE_DIR / "observado_estaciones.parquet"

METRICAS_DWLT_FILE = OUTPUT_DIR / "metricas_dwlt_estaciones.parquet"

OUT_XLSX = OUTPUT_DIR / "validacion_dwlt_hasta_observado_disponible.xlsx"
OUT_CSV = OUTPUT_DIR / "validacion_dwlt_hasta_observado_disponible.csv"


# ============================================================
# UTILIDADES
# ============================================================

def normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo requerido: {path}")


def rmse(obs, sim):
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)

    mask = np.isfinite(obs) & np.isfinite(sim)

    if mask.sum() == 0:
        return np.nan

    return float(np.sqrt(np.mean((sim[mask] - obs[mask]) ** 2)))


def mae(obs, sim):
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)

    mask = np.isfinite(obs) & np.isfinite(sim)

    if mask.sum() == 0:
        return np.nan

    return float(np.mean(np.abs(sim[mask] - obs[mask])))


def bias(obs, sim):
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)

    mask = np.isfinite(obs) & np.isfinite(sim)

    if mask.sum() == 0:
        return np.nan

    return float(np.mean(sim[mask] - obs[mask]))


def pearson_r(obs, sim):
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)

    mask = np.isfinite(obs) & np.isfinite(sim)

    if mask.sum() < 3:
        return np.nan

    return float(np.corrcoef(obs[mask], sim[mask])[0, 1])


def nse(obs, sim):
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)

    mask = np.isfinite(obs) & np.isfinite(sim)

    if mask.sum() < 3:
        return np.nan

    obs_m = obs[mask]
    sim_m = sim[mask]

    den = np.sum((obs_m - np.mean(obs_m)) ** 2)

    if den == 0:
        return np.nan

    return float(1 - np.sum((sim_m - obs_m) ** 2) / den)


def kge_2009(obs, sim):
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)

    mask = np.isfinite(obs) & np.isfinite(sim)

    if mask.sum() < 3:
        return np.nan

    obs_m = obs[mask]
    sim_m = sim[mask]

    r = np.corrcoef(obs_m, sim_m)[0, 1]

    std_obs = np.std(obs_m)
    mean_obs = np.mean(obs_m)

    alpha = np.std(sim_m) / std_obs if std_obs != 0 else np.nan
    beta = np.mean(sim_m) / mean_obs if mean_obs != 0 else np.nan

    return float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def clasificar_tendencia(delta: float, umbral: float = 0.05) -> str:
    if pd.isna(delta):
        return "Sin dato"

    if delta > umbral:
        return "Ascendente"

    if delta < -umbral:
        return "Descendente"

    return "Estable"


def clasificar_calidad(kge, rmse_m):
    if pd.isna(kge) and pd.isna(rmse_m):
        return "Sin evaluar"

    if pd.notna(kge):
        if kge >= 0.75:
            return "Buena"
        if kge >= 0.50:
            return "Moderada"
        return "Revisar"

    if pd.notna(rmse_m):
        if rmse_m <= 0.50:
            return "Buena"
        if rmse_m <= 1.00:
            return "Moderada"
        return "Revisar"

    return "Sin evaluar"


# ============================================================
# CARGA DE DATOS
# ============================================================

def cargar_historico_dwlt() -> pd.DataFrame:
    require_file(HIST_DWLT_FILE)

    df = pd.read_parquet(HIST_DWLT_FILE)
    df = normalizar_columnas(df)

    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df["comid"] = pd.to_numeric(df["comid"], errors="coerce").astype("Int64")
    df["nivel_dwlt_m"] = pd.to_numeric(df["nivel_dwlt_m"], errors="coerce")

    if "estacion" not in df.columns:
        df["estacion"] = "SIN_NOMBRE"

    # El histórico transformado puede traer observado incluido.
    # Lo quitamos para volver a cruzar limpio con el parquet observado actualizado.
    columnas_observadas_previas = [
        "nivel_observado_m",
        "estacion_observada",
        "n_obs_dia",
    ]

    df = df.drop(
        columns=[c for c in columnas_observadas_previas if c in df.columns],
        errors="ignore",
    )

    df = df.dropna(subset=["fecha", "comid", "nivel_dwlt_m"]).copy()

    return df


def cargar_observado() -> tuple[pd.DataFrame, Path]:
    require_file(OBS_FILE_CACHE)

    df = pd.read_parquet(OBS_FILE_CACHE)
    df = normalizar_columnas(df)

    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df["comid"] = pd.to_numeric(df["comid"], errors="coerce").astype("Int64")
    df["nivel_m"] = pd.to_numeric(df["nivel_m"], errors="coerce")

    if "estacion_nombre" not in df.columns:
        df["estacion_nombre"] = "SIN_NOMBRE"

    df = df.dropna(subset=["fecha", "comid", "nivel_m"]).copy()
    df = df[df["nivel_m"] > 0].copy()

    obs_diario = (
        df.groupby(["comid", "fecha"], dropna=False)
        .agg(
            estacion_observada=("estacion_nombre", "first"),
            nivel_observado_m=("nivel_m", "mean"),
            n_obs_dia=("nivel_m", "count"),
        )
        .reset_index()
    )

    return obs_diario, OBS_FILE_CACHE


def cargar_metricas_historicas() -> pd.DataFrame:
    if not METRICAS_DWLT_FILE.exists():
        return pd.DataFrame()

    df = pd.read_parquet(METRICAS_DWLT_FILE)
    df = normalizar_columnas(df)

    if "comid" in df.columns:
        df["comid"] = pd.to_numeric(df["comid"], errors="coerce").astype("Int64")

    return df


def cargar_pronostico_actual() -> pd.DataFrame:
    if not FORE_DWLT_FILE.exists():
        return pd.DataFrame()

    df = pd.read_parquet(FORE_DWLT_FILE)
    df = normalizar_columnas(df)

    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df["comid"] = pd.to_numeric(df["comid"], errors="coerce").astype("Int64")

    if "estacion" not in df.columns:
        df["estacion"] = "SIN_NOMBRE"

    for col in ["nivel_min_m", "nivel_prom_m", "nivel_max_m"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["fecha", "comid"]).copy()

    return df


# ============================================================
# VALIDACIÓN
# ============================================================

def construir_validacion_reciente(
    hist_dwlt: pd.DataFrame,
    obs_diario: pd.DataFrame,
    dias: int = 30,
    fecha_inicio: str | None = None,
    fecha_fin: str | None = None,
) -> pd.DataFrame:
    comp = hist_dwlt.merge(
        obs_diario,
        on=["comid", "fecha"],
        how="inner",
    )

    comp = comp.dropna(subset=["nivel_dwlt_m", "nivel_observado_m"]).copy()

    if comp.empty:
        return comp

    if fecha_inicio or fecha_fin:
        if fecha_inicio:
            fi = pd.to_datetime(fecha_inicio, errors="coerce")
            if pd.notna(fi):
                comp = comp[comp["fecha"] >= fi]

        if fecha_fin:
            ff = pd.to_datetime(fecha_fin, errors="coerce")
            if pd.notna(ff):
                comp = comp[comp["fecha"] <= ff]

        return comp.sort_values(["estacion", "comid", "fecha"]).reset_index(drop=True)

    partes = []

    for _, g in comp.groupby("comid", dropna=False):
        g = g.sort_values("fecha").copy()

        fecha_max = g["fecha"].max()
        fecha_min = fecha_max - pd.Timedelta(days=dias - 1)

        g_win = g[(g["fecha"] >= fecha_min) & (g["fecha"] <= fecha_max)].copy()

        partes.append(g_win)

    out = pd.concat(partes, ignore_index=True) if partes else pd.DataFrame()

    return out.sort_values(["estacion", "comid", "fecha"]).reset_index(drop=True)


def agregar_errores_y_tendencias(valid: pd.DataFrame) -> pd.DataFrame:
    if valid.empty:
        return valid

    valid = valid.copy()
    valid = valid.sort_values(["estacion", "comid", "fecha"])

    valid["error_m"] = valid["nivel_dwlt_m"] - valid["nivel_observado_m"]
    valid["error_abs_m"] = valid["error_m"].abs()

    valid["delta_obs_m"] = valid.groupby("comid")["nivel_observado_m"].diff()
    valid["delta_dwlt_m"] = valid.groupby("comid")["nivel_dwlt_m"].diff()

    valid["tendencia_obs"] = valid["delta_obs_m"].apply(clasificar_tendencia)
    valid["tendencia_dwlt"] = valid["delta_dwlt_m"].apply(clasificar_tendencia)

    valid["coincide_tendencia"] = valid["tendencia_obs"] == valid["tendencia_dwlt"]

    valid.loc[
        valid["tendencia_obs"].eq("Sin dato") | valid["tendencia_dwlt"].eq("Sin dato"),
        "coincide_tendencia",
    ] = np.nan

    return valid


def resumen_por_estacion(valid: pd.DataFrame, metricas_hist: pd.DataFrame) -> pd.DataFrame:
    if valid.empty:
        return pd.DataFrame()

    filas = []

    for (estacion, comid), g in valid.groupby(["estacion", "comid"], dropna=False):
        g = g.sort_values("fecha")

        obs = g["nivel_observado_m"]
        sim = g["nivel_dwlt_m"]

        tendencia_valid = g["coincide_tendencia"].dropna()

        coincidencia_tend = np.nan
        if len(tendencia_valid):
            coincidencia_tend = float(tendencia_valid.mean() * 100)

        kge = kge_2009(obs, sim)
        rmse_val = rmse(obs, sim)

        filas.append({
            "estacion": estacion,
            "comid": int(comid) if pd.notna(comid) else None,
            "n_dias_validados": int(len(g)),
            "fecha_inicio_validacion": g["fecha"].min(),
            "fecha_fin_validacion": g["fecha"].max(),
            "nivel_obs_inicio_m": obs.iloc[0],
            "nivel_obs_fin_m": obs.iloc[-1],
            "nivel_dwlt_inicio_m": sim.iloc[0],
            "nivel_dwlt_fin_m": sim.iloc[-1],
            "delta_obs_total_m": obs.iloc[-1] - obs.iloc[0] if len(g) >= 2 else np.nan,
            "delta_dwlt_total_m": sim.iloc[-1] - sim.iloc[0] if len(g) >= 2 else np.nan,
            "bias_m": bias(obs, sim),
            "mae_m": mae(obs, sim),
            "rmse_m": rmse_val,
            "r_pearson": pearson_r(obs, sim),
            "nse": nse(obs, sim),
            "kge_2009": kge,
            "coincidencia_tendencia_pct": coincidencia_tend,
            "calidad_validacion_reciente": clasificar_calidad(kge, rmse_val),
        })

    resumen = pd.DataFrame(filas)

    if not metricas_hist.empty and "comid" in metricas_hist.columns:
        cols = [
            "comid",
            "n_validacion",
            "r_pearson",
            "nse",
            "kge_2009",
            "rmse_m",
        ]

        cols = [c for c in cols if c in metricas_hist.columns]

        met = metricas_hist[cols].copy()
        met = met.rename(columns={
            "n_validacion": "hist_n_validacion",
            "r_pearson": "hist_r_pearson",
            "nse": "hist_nse",
            "kge_2009": "hist_kge_2009",
            "rmse_m": "hist_rmse_m",
        })

        resumen = resumen.merge(met, on="comid", how="left")

    for col in resumen.columns:
        if resumen[col].dtype.kind in "fc":
            resumen[col] = resumen[col].round(3)

    return resumen.sort_values(["calidad_validacion_reciente", "estacion"]).reset_index(drop=True)


def resumen_pronostico_actual(
    fore: pd.DataFrame,
    resumen_validacion: pd.DataFrame,
) -> pd.DataFrame:
    if fore.empty:
        return pd.DataFrame()

    if "nivel_prom_m" not in fore.columns:
        return pd.DataFrame()

    resumen_fore = (
        fore.groupby(["estacion", "comid"], dropna=False)
        .agg(
            fecha_inicio_forecast=("fecha", "min"),
            fecha_fin_forecast=("fecha", "max"),
            n_dias_forecast=("fecha", "count"),
            nivel_min_forecast_m=("nivel_min_m", "min"),
            nivel_prom_forecast_m=("nivel_prom_m", "mean"),
            nivel_max_forecast_m=("nivel_max_m", "max"),
            nivel_inicio_forecast_m=("nivel_prom_m", "first"),
            nivel_fin_forecast_m=("nivel_prom_m", "last"),
        )
        .reset_index()
    )

    resumen_fore["tendencia_forecast_m"] = (
        resumen_fore["nivel_fin_forecast_m"] - resumen_fore["nivel_inicio_forecast_m"]
    )

    resumen_fore["tendencia_forecast"] = resumen_fore["tendencia_forecast_m"].apply(
        clasificar_tendencia
    )

    cols_valid = [
        "comid",
        "fecha_fin_validacion",
        "n_dias_validados",
        "rmse_m",
        "mae_m",
        "bias_m",
        "kge_2009",
        "calidad_validacion_reciente",
    ]

    cols_valid = [c for c in cols_valid if c in resumen_validacion.columns]

    if cols_valid:
        resumen_fore = resumen_fore.merge(
            resumen_validacion[cols_valid],
            on="comid",
            how="left",
        )

    resumen_fore["nota"] = (
        "Pronóstico prospectivo. La validación directa solo aplica cuando exista observado para esas fechas."
    )

    for col in resumen_fore.columns:
        if resumen_fore[col].dtype.kind in "fc":
            resumen_fore[col] = resumen_fore[col].round(3)

    return resumen_fore


# ============================================================
# EXPORTACIÓN
# ============================================================

def convertir_fechas_texto(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.strftime("%d/%m/%Y")

    return df


def exportar(
    valid: pd.DataFrame,
    resumen_estaciones: pd.DataFrame,
    resumen_forecast: pd.DataFrame,
    hist_dwlt: pd.DataFrame,
    obs_diario: pd.DataFrame,
    obs_path: Path,
    dias: int,
) -> None:
    valid_export = convertir_fechas_texto(valid)
    resumen_export = convertir_fechas_texto(resumen_estaciones)
    forecast_export = convertir_fechas_texto(resumen_forecast)

    valid_export.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    resumen_general = pd.DataFrame([{
        "archivo_historico_dwlt": str(HIST_DWLT_FILE),
        "archivo_observado_usado": str(obs_path),
        "ventana_dias_por_defecto": dias,
        "registros_historico_dwlt": len(hist_dwlt),
        "registros_observado_diario": len(obs_diario),
        "registros_validados": len(valid),
        "fecha_inicio_historico_dwlt": hist_dwlt["fecha"].min(),
        "fecha_fin_historico_dwlt": hist_dwlt["fecha"].max(),
        "fecha_inicio_observado": obs_diario["fecha"].min(),
        "fecha_fin_observado": obs_diario["fecha"].max(),
    }])

    resumen_general = convertir_fechas_texto(resumen_general)

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        resumen_general.to_excel(writer, sheet_name="resumen_general", index=False)
        resumen_export.to_excel(writer, sheet_name="resumen_estaciones", index=False)
        valid_export.to_excel(writer, sheet_name="validacion_diaria", index=False)

        if not forecast_export.empty:
            forecast_export.to_excel(writer, sheet_name="forecast_actual_confianza", index=False)

    print("\nArchivos generados:")
    print(f" - {OUT_XLSX}")
    print(f" - {OUT_CSV}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Validación DWLT usando observado hasta donde exista por estación."
    )

    parser.add_argument(
        "--dias",
        type=int,
        default=30,
        help="Número de días recientes disponibles por estación. Por defecto: 30.",
    )

    parser.add_argument(
        "--fecha-inicio",
        default=None,
        help="Fecha inicial manual, ejemplo: 2026-06-01.",
    )

    parser.add_argument(
        "--fecha-fin",
        default=None,
        help="Fecha final manual, ejemplo: 2026-06-30.",
    )

    args = parser.parse_args()

    print("=" * 100)
    print("VALIDACIÓN DWLT HASTA OBSERVADO DISPONIBLE")
    print("=" * 100)

    hist_dwlt = cargar_historico_dwlt()
    obs_diario, obs_path = cargar_observado()
    metricas_hist = cargar_metricas_historicas()
    fore = cargar_pronostico_actual()

    print(f"Histórico DWLT: {len(hist_dwlt):,} registros")
    print(f"Observado diario: {len(obs_diario):,} registros")
    print(f"Archivo observado usado: {obs_path}")

    print(f"Periodo histórico DWLT: {hist_dwlt['fecha'].min()} a {hist_dwlt['fecha'].max()}")
    print(f"Periodo observado: {obs_diario['fecha'].min()} a {obs_diario['fecha'].max()}")

    if args.fecha_inicio or args.fecha_fin:
        print(f"Rango manual: {args.fecha_inicio} a {args.fecha_fin}")
    else:
        print(f"Validando últimos {args.dias} días disponibles por estación.")

    valid = construir_validacion_reciente(
        hist_dwlt=hist_dwlt,
        obs_diario=obs_diario,
        dias=args.dias,
        fecha_inicio=args.fecha_inicio,
        fecha_fin=args.fecha_fin,
    )

    valid = agregar_errores_y_tendencias(valid)

    resumen_estaciones = resumen_por_estacion(
        valid=valid,
        metricas_hist=metricas_hist,
    )

    resumen_forecast = resumen_pronostico_actual(
        fore=fore,
        resumen_validacion=resumen_estaciones,
    )

    exportar(
        valid=valid,
        resumen_estaciones=resumen_estaciones,
        resumen_forecast=resumen_forecast,
        hist_dwlt=hist_dwlt,
        obs_diario=obs_diario,
        obs_path=obs_path,
        dias=args.dias,
    )

    print("\nResumen por estación:")
    cols_print = [
        "estacion",
        "comid",
        "n_dias_validados",
        "fecha_fin_validacion",
        "bias_m",
        "mae_m",
        "rmse_m",
        "r_pearson",
        "kge_2009",
        "coincidencia_tendencia_pct",
        "calidad_validacion_reciente",
        "hist_kge_2009",
    ]

    cols_print = [c for c in cols_print if c in resumen_estaciones.columns]

    if cols_print and not resumen_estaciones.empty:
        print(resumen_estaciones[cols_print].to_string(index=False))
    else:
        print("No se generó resumen por estación.")

    print("\nProceso terminado correctamente.")


if __name__ == "__main__":
    main()
