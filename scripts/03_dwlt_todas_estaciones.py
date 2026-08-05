from __future__ import annotations

import math
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CACHE_DIR = BASE_DIR / "backend" / "cache"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GPKG_FILE = BASE_DIR / "Data" / "estaciones_hidrometricas_loreto_latlong_COMID_actualizado.gpkg"

HIST_FILE = CACHE_DIR / "hist_filtrado.parquet"
FORE_FILE = CACHE_DIR / "fore_filtrado.parquet"
OBS_FILE = CACHE_DIR / "observado_estaciones.parquet"

OUT_HIST = OUTPUT_DIR / "hist_nivel_transformado.parquet"
OUT_FORE = OUTPUT_DIR / "fore_nivel_transformado.parquet"
OUT_METRICAS_PARQUET = OUTPUT_DIR / "metricas_dwlt_estaciones.parquet"
OUT_METRICAS_XLSX = OUTPUT_DIR / "metricas_dwlt_estaciones.xlsx"

MIN_DATOS_MES_SIM = 30
MIN_DATOS_MES_OBS = 30

# Más parecido al correct_forecast original:
# usa el mes del inicio del pronóstico para todo el horizonte.
FORECAST_USA_MES_INICIO = True


# ============================================================
# UTILIDADES GENERALES
# ============================================================

def normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo requerido: {path}")


def get_nombre_estacion(row: pd.Series) -> str:
    for col in ["estacion", "estacion_nombre", "estacion_catalogo", "estaciones_hidro", "nombre"]:
        if col in row.index and pd.notna(row[col]):
            value = str(row[col]).strip()
            if value:
                return value

    return "SIN_NOMBRE"


# ============================================================
# LECTURA GPKG SIN GEOPANDAS
# ============================================================

def quote_sql_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def find_table_with_comid(gpkg_path: Path) -> tuple[str, list[str]]:
    conn = sqlite3.connect(str(gpkg_path))

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
        excluded_prefixes = ("gpkg_", "sqlite_", "rtree_")

        for table in tables:
            if table.lower().startswith(excluded_prefixes):
                continue

            info = conn.execute(
                f"PRAGMA table_info({quote_sql_identifier(table)})"
            ).fetchall()

            cols = [row[1] for row in info]
            cols_lower = [c.lower() for c in cols]

            if "comid" in cols_lower:
                return table, cols

        raise ValueError("No se encontró ninguna tabla del GPKG con columna COMID.")

    finally:
        conn.close()


def leer_estaciones_gpkg(gpkg_path: Path) -> pd.DataFrame:
    table, _ = find_table_with_comid(gpkg_path)

    conn = sqlite3.connect(str(gpkg_path))

    try:
        query = f"SELECT * FROM {quote_sql_identifier(table)}"
        df = pd.read_sql_query(query, conn)

    finally:
        conn.close()

    df = normalizar_columnas(df)

    if "comid" not in df.columns:
        raise ValueError("El GPKG debe tener columna COMID.")

    df["comid"] = pd.to_numeric(df["comid"], errors="coerce")
    df = df.dropna(subset=["comid"]).copy()
    df["comid"] = df["comid"].astype("int64")

    df = df.drop_duplicates(subset=["comid"], keep="first").reset_index(drop=True)

    return df


# ============================================================
# MÉTRICAS
# ============================================================

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

    return float(1 - (np.sum((sim_m - obs_m) ** 2) / den))


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


# ============================================================
# DWLT OPTIMIZADO: HISTOGRAMA + CDF MENSUAL
# ============================================================

def limpiar_serie(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return arr


def sturges_bins(n: int) -> int:
    if n <= 1:
        return 1

    return max(2, int(math.ceil(1 + math.log2(n))))


def construir_cdf_histograma(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = limpiar_serie(values)

    if len(values) < 5:
        return np.array([]), np.array([])

    vmin = np.nanmin(values)
    vmax = np.nanmax(values)

    if vmin == vmax:
        x = np.array([vmin - 1e-9, vmin, vmax + 1e-9], dtype=float)
        p = np.array([0.0, 0.5, 1.0], dtype=float)
        return x, p

    bins = sturges_bins(len(values))
    counts, bin_edges = np.histogram(values, bins=bins)

    if counts.sum() == 0:
        return np.array([]), np.array([])

    cdf = np.cumsum(counts).astype(float) / float(counts.sum())

    x_points = np.concatenate(([bin_edges[0]], bin_edges[1:]))
    p_points = np.concatenate(([0.0], cdf))

    tmp = pd.DataFrame({
        "x": x_points,
        "p": p_points,
    })

    tmp = (
        tmp.groupby("x", as_index=False)
        .agg(p=("p", "max"))
        .sort_values("x")
    )

    return tmp["x"].to_numpy(dtype=float), tmp["p"].to_numpy(dtype=float)


def preparar_inversa_cdf(x_points: np.ndarray, p_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tmp = pd.DataFrame({
        "p": p_points,
        "x": x_points,
    })

    tmp = (
        tmp.groupby("p", as_index=False)
        .agg(x=("x", "mean"))
        .sort_values("p")
    )

    return tmp["p"].to_numpy(dtype=float), tmp["x"].to_numpy(dtype=float)


def interp_lineal_extrap(x, xp, fp, extrapolate: bool = False):
    """
    Interpolación vectorizada.
    Si extrapolate=False, clipea fuera de rango.
    Si extrapolate=True, extrapola linealmente en extremos.
    """

    x = np.asarray(x, dtype=float)
    xp = np.asarray(xp, dtype=float)
    fp = np.asarray(fp, dtype=float)

    out = np.interp(x, xp, fp, left=fp[0], right=fp[-1])

    if not extrapolate or len(xp) < 2:
        return out

    left_mask = x < xp[0]
    right_mask = x > xp[-1]

    if left_mask.any():
        slope_left = (fp[1] - fp[0]) / (xp[1] - xp[0])
        out[left_mask] = fp[0] + slope_left * (x[left_mask] - xp[0])

    if right_mask.any():
        slope_right = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
        out[right_mask] = fp[-1] + slope_right * (x[right_mask] - xp[-1])

    return out


def construir_curvas_mensuales(hist_sim_est: pd.DataFrame, obs_est: pd.DataFrame) -> dict[int, dict]:
    """
    Construye una sola vez las 12 curvas mensuales por estación.

    Para cada mes guarda:
    - caudal simulado -> probabilidad
    - probabilidad -> nivel observado
    """

    curvas = {}

    for mes in range(1, 13):
        sim_mes = hist_sim_est.loc[
            hist_sim_est["mes"] == mes,
            "qr_hist",
        ].to_numpy(dtype=float)

        obs_mes = obs_est.loc[
            obs_est["mes"] == mes,
            "nivel_m",
        ].to_numpy(dtype=float)

        sim_mes = limpiar_serie(sim_mes)
        obs_mes = limpiar_serie(obs_mes)

        if len(sim_mes) < MIN_DATOS_MES_SIM or len(obs_mes) < MIN_DATOS_MES_OBS:
            curvas[mes] = {
                "ok": False,
                "n_sim": len(sim_mes),
                "n_obs": len(obs_mes),
            }
            continue

        sim_x, sim_p = construir_cdf_histograma(sim_mes)
        obs_x, obs_p = construir_cdf_histograma(obs_mes)
        obs_p_inv, obs_x_inv = preparar_inversa_cdf(obs_x, obs_p)

        if len(sim_x) < 2 or len(obs_p_inv) < 2:
            curvas[mes] = {
                "ok": False,
                "n_sim": len(sim_mes),
                "n_obs": len(obs_mes),
            }
            continue

        curvas[mes] = {
            "ok": True,
            "n_sim": len(sim_mes),
            "n_obs": len(obs_mes),
            "sim_x": sim_x,
            "sim_p": sim_p,
            "obs_p": obs_p_inv,
            "obs_x": obs_x_inv,
        }

    return curvas


def transformar_array_dwlt(
    caudales,
    meses,
    curvas: dict[int, dict],
    extrapolate: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Transforma arrays completos de caudal a nivel usando curvas ya precalculadas.
    """

    caudales = np.asarray(caudales, dtype=float)
    meses = np.asarray(meses, dtype=float)

    niveles = np.full(len(caudales), np.nan, dtype=float)
    probs = np.full(len(caudales), np.nan, dtype=float)

    for mes in range(1, 13):
        mask = (
            np.isfinite(caudales)
            & np.isfinite(meses)
            & (meses.astype(float) == float(mes))
        )

        if not mask.any():
            continue

        curva = curvas.get(mes, {"ok": False})

        if not curva.get("ok", False):
            continue

        q = caudales[mask]

        p = interp_lineal_extrap(
            x=q,
            xp=curva["sim_x"],
            fp=curva["sim_p"],
            extrapolate=extrapolate,
        )

        p = np.clip(p, 0.0, 1.0)

        h = interp_lineal_extrap(
            x=p,
            xp=curva["obs_p"],
            fp=curva["obs_x"],
            extrapolate=extrapolate,
        )

        probs[mask] = p
        niveles[mask] = h

    return niveles, probs


def transformar_historico_estacion(
    comid: int,
    estacion: str,
    hist_sim_est: pd.DataFrame,
    obs_est: pd.DataFrame,
    curvas: dict[int, dict],
) -> pd.DataFrame:
    out = hist_sim_est[["fecha", "comid", "mes", "qr_hist"]].copy()
    out["estacion"] = estacion
    out["mes_dwlt"] = out["mes"].astype(int)

    niveles, probs = transformar_array_dwlt(
        caudales=out["qr_hist"].to_numpy(dtype=float),
        meses=out["mes_dwlt"].to_numpy(dtype=float),
        curvas=curvas,
        extrapolate=False,
    )

    out["prob_no_excedencia"] = probs
    out["nivel_dwlt_m"] = niveles
    out["metodo_dwlt"] = "histograma_cdf_mensual_optimizado"

    obs_diario = obs_est[["fecha", "nivel_m"]].copy()
    obs_diario = obs_diario.rename(columns={"nivel_m": "nivel_observado_m"})

    out = out.merge(obs_diario, on="fecha", how="left")

    return out.sort_values("fecha").reset_index(drop=True)


def transformar_pronostico_estacion(
    comid: int,
    estacion: str,
    fore_est: pd.DataFrame,
    curvas: dict[int, dict],
) -> pd.DataFrame:
    out = fore_est.copy()
    out["estacion"] = estacion

    if out.empty:
        return out

    if FORECAST_USA_MES_INICIO:
        mes_inicio = int(pd.to_datetime(out["fecha"].min()).month)
        out["mes_dwlt"] = mes_inicio
    else:
        out["mes_dwlt"] = out["mes"].astype(int)

    columnas_fore = ["qr_eta_eqm", "qr_eta_scal", "qr_gfs", "qr_wrf"]

    for col in columnas_fore:
        if col not in out.columns:
            continue

        niveles, probs = transformar_array_dwlt(
            caudales=out[col].to_numpy(dtype=float),
            meses=out["mes_dwlt"].to_numpy(dtype=float),
            curvas=curvas,
            extrapolate=True,
        )

        sufijo = col.replace("qr_", "")

        out[f"nivel_{sufijo}_m"] = niveles
        out[f"prob_{sufijo}"] = probs

    nivel_cols = [
        c for c in out.columns
        if c.startswith("nivel_") and c.endswith("_m")
    ]

    if nivel_cols:
        out["nivel_min_m"] = out[nivel_cols].min(axis=1)
        out["nivel_p25_m"] = out[nivel_cols].quantile(0.25, axis=1)
        out["nivel_prom_m"] = out[nivel_cols].mean(axis=1)
        out["nivel_p75_m"] = out[nivel_cols].quantile(0.75, axis=1)
        out["nivel_max_m"] = out[nivel_cols].max(axis=1)

    out["metodo_dwlt"] = "histograma_cdf_mensual_optimizado"

    return out.sort_values("fecha").reset_index(drop=True)


# ============================================================
# MÉTRICAS POR ESTACIÓN
# ============================================================

def calcular_metricas_estacion(
    comid: int,
    estacion: str,
    hist_dwlt_est: pd.DataFrame,
    hist_sim_est: pd.DataFrame,
    obs_est: pd.DataFrame,
    fore_dwlt_est: pd.DataFrame,
    curvas: dict[int, dict],
) -> dict:
    valid = hist_dwlt_est.dropna(subset=["nivel_observado_m", "nivel_dwlt_m"]).copy()

    if len(valid) >= 3:
        obs_v = valid["nivel_observado_m"]
        sim_v = valid["nivel_dwlt_m"]

        metricas = {
            "bias_m": bias(obs_v, sim_v),
            "mae_m": mae(obs_v, sim_v),
            "rmse_m": rmse(obs_v, sim_v),
            "r_pearson": pearson_r(obs_v, sim_v),
            "nse": nse(obs_v, sim_v),
            "kge_2009": kge_2009(obs_v, sim_v),
        }
    else:
        metricas = {
            "bias_m": np.nan,
            "mae_m": np.nan,
            "rmse_m": np.nan,
            "r_pearson": np.nan,
            "nse": np.nan,
            "kge_2009": np.nan,
        }

    meses_ok = [m for m, c in curvas.items() if c.get("ok", False)]

    out = {
        "estacion": estacion,
        "comid": comid,
        "metodo_dwlt": "histograma_cdf_mensual_optimizado",
        "n_hist_sonics": int(len(hist_sim_est)),
        "n_obs_nivel": int(len(obs_est)),
        "n_fore_sonics": int(len(fore_dwlt_est)),
        "n_validacion": int(len(valid)),
        "meses_curva_ok": ",".join([str(m) for m in meses_ok]),
        "n_meses_curva_ok": int(len(meses_ok)),
        "fecha_inicio_validacion": valid["fecha"].min() if len(valid) else pd.NaT,
        "fecha_fin_validacion": valid["fecha"].max() if len(valid) else pd.NaT,
        "nivel_obs_min": valid["nivel_observado_m"].min() if len(valid) else np.nan,
        "nivel_obs_prom": valid["nivel_observado_m"].mean() if len(valid) else np.nan,
        "nivel_obs_max": valid["nivel_observado_m"].max() if len(valid) else np.nan,
        "nivel_dwlt_min": valid["nivel_dwlt_m"].min() if len(valid) else np.nan,
        "nivel_dwlt_prom": valid["nivel_dwlt_m"].mean() if len(valid) else np.nan,
        "nivel_dwlt_max": valid["nivel_dwlt_m"].max() if len(valid) else np.nan,
    }

    out.update(metricas)

    if len(fore_dwlt_est) and "nivel_prom_m" in fore_dwlt_est.columns:
        out["forecast_nivel_min_m"] = fore_dwlt_est["nivel_min_m"].min()
        out["forecast_nivel_prom_m"] = fore_dwlt_est["nivel_prom_m"].mean()
        out["forecast_nivel_max_m"] = fore_dwlt_est["nivel_max_m"].max()
    else:
        out["forecast_nivel_min_m"] = np.nan
        out["forecast_nivel_prom_m"] = np.nan
        out["forecast_nivel_max_m"] = np.nan

    return out


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 100)
    print("DWLT PARA TODAS LAS ESTACIONES - LORETO")
    print("MÉTODO: HISTOGRAMA + CDF MENSUAL OPTIMIZADO")
    print("=" * 100)

    require_file(GPKG_FILE)
    require_file(HIST_FILE)
    require_file(FORE_FILE)
    require_file(OBS_FILE)

    print("\nLeyendo archivos...")
    estaciones = leer_estaciones_gpkg(GPKG_FILE)
    hist = pd.read_parquet(HIST_FILE)
    fore = pd.read_parquet(FORE_FILE)
    obs = pd.read_parquet(OBS_FILE)

    estaciones = normalizar_columnas(estaciones)
    hist = normalizar_columnas(hist)
    fore = normalizar_columnas(fore)
    obs = normalizar_columnas(obs)

    for df in [hist, fore, obs]:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
        df["comid"] = pd.to_numeric(df["comid"], errors="coerce").astype("Int64")
        df["mes"] = df["fecha"].dt.month

    estaciones["comid"] = pd.to_numeric(estaciones["comid"], errors="coerce").astype("Int64")

    if "qr_hist" not in hist.columns:
        raise ValueError("hist_filtrado.parquet debe tener columna qr_hist.")

    hist["qr_hist"] = pd.to_numeric(hist["qr_hist"], errors="coerce")

    for col in ["qr_eta_eqm", "qr_eta_scal", "qr_gfs", "qr_wrf"]:
        if col in fore.columns:
            fore[col] = pd.to_numeric(fore[col], errors="coerce")
            fore.loc[fore[col] <= 0, col] = np.nan

    if "nivel_m" not in obs.columns:
        raise ValueError("observado_estaciones.parquet debe tener columna nivel_m.")

    obs["nivel_m"] = pd.to_numeric(obs["nivel_m"], errors="coerce")

    # Limpieza de negativos, ceros y nulos.
    hist = hist.dropna(subset=["fecha", "comid", "qr_hist", "mes"]).copy()
    hist = hist[hist["qr_hist"] > 0].copy()

    fore = fore.dropna(subset=["fecha", "comid", "mes"]).copy()

    obs = obs.dropna(subset=["fecha", "comid", "nivel_m", "mes"]).copy()
    obs = obs[obs["nivel_m"] > 0].copy()

    estaciones_validas = estaciones.dropna(subset=["comid"]).copy()
    estaciones_validas["comid"] = estaciones_validas["comid"].astype("int64")

    print(f"Estaciones válidas desde GPKG: {len(estaciones_validas)}")
    print(f"Histórico SONICS: {len(hist):,} registros")
    print(f"Pronóstico SONICS: {len(fore):,} registros")
    print(f"Observado nivel: {len(obs):,} registros")
    print(f"Forecast usa mes de inicio: {FORECAST_USA_MES_INICIO}")

    lista_hist_dwlt = []
    lista_fore_dwlt = []
    lista_metricas = []

    total_est = len(estaciones_validas)

    for idx, (_, est_row) in enumerate(estaciones_validas.iterrows(), start=1):
        comid = int(est_row["comid"])
        estacion = get_nombre_estacion(est_row)

        print("\n" + "-" * 90)
        print(f"[{idx}/{total_est}] Procesando {estacion} | COMID {comid}")

        hist_est = hist[hist["comid"] == comid].copy()
        fore_est = fore[fore["comid"] == comid].copy()
        obs_est = obs[obs["comid"] == comid].copy()

        print(f"  Histórico: {len(hist_est):,} | Observado: {len(obs_est):,} | Pronóstico: {len(fore_est):,}")

        if len(hist_est) < 365:
            print("  Saltando: histórico SONICS insuficiente.")
            continue

        if len(obs_est) < 365:
            print("  Saltando: observado insuficiente.")
            continue

        if len(fore_est) == 0:
            print("  Saltando: sin pronóstico.")
            continue

        print("  Construyendo curvas mensuales...")
        curvas = construir_curvas_mensuales(
            hist_sim_est=hist_est,
            obs_est=obs_est,
        )

        meses_ok = [m for m, c in curvas.items() if c.get("ok", False)]
        print(f"  Meses con curva válida: {meses_ok}")

        if not meses_ok:
            print("  Saltando: sin curvas mensuales válidas.")
            continue

        print("  Transformando histórico...")
        hist_dwlt_est = transformar_historico_estacion(
            comid=comid,
            estacion=estacion,
            hist_sim_est=hist_est,
            obs_est=obs_est,
            curvas=curvas,
        )

        print("  Transformando pronóstico...")
        fore_dwlt_est = transformar_pronostico_estacion(
            comid=comid,
            estacion=estacion,
            fore_est=fore_est,
            curvas=curvas,
        )

        metricas_est = calcular_metricas_estacion(
            comid=comid,
            estacion=estacion,
            hist_dwlt_est=hist_dwlt_est,
            hist_sim_est=hist_est,
            obs_est=obs_est,
            fore_dwlt_est=fore_dwlt_est,
            curvas=curvas,
        )

        lista_hist_dwlt.append(hist_dwlt_est)
        lista_fore_dwlt.append(fore_dwlt_est)
        lista_metricas.append(metricas_est)

        print(
            f"  OK | Validación: {metricas_est['n_validacion']:,} | "
            f"KGE: {metricas_est['kge_2009']:.3f} | "
            f"R: {metricas_est['r_pearson']:.3f} | "
            f"RMSE: {metricas_est['rmse_m']:.3f}"
        )

    if not lista_hist_dwlt:
        raise ValueError("No se generó ninguna serie histórica transformada.")

    if not lista_fore_dwlt:
        raise ValueError("No se generó ningún pronóstico transformado.")

    hist_dwlt = pd.concat(lista_hist_dwlt, ignore_index=True)
    fore_dwlt = pd.concat(lista_fore_dwlt, ignore_index=True)
    metricas = pd.DataFrame(lista_metricas)

    print("\nGuardando resultados...")
    hist_dwlt.to_parquet(OUT_HIST, index=False)
    fore_dwlt.to_parquet(OUT_FORE, index=False)
    metricas.to_parquet(OUT_METRICAS_PARQUET, index=False)

    with pd.ExcelWriter(OUT_METRICAS_XLSX, engine="openpyxl") as writer:
        metricas.to_excel(writer, sheet_name="metricas_dwlt", index=False)

        resumen_fore = (
            fore_dwlt.groupby(["estacion", "comid"], dropna=False)
            .agg(
                fecha_inicio_fore=("fecha", "min"),
                fecha_fin_fore=("fecha", "max"),
                registros_fore=("fecha", "count"),
                nivel_min_m=("nivel_min_m", "min"),
                nivel_prom_m=("nivel_prom_m", "mean"),
                nivel_max_m=("nivel_max_m", "max"),
            )
            .reset_index()
        )

        resumen_fore.to_excel(writer, sheet_name="resumen_forecast_nivel", index=False)

    print(f"Histórico transformado: {OUT_HIST}")
    print(f"Pronóstico transformado: {OUT_FORE}")
    print(f"Métricas Parquet: {OUT_METRICAS_PARQUET}")
    print(f"Métricas Excel: {OUT_METRICAS_XLSX}")

    print("\nResumen de métricas:")
    cols_show = [
        "estacion",
        "comid",
        "metodo_dwlt",
        "n_validacion",
        "n_meses_curva_ok",
        "r_pearson",
        "nse",
        "kge_2009",
        "rmse_m",
        "forecast_nivel_min_m",
        "forecast_nivel_prom_m",
        "forecast_nivel_max_m",
    ]

    cols_show = [c for c in cols_show if c in metricas.columns]
    print(metricas[cols_show].to_string(index=False))

    print("\nProceso terminado correctamente.")


if __name__ == "__main__":
    main()
