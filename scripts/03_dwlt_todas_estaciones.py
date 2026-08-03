from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(r"C:\Users\mgutierrez\Documents\POI-MAX\POI 2026\NivelaCaudal")

CACHE_DIR = BASE_DIR / "backend" / "cache"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ESTACIONES_FILE = CACHE_DIR / "estaciones_filtradas.csv"
HIST_FILE = CACHE_DIR / "hist_filtrado.parquet"
FORE_FILE = CACHE_DIR / "fore_filtrado.parquet"
OBS_FILE = CACHE_DIR / "observado_estaciones.parquet"

OUT_HIST = OUTPUT_DIR / "hist_nivel_transformado.parquet"
OUT_FORE = OUTPUT_DIR / "fore_nivel_transformado.parquet"
OUT_METRICAS_PARQUET = OUTPUT_DIR / "metricas_dwlt_estaciones.parquet"
OUT_METRICAS_XLSX = OUTPUT_DIR / "metricas_dwlt_estaciones.xlsx"

ENV_PATH = BASE_DIR / ".env"


# ============================================================
# CARGAR .ENV
# ============================================================

def load_local_env(env_path: Path) -> None:
    if not env_path.exists():
        return

    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if key and key not in os.environ:
                os.environ[key] = value


load_local_env(ENV_PATH)

DRIVE_FOLDER_ID = os.getenv("DRIVE_FOLDER_ID", "").strip()
GOOGLE_SERVICE_JSON = os.getenv("GOOGLE_SERVICE_JSON", "").strip()
GOOGLE_SERVICE_JSON_PATH = os.getenv("GOOGLE_SERVICE_JSON_PATH", "").strip()


# ============================================================
# UTILIDADES
# ============================================================

def normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def get_nombre_estacion(row: pd.Series) -> str:
    for col in ["estacion", "estacion_nombre", "estacion_catalogo", "estaciones_hidro", "nombre"]:
        if col in row.index and pd.notna(row[col]):
            value = str(row[col]).strip()
            if value:
                return value

    return "SIN_NOMBRE"


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo requerido: {path}")


# ============================================================
# MÉTRICAS
# ============================================================

def rmse(obs, sim):
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    return np.sqrt(np.nanmean((sim - obs) ** 2))


def mae(obs, sim):
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    return np.nanmean(np.abs(sim - obs))


def bias(obs, sim):
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    return np.nanmean(sim - obs)


def pearson_r(obs, sim):
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)

    mask = np.isfinite(obs) & np.isfinite(sim)

    if mask.sum() < 3:
        return np.nan

    return np.corrcoef(obs[mask], sim[mask])[0, 1]


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

    return 1 - (np.sum((sim_m - obs_m) ** 2) / den)


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

    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


# ============================================================
# FUNCIONES DWLT
# ============================================================

def prob_no_excedencia(valor: float, serie_simulada: np.ndarray) -> float:
    """
    Calcula P(X <= valor), es decir la probabilidad de no excedencia.
    """

    serie = np.asarray(serie_simulada, dtype=float)
    serie = serie[np.isfinite(serie)]

    if len(serie) < 5:
        return np.nan

    serie_ordenada = np.sort(serie)

    probs = np.arange(1, len(serie_ordenada) + 1, dtype=float) / (len(serie_ordenada) + 1)

    p = np.interp(
        valor,
        serie_ordenada,
        probs,
        left=probs[0],
        right=probs[-1],
    )

    return float(p)


def nivel_equivalente(probabilidad: float, serie_nivel_observado: np.ndarray) -> float:
    """
    Obtiene el nivel observado asociado a la misma probabilidad de no excedencia.
    """

    niveles = np.asarray(serie_nivel_observado, dtype=float)
    niveles = niveles[np.isfinite(niveles)]

    if len(niveles) < 5:
        return np.nan

    niveles_ordenados = np.sort(niveles)

    probs = np.arange(1, len(niveles_ordenados) + 1, dtype=float) / (len(niveles_ordenados) + 1)

    nivel = np.interp(
        probabilidad,
        probs,
        niveles_ordenados,
        left=niveles_ordenados[0],
        right=niveles_ordenados[-1],
    )

    return float(nivel)


def transformar_valor_dwlt(
    caudal: float,
    mes: int,
    hist_sim_est: pd.DataFrame,
    obs_est: pd.DataFrame,
) -> tuple[float, float]:
    """
    Transforma un caudal SONICS a nivel de agua mediante DWLT mensual.
    """

    if pd.isna(caudal):
        return np.nan, np.nan

    sim_mes = hist_sim_est.loc[hist_sim_est["mes"] == mes, "qr_hist"].to_numpy(dtype=float)
    obs_mes = obs_est.loc[obs_est["mes"] == mes, "nivel_m"].to_numpy(dtype=float)

    # Requisito mínimo para no construir curvas pobres.
    if len(sim_mes) < 30 or len(obs_mes) < 30:
        return np.nan, np.nan

    p = prob_no_excedencia(float(caudal), sim_mes)
    nivel = nivel_equivalente(p, obs_mes)

    return nivel, p


def transformar_historico_estacion(
    comid: int,
    estacion: str,
    hist_sim_est: pd.DataFrame,
    obs_est: pd.DataFrame,
) -> pd.DataFrame:
    filas = []

    for _, row in hist_sim_est.iterrows():
        nivel, p = transformar_valor_dwlt(
            caudal=row["qr_hist"],
            mes=int(row["mes"]),
            hist_sim_est=hist_sim_est,
            obs_est=obs_est,
        )

        filas.append({
            "fecha": row["fecha"],
            "comid": comid,
            "estacion": estacion,
            "mes": int(row["mes"]),
            "qr_hist": row["qr_hist"],
            "prob_no_excedencia": p,
            "nivel_dwlt_m": nivel,
        })

    out = pd.DataFrame(filas)

    obs_diario = obs_est[["fecha", "nivel_m"]].copy()
    obs_diario = obs_diario.rename(columns={"nivel_m": "nivel_observado_m"})

    out = out.merge(obs_diario, on="fecha", how="left")

    return out.sort_values("fecha").reset_index(drop=True)


def transformar_pronostico_estacion(
    comid: int,
    estacion: str,
    fore_est: pd.DataFrame,
    hist_sim_est: pd.DataFrame,
    obs_est: pd.DataFrame,
) -> pd.DataFrame:
    out = fore_est.copy()
    out["estacion"] = estacion

    columnas_fore = ["qr_eta_eqm", "qr_eta_scal", "qr_gfs", "qr_wrf"]

    for col in columnas_fore:
        if col not in out.columns:
            continue

        niveles = []
        probs = []

        for _, row in out.iterrows():
            nivel, p = transformar_valor_dwlt(
                caudal=row[col],
                mes=int(row["mes"]),
                hist_sim_est=hist_sim_est,
                obs_est=obs_est,
            )

            niveles.append(nivel)
            probs.append(p)

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

    return out.sort_values("fecha").reset_index(drop=True)


def calcular_metricas_estacion(
    comid: int,
    estacion: str,
    hist_dwlt_est: pd.DataFrame,
    hist_sim_est: pd.DataFrame,
    obs_est: pd.DataFrame,
    fore_dwlt_est: pd.DataFrame,
) -> dict:
    valid = hist_dwlt_est.dropna(subset=["nivel_observado_m", "nivel_dwlt_m"]).copy()

    if len(valid) >= 3:
        obs = valid["nivel_observado_m"]
        sim = valid["nivel_dwlt_m"]

        metricas = {
            "bias_m": bias(obs, sim),
            "mae_m": mae(obs, sim),
            "rmse_m": rmse(obs, sim),
            "r_pearson": pearson_r(obs, sim),
            "nse": nse(obs, sim),
            "kge_2009": kge_2009(obs, sim),
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

    out = {
        "estacion": estacion,
        "comid": comid,
        "n_hist_sonics": int(len(hist_sim_est)),
        "n_obs_nivel": int(len(obs_est)),
        "n_fore_sonics": int(len(fore_dwlt_est)),
        "n_validacion": int(len(valid)),
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
# GOOGLE DRIVE
# ============================================================

def get_drive_service():
    if not DRIVE_FOLDER_ID:
        raise RuntimeError("Falta DRIVE_FOLDER_ID en .env.")

    if not GOOGLE_SERVICE_JSON and not GOOGLE_SERVICE_JSON_PATH:
        raise RuntimeError("Falta GOOGLE_SERVICE_JSON o GOOGLE_SERVICE_JSON_PATH en .env.")

    scopes = ["https://www.googleapis.com/auth/drive"]

    if GOOGLE_SERVICE_JSON:
        info = json.loads(GOOGLE_SERVICE_JSON)
    else:
        json_path = Path(GOOGLE_SERVICE_JSON_PATH)

        if not json_path.exists():
            raise FileNotFoundError(f"No existe GOOGLE_SERVICE_JSON_PATH: {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            info = json.load(f)

    creds = service_account.Credentials.from_service_account_info(info, scopes=scopes)

    return build("drive", "v3", credentials=creds, cache_discovery=False)


def listar_drive(service, folder_id: str) -> list[dict]:
    q = f"'{folder_id}' in parents and trashed = false"

    archivos = []
    page_token = None

    while True:
        resp = service.files().list(
            q=q,
            spaces="drive",
            fields="nextPageToken, files(id, name, mimeType, modifiedTime, size)",
            pageSize=1000,
            pageToken=page_token,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()

        archivos.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")

        if not page_token:
            break

    return archivos


def buscar_drive(service, folder_id: str, nombre: str) -> dict | None:
    archivos = listar_drive(service, folder_id)

    for archivo in archivos:
        if archivo["name"].lower() == nombre.lower():
            return archivo

    return None


def subir_o_actualizar_drive(service, local_path: Path, drive_name: str, mime_type: str) -> None:
    existente = buscar_drive(service, DRIVE_FOLDER_ID, drive_name)

    media = MediaFileUpload(
        str(local_path),
        mimetype=mime_type,
        resumable=False,
    )

    if existente:
        service.files().update(
            fileId=existente["id"],
            media_body=media,
            fields="id,name,modifiedTime",
            supportsAllDrives=True,
        ).execute()

        print(f"Actualizado en Drive: {drive_name}")

    else:
        service.files().create(
            body={
                "name": drive_name,
                "parents": [DRIVE_FOLDER_ID],
            },
            media_body=media,
            fields="id,name",
            supportsAllDrives=True,
        ).execute()

        print(f"Creado en Drive: {drive_name}")


def subir_salidas_drive() -> None:
    print("\nSubiendo salidas DWLT a Google Drive...")

    service = get_drive_service()

    subir_o_actualizar_drive(
        service,
        OUT_HIST,
        "hist_nivel_transformado.parquet",
        "application/octet-stream",
    )

    subir_o_actualizar_drive(
        service,
        OUT_FORE,
        "fore_nivel_transformado.parquet",
        "application/octet-stream",
    )

    subir_o_actualizar_drive(
        service,
        OUT_METRICAS_PARQUET,
        "metricas_dwlt_estaciones.parquet",
        "application/octet-stream",
    )

    subir_o_actualizar_drive(
        service,
        OUT_METRICAS_XLSX,
        "metricas_dwlt_estaciones.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 100)
    print("DWLT PARA TODAS LAS ESTACIONES - LORETO")
    print("=" * 100)

    require_file(ESTACIONES_FILE)
    require_file(HIST_FILE)
    require_file(FORE_FILE)
    require_file(OBS_FILE)

    print("\nLeyendo archivos...")
    estaciones = pd.read_csv(ESTACIONES_FILE)
    hist = pd.read_parquet(HIST_FILE)
    fore = pd.read_parquet(FORE_FILE)
    obs = pd.read_parquet(OBS_FILE)

    estaciones = normalizar_columnas(estaciones)
    hist = normalizar_columnas(hist)
    fore = normalizar_columnas(fore)
    obs = normalizar_columnas(obs)

    # Normalizar tipos
    for df in [hist, fore, obs]:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
        df["comid"] = pd.to_numeric(df["comid"], errors="coerce").astype("Int64")
        df["mes"] = df["fecha"].dt.month

    estaciones["comid"] = pd.to_numeric(estaciones["comid"], errors="coerce").astype("Int64")

    hist["qr_hist"] = pd.to_numeric(hist["qr_hist"], errors="coerce")

    for col in ["qr_eta_eqm", "qr_eta_scal", "qr_gfs", "qr_wrf"]:
        if col in fore.columns:
            fore[col] = pd.to_numeric(fore[col], errors="coerce")

    obs["nivel_m"] = pd.to_numeric(obs["nivel_m"], errors="coerce")

    # Limpieza general
    hist = hist.dropna(subset=["fecha", "comid", "qr_hist"])
    hist = hist[hist["qr_hist"] > 0].copy()

    fore = fore.dropna(subset=["fecha", "comid"]).copy()

    obs = obs.dropna(subset=["fecha", "comid", "nivel_m"])
    obs = obs[obs["nivel_m"] > 0].copy()

    estaciones_validas = estaciones.dropna(subset=["comid"]).copy()
    estaciones_validas["comid"] = estaciones_validas["comid"].astype("int64")

    print(f"Estaciones válidas: {len(estaciones_validas)}")
    print(f"Histórico SONICS: {len(hist):,} registros")
    print(f"Pronóstico SONICS: {len(fore):,} registros")
    print(f"Observado nivel: {len(obs):,} registros")

    lista_hist_dwlt = []
    lista_fore_dwlt = []
    lista_metricas = []

    for idx, (_, est_row) in enumerate(estaciones_validas.iterrows(), start=1):
        comid = int(est_row["comid"])
        estacion = get_nombre_estacion(est_row)

        print("\n" + "-" * 90)
        print(f"[{idx}/{len(estaciones_validas)}] Procesando {estacion} | COMID {comid}")

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

        hist_dwlt_est = transformar_historico_estacion(
            comid=comid,
            estacion=estacion,
            hist_sim_est=hist_est,
            obs_est=obs_est,
        )

        fore_dwlt_est = transformar_pronostico_estacion(
            comid=comid,
            estacion=estacion,
            fore_est=fore_est,
            hist_sim_est=hist_est,
            obs_est=obs_est,
        )

        metricas_est = calcular_metricas_estacion(
            comid=comid,
            estacion=estacion,
            hist_dwlt_est=hist_dwlt_est,
            hist_sim_est=hist_est,
            obs_est=obs_est,
            fore_dwlt_est=fore_dwlt_est,
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
        "n_validacion",
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

    try:
        subir_salidas_drive()
    except Exception as e:
        print(f"\nADVERTENCIA: no se pudo subir a Drive: {e}")
        print("Los archivos locales sí fueron generados correctamente.")

    print("\nProceso terminado correctamente.")


if __name__ == "__main__":
    main()