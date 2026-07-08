from __future__ import annotations

import io
import json
import os
import re
import argparse
from datetime import datetime
from ftplib import FTP
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from dotenv import load_dotenv

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

GPKG_PATH = Path(
    os.getenv(
        "GPKG_PATH",
        BASE_DIR / "Data" / "estaciones_hidrometricas_loreto_latlong_COMID_actualizado.gpkg"
    )
)

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FTP_HOST = os.getenv("FTP_HOST", "").strip()
FTP_USER = os.getenv("FTP_USER", "").strip()
FTP_PASS = os.getenv("FTP_PASS", "").strip()
FTP_DIR = os.getenv("FTP_DIR", "/").strip()

DRIVE_FOLDER_ID = os.getenv("DRIVE_FOLDER_ID", "").strip()
GOOGLE_SERVICE_JSON = os.getenv("GOOGLE_SERVICE_JSON", "").strip()
GOOGLE_SERVICE_JSON_PATH = os.getenv("GOOGLE_SERVICE_JSON_PATH", "").strip()

HIST_NAME = "hist_filtrado.parquet"
FORE_NAME = "fore_filtrado.parquet"
META_NAME = "meta_filtrado.json"
ESTACIONES_NAME = "estaciones_filtradas.csv"


# ============================================================
# VALIDACIÓN
# ============================================================

def require_env_ftp() -> None:
    faltan = []

    if not FTP_HOST:
        faltan.append("FTP_HOST")
    if not FTP_USER:
        faltan.append("FTP_USER")
    if not FTP_PASS:
        faltan.append("FTP_PASS")
    if not FTP_DIR:
        faltan.append("FTP_DIR")

    if faltan:
        raise RuntimeError(f"Faltan variables FTP en .env: {', '.join(faltan)}")

    if not GPKG_PATH.exists():
        raise FileNotFoundError(f"No existe el GPKG: {GPKG_PATH}")


def require_env_drive() -> None:
    faltan = []

    if not DRIVE_FOLDER_ID:
        faltan.append("DRIVE_FOLDER_ID")

    if not GOOGLE_SERVICE_JSON and not GOOGLE_SERVICE_JSON_PATH:
        faltan.append("GOOGLE_SERVICE_JSON o GOOGLE_SERVICE_JSON_PATH")

    if faltan:
        raise RuntimeError(f"Faltan variables Drive en .env: {', '.join(faltan)}")

    if GOOGLE_SERVICE_JSON_PATH:
        ruta = Path(GOOGLE_SERVICE_JSON_PATH)
        if not ruta.exists():
            raise FileNotFoundError(f"No existe GOOGLE_SERVICE_JSON_PATH: {ruta}")


# ============================================================
# LECTURA DE GPKG
# ============================================================

def leer_estaciones_gpkg(gpkg_path: Path) -> gpd.GeoDataFrame:
    print("Leyendo estaciones desde GPKG...")

    gdf = gpd.read_file(gpkg_path)

    gdf.columns = [str(c).strip().lower() for c in gdf.columns]

    if "comid" not in gdf.columns:
        raise ValueError("El GPKG debe tener la columna COMID.")

    gdf["comid"] = pd.to_numeric(gdf["comid"], errors="coerce")
    gdf = gdf.dropna(subset=["comid"]).copy()
    gdf["comid"] = gdf["comid"].astype("int64")

    if gdf.empty:
        raise ValueError("No se encontraron COMID válidos en el GPKG.")

    print(f"Estaciones con COMID: {len(gdf)}")

    return gdf


def obtener_comids(gdf: gpd.GeoDataFrame) -> np.ndarray:
    comids = sorted(gdf["comid"].dropna().astype("int64").unique().tolist())
    return np.array(comids, dtype="float64")


def guardar_estaciones_csv(gdf: gpd.GeoDataFrame) -> Path:
    df = gdf.copy()

    if "geometry" in df.columns:
        df["longitud"] = df.geometry.x
        df["latitud"] = df.geometry.y
        df = pd.DataFrame(df.drop(columns="geometry"))

    salida = OUTPUT_DIR / ESTACIONES_NAME
    df.to_csv(salida, index=False, encoding="utf-8-sig")

    print(f"Estaciones guardadas: {salida}")

    return salida


# ============================================================
# FTP
# ============================================================

def parse_ftp_modify(value: str | None) -> datetime | None:
    if not value:
        return None

    value = value.strip()

    for fmt in ("%Y%m%d%H%M%S", "%Y%m%d%H%M%S.%f"):
        try:
            return datetime.strptime(value, fmt)
        except Exception:
            pass

    return None


def extract_ts_from_name(name: str) -> datetime | None:
    patrones = [
        r"(20\d{12})",
        r"(20\d{10})",
        r"(20\d{8})",
        r"(20\d{6})",
    ]

    for patron in patrones:
        m = re.search(patron, name)
        if not m:
            continue

        token = m.group(1)

        for fmt in ("%Y%m%d%H%M%S", "%Y%m%d%H%M", "%Y%m%d%H", "%Y%m%d"):
            try:
                return datetime.strptime(token, fmt)
            except Exception:
                pass

    return None


def list_nc_files(ftp: FTP) -> list[tuple[str, datetime | None]]:
    files: list[tuple[str, datetime | None]] = []

    try:
        for name, facts in ftp.mlsd():
            if name.lower().endswith(".nc"):
                mod = parse_ftp_modify(facts.get("modify"))
                files.append((name, mod))

        if files:
            return files

    except Exception:
        pass

    for name in ftp.nlst():
        if not name.lower().endswith(".nc"):
            continue

        mod = None

        try:
            resp = ftp.sendcmd(f"MDTM {name}")
            token = resp.split()[-1]
            mod = parse_ftp_modify(token)
        except Exception:
            mod = None

        files.append((name, mod))

    return files


def choose_latest_nc(files: list[tuple[str, datetime | None]]) -> str:
    if not files:
        raise FileNotFoundError("No se encontró ningún archivo .nc en el FTP.")

    enriquecidos = []

    for name, mod in files:
        emb = extract_ts_from_name(name)
        score = mod or emb
        enriquecidos.append((name, mod, emb, score))

    con_fecha = [x for x in enriquecidos if x[3] is not None]

    if con_fecha:
        con_fecha.sort(key=lambda x: x[3])
        return con_fecha[-1][0]

    enriquecidos.sort(key=lambda x: x[0].lower())
    return enriquecidos[-1][0]


def read_latest_nc_bytes_from_ftp() -> tuple[bytes, str]:
    print("Conectando al FTP SONICS...")

    with FTP(FTP_HOST, timeout=120) as ftp:
        ftp.login(FTP_USER, FTP_PASS)
        ftp.cwd(FTP_DIR)

        nc_files = list_nc_files(ftp)

        print(f"Archivos .nc encontrados: {len(nc_files)}")

        latest_nc = choose_latest_nc(nc_files)

        print(f"NetCDF más reciente seleccionado: {latest_nc}")
        print("Leyendo NetCDF en memoria...")

        buffer = io.BytesIO()
        ftp.retrbinary(f"RETR {latest_nc}", buffer.write)
        buffer.seek(0)

        return buffer.getvalue(), latest_nc


# ============================================================
# NETCDF
# ============================================================

def orient_tc(arr: np.ndarray, n_t: int, n_c: int) -> np.ndarray:
    arr = np.asarray(arr)

    if arr.ndim != 2:
        raise ValueError(f"La variable debe ser 2D y llegó con shape {arr.shape}")

    if arr.shape == (n_t, n_c):
        return arr

    if arr.shape == (n_c, n_t):
        return arr.T

    raise ValueError(
        f"Dimensiones inesperadas: {arr.shape}, esperado {(n_t, n_c)} o {(n_c, n_t)}"
    )


def decode_time_var(values: np.ndarray) -> pd.DatetimeIndex:
    vals = np.asarray(values)

    if np.issubdtype(vals.dtype, np.datetime64):
        return pd.to_datetime(vals).normalize()

    if np.issubdtype(vals.dtype, np.number):
        return pd.to_datetime(vals, unit="s", origin="unix").normalize()

    return pd.to_datetime(vals, errors="coerce").normalize()


def open_dataset_from_bytes(nc_bytes: bytes) -> tuple[xr.Dataset, str]:
    errores = []

    for engine in ("h5netcdf", "scipy"):
        try:
            bio = io.BytesIO(nc_bytes)
            ds = xr.open_dataset(bio, engine=engine, chunks=None, cache=False)
            return ds, engine
        except Exception as e:
            errores.append(f"{engine}: {e}")

    raise RuntimeError(
        "No se pudo abrir el NetCDF desde memoria. Errores: " + " | ".join(errores)
    )


def procesar_netcdf_filtrado(
    nc_bytes: bytes,
    source_name: str,
    comid_objetivo: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:

    ds, engine_used = open_dataset_from_bytes(nc_bytes)

    try:
        print("Variables disponibles en el NetCDF:")
        print(list(ds.variables))

        comid_nc = pd.to_numeric(
            pd.Series(np.asarray(ds["comid"].values).ravel()),
            errors="coerce",
        ).to_numpy(dtype="float64")

        time_hist = decode_time_var(ds["time_hist"].values)
        time_frst = decode_time_var(ds["time_frst"].values)

        n_c = len(comid_nc)
        n_t = len(time_hist)
        n_f = len(time_frst)

        idx_keep = np.where(np.isin(comid_nc, comid_objetivo))[0]

        if len(idx_keep) == 0:
            raise ValueError("Ningún COMID del GPKG coincide con el NetCDF.")

        comid_keep = comid_nc[idx_keep]

        print(f"COMID del GPKG: {len(comid_objetivo)}")
        print(f"COMID encontrados en NetCDF: {len(comid_keep)}")

        qr_hist = orient_tc(np.asarray(ds["qr_hist"].values), n_t, n_c)[:, idx_keep]
        qr_eta_eqm = orient_tc(np.asarray(ds["qr_eta_eqm"].values), n_f, n_c)[:, idx_keep]
        qr_eta_scal = orient_tc(np.asarray(ds["qr_eta_scal"].values), n_f, n_c)[:, idx_keep]
        qr_gfs = orient_tc(np.asarray(ds["qr_gfs"].values), n_f, n_c)[:, idx_keep]
        qr_wrf = orient_tc(np.asarray(ds["qr_wrf"].values), n_f, n_c)[:, idx_keep]

        hist_df = pd.DataFrame({
            "fecha": np.tile(time_hist.to_pydatetime(), len(comid_keep)),
            "comid": np.repeat(comid_keep.astype("int64"), len(time_hist)),
            "qr_hist": qr_hist.reshape(-1, order="F"),
        })

        hist_df = (
            hist_df
            .sort_values(["comid", "fecha"], kind="stable")
            .reset_index(drop=True)
        )

        fore_df = pd.DataFrame({
            "fecha": np.tile(time_frst.to_pydatetime(), len(comid_keep)),
            "comid": np.repeat(comid_keep.astype("int64"), len(time_frst)),
            "qr_eta_eqm": qr_eta_eqm.reshape(-1, order="F"),
            "qr_eta_scal": qr_eta_scal.reshape(-1, order="F"),
            "qr_gfs": qr_gfs.reshape(-1, order="F"),
            "qr_wrf": qr_wrf.reshape(-1, order="F"),
        })

        fore_df = (
            fore_df
            .sort_values(["comid", "fecha"], kind="stable")
            .reset_index(drop=True)
        )

        meta = {
            "nc_nombre": source_name,
            "nc_origen": f"ftp://{FTP_HOST}/{FTP_DIR.strip('/')}/{source_name}",
            "engine_usado": engine_used,
            "n_comid_gpkg": int(len(comid_objetivo)),
            "n_comid_encontrados": int(len(comid_keep)),
            "comid_encontrados": [int(x) for x in comid_keep.tolist()],
            "ult_fecha_hist": str(pd.to_datetime(time_hist).max().date()),
            "fechas_fore": [str(pd.Timestamp(x).date()) for x in pd.to_datetime(time_frst).unique()],
            "fecha_proceso": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        return hist_df, fore_df, meta

    finally:
        ds.close()


# ============================================================
# GUARDADO LOCAL
# ============================================================

def guardar_salidas_locales(
    hist_df: pd.DataFrame,
    fore_df: pd.DataFrame,
    meta: dict
) -> dict[str, Path]:

    hist_path = OUTPUT_DIR / HIST_NAME
    fore_path = OUTPUT_DIR / FORE_NAME
    meta_path = OUTPUT_DIR / META_NAME

    hist_df.to_parquet(hist_path, index=False)
    fore_df.to_parquet(fore_path, index=False)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\nArchivos generados localmente:")
    print(f" - {hist_path}")
    print(f" - {fore_path}")
    print(f" - {meta_path}")

    return {
        HIST_NAME: hist_path,
        FORE_NAME: fore_path,
        META_NAME: meta_path,
    }


# ============================================================
# GOOGLE DRIVE
# ============================================================

def get_drive_service():
    scopes = ["https://www.googleapis.com/auth/drive"]

    if GOOGLE_SERVICE_JSON:
        info = json.loads(GOOGLE_SERVICE_JSON)
        creds = service_account.Credentials.from_service_account_info(
            info,
            scopes=scopes
        )
    else:
        with open(GOOGLE_SERVICE_JSON_PATH, "r", encoding="utf-8") as f:
            info = json.load(f)

        creds = service_account.Credentials.from_service_account_info(
            info,
            scopes=scopes
        )

    return build("drive", "v3", credentials=creds, cache_discovery=False)


def list_files_in_folder(service, folder_id: str) -> list[dict]:
    q = f"'{folder_id}' in parents and trashed = false"

    files = []
    page_token = None

    while True:
        resp = service.files().list(
            q=q,
            spaces="drive",
            fields="nextPageToken, files(id, name, mimeType, modifiedTime)",
            pageSize=1000,
            pageToken=page_token,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()

        files.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")

        if not page_token:
            break

    return files


def find_file_in_folder_by_name(service, folder_id: str, filename: str) -> dict | None:
    current_files = list_files_in_folder(service, folder_id)

    for f in current_files:
        if f["name"].lower() == filename.lower():
            return f

    return None


def update_existing_drive_file(
    service,
    folder_id: str,
    local_path: Path,
    drive_name: str,
    mime_type: str
) -> None:

    existing = find_file_in_folder_by_name(service, folder_id, drive_name)

    if not existing:
        print(f"NO ENCONTRADO EN DRIVE: {drive_name}")
        print("Súbelo manualmente una vez a la carpeta Drive y vuelve a ejecutar.")
        return

    media = MediaFileUpload(
        str(local_path),
        mimetype=mime_type,
        resumable=False
    )

    service.files().update(
        fileId=existing["id"],
        media_body=media,
        fields="id, name, modifiedTime",
        supportsAllDrives=True
    ).execute()

    print(f"Actualizado en Drive: {drive_name}")


def subir_salidas_a_drive(paths: dict[str, Path]) -> None:
    require_env_drive()

    print("\nConectando a Google Drive...")
    service = get_drive_service()

    tipos = {
        HIST_NAME: "application/octet-stream",
        FORE_NAME: "application/octet-stream",
        META_NAME: "application/json",
        ESTACIONES_NAME: "text/csv",
    }

    for drive_name, local_path in paths.items():
        update_existing_drive_file(
            service=service,
            folder_id=DRIVE_FOLDER_ID,
            local_path=local_path,
            drive_name=drive_name,
            mime_type=tipos.get(drive_name, "application/octet-stream")
        )


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Actualizar datos SONICS filtrados por COMID")
    parser.add_argument(
        "--solo-gpkg",
        action="store_true",
        help="Solo revisa el GPKG y no conecta al FTP"
    )
    parser.add_argument(
        "--sin-drive",
        action="store_true",
        help="Procesa FTP y guarda localmente, pero no sube a Drive"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("ACTUALIZADOR SONICS - LORETO")
    print("=" * 80)

    gdf = leer_estaciones_gpkg(GPKG_PATH)
    comid_objetivo = obtener_comids(gdf)
    estaciones_path = guardar_estaciones_csv(gdf)

    print("\nCOMID objetivo:")
    print(comid_objetivo.astype("int64").tolist())

    if args.solo_gpkg:
        print("\nModo solo GPKG. No se conectó al FTP.")
        return

    require_env_ftp()

    nc_bytes, source_name = read_latest_nc_bytes_from_ftp()

    hist_df, fore_df, meta = procesar_netcdf_filtrado(
        nc_bytes=nc_bytes,
        source_name=source_name,
        comid_objetivo=comid_objetivo,
    )

    paths = guardar_salidas_locales(hist_df, fore_df, meta)
    paths[ESTACIONES_NAME] = estaciones_path

    print("\nResumen final local:")
    print(f"Histórico filtrado: {len(hist_df):,} filas")
    print(f"Pronóstico filtrado: {len(fore_df):,} filas")

    if args.sin_drive:
        print("\nModo sin Drive. No se subieron archivos.")
    else:
        subir_salidas_a_drive(paths)

    print("\nProceso terminado correctamente.")


if __name__ == "__main__":
    main()