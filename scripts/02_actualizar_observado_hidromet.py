from __future__ import annotations

import json
import os
import re
import sqlite3
import tempfile
import time
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CACHE_DIR = BASE_DIR / "backend" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

OUT_PARQUET = CACHE_DIR / "observado_estaciones.parquet"
OUT_CSV = CACHE_DIR / "observado_estaciones.csv"

GPKG_PATH = Path(
    os.getenv(
        "GPKG_PATH",
        str(BASE_DIR / "Data" / "estaciones_hidrometricas_loreto_latlong_COMID_actualizado.gpkg"),
    )
)

OBS_DRIVE_FOLDER_ID = os.getenv("OBS_DRIVE_FOLDER_ID", "").strip()
GOOGLE_SERVICE_JSON = os.getenv("GOOGLE_SERVICE_JSON", "").strip()

# Periodo de consulta API HidroMet.
OBS_API_START_DATE = os.getenv("OBS_API_START_DATE", "2025-01-01")
OBS_API_END_DATE = os.getenv("OBS_API_END_DATE", date.today().strftime("%Y-%m-%d"))

OBS_USAR_API = os.getenv("OBS_USAR_API", "true").strip().lower() in [
    "1",
    "true",
    "yes",
    "si",
    "sí",
]

OBS_USAR_DRIVE_MANUAL = os.getenv("OBS_USAR_DRIVE_MANUAL", "true").strip().lower() in [
    "1",
    "true",
    "yes",
    "si",
    "sí",
]

# Esta opción controla si el script también debe completar/sobrescribir
# los Excel existentes en Google Drive.
OBS_ACTUALIZAR_EXCEL_DRIVE = os.getenv("OBS_ACTUALIZAR_EXCEL_DRIVE", "true").strip().lower() in [
    "1",
    "true",
    "yes",
    "si",
    "sí",
]

CARPETAS_MANUALES_LOCALES = [
    BASE_DIR / "Data" / "observados_manuales",
    BASE_DIR / "observados_manuales",
    BASE_DIR / "backend" / "observados_manuales",
]


# ============================================================
# ESTACIONES HIDROMET API
# ============================================================

ESTACIONES_API = {
    2: "CONTAMANA",
    3: "TAMSHIYACU",
    4: "REQUENA",
    10: "NAUTA",
    14: "BELLAVISTA",
    18: "SANTA ROSA",
    30: "SANTA MARIA DE NANAY",
    34: "SAN REGIS",
    36: "SANTA CLOTILDE",
    25: "SAN LORENZO",
    26: "BORJA",
    29: "ENAPU",
    33: "LAGUNAS",
    8: "PUERTO ALMENDRAS",
    20: "GENARO HERRERA",
    24: "FLOR DE PUNGA",
    32: "ANGAMOS",
}

COMID_MANUAL_OVERRIDES = {
    "TIMICURILLO": 9036459,
    "PUERTO ALMENDRA": 9037610,
    "PUERTO ALMENDRAS": 9037610,
    "SANTA MARIA DE NANAY": 9037738,
    "SANTA MARÍA DE NANAY": 9037738,
}


# ============================================================
# UTILIDADES GENERALES
# ============================================================

def normalizar_texto(x: Any) -> str:
    if x is None or pd.isna(x):
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


def limpiar_nombre_archivo(nombre: str) -> str:
    nombre = Path(str(nombre)).stem
    nombre = normalizar_texto(nombre)
    nombre = nombre.replace("_", " ")
    nombre = nombre.replace("-", " ")
    nombre = re.sub(r"\s+", " ", nombre).strip()
    return nombre


def normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalizar_texto(c) for c in df.columns]
    return df


def rango_meses(fecha_ini: str, fecha_fin: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    ini = pd.to_datetime(fecha_ini)
    fin = pd.to_datetime(fecha_fin)

    meses = []
    actual = pd.Timestamp(year=ini.year, month=ini.month, day=1)

    while actual <= fin:
        mes_fin = actual + pd.offsets.MonthEnd(0)
        tramo_ini = max(actual, ini)
        tramo_fin = min(mes_fin, fin)

        meses.append((tramo_ini, tramo_fin))
        actual = actual + pd.offsets.MonthBegin(1)

    return meses


def reconstruir_fechas_si_necesario(
    df: pd.DataFrame,
    fecha_ini: pd.Timestamp,
    fecha_fin: pd.Timestamp,
) -> pd.DataFrame:
    df = df.copy()

    if "FECHA" not in df.columns:
        return df

    fechas = pd.to_datetime(df["FECHA"], errors="coerce")
    fechas_invalidas = fechas.isna() | (fechas.dt.year <= 1901)

    fechas_esperadas = pd.date_range(fecha_ini, fecha_fin, freq="D")

    # Si el API devuelve una fila por día, reconstruimos por orden.
    # Esto corrige fechas 0001-01-01 y casos mixtos.
    if len(fechas_esperadas) == len(df):
        df["FECHA"] = fechas_esperadas
        return df

    df["FECHA"] = fechas
    df.loc[fechas_invalidas, "FECHA"] = pd.NaT

    return df


def extraer_registros_json(obj: Any) -> list[dict]:
    registros = []

    if isinstance(obj, list):
        for item in obj:
            registros.extend(extraer_registros_json(item))

    elif isinstance(obj, dict):
        claves = {normalizar_texto(k) for k in obj.keys()}

        if (
            "H_PROM" in claves
            or "HPROM" in claves
            or "FECHA" in claves
            or "H6" in claves
            or "H10" in claves
            or "H14" in claves
            or "H18" in claves
        ):
            registros.append(obj)
        else:
            for v in obj.values():
                registros.extend(extraer_registros_json(v))

    return registros


def detectar_columna(df: pd.DataFrame, candidatos: list[str]) -> str | None:
    cols = list(df.columns)
    cols_norm = {normalizar_texto(c): c for c in cols}

    for cand in candidatos:
        cand_norm = normalizar_texto(cand)
        if cand_norm in cols_norm:
            return cols_norm[cand_norm]

    return None


def calcular_nivel_api(tmp: pd.DataFrame) -> pd.Series | None:
    """
    Calcula nivel_m desde el API o desde Excel.

    Prioridad:
    1. H_PROM / NIVEL / nivel_m.
    2. Promedio de H6, H10, H14, H18.
    """

    col_hprom = detectar_columna(
        tmp,
        [
            "H_PROM",
            "HPROM",
            "H PROM",
            "H_PROMEDIO",
            "PROMEDIO",
            "NIVEL",
            "NIVEL_M",
            "nivel_m",
        ],
    )

    if col_hprom is not None:
        return pd.to_numeric(tmp[col_hprom], errors="coerce")

    cols_horarias = []

    for c_hora in ["H6", "H10", "H14", "H18"]:
        col_detectada = detectar_columna(tmp, [c_hora])

        if col_detectada is not None:
            cols_horarias.append(col_detectada)

    if len(cols_horarias) == 0:
        return None

    valores_horarios = tmp[cols_horarias].apply(pd.to_numeric, errors="coerce")
    valores_horarios = valores_horarios.replace(-999, np.nan)

    return valores_horarios.mean(axis=1, skipna=True)


# ============================================================
# LECTURA DE GPKG
# ============================================================

def leer_estaciones_gpkg(gpkg_path: Path) -> pd.DataFrame:
    if not gpkg_path.exists():
        print(f"ADVERTENCIA: No existe GPKG: {gpkg_path}")
        return pd.DataFrame()

    try:
        conn = sqlite3.connect(str(gpkg_path))

        df = pd.read_sql_query(
            """
            SELECT
                estacion,
                COMID,
                latitud,
                longitud
            FROM estaciones_latlong
            """,
            conn,
        )

        conn.close()

    except Exception as e:
        print(f"ADVERTENCIA: No se pudo leer GPKG: {e}")
        return pd.DataFrame()

    df.columns = [c.strip().lower() for c in df.columns]

    df["estacion_nombre"] = df["estacion"].astype(str).str.strip()
    df["estacion_key"] = df["estacion_nombre"].apply(normalizar_texto)
    df["comid"] = pd.to_numeric(df["comid"], errors="coerce")

    df = df.dropna(subset=["comid", "estacion_key"]).copy()
    df["comid"] = df["comid"].astype("int64")

    return df[["estacion_nombre", "estacion_key", "comid"]].drop_duplicates()


def construir_mapa_comid(estaciones_gpkg: pd.DataFrame) -> dict[str, int]:
    mapa = {}

    if not estaciones_gpkg.empty:
        for _, row in estaciones_gpkg.iterrows():
            key = normalizar_texto(row["estacion_key"])
            comid = int(row["comid"])
            mapa[key] = comid

    for key, comid in COMID_MANUAL_OVERRIDES.items():
        mapa[normalizar_texto(key)] = int(comid)

    return mapa


# ============================================================
# API HIDROMET
# ============================================================

def obtener_hidromet_api_estacion(
    id_estacion: int,
    estacion_nombre: str,
    fecha_ini: str,
    fecha_fin: str,
    mapa_comid: dict[str, int],
) -> pd.DataFrame:
    partes = []

    estacion_key = normalizar_texto(estacion_nombre)
    comid = mapa_comid.get(estacion_key)

    if comid is None:
        print(f"  ADVERTENCIA: No se encontró COMID para {estacion_nombre}. Se omitirá API.")
        return pd.DataFrame()

    meses = rango_meses(fecha_ini, fecha_fin)

    for ini, fin in meses:
        url = (
            f"https://hidromet.net.pe/api/hidrologia/estaciones/{id_estacion}/info"
            f"?fecha1={ini.strftime('%Y-%m-%d')}&fecha2={fin.strftime('%Y-%m-%d')}"
        )

        try:
            r = requests.get(url, timeout=45)
            r.raise_for_status()
            data = r.json()

            registros = extraer_registros_json(data)

            if not registros:
                print(f"  Sin registros API: {estacion_nombre} {ini.date()} a {fin.date()}")
                continue

            tmp = pd.DataFrame(registros)
            tmp = normalizar_columnas(tmp)
            tmp = reconstruir_fechas_si_necesario(tmp, ini, fin)

            col_fecha = detectar_columna(tmp, ["FECHA", "Fecha", "fecha"])

            if col_fecha is None:
                print(
                    f"  ADVERTENCIA: API {estacion_nombre} "
                    f"{ini.date()}-{fin.date()} sin columna FECHA."
                )
                print(f"  Columnas disponibles: {list(tmp.columns)}")
                continue

            nivel_m = calcular_nivel_api(tmp)

            if nivel_m is None:
                print(
                    f"  ADVERTENCIA: API {estacion_nombre} "
                    f"{ini.date()}-{fin.date()} sin H_PROM ni H6/H10/H14/H18."
                )
                print(f"  Columnas disponibles: {list(tmp.columns)}")
                continue

            out = pd.DataFrame()
            out["fecha"] = pd.to_datetime(tmp[col_fecha], errors="coerce")
            out["nivel_m"] = pd.to_numeric(nivel_m, errors="coerce")
            out["id_estacion"] = int(id_estacion)
            out["estacion_nombre"] = estacion_nombre
            out["estacion_key"] = estacion_key
            out["comid"] = int(comid)
            out["fuente"] = "HidroMet API"

            out = out.dropna(subset=["fecha", "nivel_m"]).copy()
            out = out[out["nivel_m"] > 0].copy()
            out = out[out["nivel_m"] != -999].copy()

            if not out.empty:
                out["fecha"] = out["fecha"].dt.normalize()
                partes.append(out)

                print(
                    f"  API {estacion_nombre}: {ini.date()} a {fin.date()} | "
                    f"útiles: {len(out):,} | "
                    f"fecha max: {out['fecha'].max().date()}"
                )

            time.sleep(0.05)

        except Exception as e:
            print(f"  ADVERTENCIA API {estacion_nombre} {ini.date()}-{fin.date()}: {e}")

    if not partes:
        return pd.DataFrame()

    return pd.concat(partes, ignore_index=True)


def obtener_hidromet_api_todas(mapa_comid: dict[str, int]) -> pd.DataFrame:
    if not OBS_USAR_API:
        print("API HidroMet desactivada por OBS_USAR_API=false")
        return pd.DataFrame()

    partes = []

    print("")
    print("============================================================")
    print("ACTUALIZANDO OBSERVADO DESDE HIDROMET API")
    print("============================================================")
    print(f"Periodo API: {OBS_API_START_DATE} a {OBS_API_END_DATE}")

    for id_estacion, estacion_nombre in ESTACIONES_API.items():
        print(f"Descargando API: {estacion_nombre} | ID {id_estacion}")

        df_est = obtener_hidromet_api_estacion(
            id_estacion=id_estacion,
            estacion_nombre=estacion_nombre,
            fecha_ini=OBS_API_START_DATE,
            fecha_fin=OBS_API_END_DATE,
            mapa_comid=mapa_comid,
        )

        if not df_est.empty:
            print(
                f"  Registros API útiles: {len(df_est):,} | "
                f"{df_est['fecha'].min().date()} a {df_est['fecha'].max().date()}"
            )
            partes.append(df_est)
        else:
            print("  Registros API útiles: 0")

    if not partes:
        return pd.DataFrame()

    return pd.concat(partes, ignore_index=True)


# ============================================================
# GOOGLE DRIVE
# ============================================================

def construir_drive_service():
    if not GOOGLE_SERVICE_JSON:
        print("ADVERTENCIA: GOOGLE_SERVICE_JSON no está definido. No se leerá ni actualizará Drive.")
        return None

    try:
        info = json.loads(GOOGLE_SERVICE_JSON)
    except Exception as e:
        print(f"ADVERTENCIA: GOOGLE_SERVICE_JSON no es JSON válido: {e}")
        return None

    try:
        # Importante:
        # Usamos drive completo para poder actualizar archivos existentes.
        # No se crearán archivos nuevos; solo se hará files().update().
        creds = service_account.Credentials.from_service_account_info(
            info,
            scopes=["https://www.googleapis.com/auth/drive"],
        )

        service = build("drive", "v3", credentials=creds)
        return service

    except Exception as e:
        print(f"ADVERTENCIA: No se pudo crear servicio Drive: {e}")
        return None


def listar_archivos_drive(service, folder_id: str) -> list[dict]:
    if service is None or not folder_id:
        return []

    archivos = []
    page_token = None

    query = (
        f"'{folder_id}' in parents and trashed = false and "
        "("
        "mimeType = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' or "
        "mimeType = 'application/vnd.ms-excel' or "
        "mimeType = 'text/csv' or "
        "name contains '.xlsx' or "
        "name contains '.xls' or "
        "name contains '.csv'"
        ")"
    )

    while True:
        try:
            resp = (
                service.files()
                .list(
                    q=query,
                    spaces="drive",
                    fields="nextPageToken, files(id, name, mimeType, modifiedTime)",
                    pageToken=page_token,
                    pageSize=1000,
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                )
                .execute()
            )

            archivos.extend(resp.get("files", []))
            page_token = resp.get("nextPageToken")

            if not page_token:
                break

        except Exception as e:
            print(f"ADVERTENCIA: No se pudo listar Drive: {e}")
            break

    return archivos


def descargar_archivo_drive(service, file_id: str, nombre: str, carpeta_tmp: Path) -> Path | None:
    try:
        destino = carpeta_tmp / nombre

        request = service.files().get_media(
            fileId=file_id,
            supportsAllDrives=True,
        )

        with open(destino, "wb") as f:
            downloader = MediaIoBaseDownload(f, request)

            done = False
            while not done:
                _, done = downloader.next_chunk()

        return destino

    except Exception as e:
        print(f"  ADVERTENCIA: No se pudo descargar {nombre}: {e}")
        return None


def actualizar_archivo_excel_drive(service, file_id: str, path_excel: Path) -> bool:
    """
    Sobrescribe un archivo Excel existente en Google Drive.
    No crea archivos nuevos.
    """

    try:
        media = MediaFileUpload(
            str(path_excel),
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            resumable=True,
        )

        service.files().update(
            fileId=file_id,
            media_body=media,
            supportsAllDrives=True,
        ).execute()

        return True

    except Exception as e:
        print(f"  ADVERTENCIA: No se pudo actualizar Excel en Drive: {e}")
        return False


# ============================================================
# LECTURA DE EXCEL / CSV MANUALES
# ============================================================

def leer_archivo_observado_manual(path: Path, mapa_comid: dict[str, int]) -> pd.DataFrame:
    nombre_archivo = path.name
    estacion_desde_archivo = limpiar_nombre_archivo(nombre_archivo)

    try:
        if path.suffix.lower() in [".xlsx", ".xls"]:
            df = pd.read_excel(path)
        elif path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        else:
            return pd.DataFrame()

    except Exception as e:
        print(f"  ADVERTENCIA: No se pudo leer {nombre_archivo}: {e}")
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    df = normalizar_columnas(df)

    col_fecha = detectar_columna(df, ["FECHA", "FECHA_HORA", "DATE", "fecha"])
    col_nivel = detectar_columna(
        df,
        [
            "H_PROM",
            "HPROM",
            "H PROM",
            "H_PROMEDIO",
            "PROMEDIO",
            "NIVEL",
            "NIVEL_M",
            "nivel_m",
        ],
    )
    col_id = detectar_columna(df, ["ID_ESTACION", "ID ESTACION", "id_estacion"])
    col_estacion = detectar_columna(df, ["ESTACION_NOMBRE", "ESTACION", "NOMBRE", "estacion_nombre"])
    col_key = detectar_columna(df, ["ESTACION_KEY", "estacion_key"])

    nivel_manual = None

    if col_nivel is not None:
        nivel_manual = pd.to_numeric(df[col_nivel], errors="coerce")
    else:
        nivel_manual = calcular_nivel_api(df)

    if col_fecha is None or nivel_manual is None:
        print(f"  ADVERTENCIA: {nombre_archivo} no tiene FECHA y H_PROM/NIVEL o H6/H10/H14/H18. Se omite.")
        print(f"  Columnas detectadas: {list(df.columns)}")
        return pd.DataFrame()

    if col_estacion is not None:
        serie_nombre = df[col_estacion].dropna().astype(str).str.strip()

        if not serie_nombre.empty:
            estacion_nombre = serie_nombre.iloc[0]
        else:
            estacion_nombre = estacion_desde_archivo

    elif col_key is not None:
        serie_nombre = df[col_key].dropna().astype(str).str.strip()

        if not serie_nombre.empty:
            estacion_nombre = serie_nombre.iloc[0]
        else:
            estacion_nombre = estacion_desde_archivo

    else:
        estacion_nombre = estacion_desde_archivo

    estacion_key = normalizar_texto(estacion_nombre)
    archivo_key = normalizar_texto(estacion_desde_archivo)

    if archivo_key in COMID_MANUAL_OVERRIDES:
        estacion_key = archivo_key
        estacion_nombre = estacion_desde_archivo

    comid = mapa_comid.get(estacion_key)

    if comid is None:
        print(f"  ADVERTENCIA: {nombre_archivo}: no se encontró COMID para '{estacion_nombre}'. Se omite.")
        return pd.DataFrame()

    if col_id is not None:
        id_estacion = pd.to_numeric(df[col_id], errors="coerce")
    else:
        id_estacion = np.nan

    out = pd.DataFrame()
    out["fecha"] = pd.to_datetime(df[col_fecha], errors="coerce")
    out["nivel_m"] = pd.to_numeric(nivel_manual, errors="coerce")
    out["id_estacion"] = id_estacion
    out["estacion_nombre"] = estacion_nombre
    out["estacion_key"] = estacion_key
    out["comid"] = int(comid)
    out["fuente"] = f"Excel manual: {nombre_archivo}"

    out = out.dropna(subset=["fecha", "nivel_m"]).copy()
    out = out[out["nivel_m"] > 0].copy()
    out = out[out["nivel_m"] != -999].copy()

    if out.empty:
        return pd.DataFrame()

    out["fecha"] = out["fecha"].dt.normalize()

    return out


def leer_observados_manuales_locales(mapa_comid: dict[str, int]) -> pd.DataFrame:
    partes = []

    print("")
    print("================================================------------")
    print("LEYENDO EXCEL/CSV MANUALES LOCALES")
    print("================================================------------")

    for carpeta in CARPETAS_MANUALES_LOCALES:
        if not carpeta.exists():
            continue

        archivos = []
        archivos.extend(sorted(carpeta.glob("*.xlsx")))
        archivos.extend(sorted(carpeta.glob("*.xls")))
        archivos.extend(sorted(carpeta.glob("*.csv")))

        for path in archivos:
            print(f"Leyendo local: {path}")

            df = leer_archivo_observado_manual(path, mapa_comid)

            print(f"  Registros útiles: {len(df):,}")

            if not df.empty:
                partes.append(df)

    if not partes:
        return pd.DataFrame()

    return pd.concat(partes, ignore_index=True)


def leer_observados_manuales_drive(
    mapa_comid: dict[str, int],
    service,
    archivos_drive: list[dict],
) -> pd.DataFrame:
    if not OBS_USAR_DRIVE_MANUAL:
        print("Lectura de Drive manual desactivada por OBS_USAR_DRIVE_MANUAL=false")
        return pd.DataFrame()

    if not OBS_DRIVE_FOLDER_ID:
        print("ADVERTENCIA: OBS_DRIVE_FOLDER_ID no definido. No se leerá Drive.")
        return pd.DataFrame()

    if service is None:
        return pd.DataFrame()

    print("")
    print("============================================================")
    print("LEYENDO EXCEL/CSV MANUALES DESDE GOOGLE DRIVE")
    print("============================================================")
    print(f"Carpeta Drive: {OBS_DRIVE_FOLDER_ID}")

    if not archivos_drive:
        print("No se encontraron archivos Excel/CSV en Drive.")
        return pd.DataFrame()

    print(f"Archivos encontrados en Drive: {len(archivos_drive)}")

    partes = []

    with tempfile.TemporaryDirectory() as tmpdir:
        carpeta_tmp = Path(tmpdir)

        for f in archivos_drive:
            nombre = f.get("name", "")
            file_id = f.get("id", "")

            if not nombre or not file_id:
                continue

            if not nombre.lower().endswith((".xlsx", ".xls", ".csv")):
                continue

            print(f"Leyendo Drive: {nombre}")

            path_local = descargar_archivo_drive(
                service=service,
                file_id=file_id,
                nombre=nombre,
                carpeta_tmp=carpeta_tmp,
            )

            if path_local is None:
                continue

            df = leer_archivo_observado_manual(path_local, mapa_comid)

            if not df.empty:
                print(
                    f"  Registros útiles: {len(df):,} | "
                    f"{df['fecha'].min().date()} a {df['fecha'].max().date()}"
                )
                partes.append(df)
            else:
                print("  Registros útiles: 0")

    if not partes:
        return pd.DataFrame()

    return pd.concat(partes, ignore_index=True)


# ============================================================
# CONSOLIDACIÓN
# ============================================================

def consolidar_observados(partes: list[pd.DataFrame]) -> pd.DataFrame:
    partes_validas = [p for p in partes if p is not None and not p.empty]

    if not partes_validas:
        return pd.DataFrame(
            columns=[
                "fecha",
                "nivel_m",
                "comid",
                "estacion_nombre",
                "estacion_key",
                "id_estacion",
                "fuente",
            ]
        )

    df = pd.concat(partes_validas, ignore_index=True)

    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce").dt.normalize()
    df["nivel_m"] = pd.to_numeric(df["nivel_m"], errors="coerce")
    df["comid"] = pd.to_numeric(df["comid"], errors="coerce")
    df["id_estacion"] = pd.to_numeric(df["id_estacion"], errors="coerce")

    df = df.dropna(subset=["fecha", "nivel_m", "comid"]).copy()
    df = df[df["nivel_m"] > 0].copy()
    df = df[df["nivel_m"] != -999].copy()

    df["comid"] = df["comid"].astype("int64")
    df["estacion_nombre"] = df["estacion_nombre"].astype(str).str.strip()
    df["estacion_key"] = df["estacion_nombre"].apply(normalizar_texto)

    # API gana sobre Excel cuando hay fecha duplicada.
    df["prioridad_fuente"] = np.where(
        df["fuente"].astype(str).str.contains("HidroMet API", case=False, na=False),
        2,
        1,
    )

    df = df.sort_values(["comid", "fecha", "prioridad_fuente"]).copy()

    df = df.drop_duplicates(
        subset=["comid", "fecha"],
        keep="last",
    ).copy()

    df = df.sort_values(["estacion_nombre", "fecha"]).reset_index(drop=True)

    df = df[
        [
            "fecha",
            "nivel_m",
            "comid",
            "estacion_nombre",
            "estacion_key",
            "id_estacion",
            "fuente",
        ]
    ].copy()

    return df


# ============================================================
# ACTUALIZACIÓN DE EXCEL EN DRIVE
# ============================================================

def construir_indice_drive_por_estacion(archivos_drive: list[dict]) -> dict[str, dict]:
    indice = {}

    for f in archivos_drive:
        nombre = f.get("name", "")

        if not nombre.lower().endswith((".xlsx", ".xls")):
            continue

        key = limpiar_nombre_archivo(nombre)

        if key:
            indice[key] = f

    return indice


def preparar_excel_estacion(df_est: pd.DataFrame, estacion_nombre: str, path_salida: Path) -> None:
    df_est = df_est.copy()

    df_est["fecha"] = pd.to_datetime(df_est["fecha"], errors="coerce").dt.normalize()
    df_est["nivel_m"] = pd.to_numeric(df_est["nivel_m"], errors="coerce")

    df_est = df_est.dropna(subset=["fecha", "nivel_m"]).copy()
    df_est = df_est.sort_values("fecha").copy()

    # Usamos el último id_estacion válido si existe.
    id_validos = pd.to_numeric(df_est["id_estacion"], errors="coerce").dropna()

    if not id_validos.empty:
        id_estacion = int(id_validos.iloc[-1])
    else:
        id_estacion = ""

    salida = pd.DataFrame()
    salida["ID_ESTACION"] = id_estacion
    salida["ESTACION"] = estacion_nombre
    salida["FECHA"] = df_est["fecha"].dt.strftime("%Y-%m-%d")
    salida["H_PROM"] = df_est["nivel_m"].round(3)

    salida.to_excel(path_salida, index=False)


def sincronizar_exceles_drive(
    observado: pd.DataFrame,
    service,
    archivos_drive: list[dict],
) -> None:
    if not OBS_ACTUALIZAR_EXCEL_DRIVE:
        print("")
        print("Actualización de Excel Drive desactivada por OBS_ACTUALIZAR_EXCEL_DRIVE=false")
        return

    if service is None:
        print("")
        print("ADVERTENCIA: No hay servicio Drive. No se actualizarán Excel en Drive.")
        return

    if not archivos_drive:
        print("")
        print("ADVERTENCIA: No hay archivos Drive. No se actualizarán Excel en Drive.")
        return

    if observado.empty:
        print("")
        print("ADVERTENCIA: Observado vacío. No se actualizarán Excel en Drive.")
        return

    print("")
    print("============================================================")
    print("ACTUALIZANDO EXCEL EXISTENTES EN GOOGLE DRIVE")
    print("============================================================")
    print("Modo: sobrescribir archivos existentes. No se crearán archivos nuevos.")

    indice_drive = construir_indice_drive_por_estacion(archivos_drive)

    with tempfile.TemporaryDirectory() as tmpdir:
        carpeta_tmp = Path(tmpdir)

        resumen = (
            observado.groupby(["estacion_nombre", "estacion_key", "comid"], dropna=False)
            .agg(
                registros=("nivel_m", "count"),
                fecha_ini=("fecha", "min"),
                fecha_fin=("fecha", "max"),
            )
            .reset_index()
            .sort_values("estacion_nombre")
        )

        actualizados = 0
        omitidos = 0

        for _, row in resumen.iterrows():
            estacion_nombre = str(row["estacion_nombre"]).strip()
            estacion_key = normalizar_texto(estacion_nombre)

            # También probamos con variaciones comunes del nombre.
            posibles_keys = [
                estacion_key,
                estacion_key.replace("Á", "A"),
                estacion_key.replace("MARIA", "MARIA"),
            ]

            archivo_drive = None

            for key in posibles_keys:
                if key in indice_drive:
                    archivo_drive = indice_drive[key]
                    break

            if archivo_drive is None:
                print(f"  OMITIDO: {estacion_nombre} no tiene Excel existente en Drive.")
                omitidos += 1
                continue

            file_id = archivo_drive.get("id", "")
            nombre_drive = archivo_drive.get("name", "")

            if not file_id:
                print(f"  OMITIDO: {estacion_nombre} sin file_id.")
                omitidos += 1
                continue

            df_est = observado[observado["estacion_key"] == estacion_key].copy()

            if df_est.empty:
                print(f"  OMITIDO: {estacion_nombre} sin datos.")
                omitidos += 1
                continue

            path_excel = carpeta_tmp / nombre_drive

            # Si el archivo original era .xls, igual generamos .xlsx internamente.
            # Google Drive acepta actualizar el contenido, pero recomendamos que tus archivos sean .xlsx.
            if path_excel.suffix.lower() != ".xlsx":
                path_excel = carpeta_tmp / f"{Path(nombre_drive).stem}.xlsx"

            preparar_excel_estacion(
                df_est=df_est,
                estacion_nombre=estacion_nombre,
                path_salida=path_excel,
            )

            ok = actualizar_archivo_excel_drive(
                service=service,
                file_id=file_id,
                path_excel=path_excel,
            )

            if ok:
                actualizados += 1
                print(
                    f"  OK Drive: {nombre_drive} | "
                    f"filas: {len(df_est):,} | "
                    f"{df_est['fecha'].min().date()} a {df_est['fecha'].max().date()}"
                )
            else:
                omitidos += 1
                print(f"  ERROR Drive: {nombre_drive}")

    print("")
    print("============================================================")
    print("RESUMEN ACTUALIZACIÓN DRIVE")
    print("============================================================")
    print(f"Excel actualizados: {actualizados}")
    print(f"Excel omitidos/error: {omitidos}")


# ============================================================
# REPORTES EN CONSOLA
# ============================================================

def imprimir_resumen(df: pd.DataFrame) -> None:
    print("")
    print("============================================================")
    print("RESUMEN OBSERVADO CONSOLIDADO")
    print("============================================================")

    if df.empty:
        print("No hay datos observados consolidados.")
        return

    resumen = (
        df.groupby(["estacion_nombre", "comid"], dropna=False)
        .agg(
            registros=("nivel_m", "count"),
            fecha_inicio=("fecha", "min"),
            fecha_fin=("fecha", "max"),
            nivel_min=("nivel_m", "min"),
            nivel_prom=("nivel_m", "mean"),
            nivel_max=("nivel_m", "max"),
        )
        .reset_index()
        .sort_values("estacion_nombre")
    )

    for _, row in resumen.iterrows():
        print(
            f"{row['estacion_nombre']} | COMID {int(row['comid'])} | "
            f"Registros: {int(row['registros']):,} | "
            f"{pd.to_datetime(row['fecha_inicio']).date()} a {pd.to_datetime(row['fecha_fin']).date()} | "
            f"Nivel prom: {row['nivel_prom']:.3f} m"
        )


def imprimir_verificacion_fechas_api(df: pd.DataFrame) -> None:
    print("")
    print("============================================================")
    print("VERIFICACIÓN DE FECHAS RECIENTES")
    print("============================================================")

    if df.empty:
        print("No hay datos para verificar.")
        return

    resumen = (
        df.groupby(["estacion_nombre", "comid", "fuente"], dropna=False)
        .agg(
            registros=("nivel_m", "count"),
            fecha_min=("fecha", "min"),
            fecha_max=("fecha", "max"),
        )
        .reset_index()
        .sort_values(["estacion_nombre", "fuente"])
    )

    for _, row in resumen.iterrows():
        print(
            f"{row['estacion_nombre']} | {row['fuente']} | "
            f"Registros: {int(row['registros']):,} | "
            f"{pd.to_datetime(row['fecha_min']).date()} a {pd.to_datetime(row['fecha_max']).date()}"
        )


def imprimir_fecha_maxima_final(observado: pd.DataFrame) -> None:
    print("")
    print("============================================================")
    print("FECHA MÁXIMA FINAL POR ESTACIÓN")
    print("============================================================")

    if observado.empty:
        print("No hay datos.")
        return

    resumen_final = (
        observado.groupby(["estacion_nombre", "comid"], dropna=False)
        .agg(
            registros=("nivel_m", "count"),
            fecha_max=("fecha", "max"),
        )
        .reset_index()
        .sort_values("estacion_nombre")
    )

    for _, row in resumen_final.iterrows():
        print(
            f"{row['estacion_nombre']} | COMID {int(row['comid'])} | "
            f"fecha máxima: {pd.to_datetime(row['fecha_max']).date()} | "
            f"registros: {int(row['registros']):,}"
        )


def verificar_timicurillo(observado: pd.DataFrame) -> None:
    timi = observado[observado["comid"] == 9036459].copy()

    if not timi.empty:
        print("")
        print("============================================================")
        print("VERIFICACIÓN TIMICURILLO")
        print("============================================================")
        print(f"Registros Timicurillo: {len(timi):,}")
        print(f"Fecha inicial: {timi['fecha'].min().date()}")
        print(f"Fecha final: {timi['fecha'].max().date()}")
        print(f"Nivel mínimo: {timi['nivel_m'].min():.3f} m")
        print(f"Nivel promedio: {timi['nivel_m'].mean():.3f} m")
        print(f"Nivel máximo: {timi['nivel_m'].max():.3f} m")
    else:
        print("")
        print("ADVERTENCIA: Timicurillo sigue sin aparecer en observado_estaciones.")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("")
    print("============================================================")
    print("SCRIPT 02 - ACTUALIZAR OBSERVADO HIDROMET + EXCEL DRIVE")
    print("============================================================")
    print(f"GPKG_PATH: {GPKG_PATH}")
    print(f"OUT_PARQUET: {OUT_PARQUET}")
    print(f"OUT_CSV: {OUT_CSV}")
    print(f"OBS_API_START_DATE: {OBS_API_START_DATE}")
    print(f"OBS_API_END_DATE: {OBS_API_END_DATE}")
    print(f"OBS_USAR_API: {OBS_USAR_API}")
    print(f"OBS_USAR_DRIVE_MANUAL: {OBS_USAR_DRIVE_MANUAL}")
    print(f"OBS_ACTUALIZAR_EXCEL_DRIVE: {OBS_ACTUALIZAR_EXCEL_DRIVE}")

    estaciones_gpkg = leer_estaciones_gpkg(GPKG_PATH)
    mapa_comid = construir_mapa_comid(estaciones_gpkg)

    print("")
    print(f"Estaciones en mapa COMID: {len(mapa_comid)}")

    service = construir_drive_service()

    archivos_drive = []

    if service is not None and OBS_DRIVE_FOLDER_ID:
        archivos_drive = listar_archivos_drive(service, OBS_DRIVE_FOLDER_ID)
        print("")
        print(f"Archivos detectados en Drive: {len(archivos_drive)}")
    else:
        print("")
        print("ADVERTENCIA: No se pudo inicializar Drive o falta OBS_DRIVE_FOLDER_ID.")

    partes = []

    # 1. Excel/CSV manuales desde Drive
    obs_drive = leer_observados_manuales_drive(
        mapa_comid=mapa_comid,
        service=service,
        archivos_drive=archivos_drive,
    )

    if not obs_drive.empty:
        partes.append(obs_drive)

    # 2. Excel/CSV manuales locales opcionales
    obs_local = leer_observados_manuales_locales(mapa_comid)

    if not obs_local.empty:
        partes.append(obs_local)

    # 3. API HidroMet
    obs_api = obtener_hidromet_api_todas(mapa_comid)

    if not obs_api.empty:
        partes.append(obs_api)

    # Verificación antes de consolidar
    if partes:
        previo = pd.concat([p for p in partes if p is not None and not p.empty], ignore_index=True)
    else:
        previo = pd.DataFrame()

    imprimir_verificacion_fechas_api(previo)

    # 4. Consolidación final
    observado = consolidar_observados(partes)

    imprimir_resumen(observado)

    if observado.empty:
        raise RuntimeError("No se generó observado_estaciones porque no hay datos útiles.")

    # 5. Guardar consolidado para AMARU / GitHub
    observado.to_parquet(OUT_PARQUET, index=False)
    observado.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print("")
    print("============================================================")
    print("ARCHIVOS GUARDADOS EN REPOSITORIO")
    print("============================================================")
    print(f"OK: {OUT_PARQUET} | filas: {len(observado):,}")
    print(f"OK: {OUT_CSV} | filas: {len(observado):,}")

    imprimir_fecha_maxima_final(observado)
    verificar_timicurillo(observado)

    # 6. Actualizar Excel existentes en Google Drive
    sincronizar_exceles_drive(
        observado=observado,
        service=service,
        archivos_drive=archivos_drive,
    )


if __name__ == "__main__":
    main()
