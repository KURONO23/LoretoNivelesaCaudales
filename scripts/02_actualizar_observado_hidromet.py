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

OBS_API_START_DATE = os.getenv("OBS_API_START_DATE", "2025-01-01")
OBS_API_END_DATE = os.getenv("OBS_API_END_DATE", date.today().strftime("%Y-%m-%d"))

OBS_USAR_API = os.getenv("OBS_USAR_API", "true").strip().lower() in ["1", "true", "yes", "si", "sí"]
OBS_USAR_DRIVE_MANUAL = os.getenv("OBS_USAR_DRIVE_MANUAL", "true").strip().lower() in ["1", "true", "yes", "si", "sí"]
OBS_ACTUALIZAR_EXCEL_DRIVE = os.getenv("OBS_ACTUALIZAR_EXCEL_DRIVE", "true").strip().lower() in ["1", "true", "yes", "si", "sí"]

CARPETAS_MANUALES_LOCALES = [
    BASE_DIR / "Data" / "observados_manuales",
    BASE_DIR / "observados_manuales",
    BASE_DIR / "backend" / "observados_manuales",
]

COLUMNAS_EXCEL_ANTIGUO = [
    "ID_ESTACION",
    "ESTACION_NOMBRE",
    "FECHA",
    "ANIO",
    "MES",
    "DIA",
    "H6",
    "H10",
    "H14",
    "H18",
    "H_PROM",
    "FECHA_API",
    "FECHA_CONSULTA_INI",
    "FECHA_CONSULTA_FIN",
    "ESTADO_FECHA",
    "Id",
    "Estacion",
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
# UTILIDADES
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


def detectar_columna(df: pd.DataFrame, candidatos: list[str]) -> str | None:
    cols = list(df.columns)
    cols_norm = {normalizar_texto(c): c for c in cols}

    for cand in candidatos:
        cand_norm = normalizar_texto(cand)
        if cand_norm in cols_norm:
            return cols_norm[cand_norm]

    return None


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


def calcular_hprom_desde_horarias(df: pd.DataFrame) -> pd.Series:
    cols_h = []

    for col in ["H6", "H10", "H14", "H18"]:
        if col in df.columns:
            cols_h.append(col)

    if not cols_h:
        return pd.Series(np.nan, index=df.index)

    valores = df[cols_h].apply(pd.to_numeric, errors="coerce")
    valores = valores.replace(-999, np.nan)

    return valores.mean(axis=1, skipna=True)


def completar_columnas_fecha(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["FECHA"] = pd.to_datetime(df["FECHA"], errors="coerce")
    df = df.dropna(subset=["FECHA"]).copy()

    df["FECHA"] = df["FECHA"].dt.normalize()
    df["ANIO"] = df["FECHA"].dt.year
    df["MES"] = df["FECHA"].dt.month
    df["DIA"] = df["FECHA"].dt.day

    return df


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

def descargar_api_hidromet_formato_excel_estacion(
    id_estacion: int,
    estacion_nombre: str,
    fecha_ini: str,
    fecha_fin: str,
) -> pd.DataFrame:
    partes = []

    for ini, fin in rango_meses(fecha_ini, fecha_fin):
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
                continue

            raw = pd.DataFrame(registros)
            raw_norm = normalizar_columnas(raw)

            col_fecha = detectar_columna(raw_norm, ["FECHA", "Fecha", "fecha"])
            col_id = detectar_columna(raw_norm, ["ID", "Id", "id"])
            col_estacion = detectar_columna(raw_norm, ["ESTACION", "Estacion", "estacion"])
            col_hprom = detectar_columna(raw_norm, ["H_PROM", "HPROM", "H PROM", "H_PROMEDIO", "PROMEDIO"])

            if col_fecha is None:
                print(f"  ADVERTENCIA API {estacion_nombre}: sin FECHA {ini.date()}-{fin.date()}")
                continue

            df = pd.DataFrame()
            df["FECHA_API"] = raw_norm[col_fecha].astype(str)
            fecha_api = pd.to_datetime(raw_norm[col_fecha], errors="coerce")
            fechas_invalidas = fecha_api.isna() | (fecha_api.dt.year <= 1901)

            fechas_esperadas = pd.date_range(ini, fin, freq="D")

            if len(fechas_esperadas) == len(raw_norm):
                df["FECHA"] = fechas_esperadas
                df["ESTADO_FECHA"] = np.where(
                    fechas_invalidas,
                    "FECHA_RECONSTRUIDA_OK",
                    "FECHA_ORIGINAL_OK",
                )
            else:
                df["FECHA"] = fecha_api
                df.loc[fechas_invalidas, "FECHA"] = pd.NaT
                df["ESTADO_FECHA"] = np.where(
                    fechas_invalidas,
                    "FECHA_INVALIDA",
                    "FECHA_ORIGINAL_OK",
                )

            for col_h in ["H6", "H10", "H14", "H18"]:
                col_detectada = detectar_columna(raw_norm, [col_h])
                if col_detectada is not None:
                    df[col_h] = pd.to_numeric(raw_norm[col_detectada], errors="coerce")
                else:
                    df[col_h] = np.nan

            if col_hprom is not None:
                df["H_PROM"] = pd.to_numeric(raw_norm[col_hprom], errors="coerce")
            else:
                df["H_PROM"] = calcular_hprom_desde_horarias(df)

            df["ID_ESTACION"] = int(id_estacion)
            df["ESTACION_NOMBRE"] = estacion_nombre
            df["FECHA_CONSULTA_INI"] = ini.strftime("%Y-%m-%d")
            df["FECHA_CONSULTA_FIN"] = fin.strftime("%Y-%m-%d")

            if col_id is not None:
                df["Id"] = pd.to_numeric(raw_norm[col_id], errors="coerce").fillna(id_estacion).astype(int)
            else:
                df["Id"] = int(id_estacion)

            if col_estacion is not None:
                df["Estacion"] = raw_norm[col_estacion].astype(str)
            else:
                df["Estacion"] = estacion_nombre

            df = completar_columnas_fecha(df)

            df = df[COLUMNAS_EXCEL_ANTIGUO].copy()

            for col in ["H6", "H10", "H14", "H18", "H_PROM"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df[col].replace(-999, np.nan)

            df = df.dropna(subset=["FECHA", "H_PROM"]).copy()
            df = df[df["H_PROM"] > 0].copy()

            if not df.empty:
                partes.append(df)

                print(
                    f"  API {estacion_nombre}: {ini.date()} a {fin.date()} | "
                    f"útiles: {len(df):,} | fecha max: {df['FECHA'].max().date()}"
                )

            time.sleep(0.05)

        except Exception as e:
            print(f"  ADVERTENCIA API {estacion_nombre} {ini.date()}-{fin.date()}: {e}")

    if not partes:
        return pd.DataFrame(columns=COLUMNAS_EXCEL_ANTIGUO)

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

        df_api_excel = descargar_api_hidromet_formato_excel_estacion(
            id_estacion=id_estacion,
            estacion_nombre=estacion_nombre,
            fecha_ini=OBS_API_START_DATE,
            fecha_fin=OBS_API_END_DATE,
        )

        if df_api_excel.empty:
            print("  Registros API útiles: 0")
            continue

        estacion_key = normalizar_texto(estacion_nombre)
        comid = mapa_comid.get(estacion_key)

        if comid is None:
            print(f"  ADVERTENCIA: No se encontró COMID para {estacion_nombre}. Se omitirá API.")
            continue

        out = pd.DataFrame()
        out["fecha"] = pd.to_datetime(df_api_excel["FECHA"], errors="coerce")
        out["nivel_m"] = pd.to_numeric(df_api_excel["H_PROM"], errors="coerce")
        out["id_estacion"] = int(id_estacion)
        out["estacion_nombre"] = estacion_nombre
        out["estacion_key"] = estacion_key
        out["comid"] = int(comid)
        out["fuente"] = "HidroMet API"

        out = out.dropna(subset=["fecha", "nivel_m"]).copy()
        out = out[out["nivel_m"] > 0].copy()
        out = out[out["nivel_m"] != -999].copy()
        out["fecha"] = out["fecha"].dt.normalize()

        if not out.empty:
            print(
                f"  Registros API útiles: {len(out):,} | "
                f"{out['fecha'].min().date()} a {out['fecha'].max().date()}"
            )
            partes.append(out)

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


# ============================================================
# LECTURA EXCEL EXISTENTE
# ============================================================

def leer_excel_formato_antiguo(path: Path, mapa_comid: dict[str, int]) -> pd.DataFrame:
    nombre_archivo = path.name
    estacion_desde_archivo = limpiar_nombre_archivo(nombre_archivo)

    try:
        df = pd.read_excel(path)
    except Exception as e:
        print(f"  ADVERTENCIA: No se pudo leer {nombre_archivo}: {e}")
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    df = normalizar_columnas(df)

    col_fecha = detectar_columna(df, ["FECHA"])
    col_nivel = detectar_columna(df, ["H_PROM", "HPROM", "H PROM", "NIVEL", "NIVEL_M"])
    col_id = detectar_columna(df, ["ID_ESTACION", "ID ESTACION", "Id", "ID"])
    col_estacion = detectar_columna(df, ["ESTACION_NOMBRE", "ESTACION", "NOMBRE", "Estacion"])

    if col_fecha is None or col_nivel is None:
        print(f"  ADVERTENCIA: {nombre_archivo} no tiene FECHA y H_PROM. Se omite.")
        print(f"  Columnas detectadas: {list(df.columns)}")
        return pd.DataFrame()

    if col_estacion is not None:
        serie_nombre = df[col_estacion].dropna().astype(str).str.strip()
        estacion_nombre = serie_nombre.iloc[0] if not serie_nombre.empty else estacion_desde_archivo
    else:
        estacion_nombre = estacion_desde_archivo

    archivo_key = normalizar_texto(estacion_desde_archivo)
    estacion_key = normalizar_texto(estacion_nombre)

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
    out["nivel_m"] = pd.to_numeric(df[col_nivel], errors="coerce")
    out["id_estacion"] = id_estacion
    out["estacion_nombre"] = estacion_nombre
    out["estacion_key"] = estacion_key
    out["comid"] = int(comid)
    out["fuente"] = f"Excel manual: {nombre_archivo}"

    out = out.dropna(subset=["fecha", "nivel_m"]).copy()
    out = out[out["nivel_m"] > 0].copy()
    out = out[out["nivel_m"] != -999].copy()
    out["fecha"] = out["fecha"].dt.normalize()

    return out


def leer_observados_manuales_drive(
    mapa_comid: dict[str, int],
    service,
    archivos_drive: list[dict],
) -> pd.DataFrame:
    if not OBS_USAR_DRIVE_MANUAL:
        print("Lectura de Drive manual desactivada por OBS_USAR_DRIVE_MANUAL=false")
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

            if not nombre.lower().endswith((".xlsx", ".xls")):
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

            df = leer_excel_formato_antiguo(path_local, mapa_comid)

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

        for path in archivos:
            print(f"Leyendo local: {path}")
            df = leer_excel_formato_antiguo(path, mapa_comid)
            print(f"  Registros útiles: {len(df):,}")

            if not df.empty:
                partes.append(df)

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

    return df[
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


# ============================================================
# ACTUALIZAR EXCEL DRIVE CON FORMATO ANTIGUO
# ============================================================

def preparar_excel_antiguo_para_drive(
    df_excel_existente: pd.DataFrame,
    df_api_excel: pd.DataFrame,
    id_estacion: int,
    estacion_nombre: str,
) -> pd.DataFrame:
    existentes = df_excel_existente.copy()
    existentes = normalizar_columnas(existentes)

    # Asegurar columnas antiguas
    for col in COLUMNAS_EXCEL_ANTIGUO:
        if col not in existentes.columns:
            existentes[col] = np.nan

    existentes = existentes[COLUMNAS_EXCEL_ANTIGUO].copy()

    api = df_api_excel.copy()

    for col in COLUMNAS_EXCEL_ANTIGUO:
        if col not in api.columns:
            api[col] = np.nan

    api = api[COLUMNAS_EXCEL_ANTIGUO].copy()

    combinado = pd.concat([existentes, api], ignore_index=True)

    combinado["FECHA"] = pd.to_datetime(combinado["FECHA"], errors="coerce")
    combinado["H_PROM"] = pd.to_numeric(combinado["H_PROM"], errors="coerce")

    for col in ["H6", "H10", "H14", "H18"]:
        combinado[col] = pd.to_numeric(combinado[col], errors="coerce")

    combinado = combinado.dropna(subset=["FECHA", "H_PROM"]).copy()
    combinado = combinado[combinado["H_PROM"] > 0].copy()
    combinado = combinado[combinado["H_PROM"] != -999].copy()

    combinado["FECHA"] = combinado["FECHA"].dt.normalize()

    # Completar columnas de identificación
    combinado["ID_ESTACION"] = pd.to_numeric(combinado["ID_ESTACION"], errors="coerce")
    combinado["ID_ESTACION"] = combinado["ID_ESTACION"].fillna(id_estacion).astype(int)

    combinado["ESTACION_NOMBRE"] = combinado["ESTACION_NOMBRE"].replace("", np.nan)
    combinado["ESTACION_NOMBRE"] = combinado["ESTACION_NOMBRE"].fillna(estacion_nombre)

    combinado["Id"] = pd.to_numeric(combinado["Id"], errors="coerce")
    combinado["Id"] = combinado["Id"].fillna(id_estacion).astype(int)

    combinado["Estacion"] = combinado["Estacion"].replace("", np.nan)
    combinado["Estacion"] = combinado["Estacion"].fillna(estacion_nombre)

    combinado = completar_columnas_fecha(combinado)

    combinado["FECHA_CONSULTA_INI"] = combinado["FECHA_CONSULTA_INI"].fillna("")
    combinado["FECHA_CONSULTA_FIN"] = combinado["FECHA_CONSULTA_FIN"].fillna("")
    combinado["FECHA_API"] = combinado["FECHA_API"].fillna("")
    combinado["ESTADO_FECHA"] = combinado["ESTADO_FECHA"].fillna("EXCEL_HISTORICO")

    # API gana sobre Excel en fechas duplicadas
    combinado["prioridad"] = np.where(
        combinado["ESTADO_FECHA"].astype(str).str.contains("API|ORIGINAL|RECONSTRUIDA", case=False, na=False),
        2,
        1,
    )

    combinado = combinado.sort_values(["FECHA", "prioridad"]).drop_duplicates(
        subset=["FECHA"],
        keep="last",
    )

    combinado = combinado.sort_values("FECHA").reset_index(drop=True)
    combinado = combinado[COLUMNAS_EXCEL_ANTIGUO].copy()

    # FECHA como texto con hora, parecido a lo que tenías antes en Excel
    combinado["FECHA"] = pd.to_datetime(combinado["FECHA"], errors="coerce").dt.strftime("%Y-%m-%d 00:00:00")

    return combinado


def guardar_excel_drive_formato_original(df: pd.DataFrame, path_salida: Path) -> None:
    with pd.ExcelWriter(path_salida, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Sheet1")

        ws = writer.book["Sheet1"]

        anchos = {
            "A": 14,
            "B": 24,
            "C": 22,
            "D": 10,
            "E": 8,
            "F": 8,
            "G": 10,
            "H": 10,
            "I": 10,
            "J": 10,
            "K": 12,
            "L": 22,
            "M": 18,
            "N": 18,
            "O": 24,
            "P": 10,
            "Q": 24,
        }

        for col, ancho in anchos.items():
            ws.column_dimensions[col].width = ancho

        for row in ws.iter_rows(min_row=2):
            # D, E, F, P enteros
            for idx in [4, 5, 6, 16]:
                row[idx - 1].number_format = "0"

            # G-K niveles
            for idx in [7, 8, 9, 10, 11]:
                row[idx - 1].number_format = "0.000"


def sincronizar_exceles_drive(
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

    print("")
    print("============================================================")
    print("ACTUALIZANDO EXCEL EXISTENTES EN GOOGLE DRIVE")
    print("============================================================")
    print("Formato conservado:")
    print("ID_ESTACION | ESTACION_NOMBRE | FECHA | ANIO | MES | DIA | H6 | H10 | H14 | H18 | H_PROM | FECHA_API | FECHA_CONSULTA_INI | FECHA_CONSULTA_FIN | ESTADO_FECHA | Id | Estacion")

    indice_drive = construir_indice_drive_por_estacion(archivos_drive)

    actualizados = 0
    omitidos = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        carpeta_tmp = Path(tmpdir)

        for id_estacion, estacion_nombre in ESTACIONES_API.items():
            estacion_key = normalizar_texto(estacion_nombre)
            archivo_drive = indice_drive.get(estacion_key)

            if archivo_drive is None:
                print(f"  OMITIDO: {estacion_nombre} no tiene Excel existente en Drive.")
                omitidos += 1
                continue

            file_id = archivo_drive.get("id", "")
            nombre_drive = archivo_drive.get("name", "")

            if not file_id or not nombre_drive:
                print(f"  OMITIDO: {estacion_nombre} sin file_id/nombre.")
                omitidos += 1
                continue

            if not nombre_drive.lower().endswith(".xlsx"):
                print(f"  OMITIDO: {nombre_drive} no es .xlsx.")
                omitidos += 1
                continue

            print(f"Actualizando Drive: {nombre_drive}")

            path_original = descargar_archivo_drive(
                service=service,
                file_id=file_id,
                nombre=nombre_drive,
                carpeta_tmp=carpeta_tmp,
            )

            if path_original is None:
                omitidos += 1
                continue

            try:
                df_existente = pd.read_excel(path_original)
            except Exception as e:
                print(f"  ERROR leyendo Excel existente {nombre_drive}: {e}")
                omitidos += 1
                continue

            df_api_excel = descargar_api_hidromet_formato_excel_estacion(
                id_estacion=id_estacion,
                estacion_nombre=estacion_nombre,
                fecha_ini=OBS_API_START_DATE,
                fecha_fin=OBS_API_END_DATE,
            )

            if df_api_excel.empty:
                print(f"  OMITIDO: {nombre_drive} sin datos API nuevos.")
                omitidos += 1
                continue

            df_final_excel = preparar_excel_antiguo_para_drive(
                df_excel_existente=df_existente,
                df_api_excel=df_api_excel,
                id_estacion=id_estacion,
                estacion_nombre=estacion_nombre,
            )

            path_salida = carpeta_tmp / f"actualizado_{nombre_drive}"
            guardar_excel_drive_formato_original(df_final_excel, path_salida)

            ok = actualizar_archivo_excel_drive(
                service=service,
                file_id=file_id,
                path_excel=path_salida,
            )

            if ok:
                actualizados += 1
                print(
                    f"  OK Drive: {nombre_drive} | "
                    f"filas: {len(df_final_excel):,} | "
                    f"{df_final_excel['FECHA'].min()} a {df_final_excel['FECHA'].max()}"
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
# REPORTES
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

    obs_drive = leer_observados_manuales_drive(
        mapa_comid=mapa_comid,
        service=service,
        archivos_drive=archivos_drive,
    )

    if not obs_drive.empty:
        partes.append(obs_drive)

    obs_local = leer_observados_manuales_locales(mapa_comid)

    if not obs_local.empty:
        partes.append(obs_local)

    obs_api = obtener_hidromet_api_todas(mapa_comid)

    if not obs_api.empty:
        partes.append(obs_api)

    observado = consolidar_observados(partes)

    imprimir_resumen(observado)

    if observado.empty:
        raise RuntimeError("No se generó observado_estaciones porque no hay datos útiles.")

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

    sincronizar_exceles_drive(
        service=service,
        archivos_drive=archivos_drive,
    )


if __name__ == "__main__":
    main()
