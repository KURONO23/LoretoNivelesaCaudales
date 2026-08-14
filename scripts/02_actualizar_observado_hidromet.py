from __future__ import annotations

import io
import json
import os
import re
import sqlite3
import tempfile
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload


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

# Fecha inicial para actualización por API.
# Se usa 2025-01-01 para tener más de 365 días sin hacer demasiadas llamadas.
OBS_API_START_DATE = os.getenv("OBS_API_START_DATE", "2025-01-01")
OBS_API_END_DATE = os.getenv("OBS_API_END_DATE", date.today().strftime("%Y-%m-%d"))

# Si quieres forzar que solo lea Drive y no API:
# OBS_USAR_API=false
OBS_USAR_API = os.getenv("OBS_USAR_API", "true").strip().lower() in ["1", "true", "yes", "si", "sí"]

# Si quieres forzar que lea archivos manuales de Drive:
OBS_USAR_DRIVE_MANUAL = os.getenv("OBS_USAR_DRIVE_MANUAL", "true").strip().lower() in ["1", "true", "yes", "si", "sí"]

# Carpetas locales opcionales donde también puede buscar Excel manuales
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

# Estaciones manuales o estaciones sin ID HidroMet claro.
# Aquí agregamos Timicurillo explícitamente.
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


def parse_fecha_segura(x) -> pd.Timestamp | pd.NaT:
    if x is None or pd.isna(x):
        return pd.NaT

    try:
        return pd.to_datetime(x, errors="coerce")
    except Exception:
        return pd.NaT


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

    # Caso típico del API: Fecha = 0001-01-01T00:00:00
    fechas_invalidas = fechas.isna() | (fechas.dt.year <= 1901)

    if fechas_invalidas.all():
        fechas_esperadas = pd.date_range(fecha_ini, fecha_fin, freq="D")

        if len(fechas_esperadas) == len(df):
            df["FECHA"] = fechas_esperadas
        else:
            df["FECHA"] = pd.NaT

    else:
        df["FECHA"] = fechas

    return df


def extraer_registros_json(obj: Any) -> list[dict]:
    registros = []

    if isinstance(obj, list):
        for item in obj:
            registros.extend(extraer_registros_json(item))

    elif isinstance(obj, dict):
        # Si el diccionario parece ser un registro tabular
        claves = {normalizar_texto(k) for k in obj.keys()}

        if (
            "H_PROM" in claves
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
                continue

            tmp = pd.DataFrame(registros)
            tmp = normalizar_columnas(tmp)
            tmp = reconstruir_fechas_si_necesario(tmp, ini, fin)

            col_fecha = detectar_columna(tmp, ["FECHA", "Fecha", "fecha"])
            col_hprom = detectar_columna(tmp, ["H_PROM", "HPROM", "H PROM", "H_PROMEDIO", "NIVEL", "nivel_m"])

            if col_fecha is None or col_hprom is None:
                continue

            out = pd.DataFrame()
            out["fecha"] = pd.to_datetime(tmp[col_fecha], errors="coerce")
            out["nivel_m"] = pd.to_numeric(tmp[col_hprom], errors="coerce")
            out["id_estacion"] = int(id_estacion)
            out["estacion_nombre"] = estacion_nombre
            out["estacion_key"] = estacion_key
            out["comid"] = int(comid)
            out["fuente"] = "HidroMet API"

            out = out.dropna(subset=["fecha", "nivel_m"]).copy()
            out = out[out["nivel_m"] > 0].copy()
            out = out[out["nivel_m"] != -999].copy()

            if not out.empty:
                partes.append(out)

            time.sleep(0.05)

        except Exception as e:
            print(f"  ADVERTENCIA API {estacion_nombre} {ini.date()}-{fin.date()}: {e}")

    if not partes:
        return pd.DataFrame()

    df = pd.concat(partes, ignore_index=True)
    return df


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

        print(f"  Registros API útiles: {len(df_est):,}")

        if not df_est.empty:
            partes.append(df_est)

    if not partes:
        return pd.DataFrame()

    return pd.concat(partes, ignore_index=True)


# ============================================================
# GOOGLE DRIVE — LECTURA DE EXCEL MANUALES
# ============================================================

def construir_drive_service():
    if not GOOGLE_SERVICE_JSON:
        print("ADVERTENCIA: GOOGLE_SERVICE_JSON no está definido. No se leerá Drive.")
        return None

    try:
        info = json.loads(GOOGLE_SERVICE_JSON)
    except Exception as e:
        print(f"ADVERTENCIA: GOOGLE_SERVICE_JSON no es JSON válido: {e}")
        return None

    try:
        creds = service_account.Credentials.from_service_account_info(
            info,
            scopes=["https://www.googleapis.com/auth/drive.readonly"],
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
    col_nivel = detectar_columna(df, ["H_PROM", "HPROM", "H PROM", "NIVEL", "NIVEL_M", "nivel_m"])
    col_id = detectar_columna(df, ["ID_ESTACION", "ID ESTACION", "id_estacion"])
    col_estacion = detectar_columna(df, ["ESTACION_NOMBRE", "ESTACION", "NOMBRE", "estacion_nombre"])
    col_key = detectar_columna(df, ["ESTACION_KEY", "estacion_key"])

    if col_fecha is None or col_nivel is None:
        print(f"  ADVERTENCIA: {nombre_archivo} no tiene FECHA y H_PROM/NIVEL. Se omite.")
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

    # Si el nombre del archivo es más confiable que el contenido, usar override.
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
    out["nivel_m"] = pd.to_numeric(df[col_nivel], errors="coerce")
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


def leer_observados_manuales_drive(mapa_comid: dict[str, int]) -> pd.DataFrame:
    if not OBS_USAR_DRIVE_MANUAL:
        print("Lectura de Drive manual desactivada por OBS_USAR_DRIVE_MANUAL=false")
        return pd.DataFrame()

    if not OBS_DRIVE_FOLDER_ID:
        print("ADVERTENCIA: OBS_DRIVE_FOLDER_ID no definido. No se leerá Drive.")
        return pd.DataFrame()

    service = construir_drive_service()

    if service is None:
        return pd.DataFrame()

    print("")
    print("============================================================")
    print("LEYENDO EXCEL/CSV MANUALES DESDE GOOGLE DRIVE")
    print("============================================================")
    print(f"Carpeta Drive: {OBS_DRIVE_FOLDER_ID}")

    archivos = listar_archivos_drive(service, OBS_DRIVE_FOLDER_ID)

    if not archivos:
        print("No se encontraron archivos Excel/CSV en Drive.")
        return pd.DataFrame()

    print(f"Archivos encontrados en Drive: {len(archivos)}")

    partes = []

    with tempfile.TemporaryDirectory() as tmpdir:
        carpeta_tmp = Path(tmpdir)

        for f in archivos:
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

    # Prioridad:
    # 1. Excel manual puede aportar histórico completo.
    # 2. API puede actualizar valores recientes.
    # Al concatenar manual primero y API después, keep='last' prioriza API en duplicados.
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


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("")
    print("============================================================")
    print("SCRIPT 02 - ACTUALIZAR OBSERVADO HIDROMET + EXCEL MANUALES")
    print("============================================================")
    print(f"GPKG_PATH: {GPKG_PATH}")
    print(f"OUT_PARQUET: {OUT_PARQUET}")
    print(f"OUT_CSV: {OUT_CSV}")

    estaciones_gpkg = leer_estaciones_gpkg(GPKG_PATH)
    mapa_comid = construir_mapa_comid(estaciones_gpkg)

    print("")
    print(f"Estaciones en mapa COMID: {len(mapa_comid)}")

    partes = []

    # 1. Excel/CSV manuales desde Drive
    obs_drive = leer_observados_manuales_drive(mapa_comid)

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

    # 4. Consolidación final
    observado = consolidar_observados(partes)

    imprimir_resumen(observado)

    if observado.empty:
        raise RuntimeError("No se generó observado_estaciones porque no hay datos útiles.")

    observado.to_parquet(OUT_PARQUET, index=False)
    observado.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print("")
    print("============================================================")
    print("ARCHIVOS GUARDADOS")
    print("============================================================")
    print(f"OK: {OUT_PARQUET} | filas: {len(observado):,}")
    print(f"OK: {OUT_CSV} | filas: {len(observado):,}")

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


if __name__ == "__main__":
    main()
