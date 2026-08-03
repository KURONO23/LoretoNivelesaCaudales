from __future__ import annotations

import calendar
import json
import os
import re
import sqlite3
import tempfile
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ============================================================
# ACTUALIZACIÓN OBSERVADA HIDROMET DESDE GOOGLE DRIVE
# ============================================================
# Lee Excel por estación desde Drive, actualiza con WebService
# HidroMet, actualiza Excel existentes en Drive y genera:
# backend/cache/observado_estaciones.parquet
# backend/cache/observado_estaciones.csv
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CACHE_DIR = BASE_DIR / "backend" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = BASE_DIR / "Data"

GPKG_PATH = Path(
    os.getenv(
        "GPKG_PATH",
        DATA_DIR / "estaciones_hidrometricas_loreto_latlong_COMID_actualizado.gpkg",
    )
)

OBS_DRIVE_FOLDER_ID = os.getenv("OBS_DRIVE_FOLDER_ID", "").strip()
GOOGLE_SERVICE_JSON = os.getenv("GOOGLE_SERVICE_JSON", "").strip()
GOOGLE_SERVICE_JSON_PATH = os.getenv("GOOGLE_SERVICE_JSON_PATH", "").strip()

BASE_URL = "https://hidromet.net.pe/api/hidrologia/estaciones/{id_estacion}/info"

FECHA_INICIO_BASE = date(1981, 1, 1)
FECHA_FINAL = date.today()

MODO_DESCARGA = "mensual"
PAUSA_SEGUNDOS = 0.15
FORZAR_ACTUALIZACION = False

OBS_PARQUET_NAME = "observado_estaciones.parquet"
OBS_CSV_NAME = "observado_estaciones.csv"
LOG_NAME = "LOG_ACTUALIZACION_OBSERVADO.xlsx"
CONSOLIDADO_NAME = "CONSOLIDADO.xlsx"

COLUMNAS_NIVEL = ["H6", "H10", "H14", "H18"]


# ============================================================
# ESTACIONES HIDROMET
# ============================================================

ESTACIONES = {
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


# ============================================================
# VALIDACIÓN DE VARIABLES
# ============================================================

def require_env() -> None:
    faltan = []

    if not OBS_DRIVE_FOLDER_ID:
        faltan.append("OBS_DRIVE_FOLDER_ID")

    if not GOOGLE_SERVICE_JSON and not GOOGLE_SERVICE_JSON_PATH:
        faltan.append("GOOGLE_SERVICE_JSON o GOOGLE_SERVICE_JSON_PATH")

    if faltan:
        raise RuntimeError(f"Faltan variables de entorno: {', '.join(faltan)}")

    if not GPKG_PATH.exists():
        raise FileNotFoundError(f"No existe el GPKG: {GPKG_PATH}")


# ============================================================
# GOOGLE DRIVE
# ============================================================

def get_drive_service():
    scopes = ["https://www.googleapis.com/auth/drive"]

    if GOOGLE_SERVICE_JSON:
        info = json.loads(GOOGLE_SERVICE_JSON)
    else:
        with open(GOOGLE_SERVICE_JSON_PATH, "r", encoding="utf-8") as f:
            info = json.load(f)

    creds = service_account.Credentials.from_service_account_info(
        info,
        scopes=scopes,
    )

    return build("drive", "v3", credentials=creds, cache_discovery=False)


def listar_archivos_drive(service, folder_id: str) -> list[dict]:
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


def buscar_archivo_por_nombre(service, folder_id: str, nombre: str) -> dict | None:
    archivos = listar_archivos_drive(service, folder_id)

    for archivo in archivos:
        if archivo["name"].lower() == nombre.lower():
            return archivo

    return None


def descargar_archivo_drive(service, file_id: str, destino: Path) -> None:
    request = service.files().get_media(
        fileId=file_id,
        supportsAllDrives=True,
    )

    with open(destino, "wb") as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False

        while not done:
            _, done = downloader.next_chunk()


def subir_o_actualizar_archivo(
    service,
    folder_id: str,
    local_path: Path,
    drive_name: str,
    mime_type: str,
    crear_si_no_existe: bool = False,
) -> None:
    """
    Por defecto NO crea archivos nuevos en Drive para evitar error de cuota
    de Service Account. Solo actualiza si el archivo ya existe.
    """
    existente = buscar_archivo_por_nombre(service, folder_id, drive_name)

    if not existente and not crear_si_no_existe:
        print(f"No existe en Drive y no se creará por cuota: {drive_name}")
        return

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
                "parents": [folder_id],
            },
            media_body=media,
            fields="id,name,modifiedTime",
            supportsAllDrives=True,
        ).execute()

        print(f"Creado en Drive: {drive_name}")


# ============================================================
# HTTP HIDROMET
# ============================================================

def crear_sesion():
    session = requests.Session()

    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )

    adapter = HTTPAdapter(max_retries=retry)

    session.mount("https://", adapter)
    session.mount("http://", adapter)

    session.headers.update({
        "User-Agent": "Mozilla/5.0 HidroMet-Updater",
    })

    return session


SESSION = crear_sesion()


# ============================================================
# UTILIDADES
# ============================================================

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


def limpiar_nombre_archivo(nombre):
    nombre = str(nombre).strip()
    nombre = re.sub(r'[\\/*?:"<>|]', "_", nombre)
    nombre = re.sub(r"\s+", " ", nombre)
    return nombre


def nombre_excel_estacion(nombre_estacion: str) -> str:
    return f"{limpiar_nombre_archivo(nombre_estacion)}.xlsx"


def generar_periodos_por_fechas(fecha_inicio, fecha_final, modo="mensual"):
    periodos = []

    if fecha_inicio > fecha_final:
        return periodos

    if modo == "anual":
        for anio in range(fecha_inicio.year, fecha_final.year + 1):
            f1 = date(anio, 1, 1)
            f2 = date(anio, 12, 31)

            if anio == fecha_inicio.year:
                f1 = fecha_inicio

            if anio == fecha_final.year:
                f2 = fecha_final

            periodos.append({
                "anio": anio,
                "mes": None,
                "fecha1": f1.strftime("%Y-%m-%d"),
                "fecha2": f2.strftime("%Y-%m-%d"),
                "periodo": f"{f1.strftime('%Y-%m-%d')}_a_{f2.strftime('%Y-%m-%d')}",
            })

    elif modo == "mensual":
        anio = fecha_inicio.year
        mes = fecha_inicio.month

        while True:
            primer_dia_mes = date(anio, mes, 1)
            ultimo_dia_mes = date(anio, mes, calendar.monthrange(anio, mes)[1])

            f1 = max(fecha_inicio, primer_dia_mes)
            f2 = min(fecha_final, ultimo_dia_mes)

            if f1 <= f2:
                periodos.append({
                    "anio": anio,
                    "mes": mes,
                    "fecha1": f1.strftime("%Y-%m-%d"),
                    "fecha2": f2.strftime("%Y-%m-%d"),
                    "periodo": f"{anio}-{mes:02d}",
                })

            if anio == fecha_final.year and mes == fecha_final.month:
                break

            mes += 1

            if mes == 13:
                mes = 1
                anio += 1

    else:
        raise ValueError("MODO_DESCARGA debe ser mensual o anual.")

    return periodos


# ============================================================
# GPKG / COMID
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

        excluded_prefixes = (
            "gpkg_",
            "sqlite_",
            "rtree_",
        )

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

        raise ValueError("No se encontró tabla con columna COMID en el GPKG.")

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

    df.columns = [str(c).strip().lower() for c in df.columns]

    if "comid" not in df.columns:
        raise ValueError("No existe columna COMID en el GPKG.")

    df["comid"] = pd.to_numeric(df["comid"], errors="coerce")
    df = df.dropna(subset=["comid"]).copy()
    df["comid"] = df["comid"].astype("int64")

    nombre_col = None

    for col in ["estacion", "estacion_nombre", "nombre", "estaciones_hidro"]:
        if col in df.columns:
            nombre_col = col
            break

    if nombre_col is None:
        raise ValueError(
            "No se encontró columna de nombre de estación en el GPKG."
        )

    df["estacion_norm"] = df[nombre_col].apply(normalizar_texto)

    return df


def construir_mapa_comid(gpkg_path: Path) -> dict[str, int]:
    est = leer_estaciones_gpkg(gpkg_path)

    mapa = {}

    for _, row in est.iterrows():
        estacion_norm = row["estacion_norm"]
        comid = row["comid"]

        if estacion_norm and pd.notna(comid):
            mapa[estacion_norm] = int(comid)

    return mapa


# ============================================================
# EXCEL / WEBSERVICE
# ============================================================

def leer_excel_estacion(path_excel: Path):
    if not path_excel.exists():
        return pd.DataFrame(), None

    df = pd.read_excel(path_excel)

    if "FECHA" not in df.columns:
        return df, None

    df["FECHA"] = pd.to_datetime(df["FECHA"], errors="coerce")
    fechas_validas = df["FECHA"].dropna()

    if fechas_validas.empty:
        return df, None

    ultima_fecha = fechas_validas.max().date()

    return df, ultima_fecha


def descargar_periodo(id_estacion, nombre_estacion, fecha1, fecha2):
    url = BASE_URL.format(id_estacion=id_estacion)

    params = {
        "fecha1": fecha1,
        "fecha2": fecha2,
    }

    r = SESSION.get(url, params=params, timeout=90)
    r.raise_for_status()

    data = r.json()
    df = pd.DataFrame(data)

    if df.empty:
        return pd.DataFrame(), "SIN_DATOS"

    if "Fecha" in df.columns:
        df = df.rename(columns={"Fecha": "FECHA_API"})

    fechas_reales = pd.date_range(fecha1, fecha2, freq="D")

    if len(df) == len(fechas_reales):
        df.insert(0, "FECHA", fechas_reales)
        estado_fecha = "FECHA_RECONSTRUIDA_OK"
    else:
        df.insert(0, "FECHA", pd.NaT)
        estado_fecha = f"REVISAR_FILAS_API_{len(df)}_DIAS_ESPERADOS_{len(fechas_reales)}"

    df.insert(1, "ID_ESTACION", id_estacion)
    df.insert(2, "ESTACION_NOMBRE", nombre_estacion)

    df["FECHA_CONSULTA_INI"] = fecha1
    df["FECHA_CONSULTA_FIN"] = fecha2
    df["ESTADO_FECHA"] = estado_fecha

    for col in COLUMNAS_NIVEL:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    cols_existentes = [c for c in COLUMNAS_NIVEL if c in df.columns]

    if cols_existentes:
        df["H_PROM"] = df[cols_existentes].mean(axis=1)

    df["ANIO"] = df["FECHA"].dt.year
    df["MES"] = df["FECHA"].dt.month
    df["DIA"] = df["FECHA"].dt.day

    columnas_inicio = [
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
    ]

    columnas_existentes = [c for c in columnas_inicio if c in df.columns]
    columnas_restantes = [c for c in df.columns if c not in columnas_existentes]

    df = df[columnas_existentes + columnas_restantes]

    return df, estado_fecha


def unir_y_limpiar(df_antiguo, df_nuevo):
    if df_antiguo is None or df_antiguo.empty:
        df_final = df_nuevo.copy()
    elif df_nuevo is None or df_nuevo.empty:
        df_final = df_antiguo.copy()
    else:
        df_final = pd.concat([df_antiguo, df_nuevo], ignore_index=True)

    if "FECHA" in df_final.columns:
        df_final["FECHA"] = pd.to_datetime(df_final["FECHA"], errors="coerce")

    if "ID_ESTACION" in df_final.columns and "FECHA" in df_final.columns:
        df_final = df_final.sort_values(["ID_ESTACION", "FECHA"])
        df_final = df_final.drop_duplicates(
            subset=["ID_ESTACION", "FECHA"],
            keep="last",
        )

    return df_final


def preparar_observado_parquet(
    df_total: pd.DataFrame,
    mapa_comid: dict[str, int],
) -> pd.DataFrame:
    df = df_total.copy()

    df.columns = [str(c).strip().upper() for c in df.columns]

    requeridas = ["ID_ESTACION", "ESTACION_NOMBRE", "FECHA", "H_PROM"]

    for col in requeridas:
        if col not in df.columns:
            raise ValueError(f"Falta columna requerida en consolidado: {col}")

    df["FECHA"] = pd.to_datetime(df["FECHA"], errors="coerce")
    df["H_PROM"] = pd.to_numeric(df["H_PROM"], errors="coerce")

    df["estacion_nombre"] = df["ESTACION_NOMBRE"].astype(str).str.strip()
    df["estacion_norm"] = df["estacion_nombre"].apply(normalizar_texto)

    df["comid"] = df["estacion_norm"].map(mapa_comid)

    sin_comid = (
        df.loc[df["comid"].isna(), "estacion_nombre"]
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    if sin_comid:
        print("ADVERTENCIA: estaciones observadas sin COMID:")
        for nombre in sin_comid:
            print(f" - {nombre}")

    df = df.dropna(subset=["FECHA", "comid", "H_PROM"]).copy()

    # Limpieza de valores inválidos.
    df = df[df["H_PROM"] > 0].copy()
    df = df[df["H_PROM"] != -999].copy()

    df["comid"] = df["comid"].astype("int64")

    out = pd.DataFrame({
        "fecha": df["FECHA"],
        "comid": df["comid"],
        "estacion_nombre": df["estacion_nombre"],
        "id_estacion": pd.to_numeric(df["ID_ESTACION"], errors="coerce"),
        "nivel_m": df["H_PROM"],
    })

    out = out.drop_duplicates(
        subset=["comid", "fecha"],
        keep="last",
    )

    out = out.sort_values(["comid", "fecha"]).reset_index(drop=True)

    return out


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    require_env()

    print("=" * 100)
    print("ACTUALIZACIÓN OBSERVADO HIDROMET DESDE DRIVE")
    print("=" * 100)
    print(f"Fecha final: {FECHA_FINAL}")
    print(f"Carpeta Drive observados: {OBS_DRIVE_FOLDER_ID}")

    service = get_drive_service()
    mapa_comid = construir_mapa_comid(GPKG_PATH)

    print(f"COMID en GPKG: {len(mapa_comid)}")

    logs = []
    dfs_consolidado = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        for id_estacion, nombre_estacion in ESTACIONES.items():
            nombre_xlsx = nombre_excel_estacion(nombre_estacion)
            local_xlsx = tmpdir / nombre_xlsx

            print("\n" + "=" * 100)
            print(f"ESTACIÓN {id_estacion} - {nombre_estacion}")
            print("=" * 100)

            archivo_drive = buscar_archivo_por_nombre(
                service=service,
                folder_id=OBS_DRIVE_FOLDER_ID,
                nombre=nombre_xlsx,
            )

            if archivo_drive:
                print(f"Descargando desde Drive: {nombre_xlsx}")
                descargar_archivo_drive(
                    service=service,
                    file_id=archivo_drive["id"],
                    destino=local_xlsx,
                )
            else:
                print(f"No existe en Drive. Se trabajará desde cero: {nombre_xlsx}")

            df_existente, ultima_fecha = leer_excel_estacion(local_xlsx)

            if ultima_fecha is None:
                fecha_inicio_actualizacion = FECHA_INICIO_BASE
                estado_archivo = "NO_EXISTE_O_SIN_FECHA"
            else:
                fecha_inicio_actualizacion = ultima_fecha + timedelta(days=1)
                estado_archivo = "EXISTE"

            if FORZAR_ACTUALIZACION:
                fecha_inicio_actualizacion = FECHA_INICIO_BASE

            if fecha_inicio_actualizacion > FECHA_FINAL:
                print(f"YA ACTUALIZADO. Última fecha: {ultima_fecha}")
                df_final = df_existente.copy()

                logs.append({
                    "fecha_hora": datetime.now(),
                    "id_estacion": id_estacion,
                    "estacion": nombre_estacion,
                    "ultima_fecha_previa": ultima_fecha,
                    "estado": "YA_ACTUALIZADO",
                    "filas_finales": len(df_final),
                })

            else:
                periodos = generar_periodos_por_fechas(
                    fecha_inicio=fecha_inicio_actualizacion,
                    fecha_final=FECHA_FINAL,
                    modo=MODO_DESCARGA,
                )

                print(f"Estado archivo: {estado_archivo}")
                print(f"Última fecha previa: {ultima_fecha}")
                print(f"Descargando desde: {fecha_inicio_actualizacion}")
                print(f"Hasta: {FECHA_FINAL}")
                print(f"Consultas: {len(periodos)}")

                datos_nuevos = []

                for periodo in periodos:
                    t0 = time.time()

                    fecha1 = periodo["fecha1"]
                    fecha2 = periodo["fecha2"]
                    nombre_periodo = periodo["periodo"]

                    try:
                        df_periodo, estado_fecha = descargar_periodo(
                            id_estacion=id_estacion,
                            nombre_estacion=nombre_estacion,
                            fecha1=fecha1,
                            fecha2=fecha2,
                        )

                        filas = len(df_periodo)

                        if filas > 0:
                            datos_nuevos.append(df_periodo)

                        segundos = round(time.time() - t0, 2)

                        print(
                            f"{id_estacion} - {nombre_estacion} | "
                            f"{nombre_periodo} | {fecha1} a {fecha2} | "
                            f"filas: {filas} | {estado_fecha} | {segundos}s"
                        )

                        logs.append({
                            "fecha_hora": datetime.now(),
                            "id_estacion": id_estacion,
                            "estacion": nombre_estacion,
                            "estado": "OK",
                            "periodo": nombre_periodo,
                            "fecha1": fecha1,
                            "fecha2": fecha2,
                            "estado_fecha": estado_fecha,
                            "filas_nuevas": filas,
                        })

                        time.sleep(PAUSA_SEGUNDOS)

                    except Exception as e:
                        print(
                            f"ERROR | {id_estacion} - {nombre_estacion} | "
                            f"{nombre_periodo} | {e}"
                        )

                        logs.append({
                            "fecha_hora": datetime.now(),
                            "id_estacion": id_estacion,
                            "estacion": nombre_estacion,
                            "estado": "ERROR",
                            "periodo": nombre_periodo,
                            "fecha1": fecha1,
                            "fecha2": fecha2,
                            "mensaje": str(e),
                        })

                        time.sleep(0.8)

                if datos_nuevos:
                    df_nuevo = pd.concat(datos_nuevos, ignore_index=True)
                else:
                    df_nuevo = pd.DataFrame()

                df_final = unir_y_limpiar(df_existente, df_nuevo)

                print(f"Filas antiguas: {len(df_existente)}")
                print(f"Filas nuevas: {len(df_nuevo)}")
                print(f"Filas finales: {len(df_final)}")

            # Guardar Excel actualizado local temporal.
            df_final.to_excel(local_xlsx, index=False)

            # Actualizar Excel en Drive solo si ya existe.
            subir_o_actualizar_archivo(
                service=service,
                folder_id=OBS_DRIVE_FOLDER_ID,
                local_path=local_xlsx,
                drive_name=nombre_xlsx,
                mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                crear_si_no_existe=False,
            )

            if not df_final.empty:
                dfs_consolidado.append(df_final)

        if not dfs_consolidado:
            raise RuntimeError("No se pudo construir consolidado observado.")

        df_total = pd.concat(dfs_consolidado, ignore_index=True)

        if "FECHA" in df_total.columns:
            df_total["FECHA"] = pd.to_datetime(df_total["FECHA"], errors="coerce")

        if "ID_ESTACION" in df_total.columns and "FECHA" in df_total.columns:
            df_total = df_total.sort_values(["ID_ESTACION", "FECHA"])
            df_total = df_total.drop_duplicates(
                subset=["ID_ESTACION", "FECHA"],
                keep="last",
            )

        consolidado_xlsx = tmpdir / CONSOLIDADO_NAME
        log_xlsx = tmpdir / LOG_NAME

        if len(df_total) <= 1_048_000:
            df_total.to_excel(consolidado_xlsx, index=False)
            print(f"Consolidado temporal generado: {consolidado_xlsx}")

        df_log = pd.DataFrame(logs)
        df_log.to_excel(log_xlsx, index=False)
        print(f"Log temporal generado: {log_xlsx}")

        print("No se suben CONSOLIDADO ni LOG a Drive para evitar error de cuota de Service Account.")

        observado = preparar_observado_parquet(
            df_total=df_total,
            mapa_comid=mapa_comid,
        )

        obs_parquet_path = CACHE_DIR / OBS_PARQUET_NAME
        obs_csv_path = CACHE_DIR / OBS_CSV_NAME

        observado.to_parquet(obs_parquet_path, index=False)
        observado.to_csv(obs_csv_path, index=False, encoding="utf-8-sig")

        print("\n" + "=" * 100)
        print("OBSERVADO PARQUET GENERADO")
        print("=" * 100)
        print(f"Archivo: {obs_parquet_path}")
        print(f"Filas: {len(observado):,}")
        print(f"Estaciones: {observado['estacion_nombre'].nunique()}")
        print(f"COMID: {observado['comid'].nunique()}")
        print(f"Fecha mínima: {observado['fecha'].min()}")
        print(f"Fecha máxima: {observado['fecha'].max()}")

        print("No se sube observado_estaciones a Drive.")
        print("El observado actualizado queda guardado en backend/cache para commit en GitHub.")

    print("\nProceso terminado correctamente.")


if __name__ == "__main__":
    main()
