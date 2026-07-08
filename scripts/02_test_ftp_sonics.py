from __future__ import annotations

import os
import re
from datetime import datetime
from ftplib import FTP
from pathlib import Path

from dotenv import load_dotenv


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

FTP_HOST = os.getenv("FTP_HOST", "").strip()
FTP_USER = os.getenv("FTP_USER", "").strip()
FTP_PASS = os.getenv("FTP_PASS", "").strip()
FTP_DIR = os.getenv("FTP_DIR", "/").strip()


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def validar_env():
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
        raise RuntimeError(f"Faltan variables en .env: {', '.join(faltan)}")


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
        r"(20\d{12})",  # YYYYMMDDHHMMSS
        r"(20\d{10})",  # YYYYMMDDHHMM
        r"(20\d{8})",   # YYYYMMDDHH
        r"(20\d{6})",   # YYYYMMDD
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


def listar_nc_files(ftp: FTP) -> list[tuple[str, datetime | None]]:
    archivos = []

    # Primer intento: MLSD
    try:
        for name, facts in ftp.mlsd():
            if name.lower().endswith(".nc"):
                mod = parse_ftp_modify(facts.get("modify"))
                archivos.append((name, mod))

        if archivos:
            return archivos

    except Exception as e:
        print(f"MLSD no disponible o falló: {e}")

    # Segundo intento: NLST
    nombres = ftp.nlst()

    for name in nombres:
        if not name.lower().endswith(".nc"):
            continue

        mod = None

        try:
            resp = ftp.sendcmd(f"MDTM {name}")
            token = resp.split()[-1]
            mod = parse_ftp_modify(token)
        except Exception:
            mod = None

        archivos.append((name, mod))

    return archivos


def elegir_nc_mas_reciente(files: list[tuple[str, datetime | None]]) -> str:
    if not files:
        raise FileNotFoundError("No se encontró ningún archivo .nc en el FTP.")

    enriquecidos = []

    for name, mod in files:
        fecha_nombre = extract_ts_from_name(name)
        fecha_final = mod or fecha_nombre
        enriquecidos.append((name, mod, fecha_nombre, fecha_final))

    con_fecha = [x for x in enriquecidos if x[3] is not None]

    if con_fecha:
        con_fecha.sort(key=lambda x: x[3])
        return con_fecha[-1][0]

    enriquecidos.sort(key=lambda x: x[0].lower())
    return enriquecidos[-1][0]


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 80)
    print("PRUEBA DE CONEXIÓN FTP SONICS")
    print("=" * 80)

    validar_env()

    print(f"FTP_HOST: {FTP_HOST}")
    print(f"FTP_USER: {FTP_USER}")
    print(f"FTP_DIR : {FTP_DIR}")
    print("FTP_PASS: ********")

    print("\nConectando al FTP...")

    with FTP(FTP_HOST, timeout=60) as ftp:
        ftp.login(FTP_USER, FTP_PASS)

        print("Conexión correcta.")

        print(f"\nEntrando a carpeta FTP: {FTP_DIR}")
        ftp.cwd(FTP_DIR)

        print("Carpeta actual:")
        try:
            print(ftp.pwd())
        except Exception:
            print("(No se pudo obtener pwd, pero continuamos)")

        print("\nListando archivos .nc...")
        archivos_nc = listar_nc_files(ftp)

        print(f"Total de archivos .nc encontrados: {len(archivos_nc)}")

        if not archivos_nc:
            print("\nNo se encontraron archivos .nc.")
            print("Puede ser que FTP_DIR no sea la carpeta correcta.")
            return

        print("\nPrimeros archivos encontrados:")
        for i, (name, mod) in enumerate(archivos_nc[:20], start=1):
            print(f"{i:02d}. {name} | modificado: {mod}")

        latest = elegir_nc_mas_reciente(archivos_nc)

        print("\nArchivo .nc más reciente seleccionado:")
        print(latest)

    print("\nPrueba FTP terminada correctamente.")


if __name__ == "__main__":
    main()