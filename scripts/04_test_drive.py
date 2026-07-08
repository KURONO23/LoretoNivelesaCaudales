from __future__ import annotations

import io
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DRIVE_FOLDER_ID = os.getenv("DRIVE_FOLDER_ID", "").strip()
GOOGLE_SERVICE_JSON = os.getenv("GOOGLE_SERVICE_JSON", "").strip()
GOOGLE_SERVICE_JSON_PATH = os.getenv("GOOGLE_SERVICE_JSON_PATH", "").strip()


# ============================================================
# VALIDACIÓN
# ============================================================

def validar_env_drive():
    faltan = []

    if not DRIVE_FOLDER_ID:
        faltan.append("DRIVE_FOLDER_ID")

    if not GOOGLE_SERVICE_JSON and not GOOGLE_SERVICE_JSON_PATH:
        faltan.append("GOOGLE_SERVICE_JSON o GOOGLE_SERVICE_JSON_PATH")

    if faltan:
        raise RuntimeError(f"Faltan variables en .env: {', '.join(faltan)}")

    if GOOGLE_SERVICE_JSON_PATH:
        ruta = Path(GOOGLE_SERVICE_JSON_PATH)

        if not ruta.exists():
            raise FileNotFoundError(f"No existe GOOGLE_SERVICE_JSON_PATH: {ruta}")


# ============================================================
# SERVICIO DRIVE
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


def buscar_archivo_en_carpeta(service, folder_id: str, filename: str):
    query = (
        f"'{folder_id}' in parents and "
        f"name = '{filename}' and "
        f"trashed = false"
    )

    resp = service.files().list(
        q=query,
        spaces="drive",
        fields="files(id, name)",
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()

    archivos = resp.get("files", [])

    if archivos:
        return archivos[0]

    return None


def subir_o_actualizar_texto(service, folder_id: str, filename: str, contenido: str):
    buffer = io.BytesIO(contenido.encode("utf-8"))
    media = MediaIoBaseUpload(
        buffer,
        mimetype="text/plain",
        resumable=False
    )

    existente = buscar_archivo_en_carpeta(service, folder_id, filename)

    if existente:
        service.files().update(
            fileId=existente["id"],
            media_body=media,
            fields="id, name",
            supportsAllDrives=True,
        ).execute()

        print(f"Archivo actualizado en Drive: {filename}")

    else:
        service.files().create(
            body={
                "name": filename,
                "parents": [folder_id],
            },
            media_body=media,
            fields="id, name",
            supportsAllDrives=True,
        ).execute()

        print(f"Archivo creado en Drive: {filename}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 80)
    print("PRUEBA DE CONEXIÓN A GOOGLE DRIVE")
    print("=" * 80)

    validar_env_drive()

    print("DRIVE_FOLDER_ID: configurado")
    print("GOOGLE_SERVICE_JSON_PATH:", GOOGLE_SERVICE_JSON_PATH if GOOGLE_SERVICE_JSON_PATH else "(no usado)")
    print("GOOGLE_SERVICE_JSON:", "configurado" if GOOGLE_SERVICE_JSON else "(no usado)")

    print("\nConectando a Google Drive...")
    service = get_drive_service()

    contenido = (
        "Prueba correcta desde Python.\n"
        "Proyecto: NivelaCaudal SONICS Loreto.\n"
    )

    subir_o_actualizar_texto(
        service=service,
        folder_id=DRIVE_FOLDER_ID,
        filename="prueba_conexion_drive_nivelacaudal.txt",
        contenido=contenido,
    )

    print("\nPrueba de Drive terminada correctamente.")


if __name__ == "__main__":
    main()