# NivelaCaudal - SONICS Loreto

Proyecto en Python para extraer información SONICS desde FTP, filtrar por COMID de estaciones hidrológicas y guardar archivos procesados para consulta posterior.

## Flujo general

FTP SONICS → NetCDF reciente → filtro por COMID → archivos filtrados → Google Drive

## Insumos

- GPKG de estaciones hidrológicas con campo COMID.
- Acceso FTP SONICS.
- Carpeta de Google Drive.
- Credenciales de servicio de Google.

## Salidas esperadas

- hist_filtrado.parquet
- fore_filtrado.parquet
- meta_filtrado.rds

## Ejecución local

```bash
pip install -r requirements.txt
python scripts/01_actualizar_sonics_ftp_drive.py