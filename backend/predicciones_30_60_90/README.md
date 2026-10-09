# Backend de predicciones 30/60/90

Este módulo es independiente del backend DWLT.

## Flujo

1. `scripts/12_actualizar_predicciones_drive.py` autentica con la cuenta de servicio configurada en `GOOGLE_SERVICE_JSON`.
2. Lee los 12 Excel de la carpeta `PREDICCIONES306090`.
3. Consulta HidroMet para la ventana reciente (`PRED_API_START_DATE`–`PRED_API_END_DATE`).
4. Actualiza el mismo archivo de Drive conservando hoja, columnas, filas históricas y estilos.
5. Genera la entrada propia del backend:
   - `cache/niveles_drive_predicciones.parquet`
   - `cache/niveles_drive_predicciones.csv`

## Variables

- `PRED_DRIVE_FOLDER_ID`: ID de `PREDICCIONES306090`.
- `GOOGLE_SERVICE_JSON` o `PRED_GOOGLE_SERVICE_JSON`: credencial de la cuenta de servicio.
- `PRED_API_START_DATE`: fecha inicial de consulta; por defecto, hoy menos siete días.
- `PRED_API_END_DATE`: fecha final; por defecto, hoy.
- `PRED_DRIVE_WRITE`: `true` para subir cambios a Drive; `false` para ensayo.

## Separación con DWLT

El script no llama a `03_dwlt_todas_estaciones.py`, no escribe en `backend/cache/observado_estaciones.parquet` y no modifica `outputs/fore_nivel_transformado.parquet`. El futuro cálculo de análogos y horizontes 30/60/90 consumirá esta caché propia.

## Control de continuidad

El predictor ancla H+1 al último nivel observado mediante un desplazamiento
constante que conserva la dinámica futura y el ancho del intervalo P10–P90.
La corrección se registra en `correccion_continuidad_m` y
`continuidad_estado`. ENAPU queda excluida (`EXCLUIDA_UNIDADES_MIXTAS`) porque
combina cotas y tirantes; su salto no debe corregirse de forma automática.

## Automatización diaria

`.github/workflows/predicciones_30_60_90.yml` ejecuta el actualizador a las 18:00 UTC (13:00 en Perú). Usa `PRED_DRIVE_FOLDER_ID` —o temporalmente `OBS_DRIVE_FOLDER_ID`— y `GOOGLE_SERVICE_JSON`. Esta Action no modifica el repositorio ni el flujo DWLT; actualiza directamente los mismos archivos de Drive.
