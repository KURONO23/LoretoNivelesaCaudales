# -*- coding: utf-8 -*-
"""Línea ARIMA(p,1,0) experimental para niveles de agua.

Implementación reproducible con numpy, sin dependencias de statsmodels. Se
ajusta un autorregresivo sobre la primera diferencia de los niveles, selecciona
el orden por BIC y proyecta H+30/H+60/H+90. Los vacíos observados no se
rellenan en los datos ni en el visor.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = BASE_DIR / "backend" / "predicciones_30_60_90"
OUTPUT_DIR = BACKEND_DIR / "output"
INPUT_PARQUET = BACKEND_DIR / "cache" / "niveles_drive_predicciones.parquet"
BASE_OUTPUT = OUTPUT_DIR / "pronostico_30_60_90_actual.parquet"
OUT_PARQUET = OUTPUT_DIR / "pronostico_30_60_90_arima_actual.parquet"
OUT_CSV = OUTPUT_DIR / "pronostico_30_60_90_arima_actual.csv"
OUT_RESUMEN = OUTPUT_DIR / "resumen_pronostico_30_60_90_arima.csv"

HORIZONTES = (30, 60, 90)
ORDENES = (1, 2, 3, 5, 7, 10)
MAX_HISTORIA = 730
MIN_DIFERENCIAS = 60
ESTACIONES_EXCLUIDAS = {"ENAPU"}


def serie_diaria(grupo: pd.DataFrame) -> tuple[pd.DatetimeIndex, np.ndarray]:
    tmp = grupo.copy()
    tmp["fecha"] = pd.to_datetime(tmp["fecha"], errors="coerce").dt.normalize()
    tmp["nivel_m"] = pd.to_numeric(tmp["nivel_m"], errors="coerce")
    tmp = tmp.dropna(subset=["fecha"]).sort_values("fecha").drop_duplicates("fecha", keep="last")
    fechas = pd.date_range(tmp["fecha"].min(), tmp["fecha"].max(), freq="D")
    niveles = tmp.set_index("fecha")["nivel_m"].reindex(fechas).to_numpy(dtype=float)
    return fechas, niveles


def ajustar_arima_d1(niveles: np.ndarray, horizonte: int) -> tuple[np.ndarray, int, str]:
    validos = np.flatnonzero(np.isfinite(niveles))
    if len(validos) < MIN_DIFERENCIAS + 1:
        return np.full(horizonte, np.nan), 0, "SIN_DATOS_SUFICIENTES"

    y = niveles[validos][-MAX_HISTORIA:]
    diferencias = np.diff(y)
    if len(diferencias) < MIN_DIFERENCIAS:
        return np.full(horizonte, np.nan), 0, "SIN_DATOS_SUFICIENTES"

    candidatos = []
    for orden in ORDENES:
        if len(diferencias) <= orden + 10:
            continue
        X = []
        objetivo = []
        for i in range(orden, len(diferencias)):
            X.append([1.0, *diferencias[i - orden : i][::-1]])
            objetivo.append(diferencias[i])
        X = np.asarray(X, dtype=float)
        objetivo = np.asarray(objetivo, dtype=float)
        coef, *_ = np.linalg.lstsq(X, objetivo, rcond=None)
        residuos = objetivo - X @ coef
        varianza = max(float(np.mean(residuos**2)), 1e-8)
        bic = len(objetivo) * np.log(varianza) + len(coef) * np.log(len(objetivo))
        candidatos.append((bic, orden, coef))

    if not candidatos:
        return np.full(horizonte, np.nan), 0, "SIN_AJUSTE"

    _, orden, coef = min(candidatos, key=lambda x: x[0])
    historial = list(diferencias[-orden:])
    limite_inferior, limite_superior = np.nanpercentile(diferencias, [1, 99])
    pron_diferencias = []
    for _ in range(horizonte):
        valor = float(coef[0] + np.dot(coef[1:], np.asarray(historial[-orden:][::-1])))
        valor = float(np.clip(valor, limite_inferior, limite_superior))
        pron_diferencias.append(valor)
        historial.append(valor)

    pronostico = float(y[-1]) + np.cumsum(np.asarray(pron_diferencias, dtype=float))
    return pronostico, int(orden), "OK_ARIMA_D1"


def main() -> int:
    if not INPUT_PARQUET.exists() or not BASE_OUTPUT.exists():
        raise FileNotFoundError("Faltan la caché de niveles o la salida vigente.")
    datos = pd.read_parquet(INPUT_PARQUET)
    datos.columns = [str(c).strip().lower() for c in datos.columns]
    datos["fecha"] = pd.to_datetime(datos["fecha"], errors="coerce").dt.normalize()
    datos["nivel_m"] = pd.to_numeric(datos["nivel_m"], errors="coerce")
    datos = datos.dropna(subset=["estacion", "fecha"]).copy()
    salida = pd.read_parquet(BASE_OUTPUT)
    salida["fecha_pronostico"] = pd.to_datetime(salida["fecha_pronostico"], errors="coerce").dt.normalize()
    salida["nivel_arima_m"] = np.nan
    salida["orden_arima_p"] = np.nan
    salida["estado_arima"] = "SIN_CALCULAR"
    salida["correccion_continuidad_arima_m"] = 0.0
    resumen = []

    for estacion, grupo in sorted(datos.groupby("estacion", sort=True)):
        estacion_txt = str(estacion).strip()
        fechas, niveles = serie_diaria(grupo)
        observado = niveles[-1] if len(niveles) and np.isfinite(niveles[-1]) else np.nan
        bloque_mask = salida["estacion"].astype(str).str.strip() == estacion_txt
        bloque = salida[bloque_mask].copy()
        excluida = estacion_txt.upper() in ESTACIONES_EXCLUIDAS
        if excluida:
            bloque["estado_arima"] = "EXCLUIDA_UNIDADES_MIXTAS"
            resumen.extend({"estacion": estacion_txt, "horizonte_dias": h, "orden_arima_p": np.nan, "estado_arima": "EXCLUIDA_UNIDADES_MIXTAS"} for h in HORIZONTES)
            salida.loc[bloque_mask, bloque.columns] = bloque
            continue

        for horizonte in HORIZONTES:
            pred, orden, estado = ajustar_arima_d1(niveles, horizonte)
            mask_h = bloque["horizonte_dias"].astype(int) == horizonte
            if estado == "OK_ARIMA_D1" and len(pred) == horizonte:
                offset = float(observado - pred[0]) if np.isfinite(observado) else 0.0
                valores = pred + offset
                bloque.loc[mask_h, "nivel_arima_m"] = valores
                bloque.loc[mask_h, "orden_arima_p"] = orden
                bloque.loc[mask_h, "estado_arima"] = estado
                bloque.loc[mask_h, "correccion_continuidad_arima_m"] = offset
            else:
                bloque.loc[mask_h, "estado_arima"] = estado
            resumen.append({"estacion": estacion_txt, "horizonte_dias": horizonte, "orden_arima_p": orden if estado == "OK_ARIMA_D1" else np.nan, "estado_arima": estado})
        salida.loc[bloque_mask, bloque.columns] = bloque

    salida["metodo_arima"] = "ARIMA_P_D1_EXPERIMENTAL"
    salida = salida.sort_values(["estacion", "horizonte_dias", "fecha_pronostico"]).reset_index(drop=True)
    resumen_df = pd.DataFrame(resumen)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    salida.to_parquet(OUT_PARQUET, index=False)
    salida.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    resumen_df.to_csv(OUT_RESUMEN, index=False, encoding="utf-8-sig")
    print(f"OK: {OUT_PARQUET} | filas: {len(salida):,}")
    print(f"OK: {OUT_CSV}")
    print(f"OK: {OUT_RESUMEN}")
    print(resumen_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
