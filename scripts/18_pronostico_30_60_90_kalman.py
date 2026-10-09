# -*- coding: utf-8 -*-
"""Línea experimental de niveles mediante filtro de Kalman estructural.

Modelo local de nivel y tendencia con observaciones diarias. Se usa solo la
serie de niveles de cada estación; las brechas se omiten en la actualización
del filtro y no se rellenan en los datos observados.
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
OUT_PARQUET = OUTPUT_DIR / "pronostico_30_60_90_kalman_actual.parquet"
OUT_CSV = OUTPUT_DIR / "pronostico_30_60_90_kalman_actual.csv"
OUT_RESUMEN = OUTPUT_DIR / "resumen_pronostico_30_60_90_kalman.csv"

HORIZONTES = (30, 60, 90)
MAX_HISTORIA = 365
ESTACIONES_EXCLUIDAS = {"ENAPU"}


def serie_diaria(grupo: pd.DataFrame) -> tuple[pd.DatetimeIndex, np.ndarray]:
    tmp = grupo.copy()
    tmp["fecha"] = pd.to_datetime(tmp["fecha"], errors="coerce").dt.normalize()
    tmp["nivel_m"] = pd.to_numeric(tmp["nivel_m"], errors="coerce")
    tmp = tmp.dropna(subset=["fecha"]).sort_values("fecha").drop_duplicates("fecha", keep="last")
    fechas = pd.date_range(tmp["fecha"].min(), tmp["fecha"].max(), freq="D")
    niveles = tmp.set_index("fecha")["nivel_m"].reindex(fechas).to_numpy(dtype=float)
    return fechas, niveles


def kalman_local_tendencia(niveles: np.ndarray, horizonte: int) -> tuple[np.ndarray, str, float]:
    """Filtra nivel/tendencia y proyecta con tendencia amortiguada."""
    y = np.asarray(niveles, dtype=float)
    validos = np.flatnonzero(np.isfinite(y))
    if len(validos) < 30:
        return np.full(horizonte, np.nan), "SIN_DATOS_SUFICIENTES", np.nan

    # El filtro se ajusta con el último año disponible para evitar que cambios
    # de régimen antiguos dominen la predicción operativa.
    y = y[max(0, len(y) - MAX_HISTORIA) :]
    validos = np.flatnonzero(np.isfinite(y))
    if len(validos) < 30:
        return np.full(horizonte, np.nan), "SIN_DATOS_SUFICIENTES", np.nan

    diferencias = np.diff(y[validos])
    var_obs = float(max(np.nanvar(diferencias), 1e-4))
    var_tend = float(max(np.nanvar(np.diff(diferencias)), 1e-6)) if len(diferencias) > 2 else 1e-5
    F = np.array([[1.0, 1.0], [0.0, 0.97]], dtype=float)
    H = np.array([[1.0, 0.0]], dtype=float)
    Q = np.array([[var_tend * 0.25, 0.0], [0.0, var_tend]], dtype=float)
    R = np.array([[var_obs]], dtype=float)

    primero = validos[0]
    pendiente = float(np.nanmedian(diferencias[-30:])) if len(diferencias) else 0.0
    estado = np.array([y[primero], pendiente], dtype=float)
    cov = np.eye(2, dtype=float)
    for observacion in y[primero:]:
        estado = F @ estado
        cov = F @ cov @ F.T + Q
        if not np.isfinite(observacion):
            continue
        innovacion = np.array([observacion]) - H @ estado
        S = H @ cov @ H.T + R
        K = cov @ H.T @ np.linalg.inv(S)
        estado = estado + (K @ innovacion).ravel()
        cov = (np.eye(2) - K @ H) @ cov

    ultimo_observado = float(estado[0])
    pronostico = []
    for _ in range(horizonte):
        estado = F @ estado
        pronostico.append(float(estado[0]))
    return np.asarray(pronostico, dtype=float), "OK_KALMAN", ultimo_observado


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
    salida["nivel_kalman_m"] = np.nan
    salida["estado_kalman"] = "SIN_CALCULAR"
    salida["correccion_continuidad_kalman_m"] = 0.0
    resumen = []

    for estacion, grupo in sorted(datos.groupby("estacion", sort=True)):
        estacion_txt = str(estacion).strip()
        fechas, niveles = serie_diaria(grupo)
        origen = fechas[-1]
        observado = niveles[-1] if np.isfinite(niveles[-1]) else np.nan
        bloque_mask = salida["estacion"].astype(str).str.strip() == estacion_txt
        bloque = salida[bloque_mask].copy()
        excluida = estacion_txt.upper() in ESTACIONES_EXCLUIDAS
        if excluida:
            bloque["estado_kalman"] = "EXCLUIDA_UNIDADES_MIXTAS"
            resumen.extend({"estacion": estacion_txt, "horizonte_dias": h, "estado_kalman": "EXCLUIDA_UNIDADES_MIXTAS", "pendiente_final_m_dia": np.nan} for h in HORIZONTES)
            salida.loc[bloque_mask, bloque.columns] = bloque
            continue

        for horizonte in HORIZONTES:
            pred, estado, ultimo_filtrado = kalman_local_tendencia(niveles, horizonte)
            mask_h = bloque["horizonte_dias"].astype(int) == horizonte
            fechas_pred = bloque.loc[mask_h, "fecha_pronostico"].sort_values().to_numpy()
            if estado == "OK_KALMAN" and len(fechas_pred) == horizonte:
                offset = float(observado - pred[0]) if np.isfinite(observado) else 0.0
                valores = pred + offset
                bloque.loc[mask_h, "nivel_kalman_m"] = valores
                bloque.loc[mask_h, "estado_kalman"] = estado
                bloque.loc[mask_h, "correccion_continuidad_kalman_m"] = offset
                pendiente = float((valores[-1] - valores[0]) / max(horizonte - 1, 1))
            else:
                pendiente = np.nan
                bloque.loc[mask_h, "estado_kalman"] = estado
            resumen.append({"estacion": estacion_txt, "horizonte_dias": horizonte, "estado_kalman": estado, "pendiente_final_m_dia": pendiente})
        salida.loc[bloque_mask, bloque.columns] = bloque

    salida["metodo_kalman"] = "KALMAN_LOCAL_TENDENCIA_EXPERIMENTAL"
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
