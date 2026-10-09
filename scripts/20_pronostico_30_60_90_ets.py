# -*- coding: utf-8 -*-
"""Línea ETS/Holt experimental para pronósticos 30/60/90 de niveles."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = BASE_DIR / "backend" / "predicciones_30_60_90"
OUTPUT_DIR = BACKEND_DIR / "output"
INPUT_PARQUET = BACKEND_DIR / "cache" / "niveles_drive_predicciones.parquet"
BASE_OUTPUT = OUTPUT_DIR / "pronostico_30_60_90_actual.parquet"
OUT_PARQUET = OUTPUT_DIR / "pronostico_30_60_90_ets_actual.parquet"
OUT_CSV = OUTPUT_DIR / "pronostico_30_60_90_ets_actual.csv"
OUT_RESUMEN = OUTPUT_DIR / "resumen_pronostico_30_60_90_ets.csv"

HORIZONTES = (30, 60, 90)
MAX_HISTORIA = 730
ESTACIONES_EXCLUIDAS = {"ENAPU"}


def serie_diaria(grupo: pd.DataFrame) -> tuple[pd.DatetimeIndex, np.ndarray]:
    tmp = grupo.copy()
    tmp["fecha"] = pd.to_datetime(tmp["fecha"], errors="coerce").dt.normalize()
    tmp["nivel_m"] = pd.to_numeric(tmp["nivel_m"], errors="coerce")
    tmp = tmp.dropna(subset=["fecha"]).sort_values("fecha").drop_duplicates("fecha", keep="last")
    fechas = pd.date_range(tmp["fecha"].min(), tmp["fecha"].max(), freq="D")
    niveles = tmp.set_index("fecha")["nivel_m"].reindex(fechas).to_numpy(dtype=float)
    return fechas, niveles


def holt_pronostico(serie: np.ndarray, horizonte: int, alpha: float, beta: float, phi: float) -> np.ndarray:
    validos = np.flatnonzero(np.isfinite(serie))
    if len(validos) < 30:
        return np.full(horizonte, np.nan)
    y = serie[max(0, len(serie) - MAX_HISTORIA) :]
    idx = np.flatnonzero(np.isfinite(y))
    if len(idx) < 30:
        return np.full(horizonte, np.nan)
    primero = idx[0]
    nivel = float(y[primero])
    pendiente = 0.0
    if len(idx) > 1:
        pendiente = float(np.nanmedian(np.diff(y[idx[: min(30, len(idx))]])))
    for observacion in y[primero:]:
        nivel_previo = nivel
        pendiente_previa = pendiente
        if np.isfinite(observacion):
            nivel = alpha * observacion + (1.0 - alpha) * (nivel_previo + phi * pendiente_previa)
            pendiente = beta * (nivel - nivel_previo) + (1.0 - beta) * phi * pendiente_previa
        else:
            nivel = nivel_previo + phi * pendiente_previa
            pendiente = phi * pendiente_previa

    factores = np.array([sum(phi ** j for j in range(1, h + 1)) for h in range(1, horizonte + 1)])
    return nivel + factores * pendiente


def seleccionar_ets(serie: np.ndarray) -> tuple[float, float, float, str]:
    """Selecciona SES, Holt o Holt amortiguado por SSE de un paso."""
    y = serie[max(0, len(serie) - MAX_HISTORIA) :]
    candidatos = []
    configuraciones = [(a, b, p, nombre) for a in (0.2, 0.4, 0.6, 0.8) for b in (0.05, 0.15, 0.30) for p, nombre in ((0.85, "HOLT_AMORTIGUADO"), (0.95, "HOLT_AMORTIGUADO"), (1.0, "HOLT_TENDENCIA"))]
    configuraciones.extend((a, 0.0, 0.0, "SES") for a in (0.2, 0.4, 0.6, 0.8))
    for alpha, beta, phi, nombre in configuraciones:
        validos = np.flatnonzero(np.isfinite(y))
        if len(validos) < 40:
            continue
        primero = validos[0]
        nivel = float(y[primero])
        pendiente = float(np.nanmedian(np.diff(y[validos[: min(30, len(validos))]]))) if len(validos) > 1 else 0.0
        errores = []
        pasos = 0
        for observacion in y[primero + 1 :]:
            pred = nivel + phi * pendiente
            if np.isfinite(observacion):
                if pasos >= 20:
                    errores.append((pred - observacion) ** 2)
                nivel_previo = nivel
                pendiente_previa = pendiente
                nivel = alpha * observacion + (1.0 - alpha) * pred
                pendiente = beta * (nivel - nivel_previo) + (1.0 - beta) * phi * pendiente_previa
                pasos += 1
            else:
                nivel = pred
                pendiente = phi * pendiente
        if errores:
            candidatos.append((float(np.mean(errores)), alpha, beta, phi, nombre))
    if not candidatos:
        return 0.4, 0.15, 0.95, "HOLT_AMORTIGUADO"
    _, alpha, beta, phi, nombre = min(candidatos, key=lambda x: x[0])
    return alpha, beta, phi, nombre


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
    salida["nivel_ets_m"] = np.nan
    salida["modelo_ets"] = "SIN_CALCULAR"
    salida["alpha_ets"] = np.nan
    salida["beta_ets"] = np.nan
    salida["phi_ets"] = np.nan
    salida["correccion_continuidad_ets_m"] = 0.0
    resumen = []

    for estacion, grupo in sorted(datos.groupby("estacion", sort=True)):
        estacion_txt = str(estacion).strip()
        fechas, niveles = serie_diaria(grupo)
        observado = niveles[-1] if len(niveles) and np.isfinite(niveles[-1]) else np.nan
        bloque_mask = salida["estacion"].astype(str).str.strip() == estacion_txt
        bloque = salida[bloque_mask].copy()
        if estacion_txt.upper() in ESTACIONES_EXCLUIDAS:
            bloque["modelo_ets"] = "EXCLUIDA_UNIDADES_MIXTAS"
            resumen.extend({"estacion": estacion_txt, "horizonte_dias": h, "modelo_ets": "EXCLUIDA_UNIDADES_MIXTAS"} for h in HORIZONTES)
            salida.loc[bloque_mask, bloque.columns] = bloque
            continue

        alpha, beta, phi, nombre = seleccionar_ets(niveles)
        for horizonte in HORIZONTES:
            pred = holt_pronostico(niveles, horizonte, alpha, beta, phi)
            mask_h = bloque["horizonte_dias"].astype(int) == horizonte
            if np.isfinite(pred).all() and len(pred) == horizonte:
                offset = float(observado - pred[0]) if np.isfinite(observado) else 0.0
                bloque.loc[mask_h, "nivel_ets_m"] = pred + offset
                bloque.loc[mask_h, "correccion_continuidad_ets_m"] = offset
                bloque.loc[mask_h, "modelo_ets"] = nombre
            else:
                bloque.loc[mask_h, "modelo_ets"] = "SIN_DATOS_SUFICIENTES"
            bloque.loc[mask_h, "alpha_ets"] = alpha
            bloque.loc[mask_h, "beta_ets"] = beta
            bloque.loc[mask_h, "phi_ets"] = phi
            resumen.append({"estacion": estacion_txt, "horizonte_dias": horizonte, "modelo_ets": nombre, "alpha_ets": alpha, "beta_ets": beta, "phi_ets": phi})
        salida.loc[bloque_mask, bloque.columns] = bloque

    salida["metodo_ets"] = "ETS_HOLT_AMORTIGUADO_EXPERIMENTAL"
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
