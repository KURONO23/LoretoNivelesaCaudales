# -*- coding: utf-8 -*-
"""Calibra P10/P90 del pronóstico 30/60/90 con conformalización temporal.

El pronóstico central sigue siendo el método de análogos vigente. Se generan
hindcasts recientes, se calculan errores absolutos por horizonte y se usa un
cuantil conformal de esos errores para construir una banda alternativa. La
salida se conserva separada para compararla en el visor.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = BASE_DIR / "backend" / "predicciones_30_60_90"
OUTPUT_DIR = BACKEND_DIR / "output"
OUT_PARQUET = OUTPUT_DIR / "pronostico_30_60_90_conformal_actual.parquet"
OUT_CSV = OUTPUT_DIR / "pronostico_30_60_90_conformal_actual.csv"
OUT_RESUMEN = OUTPUT_DIR / "resumen_pronostico_30_60_90_conformal.csv"

HORIZONTES = (30, 60, 90)
MAX_ORIGENES = 8
ESPACIADO_ORIGENES_DIAS = 30
MIN_CALIBRACION = 20
ALPHA = 0.10


def cargar_base():
    path = Path(__file__).with_name("13_pronostico_30_60_90_analogos.py")
    spec = importlib.util.spec_from_file_location("amaru_analogos_base_conformal", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo cargar el predictor base: {path}")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def serie_diaria(datos: pd.DataFrame) -> tuple[pd.DatetimeIndex, np.ndarray]:
    datos = datos.copy()
    datos["fecha"] = pd.to_datetime(datos["fecha"], errors="coerce").dt.normalize()
    datos["nivel_m"] = pd.to_numeric(datos["nivel_m"], errors="coerce")
    datos = datos.dropna(subset=["fecha"]).sort_values("fecha").drop_duplicates("fecha", keep="last")
    fechas = pd.date_range(datos["fecha"].min(), datos["fecha"].max(), freq="D")
    niveles = datos.set_index("fecha")["nivel_m"].reindex(fechas).to_numpy(dtype=float)
    return fechas, niveles


def origenes_calibracion(fechas: pd.DatetimeIndex, niveles: np.ndarray) -> list[int]:
    """Selecciona orígenes recientes con historia y futuro observados completos."""
    ultimo = len(fechas) - 1
    candidatos = []
    inicio = ultimo - 90 - (MAX_ORIGENES + 2) * ESPACIADO_ORIGENES_DIAS
    for i in range(max(90, inicio), ultimo - 90 + 1, ESPACIADO_ORIGENES_DIAS):
        historia = niveles[i - 89 : i + 1]
        futuro = niveles[i + 1 : i + 91]
        if len(historia) == 90 and len(futuro) == 90 and np.isfinite(historia).all() and np.isfinite(futuro).all():
            candidatos.append(i)
    return candidatos[-MAX_ORIGENES:]


def cuantil_conformal(scores: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if len(scores) < MIN_CALIBRACION:
        return np.nan
    # Cuantil finito conservador: ceil((n+1)*(1-alpha))/n.
    nivel = min(1.0, np.ceil((len(scores) + 1) * (1.0 - ALPHA)) / len(scores))
    return float(np.quantile(scores, nivel, method="higher"))


def calibrar_estacion(base, estacion: str, id_estacion: float, grupo: pd.DataFrame) -> dict[int, float | int]:
    fechas, niveles = serie_diaria(grupo)
    origenes = origenes_calibracion(fechas, niveles)
    errores = {h: [] for h in HORIZONTES}
    for indice in origenes:
        fecha_origen = fechas[indice]
        historico = grupo[pd.to_datetime(grupo["fecha"], errors="coerce").dt.normalize() <= fecha_origen].copy()
        pron, _ = base.pronosticar_estacion(
            estacion=estacion,
            id_estacion=id_estacion,
            datos=historico[["fecha", "nivel_m"]],
            fecha_emision=fecha_origen,
        )
        if pron.empty:
            continue
        for horizonte in HORIZONTES:
            pred = pron[pron["horizonte_dias"] == horizonte].copy()
            pred["fecha_pronostico"] = pd.to_datetime(pred["fecha_pronostico"], errors="coerce").dt.normalize()
            pred["nivel_hibrido_m"] = pd.to_numeric(pred["nivel_hibrido_m"], errors="coerce")
            real = pd.DataFrame({"fecha_pronostico": fechas[indice + 1 : indice + horizonte + 1], "real": niveles[indice + 1 : indice + horizonte + 1]})
            unido = pred.merge(real, on="fecha_pronostico", how="inner")
            if not unido.empty:
                errores[horizonte].extend(
                    np.abs(unido["real"].to_numpy(dtype=float) - unido["nivel_hibrido_m"].to_numpy(dtype=float)).tolist()
                )

    resultado = {}
    for horizonte in HORIZONTES:
        scores = np.asarray(errores[horizonte], dtype=float)
        q = cuantil_conformal(scores)
        resultado[horizonte] = {"q": q, "n": int(np.isfinite(scores).sum())}
    return resultado


def main() -> int:
    base = cargar_base()
    if not base.INPUT_PARQUET.exists():
        raise FileNotFoundError(f"No existe la caché de entrada: {base.INPUT_PARQUET}")

    datos = pd.read_parquet(base.INPUT_PARQUET)
    datos.columns = [str(c).strip().lower() for c in datos.columns]
    requeridas = {"estacion", "fecha", "nivel_m"}
    if not requeridas.issubset(datos.columns):
        raise ValueError(f"La caché no tiene columnas requeridas: {requeridas}")
    datos["fecha"] = pd.to_datetime(datos["fecha"], errors="coerce").dt.normalize()
    datos["nivel_m"] = pd.to_numeric(datos["nivel_m"], errors="coerce")
    datos = datos.dropna(subset=["estacion", "fecha", "nivel_m"]).copy()

    actual = pd.read_parquet(base.OUT_PARQUET)
    actual["fecha_pronostico"] = pd.to_datetime(actual["fecha_pronostico"], errors="coerce").dt.normalize()
    salida_partes = []
    resumen = []
    for estacion, grupo in sorted(datos.groupby("estacion", sort=True)):
        id_col = pd.to_numeric(grupo["id_estacion"], errors="coerce").dropna() if "id_estacion" in grupo.columns else pd.Series(dtype=float)
        id_value = float(id_col.iloc[0]) if not id_col.empty else np.nan
        calibracion = calibrar_estacion(base, str(estacion), id_value, grupo)
        bloque = actual[actual["estacion"].astype(str).str.strip() == str(estacion).strip()].copy()
        if bloque.empty:
            continue
        q_por_h = {}
        for horizonte in HORIZONTES:
            q = calibracion[horizonte]["q"]
            n = calibracion[horizonte]["n"]
            q_por_h[horizonte] = q
            mask = bloque["horizonte_dias"].astype(int) == horizonte
            centro = pd.to_numeric(bloque.loc[mask, "nivel_hibrido_m"], errors="coerce")
            if np.isfinite(q):
                bloque.loc[mask, "p10_conformal_m"] = centro - q
                bloque.loc[mask, "p90_conformal_m"] = centro + q
                bloque.loc[mask, "conformal_q_m"] = q
                bloque.loc[mask, "n_calibracion"] = n
                bloque.loc[mask, "estado_conformal"] = "OK_CALIBRADO"
            else:
                bloque.loc[mask, "p10_conformal_m"] = bloque.loc[mask, "p10_m"]
                bloque.loc[mask, "p90_conformal_m"] = bloque.loc[mask, "p90_m"]
                bloque.loc[mask, "conformal_q_m"] = np.nan
                bloque.loc[mask, "n_calibracion"] = n
                bloque.loc[mask, "estado_conformal"] = "SIN_CALIBRACION_USA_P10_P90"
        bloque["metodo"] = "ANALOGO_ESTADO_V2_CONFORMAL"
        salida_partes.append(bloque)
        for horizonte in HORIZONTES:
            resumen.append({
                "estacion": estacion,
                "horizonte_dias": horizonte,
                "fecha_emision": bloque["fecha_emision"].iloc[0],
                "conformal_q_m": q_por_h[horizonte],
                "n_calibracion": calibracion[horizonte]["n"],
                "estado_conformal": "OK_CALIBRADO" if np.isfinite(q_por_h[horizonte]) else "SIN_CALIBRACION_USA_P10_P90",
            })

    if not salida_partes:
        raise RuntimeError("No se generó salida conformal.")
    salida = pd.concat(salida_partes, ignore_index=True)
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
