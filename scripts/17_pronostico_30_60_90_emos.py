# -*- coding: utf-8 -*-
"""Línea EMOS/BMA lineal experimental para pronósticos 30/60/90.

La línea combina cuatro miembros disponibles en AMARU: análogo de estado,
análogo DTW, persistencia y climatología. Los pesos se estiman con hindcasts
recientes por estación y horizonte mediante una regresión no negativa
normalizada. Es una aproximación operativa a EMOS/BMA y se conserva separada
de la línea vigente hasta validar su desempeño por estación.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = BASE_DIR / "backend" / "predicciones_30_60_90"
OUTPUT_DIR = BACKEND_DIR / "output"
OUT_PARQUET = OUTPUT_DIR / "pronostico_30_60_90_emos_actual.parquet"
OUT_CSV = OUTPUT_DIR / "pronostico_30_60_90_emos_actual.csv"
OUT_RESUMEN = OUTPUT_DIR / "resumen_pronostico_30_60_90_emos.csv"

HORIZONTES = (30, 60, 90)
MIEMBROS = ("analogo", "dtw", "persistencia", "climatologia")
MAX_ORIGENES = 6
ESPACIADO_ORIGENES_DIAS = 30
MIN_CALIBRACION = 24


def cargar_modulo(nombre: str, archivo: str):
    path = Path(__file__).with_name(archivo)
    spec = importlib.util.spec_from_file_location(nombre, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo cargar {path}")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def preparar_serie(grupo: pd.DataFrame) -> tuple[pd.DatetimeIndex, np.ndarray]:
    tmp = grupo.copy()
    tmp["fecha"] = pd.to_datetime(tmp["fecha"], errors="coerce").dt.normalize()
    tmp["nivel_m"] = pd.to_numeric(tmp["nivel_m"], errors="coerce")
    tmp = tmp.dropna(subset=["fecha"]).sort_values("fecha").drop_duplicates("fecha", keep="last")
    fechas = pd.date_range(tmp["fecha"].min(), tmp["fecha"].max(), freq="D")
    niveles = tmp.set_index("fecha")["nivel_m"].reindex(fechas).to_numpy(dtype=float)
    return fechas, niveles


def seleccionar_origenes(fechas: pd.DatetimeIndex, niveles: np.ndarray) -> list[int]:
    ultimo = len(fechas) - 1
    inicio = ultimo - 90 - (MAX_ORIGENES + 2) * ESPACIADO_ORIGENES_DIAS
    candidatos = []
    for i in range(max(90, inicio), ultimo - 90 + 1, ESPACIADO_ORIGENES_DIAS):
        historia = niveles[i - 89 : i + 1]
        futuro = niveles[i + 1 : i + 91]
        if len(historia) == 90 and len(futuro) == 90 and np.isfinite(historia).all() and np.isfinite(futuro).all():
            candidatos.append(i)
    return candidatos[-MAX_ORIGENES:]


def ajustar_pesos(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Ajuste lineal con pesos no negativos que suman uno."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
    if mask.sum() < MIN_CALIBRACION:
        return np.full(len(MIEMBROS), np.nan), np.nan
    diseño = np.column_stack([np.ones(mask.sum()), X[mask]])
    coef, *_ = np.linalg.lstsq(diseño, y[mask], rcond=None)
    pesos = np.clip(coef[1:], 0.0, None)
    if pesos.sum() <= 0:
        pesos = np.full(len(MIEMBROS), 1.0 / len(MIEMBROS))
    else:
        pesos = pesos / pesos.sum()
    intercepto = float(np.mean(y[mask] - X[mask] @ pesos))
    return pesos, intercepto


def pronostico_miembro(base, selector_dtw, historico: pd.DataFrame, estacion: str, id_estacion: float, fecha_origen: pd.Timestamp, horizonte: int, usar_dtw: bool) -> pd.DataFrame:
    selector_original = base.seleccionar_analogos
    if usar_dtw:
        base.METODO = "ANALOGO_DTW_HIBRIDO_CLIMATOLOGIA"
        base.seleccionar_analogos = selector_dtw
    else:
        base.METODO = "ANALOGO_ESTADO_V2_HIBRIDO_CLIMATOLOGIA"
    pron, _ = base.pronosticar_estacion(
        estacion=estacion,
        id_estacion=id_estacion,
        datos=historico[["fecha", "nivel_m"]],
        fecha_emision=fecha_origen,
    )
    base.seleccionar_analogos = selector_original
    return pron[pron["horizonte_dias"] == horizonte].copy()


def construir_selector_dtw():
    modulo = cargar_modulo("amaru_dtw_emos", "15_pronostico_30_60_90_dtw.py")
    base = cargar_modulo("amaru_base_emos", "13_pronostico_30_60_90_analogos.py")
    return base, modulo, modulo.construir_selector_dtw(base)


def calibrar_estacion(base, selector_dtw, estacion: str, id_estacion: float, grupo: pd.DataFrame) -> dict[int, dict]:
    fechas, niveles = preparar_serie(grupo)
    errores = {h: [] for h in HORIZONTES}
    for indice in seleccionar_origenes(fechas, niveles):
        fecha_origen = fechas[indice]
        historico = grupo[pd.to_datetime(grupo["fecha"], errors="coerce").dt.normalize() <= fecha_origen].copy()
        for horizonte in HORIZONTES:
            actual = pronostico_miembro(base, selector_dtw, historico, estacion, id_estacion, fecha_origen, horizonte, False)
            analog_dtw = pronostico_miembro(base, selector_dtw, historico, estacion, id_estacion, fecha_origen, horizonte, True)
            if actual.empty or analog_dtw.empty:
                continue
            actual = actual.sort_values("fecha_pronostico")
            analog_dtw = analog_dtw.sort_values("fecha_pronostico")
            futuro = pd.DataFrame({
                "fecha_pronostico": fechas[indice + 1 : indice + horizonte + 1],
                "real": niveles[indice + 1 : indice + horizonte + 1],
            })
            a = actual[["fecha_pronostico", "nivel_hibrido_m", "nivel_persistencia_m", "nivel_climatologia_m"]].rename(columns={"nivel_hibrido_m": "analogo"})
            d = analog_dtw[["fecha_pronostico", "nivel_hibrido_m"]].rename(columns={"nivel_hibrido_m": "dtw"})
            unido = futuro.merge(a, on="fecha_pronostico", how="inner").merge(d, on="fecha_pronostico", how="inner")
            if not unido.empty:
                errores[horizonte].append(unido[["analogo", "dtw", "nivel_persistencia_m", "nivel_climatologia_m", "real"]].to_numpy(dtype=float))

    resultado = {}
    for horizonte in HORIZONTES:
        matrices = errores[horizonte]
        if matrices:
            tabla = np.vstack(matrices)
            pesos, intercepto = ajustar_pesos(tabla[:, :4], tabla[:, 4])
            n = int(np.isfinite(tabla[:, 4]).sum())
        else:
            pesos, intercepto, n = np.full(len(MIEMBROS), np.nan), np.nan, 0
        resultado[horizonte] = {"pesos": pesos, "intercepto": intercepto, "n": n}
    return resultado


def main() -> int:
    base, dtw, selector_dtw = construir_selector_dtw()
    if not base.INPUT_PARQUET.exists() or not base.OUT_PARQUET.exists() or not dtw.OUT_PARQUET.exists():
        raise FileNotFoundError("Faltan entradas o salidas vigentes para EMOS/BMA.")

    datos = pd.read_parquet(base.INPUT_PARQUET)
    datos.columns = [str(c).strip().lower() for c in datos.columns]
    datos["fecha"] = pd.to_datetime(datos["fecha"], errors="coerce").dt.normalize()
    datos["nivel_m"] = pd.to_numeric(datos["nivel_m"], errors="coerce")
    datos = datos.dropna(subset=["estacion", "fecha", "nivel_m"]).copy()
    actual = pd.read_parquet(base.OUT_PARQUET)
    dtw_salida = pd.read_parquet(dtw.OUT_PARQUET)
    claves = ["estacion", "fecha_pronostico", "horizonte_dias"]
    actual["fecha_pronostico"] = pd.to_datetime(actual["fecha_pronostico"], errors="coerce").dt.normalize()
    dtw_salida["fecha_pronostico"] = pd.to_datetime(dtw_salida["fecha_pronostico"], errors="coerce").dt.normalize()
    dtw_salida = dtw_salida[claves + ["nivel_hibrido_m"]].rename(columns={"nivel_hibrido_m": "nivel_hibrido_dtw_m"})
    salida = actual.merge(dtw_salida, on=claves, how="left")
    resumen = []

    for estacion, grupo in sorted(datos.groupby("estacion", sort=True)):
        id_col = pd.to_numeric(grupo["id_estacion"], errors="coerce").dropna() if "id_estacion" in grupo.columns else pd.Series(dtype=float)
        id_value = float(id_col.iloc[0]) if not id_col.empty else np.nan
        es_enapu = str(estacion).strip().upper() == "ENAPU"
        calibracion = {h: {"pesos": np.full(len(MIEMBROS), np.nan), "intercepto": np.nan, "n": 0} for h in HORIZONTES} if es_enapu else calibrar_estacion(base, selector_dtw, str(estacion), id_value, grupo)
        mask_est = salida["estacion"].astype(str).str.strip() == str(estacion).strip()
        bloque = salida[mask_est].copy()
        for horizonte in HORIZONTES:
            mask = bloque["horizonte_dias"].astype(int) == horizonte
            cal = calibracion[horizonte]
            pesos = cal["pesos"]
            if np.isfinite(pesos).all():
                X = bloque.loc[mask, ["nivel_hibrido_m", "nivel_hibrido_dtw_m", "nivel_persistencia_m", "nivel_climatologia_m"]].to_numpy(dtype=float)
                emos = cal["intercepto"] + X @ pesos
                bloque.loc[mask, "nivel_emos_m"] = emos
                bloque.loc[mask, "estado_emos"] = "OK_CALIBRADO"
            else:
                bloque.loc[mask, "nivel_emos_m"] = bloque.loc[mask, "nivel_hibrido_m"]
                bloque.loc[mask, "estado_emos"] = "EXCLUIDA_UNIDADES_MIXTAS" if es_enapu else "SIN_CALIBRACION_USA_VIGENTE"
            bloque.loc[mask, "peso_analogo"] = pesos[0]
            bloque.loc[mask, "peso_dtw"] = pesos[1]
            bloque.loc[mask, "peso_persistencia"] = pesos[2]
            bloque.loc[mask, "peso_climatologia"] = pesos[3]
            bloque.loc[mask, "intercepto_emos_m"] = cal["intercepto"]
            bloque.loc[mask, "n_calibracion_emos"] = cal["n"]
        if not es_enapu and "nivel_emos_m" in bloque.columns and np.isfinite(bloque["nivel_emos_m"]).any():
            origen = pd.to_datetime(bloque["fecha_origen"], errors="coerce").dropna().iloc[0]
            observado = grupo[pd.to_datetime(grupo["fecha"], errors="coerce").dt.normalize() == origen]["nivel_m"]
            if not observado.empty:
                primer = bloque.sort_values("fecha_pronostico").iloc[0]["nivel_emos_m"]
                bloque["correccion_continuidad_emos_m"] = float(observado.iloc[-1] - primer)
                bloque["nivel_emos_m"] = bloque["nivel_emos_m"] + float(observado.iloc[-1] - primer)
        else:
            bloque["correccion_continuidad_emos_m"] = 0.0
        salida.loc[mask_est, bloque.columns] = bloque
        for horizonte in HORIZONTES:
            cal = calibracion[horizonte]
            resumen.append({"estacion": estacion, "horizonte_dias": horizonte, "n_calibracion_emos": cal["n"], "peso_analogo": cal["pesos"][0], "peso_dtw": cal["pesos"][1], "peso_persistencia": cal["pesos"][2], "peso_climatologia": cal["pesos"][3], "estado_emos": "EXCLUIDA_UNIDADES_MIXTAS" if es_enapu else "OK_CALIBRADO" if np.isfinite(cal["pesos"]).all() else "SIN_CALIBRACION_USA_VIGENTE"})

    salida["metodo_emos"] = "EMOS_LINEAL_EXPERIMENTAL"
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
