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
UMBRAL_EQUIVALENCIA_DTW = 0.95
MAX_PESO_DIVERSIFICADO = 0.85
ORIGENES_VALIDACION = 2


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


def proyectar_pesos(v: np.ndarray, max_peso: float = 1.0) -> np.ndarray:
    """Proyecta un vector sobre pesos no negativos que suman uno.

    ``max_peso`` permite evaluar una alternativa diversificada sin inventar
    miembros: cada peso sigue siendo una combinación convexa de los cuatro
    pronósticos ya disponibles.
    """
    v = np.asarray(v, dtype=float)
    if not (1.0 / len(v) <= max_peso <= 1.0):
        raise ValueError("max_peso debe permitir que los pesos sumen uno")
    bajo = float(np.min(v) - max_peso)
    alto = float(np.max(v))
    for _ in range(80):
        umbral = (bajo + alto) / 2.0
        pesos = np.clip(v - umbral, 0.0, max_peso)
        if pesos.sum() > 1.0:
            bajo = umbral
        else:
            alto = umbral
    pesos = np.clip(v - alto, 0.0, max_peso)
    return pesos / pesos.sum()


def ajustar_pesos_simplex(X: np.ndarray, y: np.ndarray, max_peso: float = 1.0) -> tuple[np.ndarray, float]:
    """Ajusta EMOS directamente en el simplex, con intercepto recalculado.

    La rutina previa ajustaba una regresión libre y después truncaba sus
    coeficientes negativos. Ese truncamiento no resuelve el problema
    restringido y puede concentrar artificialmente todo el peso en un miembro.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
    if mask.sum() < MIN_CALIBRACION:
        return np.full(len(MIEMBROS), np.nan), np.nan
    X = X[mask]
    y = y[mask]
    X_centrado = X - X.mean(axis=0)
    y_centrado = y - y.mean()
    pesos = proyectar_pesos(np.full(X.shape[1], 1.0 / X.shape[1]), max_peso)
    norma = float(np.linalg.norm(X_centrado, ord=2) ** 2)
    paso = 1.0 / max(2.0 * norma / len(X), 1e-12)
    for _ in range(2000):
        gradiente = 2.0 * (X_centrado.T @ (X_centrado @ pesos - y_centrado)) / len(X)
        nuevo = proyectar_pesos(pesos - paso * gradiente, max_peso)
        if np.max(np.abs(nuevo - pesos)) < 1e-11:
            pesos = nuevo
            break
        pesos = nuevo
    intercepto = float(np.mean(y - X @ pesos))
    return pesos, intercepto


def rmse(X: np.ndarray, y: np.ndarray, pesos: np.ndarray, intercepto: float) -> float:
    mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
    if not mask.any() or not np.isfinite(pesos).all() or not np.isfinite(intercepto):
        return np.nan
    return float(np.sqrt(np.mean((y[mask] - (intercepto + X[mask] @ pesos)) ** 2)))


def preparar_miembros_emos(estacion: str, X: np.ndarray) -> np.ndarray:
    """Devuelve miembros comparables en una misma escala por estación."""
    miembros = np.asarray(X, dtype=float).copy()
    return miembros


def consolidar_pesos_estacion(estacion: str, pesos: np.ndarray) -> np.ndarray:
    return np.asarray(pesos, dtype=float).copy()


def elegir_calibracion(matrices: list[np.ndarray]) -> dict:
    """Elige el ajuste por validación temporal, sin mezclar orígenes futuros.

    Los dos orígenes más recientes se reservan para validar. La alternativa
    diversificada solo se adopta si reduce el RMSE frente al método heredado;
    en caso contrario se conserva el mejor modelo validado y queda trazado.
    """
    if not matrices:
        nan = np.full(len(MIEMBROS), np.nan)
        return {"pesos": nan, "intercepto": np.nan, "n": 0, "estado": "SIN_CALIBRACION_USA_VIGENTE", "metodo": "SIN_CALIBRACION", "rmse_validacion": np.nan, "rmse_legacy": np.nan}
    tabla = np.vstack(matrices)
    n = int(np.isfinite(tabla[:, 4]).sum())
    if len(matrices) <= ORIGENES_VALIDACION:
        pesos, intercepto = ajustar_pesos_simplex(tabla[:, :4], tabla[:, 4])
        return {"pesos": pesos, "intercepto": intercepto, "n": n, "estado": estado_calibracion(pesos), "metodo": "SIMPLEX_SIN_VALIDACION", "rmse_validacion": np.nan, "rmse_legacy": np.nan}

    entrenamiento = np.vstack(matrices[:-ORIGENES_VALIDACION])
    validacion = np.vstack(matrices[-ORIGENES_VALIDACION:])
    X_train, y_train = entrenamiento[:, :4], entrenamiento[:, 4]
    X_val, y_val = validacion[:, :4], validacion[:, 4]
    candidatos = [
        ("LEGACY_TRUNCADO",) + ajustar_pesos(X_train, y_train),
        ("SIMPLEX",) + ajustar_pesos_simplex(X_train, y_train),
        ("SIMPLEX_DIVERSIFICADO",) + ajustar_pesos_simplex(X_train, y_train, MAX_PESO_DIVERSIFICADO),
    ]
    evaluados = [(nombre, pesos, intercepto, rmse(X_val, y_val, pesos, intercepto)) for nombre, pesos, intercepto in candidatos]
    validos = [fila for fila in evaluados if np.isfinite(fila[3])]
    elegido = min(validos, key=lambda fila: fila[3]) if validos else evaluados[0]
    metodo, _, _, rmse_validacion = elegido
    if metodo == "LEGACY_TRUNCADO":
        pesos, intercepto = ajustar_pesos(tabla[:, :4], tabla[:, 4])
    elif metodo == "SIMPLEX":
        pesos, intercepto = ajustar_pesos_simplex(tabla[:, :4], tabla[:, 4])
    else:
        pesos, intercepto = ajustar_pesos_simplex(tabla[:, :4], tabla[:, 4], MAX_PESO_DIVERSIFICADO)
    estado = estado_calibracion(pesos)
    if metodo == "SIMPLEX_DIVERSIFICADO":
        estado = "OK_CALIBRADO_DIVERSIFICADO"
    return {"pesos": pesos, "intercepto": intercepto, "n": n, "estado": estado, "metodo": metodo, "rmse_validacion": rmse_validacion, "rmse_legacy": evaluados[0][3]}


def estado_calibracion(pesos: np.ndarray) -> str:
    """Describe una calibración que colapsa en un único miembro.

    No modifica los pesos ni fabrica una separación gráfica. La marca permite
    que la salida documente cuándo EMOS es matemáticamente equivalente a DTW.
    """
    if (
        np.isfinite(pesos).all()
        and pesos[1] >= UMBRAL_EQUIVALENCIA_DTW
        and np.delete(pesos, 1).sum() <= 1.0 - UMBRAL_EQUIVALENCIA_DTW + 1e-9
    ):
        return "EMOS_EQUIVALENTE_DTW"
    if np.isfinite(pesos).all():
        return "OK_CALIBRADO"
    return "SIN_CALIBRACION_USA_VIGENTE"


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
        matrices = [np.column_stack([preparar_miembros_emos(estacion, matriz[:, :4]), matriz[:, 4]]) for matriz in matrices]
        resultado[horizonte] = elegir_calibracion(matrices)
        resultado[horizonte]["pesos"] = consolidar_pesos_estacion(estacion, resultado[horizonte]["pesos"])
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
        calibracion = calibrar_estacion(base, selector_dtw, str(estacion), id_value, grupo)
        mask_est = salida["estacion"].astype(str).str.strip() == str(estacion).strip()
        bloque = salida[mask_est].copy()
        for horizonte in HORIZONTES:
            mask = bloque["horizonte_dias"].astype(int) == horizonte
            cal = calibracion[horizonte]
            pesos = cal["pesos"]
            if np.isfinite(pesos).all():
                X = bloque.loc[mask, ["nivel_hibrido_m", "nivel_hibrido_dtw_m", "nivel_persistencia_m", "nivel_climatologia_m"]].to_numpy(dtype=float)
                X = preparar_miembros_emos(str(estacion), X)
                emos = cal["intercepto"] + X @ pesos
                bloque.loc[mask, "nivel_emos_m"] = emos
                bloque.loc[mask, "estado_emos"] = cal["estado"]
            else:
                # Si aún no tiene suficientes ventanas retrospectivas completas de calibración,
                # EMOS combina equitativamente los miembros disponibles (Híbrido + DTW + Climatología)
                h_miembro = bloque.loc[mask, "nivel_hibrido_m"]
                d_miembro = bloque.loc[mask, "nivel_hibrido_dtw_m"]
                c_miembro = bloque.loc[mask, "nivel_climatologia_m"]
                bloque.loc[mask, "nivel_emos_m"] = 0.45 * h_miembro + 0.45 * d_miembro + 0.10 * c_miembro
                bloque.loc[mask, "estado_emos"] = "ENSAMBLE_EQUILIBRADO_SIN_CALIBRACION"
            bloque.loc[mask, "peso_analogo"] = pesos[0]
            bloque.loc[mask, "peso_dtw"] = pesos[1]
            bloque.loc[mask, "peso_persistencia"] = pesos[2]
            bloque.loc[mask, "peso_climatologia"] = pesos[3]
            bloque.loc[mask, "intercepto_emos_m"] = cal["intercepto"]
            bloque.loc[mask, "n_calibracion_emos"] = cal["n"]
        if "nivel_emos_m" in bloque.columns and np.isfinite(bloque["nivel_emos_m"]).any():
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
            resumen.append({"estacion": estacion, "horizonte_dias": horizonte, "n_calibracion_emos": cal["n"], "peso_analogo": cal["pesos"][0], "peso_dtw": cal["pesos"][1], "peso_persistencia": cal["pesos"][2], "peso_climatologia": cal["pesos"][3], "estado_emos": cal["estado"], "metodo_calibracion_emos": cal["metodo"], "rmse_validacion_emos_m": cal["rmse_validacion"], "rmse_validacion_legacy_m": cal["rmse_legacy"]})

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
