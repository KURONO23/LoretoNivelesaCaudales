# -*- coding: utf-8 -*-
"""Pronóstico 30/60/90 con selección de análogos por distancia DTW.

Este módulo es una variante experimental del flujo operativo de
``13_pronostico_30_60_90_analogos.py``. Reutiliza su preparación de datos,
corrección de continuidad, intervalos P10–P90 y exclusión de ENAPU, pero
reemplaza la selección de análogos de estado por una comparación de forma
mediante Dynamic Time Warping (DTW).

La salida se escribe en archivos separados para que el método vigente y DWLT
no se modifiquen mientras se valida la nueva opción en el visor.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = BASE_DIR / "backend" / "predicciones_30_60_90"
OUTPUT_DIR = BACKEND_DIR / "output"
OUT_PARQUET = OUTPUT_DIR / "pronostico_30_60_90_dtw_actual.parquet"
OUT_CSV = OUTPUT_DIR / "pronostico_30_60_90_dtw_actual.csv"
OUT_RESUMEN = OUTPUT_DIR / "resumen_pronostico_30_60_90_dtw.csv"


def cargar_flujo_base():
    """Importa el flujo base sin duplicar sus reglas operativas."""
    path = Path(__file__).with_name("13_pronostico_30_60_90_analogos.py")
    spec = importlib.util.spec_from_file_location("amaru_analogos_base", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo cargar el flujo base: {path}")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def _normalizar(serie: np.ndarray) -> np.ndarray:
    """Normaliza nivel y conserva la forma relativa del hidrograma."""
    x = np.asarray(serie, dtype=float)
    centro = float(np.mean(x))
    escala = float(np.std(x))
    if not np.isfinite(escala) or escala < 1e-6:
        escala = 1.0
    return (x - centro) / escala


def distancia_dtw(a: np.ndarray, b: np.ndarray, banda: int = 10) -> float:
    """Distancia DTW normalizada con banda Sakoe–Chiba.

    La banda evita alineaciones físicamente poco plausibles y mantiene el
    cálculo acotado para las ventanas diarias de 30, 60 y 90 días.
    """
    x = _normalizar(a)
    y = _normalizar(b)
    n, m = len(x), len(y)
    coste = np.full((n + 1, m + 1), np.inf, dtype=float)
    coste[0, 0] = 0.0

    for i in range(1, n + 1):
        j0 = max(1, i - banda)
        j1 = min(m, i + banda)
        for j in range(j0, j1 + 1):
            distancia = abs(x[i - 1] - y[j - 1])
            coste[i, j] = distancia + min(
                coste[i - 1, j],
                coste[i, j - 1],
                coste[i - 1, j - 1],
            )

    valor = coste[n, m]
    return float(valor / max(n + m, 1)) if np.isfinite(valor) else np.inf


def construir_selector_dtw(base):
    """Devuelve un selector compatible con ``pronosticar_estacion``."""

    def seleccionar_analogos_dtw(a, fechas, i, ventana, horizonte):
        actual = base.estado(a, i)
        historial = a[i - ventana + 1 : i + 1]
        if actual is None or len(historial) != ventana or np.isnan(historial).any():
            return None

        escalas = np.array(
            [max(np.std(historial), 0.5)]
            + [max(np.std(np.diff(historial)), 0.05)] * 5,
            dtype=float,
        )
        candidatos = []
        for j in range(ventana - 1, i - horizonte + 1, base.CANDIDATE_STRIDE):
            analog = base.estado(a, j)
            futuro = a[j + 1 : j + horizonte + 1]
            ventana_analog = a[j - ventana + 1 : j + 1]
            if (
                analog is None
                or len(futuro) != horizonte
                or len(ventana_analog) != ventana
                or np.isnan(futuro).any()
                or np.isnan(ventana_analog).any()
            ):
                continue

            distancia_estado = float(
                np.sqrt(np.mean(((actual - analog) / escalas) ** 2))
            )
            # Preselección barata: solo los mejores candidatos de estado pasan
            # al DTW, evitando recalcular toda la historia para cada ventana.
            diferencia_doy = abs(fechas[j].dayofyear - fechas[i].dayofyear)
            diferencia_doy = min(diferencia_doy, 365 - diferencia_doy) / 90.0
            preseleccion = float(
                np.sqrt(distancia_estado**2 + 0.25 * diferencia_doy**2)
            )
            candidatos.append(
                {
                    "indice": j,
                    "preseleccion": preseleccion,
                    "diferencia_doy": diferencia_doy,
                    "ultimo": float(a[j]),
                    "futuro": futuro.astype(float),
                    "ventana_analog": ventana_analog.astype(float),
                }
            )

        if len(candidatos) < base.K:
            return None

        candidatos = sorted(candidatos, key=lambda x: x["preseleccion"])
        candidatos = candidatos[: max(base.K * 5, 50)]
        for candidato in candidatos:
            candidato["distancia_dtw"] = distancia_dtw(
                historial, candidato["ventana_analog"], banda=min(10, ventana // 4)
            )
            candidato["distancia"] = float(
                candidato["distancia_dtw"] + 0.15 * candidato["diferencia_doy"]
            )

        tabla = (
            pd.DataFrame(candidatos)
            .sort_values("distancia")
            .head(base.K)
            .reset_index(drop=True)
        )
        return {"tabla": tabla, "actual": float(a[i])}

    return seleccionar_analogos_dtw


def main() -> int:
    base = cargar_flujo_base()
    base.METODO = "ANALOGO_DTW_HIBRIDO_CLIMATOLOGIA"
    base.seleccionar_analogos = construir_selector_dtw(base)

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
    fecha_emision = pd.Timestamp.today().normalize()

    partes = []
    resumen = []
    for estacion, grupo in sorted(datos.groupby("estacion", sort=True)):
        id_estacion = (
            pd.to_numeric(grupo["id_estacion"], errors="coerce").dropna()
            if "id_estacion" in grupo.columns
            else pd.Series(dtype=float)
        )
        id_value = float(id_estacion.iloc[0]) if not id_estacion.empty else np.nan
        pron, estado = base.pronosticar_estacion(
            estacion=str(estacion),
            id_estacion=id_value,
            datos=grupo,
            fecha_emision=fecha_emision,
        )
        if not pron.empty:
            partes.append(pron)
        if "estados" in estado:
            resumen.extend(estado["estados"])
        else:
            resumen.append(estado)

    if not partes:
        raise RuntimeError("No se generó ningún pronóstico DTW.")

    salida = pd.concat(partes, ignore_index=True).sort_values(
        ["estacion", "horizonte_dias", "fecha_pronostico"]
    ).reset_index(drop=True)
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
