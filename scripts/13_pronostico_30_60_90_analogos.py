# -*- coding: utf-8 -*-
"""
Pronóstico diario de niveles a 30, 60 y 90 días mediante análogos de estado v2.

Entrada:
    backend/predicciones_30_60_90/cache/niveles_drive_predicciones.parquet

Salida fija y pequeña para el frontend:
    backend/predicciones_30_60_90/output/pronostico_30_60_90_actual.parquet
    backend/predicciones_30_60_90/output/pronostico_30_60_90_actual.csv
    backend/predicciones_30_60_90/output/resumen_pronostico_30_60_90.csv

Solo usa niveles H_PROM. No usa caudales ni modifica DWLT.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = BASE_DIR / "backend" / "predicciones_30_60_90"
CACHE_DIR = BACKEND_DIR / "cache"
OUTPUT_DIR = BACKEND_DIR / "output"
INPUT_PARQUET = Path(
    os.getenv(
        "PRED_INPUT_PARQUET",
        str(CACHE_DIR / "niveles_drive_predicciones.parquet"),
    )
)
OUT_PARQUET = OUTPUT_DIR / "pronostico_30_60_90_actual.parquet"
OUT_CSV = OUTPUT_DIR / "pronostico_30_60_90_actual.csv"
OUT_RESUMEN = OUTPUT_DIR / "resumen_pronostico_30_60_90.csv"

HORIZONTES = (30, 60, 90)
VENTANAS = (30, 60, 90)
K = 10
CANDIDATE_STRIDE = 14
METODO = "ANALOGO_ESTADO_V2_HIBRIDO_CLIMATOLOGIA"


def _metricas_basicas(observado: np.ndarray, predicho: np.ndarray) -> dict:
    mask = np.isfinite(observado) & np.isfinite(predicho)
    if mask.sum() == 0:
        return {"mae": np.nan, "rmse": np.nan, "nse": np.nan}
    o = observado[mask]
    p = predicho[mask]
    mae = float(np.mean(np.abs(p - o)))
    rmse = float(np.sqrt(np.mean((p - o) ** 2)))
    den = float(np.sum((o - o.mean()) ** 2))
    nse = float(1 - np.sum((p - o) ** 2) / den) if den > 0 else np.nan
    return {"mae": mae, "rmse": rmse, "nse": nse}


def estado(a: np.ndarray, i: int) -> np.ndarray | None:
    if i < 30:
        return None
    bloque = a[i - 29 : i + 1]
    if len(bloque) != 30 or np.isnan(bloque).any():
        return None
    return np.array(
        [
            a[i],
            a[i] - a[i - 7],
            a[i] - a[i - 15],
            a[i] - a[i - 30],
            (bloque[-1] - bloque[0]) / 29,
            np.std(np.diff(bloque)),
        ],
        dtype=float,
    )


def climatologia(
    a: np.ndarray,
    fechas: pd.DatetimeIndex,
    i: int,
    horizonte: int,
) -> np.ndarray:
    historico = pd.DataFrame({"fecha": fechas[: i + 1], "nivel": a[: i + 1]}).dropna()
    if historico.empty:
        return np.full(horizonte, np.nan)

    historico["dia_ano"] = historico["fecha"].dt.dayofyear
    clim = historico.groupby("dia_ano")["nivel"].mean()
    fechas_futuras = fechas[i + 1 : i + horizonte + 1]
    resultado = np.array(
        [clim.get(f.dayofyear, np.nan) for f in fechas_futuras],
        dtype=float,
    )

    media = float(historico["nivel"].mean())
    return np.where(np.isfinite(resultado), resultado, media)


def seleccionar_analogos(
    a: np.ndarray,
    fechas: pd.DatetimeIndex,
    i: int,
    ventana: int,
    horizonte: int,
) -> dict | None:
    actual = estado(a, i)
    historial = a[i - ventana + 1 : i + 1]
    if actual is None or len(historial) != ventana or np.isnan(historial).any():
        return None

    escalas = np.array(
        [max(np.std(historial), 0.5)]
        + [max(np.std(np.diff(historial)), 0.05)] * 5,
        dtype=float,
    )
    candidatos = []

    for j in range(ventana - 1, i - horizonte + 1, CANDIDATE_STRIDE):
        analog = estado(a, j)
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
        diferencia_doy = abs(fechas[j].dayofyear - fechas[i].dayofyear)
        diferencia_doy = min(diferencia_doy, 365 - diferencia_doy) / 90.0
        distancia = float(
            np.sqrt(distancia_estado**2 + 0.25 * diferencia_doy**2)
        )
        candidatos.append(
            {
                "indice": j,
                "distancia": distancia,
                "ultimo": float(a[j]),
                "futuro": futuro.astype(float),
            }
        )

    if len(candidatos) < K:
        return None

    tabla = (
        pd.DataFrame(candidatos)
        .sort_values("distancia")
        .head(K)
        .reset_index(drop=True)
    )
    return {"tabla": tabla, "actual": float(a[i])}


def predecir_analogos(
    seleccion: dict,
    climatologia_futura: np.ndarray,
) -> dict[str, np.ndarray | float | int]:
    tabla = seleccion["tabla"]
    matriz = np.vstack(tabla["futuro"].to_numpy())
    pesos = 1.0 / (tabla["distancia"].to_numpy(dtype=float) + 1e-6)
    pesos = pesos / pesos.sum()

    ajustados = (
        matriz
        + (
            seleccion["actual"] - tabla["ultimo"].to_numpy(dtype=float)
        )[:, None]
    )
    analogos = np.sum(ajustados * pesos[:, None], axis=0)

    plazos = np.arange(1, len(analogos) + 1)
    alpha = np.clip(1 - plazos / 120.0, 0.25, 0.75)
    hibrido = alpha * analogos + (1 - alpha) * climatologia_futura

    return {
        "analogos": analogos,
        "hibrido": hibrido,
        "p10": np.nanpercentile(ajustados, 10, axis=0),
        "p90": np.nanpercentile(ajustados, 90, axis=0),
        "n_analogos": int(len(tabla)),
        "distancia_media": float(tabla["distancia"].mean()),
    }


def pronosticar_estacion(
    estacion: str,
    id_estacion: float,
    datos: pd.DataFrame,
    fecha_emision: pd.Timestamp,
) -> tuple[pd.DataFrame, dict]:
    datos = datos.copy()
    print(
        f"[PRED] {estacion}: entrada={len(datos):,} | "
        f"fecha_dtype={datos['fecha'].dtype if 'fecha' in datos else 'ausente'} | "
        f"nivel_validos_pre={pd.to_numeric(datos.get('nivel_m'), errors='coerce').notna().sum() if 'nivel_m' in datos else 0:,}"
    )
    datos["fecha"] = pd.to_datetime(datos["fecha"], errors="coerce").dt.normalize()
    datos["nivel_m"] = pd.to_numeric(datos["nivel_m"], errors="coerce")
    datos = (
        datos.dropna(subset=["fecha", "nivel_m"])
        .sort_values("fecha")
        .drop_duplicates("fecha", keep="last")
    )

    print(
        f"[PRED] {estacion}: después_limpieza={len(datos):,} | "
        f"rango={datos['fecha'].min() if not datos.empty else 'NaT'} a "
        f"{datos['fecha'].max() if not datos.empty else 'NaT'}"
    )

    if datos.empty:
        return pd.DataFrame(), {
            "estacion": estacion,
            "estado": "SIN_DATOS",
            "fecha_origen": pd.NaT,
            "n_datos": 0,
        }

    fechas = pd.date_range(datos["fecha"].min(), datos["fecha"].max(), freq="D")
    serie = datos.set_index("fecha")["nivel_m"].reindex(fechas).astype(float)
    a = serie.to_numpy()
    origen_i = len(a) - 1
    fecha_origen = fechas[origen_i]

    filas = []
    estados = []
    for horizonte in HORIZONTES:
        mejor = None
        ventana_usada = None
        for ventana in VENTANAS:
            seleccion = seleccionar_analogos(
                a, fechas, origen_i, ventana, horizonte
            )
            if seleccion is not None:
                mejor = seleccion
                ventana_usada = ventana
                break

        fechas_futuras = pd.date_range(
            fecha_origen + pd.Timedelta(days=1),
            periods=horizonte,
            freq="D",
        )
        clima = climatologia(a, fechas.append(fechas_futuras), origen_i, horizonte)
        persistencia = np.repeat(float(a[origen_i]), horizonte)

        if mejor is None:
            analogos = np.full(horizonte, np.nan)
            hibrido = np.full(horizonte, np.nan)
            p10 = np.full(horizonte, np.nan)
            p90 = np.full(horizonte, np.nan)
            modelo = "PERSISTENCIA"
            n_analogos = 0
            distancia_media = np.nan
            estado = "SIN_10_ANALOGOS_USA_PERSISTENCIA"
            nivel_seleccionado = persistencia
            ventana_usada = np.nan
        else:
            pred = predecir_analogos(mejor, clima)
            analogos = pred["analogos"]
            hibrido = pred["hibrido"]
            p10 = pred["p10"]
            p90 = pred["p90"]
            modelo = "HIBRIDO_ANALOGO_CLIM"
            n_analogos = pred["n_analogos"]
            distancia_media = pred["distancia_media"]
            estado = "OK_ANALOGOS"
            nivel_seleccionado = hibrido

        for plazo, fecha_pron in enumerate(fechas_futuras):
            filas.append(
                {
                    "fecha_emision": fecha_emision,
                    "estacion": estacion,
                    "id_estacion": id_estacion,
                    "fecha_origen": fecha_origen,
                    "fecha_pronostico": fecha_pron,
                    "horizonte_dias": horizonte,
                    "dia_adelante": plazo + 1,
                    "nivel_pronosticado_m": float(nivel_seleccionado[plazo]),
                    "nivel_analogos_m": float(analogos[plazo]) if np.isfinite(analogos[plazo]) else np.nan,
                    "nivel_hibrido_m": float(hibrido[plazo]) if np.isfinite(hibrido[plazo]) else np.nan,
                    "nivel_persistencia_m": float(persistencia[plazo]),
                    "nivel_climatologia_m": float(clima[plazo]) if np.isfinite(clima[plazo]) else np.nan,
                    "p10_m": float(p10[plazo]) if np.isfinite(p10[plazo]) else np.nan,
                    "p90_m": float(p90[plazo]) if np.isfinite(p90[plazo]) else np.nan,
                    "modelo": modelo,
                    "metodo": METODO,
                    "ventana_analogos_dias": ventana_usada,
                    "n_analogos": n_analogos,
                    "distancia_media": distancia_media,
                    "estado": estado,
                }
            )

        estados.append(
            {
                "estacion": estacion,
                "id_estacion": id_estacion,
                "fecha_emision": fecha_emision,
                "fecha_origen": fecha_origen,
                "n_datos": int(datos["nivel_m"].notna().sum()),
                "horizonte_dias": horizonte,
                "modelo": modelo,
                "estado": estado,
                "n_analogos": n_analogos,
                "ventana_analogos_dias": ventana_usada,
                "distancia_media": distancia_media,
            }
        )

    return pd.DataFrame(filas), {"estados": estados}


def main() -> int:
    if not INPUT_PARQUET.exists():
        raise FileNotFoundError(f"No existe la caché de entrada: {INPUT_PARQUET}")

    datos = pd.read_parquet(INPUT_PARQUET)
    print(
        f"[PRED] parquet bruto filas={len(datos):,} | "
        f"fecha_dtype={datos['fecha'].dtype if 'fecha' in datos else 'ausente'} | "
        f"fechas_no_nulas={datos['fecha'].notna().sum():,} | "
        f"niveles_no_nulos={datos['nivel_m'].notna().sum():,}"
    )
    datos.columns = [str(c).strip().lower() for c in datos.columns]
    requeridas = {"estacion", "fecha", "nivel_m"}
    if not requeridas.issubset(datos.columns):
        raise ValueError(f"La caché no tiene columnas requeridas: {requeridas}")

    datos["fecha"] = pd.to_datetime(datos["fecha"], errors="coerce").dt.normalize()
    datos["nivel_m"] = pd.to_numeric(datos["nivel_m"], errors="coerce")
    datos = datos.dropna(subset=["estacion", "fecha", "nivel_m"]).copy()
    print(
        f"[PRED] entrada total={len(datos):,} | estaciones={datos['estacion'].nunique()} | "
        f"niveles_validos={datos['nivel_m'].notna().sum():,}"
    )
    fecha_emision = pd.Timestamp(date.today())

    partes = []
    resumen = []
    for estacion, grupo in sorted(datos.groupby("estacion", sort=True)):
        id_estacion = pd.to_numeric(grupo["id_estacion"], errors="coerce").dropna()
        id_value = float(id_estacion.iloc[0]) if not id_estacion.empty else np.nan
        pron, estado = pronosticar_estacion(
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
        raise RuntimeError("No se generó ningún pronóstico.")

    salida = pd.concat(partes, ignore_index=True)
    salida = salida.sort_values(
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

