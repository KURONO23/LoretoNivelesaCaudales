# -*- coding: utf-8 -*-
"""Verifica la continuidad del pronóstico 30/60/90 por estación.

Compara la salida vigente anterior con la salida generada por el predictor
que ancla H+1 al último observado. No modifica los Excel ni los archivos
oficiales de salida; guarda gráficos y una tabla de diagnóstico local.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PARQUET = BASE_DIR / "backend" / "cache" / "observado_estaciones.parquet"
SALIDA_ANTERIOR = (
    BASE_DIR
    / "backend"
    / "predicciones_30_60_90"
    / "output"
    / "pronostico_30_60_90_actual.csv"
)
PREDICTOR = BASE_DIR / "scripts" / "13_pronostico_30_60_90_analogos.py"
OUT_DIR = BASE_DIR / "outputs" / "validacion_continuidad_30_60_90"
EXCLUIDAS = {"ENAPU"}


def cargar_predictor():
    spec = importlib.util.spec_from_file_location("predictor_306090", PREDICTOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"No se pudo cargar {PREDICTOR}")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def preparar_datos() -> tuple[pd.DataFrame, pd.DataFrame]:
    obs = pd.read_parquet(INPUT_PARQUET)
    obs.columns = [str(c).strip().lower() for c in obs.columns]
    obs["fecha"] = pd.to_datetime(obs["fecha"], errors="coerce").dt.normalize()
    obs["nivel_m"] = pd.to_numeric(obs["nivel_m"], errors="coerce")
    obs["estacion_key"] = obs["estacion_key"].astype(str).str.strip().str.upper()
    obs = obs.dropna(subset=["fecha", "nivel_m", "estacion_key"]).copy()

    anterior = pd.read_csv(SALIDA_ANTERIOR)
    anterior.columns = [str(c).strip().lower() for c in anterior.columns]
    for col in ["fecha_emision", "fecha_origen", "fecha_pronostico"]:
        anterior[col] = pd.to_datetime(anterior[col], errors="coerce").dt.normalize()
    for col in [
        "horizonte_dias",
        "dia_adelante",
        "nivel_hibrido_m",
        "nivel_pronosticado_m",
        "p10_m",
        "p90_m",
    ]:
        anterior[col] = pd.to_numeric(anterior[col], errors="coerce")
    anterior["estacion"] = anterior["estacion"].astype(str).str.strip().str.upper()
    return obs, anterior


def salto_inicial(df: pd.DataFrame, observado: float) -> float:
    if df.empty or pd.isna(observado):
        return float("nan")
    valor = pd.to_numeric(df.iloc[0]["nivel_hibrido_m"], errors="coerce")
    return float(valor - observado) if pd.notna(valor) else float("nan")


def main() -> int:
    predictor = cargar_predictor()
    obs, anterior = preparar_datos()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    estaciones = sorted(set(anterior["estacion"].dropna()) - EXCLUIDAS)
    resumen = []
    graficos = []

    for estacion in estaciones:
        obs_est = obs[obs["estacion_key"] == estacion].copy()
        old_est = anterior[anterior["estacion"] == estacion].copy()
        if obs_est.empty or old_est.empty:
            continue

        fecha_emision = old_est["fecha_emision"].dropna().max()
        id_estacion = pd.to_numeric(old_est["id_estacion"], errors="coerce").dropna()
        id_value = float(id_estacion.iloc[0]) if not id_estacion.empty else float("nan")

        nuevo, _ = predictor.pronosticar_estacion(
            estacion=estacion,
            id_estacion=id_value,
            datos=obs_est[["fecha", "nivel_m"]],
            fecha_emision=fecha_emision,
        )
        if nuevo.empty:
            continue

        origen = old_est["fecha_origen"].dropna().iloc[0]
        observado = obs_est.loc[obs_est["fecha"] == origen, "nivel_m"]
        if observado.empty:
            continue
        observado = float(observado.iloc[-1])

        for horizonte in [30, 60, 90]:
            old_h = old_est[old_est["horizonte_dias"] == horizonte].sort_values("fecha_pronostico")
            new_h = nuevo[nuevo["horizonte_dias"] == horizonte].sort_values("fecha_pronostico")
            if old_h.empty or new_h.empty:
                continue
            resumen.append(
                {
                    "estacion": estacion,
                    "horizonte_dias": horizonte,
                    "fecha_origen": origen,
                    "nivel_observado_origen_m": observado,
                    "salto_original_m": salto_inicial(old_h, observado),
                    "salto_corregido_m": salto_inicial(new_h, observado),
                    "correccion_m": float(new_h.iloc[0]["correccion_continuidad_m"]),
                    "estado_nuevo": new_h.iloc[0]["estado"],
                    "continuidad_estado": new_h.iloc[0]["continuidad_estado"],
                }
            )

        # Gráfico operativo de 30 días: observado reciente + salida original y corregida.
        old_30 = old_est[old_est["horizonte_dias"] == 30].sort_values("fecha_pronostico")
        new_30 = nuevo[nuevo["horizonte_dias"] == 30].sort_values("fecha_pronostico")
        fecha_min = origen - pd.Timedelta(days=14)
        obs_plot = obs_est[(obs_est["fecha"] >= fecha_min) & (obs_est["fecha"] <= origen)]

        fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
        ax.plot(obs_plot["fecha"], obs_plot["nivel_m"], "o-", color="black", label="Observado")
        ax.plot(old_30["fecha_pronostico"], old_30["nivel_hibrido_m"], "--", color="#94a3b8", label="Pronóstico original")
        ax.plot(new_30["fecha_pronostico"], new_30["nivel_hibrido_m"], "o-", color="#2563eb", label="Pronóstico corregido")
        ax.axvline(origen, color="#64748b", linestyle="--", linewidth=1.5, label="Origen")
        ax.set_title(f"{estacion} — continuidad H+1 y horizonte de 30 días")
        ax.set_ylabel("Nivel (m)")
        ax.set_xlabel("Fecha")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
        fig.autofmt_xdate()
        path = OUT_DIR / f"{estacion.lower()}_continuidad_30d.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        graficos.append(path)

    resumen_df = pd.DataFrame(resumen)
    resumen_df.to_csv(OUT_DIR / "resumen_continuidad.csv", index=False, encoding="utf-8-sig")

    if not resumen_df.empty:
        fig, axes = plt.subplots(3, 4, figsize=(16, 10), constrained_layout=True)
        axes = axes.ravel()
        for ax, estacion in zip(axes, estaciones):
            datos = resumen_df[resumen_df["estacion"] == estacion]
            if datos.empty:
                ax.axis("off")
                continue
            ax.axhline(0, color="#64748b", linewidth=1)
            ax.plot(datos["horizonte_dias"], datos["salto_original_m"], "o--", color="#94a3b8", label="Original")
            ax.plot(datos["horizonte_dias"], datos["salto_corregido_m"], "o-", color="#2563eb", label="Corregido")
            ax.set_title(estacion, fontsize=9)
            ax.set_xlabel("H+ días", fontsize=8)
            ax.set_ylabel("Salto inicial (m)", fontsize=8)
            ax.grid(True, alpha=0.2)
        for ax in axes[len(estaciones):]:
            ax.axis("off")
        axes[0].legend(fontsize=8)
        fig.savefig(OUT_DIR / "resumen_graficos_continuidad.png", dpi=150)
        plt.close(fig)

    print(f"OK: {OUT_DIR}")
    print(resumen_df.to_string(index=False))
    print(f"Gráficos generados: {len(graficos)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
