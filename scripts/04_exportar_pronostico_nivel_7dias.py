from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FORE_DWLT = OUTPUT_DIR / "fore_nivel_transformado.parquet"
METRICAS = OUTPUT_DIR / "metricas_dwlt_estaciones.xlsx"

OUT_XLSX = OUTPUT_DIR / "pronostico_nivel_7dias_estaciones.xlsx"
OUT_CSV = OUTPUT_DIR / "pronostico_nivel_7dias_estaciones.csv"

DIAS_EXPORTAR = 7


# ============================================================
# UTILIDADES
# ============================================================

def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo requerido: {path}")


def normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def clasificar_tendencia(x):
    if pd.isna(x):
        return "Sin dato"

    if x > 0.10:
        return "Ascendente"

    if x < -0.10:
        return "Descendente"

    return "Estable"


def calidad_kge(kge):
    if pd.isna(kge):
        return "Sin evaluar"

    if kge >= 0.75:
        return "Buena"

    if kge >= 0.50:
        return "Moderada"

    return "Revisar"


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 90)
    print("EXPORTANDO PRONÓSTICO DE NIVEL A 7 DÍAS - DWLT")
    print("=" * 90)

    require_file(FORE_DWLT)

    print(f"Leyendo pronóstico DWLT: {FORE_DWLT}")
    df = pd.read_parquet(FORE_DWLT)
    df = normalizar_columnas(df)

    if "fecha" not in df.columns:
        raise ValueError("El archivo fore_nivel_transformado.parquet no tiene columna 'fecha'.")

    if "comid" not in df.columns:
        raise ValueError("El archivo fore_nivel_transformado.parquet no tiene columna 'comid'.")

    if "estacion" not in df.columns:
        df["estacion"] = "SIN_NOMBRE"

    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df["comid"] = pd.to_numeric(df["comid"], errors="coerce").astype("Int64")

    df = df.dropna(subset=["fecha", "comid"]).copy()

    # Exportar solo los primeros 7 días disponibles del pronóstico.
    fechas_disponibles = sorted(df["fecha"].dropna().unique())

    if not fechas_disponibles:
        raise ValueError("No hay fechas válidas en el pronóstico DWLT.")

    fechas_exportar = fechas_disponibles[:DIAS_EXPORTAR]

    df = df[df["fecha"].isin(fechas_exportar)].copy()

    print(f"Fechas disponibles en forecast: {len(fechas_disponibles)}")
    print(f"Fechas exportadas: {len(fechas_exportar)}")
    print(f"Inicio: {pd.to_datetime(fechas_exportar[0]).date()}")
    print(f"Fin: {pd.to_datetime(fechas_exportar[-1]).date()}")

    df = df.sort_values(["estacion", "comid", "fecha"]).copy()

    # Columnas principales del pronóstico de nivel.
    cols_base = [
        "estacion",
        "comid",
        "fecha",
        "nivel_eta_eqm_m",
        "nivel_eta_scal_m",
        "nivel_gfs_m",
        "nivel_wrf_m",
        "nivel_min_m",
        "nivel_p25_m",
        "nivel_prom_m",
        "nivel_p75_m",
        "nivel_max_m",
    ]

    cols = [c for c in cols_base if c in df.columns]
    pron = df[cols].copy()

    if "nivel_prom_m" not in pron.columns:
        raise ValueError("No existe la columna 'nivel_prom_m' en el pronóstico DWLT.")

    # Redondear niveles.
    for col in pron.columns:
        if col.startswith("nivel_"):
            pron[col] = pd.to_numeric(pron[col], errors="coerce").round(3)

    pron["fecha_texto"] = pron["fecha"].dt.strftime("%d/%m/%Y")

    cols_final = [
        "estacion",
        "comid",
        "fecha",
        "fecha_texto",
        "nivel_eta_eqm_m",
        "nivel_eta_scal_m",
        "nivel_gfs_m",
        "nivel_wrf_m",
        "nivel_min_m",
        "nivel_p25_m",
        "nivel_prom_m",
        "nivel_p75_m",
        "nivel_max_m",
    ]

    cols_final = [c for c in cols_final if c in pron.columns]
    pron = pron[cols_final].copy()

    # Resumen por estación.
    resumen = (
        pron.groupby(["estacion", "comid"], dropna=False)
        .agg(
            fecha_inicio=("fecha", "min"),
            fecha_fin=("fecha", "max"),
            nivel_min_7dias=("nivel_min_m", "min"),
            nivel_prom_7dias=("nivel_prom_m", "mean"),
            nivel_max_7dias=("nivel_max_m", "max"),
            nivel_inicio=("nivel_prom_m", "first"),
            nivel_fin=("nivel_prom_m", "last"),
        )
        .reset_index()
    )

    resumen["fecha_inicio_texto"] = resumen["fecha_inicio"].dt.strftime("%d/%m/%Y")
    resumen["fecha_fin_texto"] = resumen["fecha_fin"].dt.strftime("%d/%m/%Y")

    resumen["tendencia_7dias_m"] = resumen["nivel_fin"] - resumen["nivel_inicio"]
    resumen["tendencia"] = resumen["tendencia_7dias_m"].apply(clasificar_tendencia)

    for col in [
        "nivel_min_7dias",
        "nivel_prom_7dias",
        "nivel_max_7dias",
        "nivel_inicio",
        "nivel_fin",
        "tendencia_7dias_m",
    ]:
        resumen[col] = pd.to_numeric(resumen[col], errors="coerce").round(3)

    # Agregar métricas DWLT si existen.
    if METRICAS.exists():
        print(f"Leyendo métricas DWLT: {METRICAS}")

        met = pd.read_excel(METRICAS, sheet_name="metricas_dwlt")
        met = normalizar_columnas(met)

        if "comid" in met.columns:
            met["comid"] = pd.to_numeric(met["comid"], errors="coerce").astype("Int64")

            met_cols = [
                "comid",
                "n_validacion",
                "r_pearson",
                "nse",
                "kge_2009",
                "rmse_m",
            ]

            met_cols = [c for c in met_cols if c in met.columns]

            resumen = resumen.merge(
                met[met_cols].drop_duplicates(subset=["comid"], keep="first"),
                on="comid",
                how="left",
            )

            for col in ["r_pearson", "nse", "kge_2009", "rmse_m"]:
                if col in resumen.columns:
                    resumen[col] = pd.to_numeric(resumen[col], errors="coerce").round(3)

            if "kge_2009" in resumen.columns:
                resumen["calidad_dwlt"] = resumen["kge_2009"].apply(calidad_kge)
            else:
                resumen["calidad_dwlt"] = "Sin evaluar"

        else:
            print("ADVERTENCIA: metricas_dwlt_estaciones.xlsx no tiene columna comid.")
            resumen["calidad_dwlt"] = "Sin evaluar"

    else:
        print("ADVERTENCIA: no existe metricas_dwlt_estaciones.xlsx. Se exportará sin métricas.")
        resumen["calidad_dwlt"] = "Sin evaluar"

    # Tabla pivote: estaciones x fechas con nivel promedio.
    pivot_prom = (
        pron.pivot_table(
            index=["estacion", "comid"],
            columns="fecha_texto",
            values="nivel_prom_m",
            aggfunc="mean",
        )
        .reset_index()
    )

    # Orden de columnas del resumen.
    cols_resumen = [
        "estacion",
        "comid",
        "fecha_inicio_texto",
        "fecha_fin_texto",
        "nivel_min_7dias",
        "nivel_prom_7dias",
        "nivel_max_7dias",
        "nivel_inicio",
        "nivel_fin",
        "tendencia_7dias_m",
        "tendencia",
        "n_validacion",
        "r_pearson",
        "nse",
        "kge_2009",
        "rmse_m",
        "calidad_dwlt",
    ]

    cols_resumen = [c for c in cols_resumen if c in resumen.columns]
    resumen = resumen[cols_resumen].copy()

    # Guardar CSV y Excel.
    pron.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        resumen.to_excel(writer, sheet_name="resumen_estaciones", index=False)
        pron.to_excel(writer, sheet_name="pronostico_7dias", index=False)
        pivot_prom.to_excel(writer, sheet_name="pivot_nivel_prom", index=False)

    print("\nArchivos generados:")
    print(f" - {OUT_XLSX}")
    print(f" - {OUT_CSV}")

    print("\nResumen por estación:")
    cols_print = [
        "estacion",
        "comid",
        "nivel_prom_7dias",
        "nivel_inicio",
        "nivel_fin",
        "tendencia_7dias_m",
        "tendencia",
        "kge_2009",
        "calidad_dwlt",
    ]

    cols_print = [c for c in cols_print if c in resumen.columns]

    print(resumen[cols_print].to_string(index=False))

    print("\nProceso terminado correctamente.")


if __name__ == "__main__":
    main()
