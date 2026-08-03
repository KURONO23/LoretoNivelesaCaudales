from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(r"C:\Users\mgutierrez\Documents\POI-MAX\POI 2026\NivelaCaudal")
OUTPUT_DIR = BASE_DIR / "outputs"

FORE_DWLT = OUTPUT_DIR / "fore_nivel_transformado.parquet"
METRICAS = OUTPUT_DIR / "metricas_dwlt_estaciones.xlsx"

OUT_XLSX = OUTPUT_DIR / "pronostico_nivel_7dias_estaciones.xlsx"
OUT_CSV = OUTPUT_DIR / "pronostico_nivel_7dias_estaciones.csv"

print("=" * 90)
print("EXPORTANDO PRONÓSTICO DE NIVEL A 7 DÍAS - DWLT")
print("=" * 90)

if not FORE_DWLT.exists():
    raise FileNotFoundError(f"No existe: {FORE_DWLT}")

df = pd.read_parquet(FORE_DWLT)
df.columns = [str(c).strip().lower() for c in df.columns]

df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
df["comid"] = pd.to_numeric(df["comid"], errors="coerce").astype("Int64")

# Ordenar
df = df.sort_values(["estacion", "comid", "fecha"]).copy()

# Columnas principales del pronóstico de nivel
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

# Redondear niveles
for col in pron.columns:
    if col.startswith("nivel_"):
        pron[col] = pd.to_numeric(pron[col], errors="coerce").round(3)

# Agregar fecha texto
pron["fecha_texto"] = pron["fecha"].dt.strftime("%d/%m/%Y")

# Reordenar con fecha texto
cols_final = [
    "estacion",
    "comid",
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

# Resumen por estación
resumen = (
    pron.groupby(["estacion", "comid"], dropna=False)
    .agg(
        fecha_inicio=("fecha_texto", "first"),
        fecha_fin=("fecha_texto", "last"),
        nivel_min_7dias=("nivel_min_m", "min"),
        nivel_prom_7dias=("nivel_prom_m", "mean"),
        nivel_max_7dias=("nivel_max_m", "max"),
        nivel_inicio=("nivel_prom_m", "first"),
        nivel_fin=("nivel_prom_m", "last"),
    )
    .reset_index()
)

resumen["tendencia_7dias_m"] = resumen["nivel_fin"] - resumen["nivel_inicio"]

def clasificar_tendencia(x):
    if pd.isna(x):
        return "Sin dato"
    if x > 0.10:
        return "Ascendente"
    if x < -0.10:
        return "Descendente"
    return "Estable"

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

# Agregar métricas DWLT si existen
if METRICAS.exists():
    met = pd.read_excel(METRICAS, sheet_name="metricas_dwlt")
    met.columns = [str(c).strip().lower() for c in met.columns]
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

    resumen = resumen.merge(met[met_cols], on="comid", how="left")

    for col in ["r_pearson", "nse", "kge_2009", "rmse_m"]:
        if col in resumen.columns:
            resumen[col] = pd.to_numeric(resumen[col], errors="coerce").round(3)

    def calidad_kge(kge):
        if pd.isna(kge):
            return "Sin evaluar"
        if kge >= 0.75:
            return "Buena"
        if kge >= 0.50:
            return "Moderada"
        return "Revisar"

    resumen["calidad_dwlt"] = resumen["kge_2009"].apply(calidad_kge)

# Tabla pivote: estaciones x fechas con nivel promedio
pivot_prom = pron.pivot_table(
    index=["estacion", "comid"],
    columns="fecha_texto",
    values="nivel_prom_m",
    aggfunc="mean"
).reset_index()

# Guardar CSV y Excel
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