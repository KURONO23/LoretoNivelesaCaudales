from pathlib import Path
import json
import pandas as pd

BASE_DIR = Path(r"C:\Users\mgutierrez\Documents\POI-MAX\POI 2026\NivelaCaudal")

ARCHIVOS = {
    "CACHE hist_filtrado": BASE_DIR / "backend" / "cache" / "hist_filtrado.parquet",
    "CACHE fore_filtrado": BASE_DIR / "backend" / "cache" / "fore_filtrado.parquet",
    "OUTPUT hist_nivel_transformado": BASE_DIR / "outputs" / "hist_nivel_transformado.parquet",
    "OUTPUT fore_nivel_transformado": BASE_DIR / "outputs" / "fore_nivel_transformado.parquet",
    "OUTPUT observado_estaciones": BASE_DIR / "outputs" / "observado_estaciones.parquet",
    "CACHE observado_estaciones": BASE_DIR / "backend" / "cache" / "observado_estaciones.parquet",
}

META = BASE_DIR / "backend" / "cache" / "meta_filtrado.json"

print("=" * 100)
print("REVISIÓN DE FECHAS SONICS / DWLT / OBSERVADO")
print("=" * 100)

for nombre, path in ARCHIVOS.items():
    print("\n" + "-" * 100)
    print(nombre)
    print(path)

    if not path.exists():
        print("NO EXISTE")
        continue

    try:
        df = pd.read_parquet(path)
        df.columns = [str(c).strip().lower() for c in df.columns]

        print(f"Filas: {len(df):,}")
        print(f"Columnas: {list(df.columns)}")

        if "fecha" in df.columns:
            df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
            print(f"Fecha mínima: {df['fecha'].min()}")
            print(f"Fecha máxima: {df['fecha'].max()}")

        if "comid" in df.columns:
            print(f"COMID únicos: {df['comid'].nunique()}")

        if "estacion" in df.columns:
            print(f"Estaciones únicas: {df['estacion'].nunique()}")

        if "estacion_nombre" in df.columns:
            print(f"Estaciones únicas: {df['estacion_nombre'].nunique()}")

    except Exception as e:
        print(f"ERROR LEYENDO ARCHIVO: {e}")

print("\n" + "=" * 100)
print("META FILTRADO")
print("=" * 100)

if META.exists():
    with open(META, "r", encoding="utf-8") as f:
        meta = json.load(f)

    print(json.dumps(meta, indent=2, ensure_ascii=False))
else:
    print("No existe meta_filtrado.json")
