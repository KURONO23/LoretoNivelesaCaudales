from pathlib import Path
import geopandas as gpd
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent

GPKG_PATH = BASE_DIR / "Data" / "estaciones_hidrometricas_loreto_latlong_COMID_actualizado.gpkg"


def main():
    print("=" * 80)
    print("REVISIÓN DEL GPKG DE ESTACIONES")
    print("=" * 80)

    if not GPKG_PATH.exists():
        raise FileNotFoundError(f"No existe el archivo GPKG: {GPKG_PATH}")

    print(f"GPKG encontrado:")
    print(GPKG_PATH)

    gdf = gpd.read_file(GPKG_PATH)

    print("\nColumnas encontradas:")
    for col in gdf.columns:
        print(f" - {col}")

    print("\nSistema de coordenadas:")
    print(gdf.crs)

    # Normalizar nombres de columnas para buscar COMID
    columnas_originales = list(gdf.columns)
    columnas_lower = {c.lower(): c for c in columnas_originales}

    if "comid" not in columnas_lower:
        raise ValueError("No se encontró la columna COMID en el GPKG.")

    col_comid = columnas_lower["comid"]

    gdf[col_comid] = pd.to_numeric(gdf[col_comid], errors="coerce")

    total_estaciones = len(gdf)
    estaciones_con_comid = gdf[col_comid].notna().sum()
    estaciones_sin_comid = total_estaciones - estaciones_con_comid

    print("\nResumen:")
    print(f"Total de estaciones: {total_estaciones}")
    print(f"Estaciones con COMID: {estaciones_con_comid}")
    print(f"Estaciones sin COMID: {estaciones_sin_comid}")

    print("\nPrimeras estaciones:")
    columnas_mostrar = []

    for posible in ["estacion", "Estacion", "estaciones", "Estaciones", "nombre", "Nombre"]:
        if posible in gdf.columns:
            columnas_mostrar.append(posible)
            break

    for posible in ["cuenca", "Cuenca"]:
        if posible in gdf.columns:
            columnas_mostrar.append(posible)
            break

    columnas_mostrar.append(col_comid)

    if "geometry" in gdf.columns:
        gdf["longitud"] = gdf.geometry.x
        gdf["latitud"] = gdf.geometry.y
        columnas_mostrar.extend(["latitud", "longitud"])

    print(gdf[columnas_mostrar].head(30).to_string(index=False))

    print("\nProceso terminado correctamente.")


if __name__ == "__main__":
    main()