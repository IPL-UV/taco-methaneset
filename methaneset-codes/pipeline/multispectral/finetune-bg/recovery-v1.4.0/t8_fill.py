"""T8 · rellenar las columnas de las muestras nuevas en el level0 (v1.4.0).

El repaquetado deja las filas nuevas heredando valores de la primera fila publicada, que es
peor que dejarlas vacias: todas compartirian geometria. Esto escribe cada columna con su dato
real y deja en nulo solo lo que de verdad no tenemos.

Fuentes:
  - Cesar (level0): sensor, tile, angulos, sector, viento, pais, split, crs, tiempo
  - validated_images_all.csv: la geometria de la pluma (WKT) por id_loc_image
  - seleccion_<col>_<sensor>.csv: fecha, diferencia, nube y metano de las 3 referencias

Uso:
  python v140/t8_fill.py --coleccion l89
"""
import argparse
import pathlib

import numpy as np
import pandas as pd

PUB = pathlib.Path("/data/databases/METHANESET_TACOS")
NEW = PUB / "_v140/T7"
CE = pathlib.Path("/data/databases/METHANE_DATASETS_TACO")
RAW = PUB / "_v140/T2/raw"
CSV_PLUMAS = pathlib.Path("/data/users/julio/IMEO_MARS/assets/table/validated_images_all.csv")


def _fecha(texto):
    """Nombre de escena -> fecha, o None."""
    import re
    if isinstance(texto, (int, np.integer)):
        return pd.to_datetime(str(int(texto)), format="%Y%m%d")
    if not isinstance(texto, str):
        return None
    m = re.search(r"(20\d{6})", texto)
    if m:
        return pd.to_datetime(m.group(1), format="%Y%m%d")
    m = re.search(r"(20\d{2})-(\d{2})-(\d{2})", texto)
    return pd.Timestamp(m.group(0)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coleccion", required=True, choices=["l89", "s2"])
    args = ap.parse_args()

    destino = NEW / args.coleccion / ".tacocat" / "level0.parquet"
    publicado = PUB / f"methaneset-{args.coleccion}-finetune" / ".tacocat" / "level0.parquet"
    l0_pub = pd.read_parquet(publicado)
    l0 = pd.read_parquet(destino)
    nuevas = l0.iloc[len(l0_pub):]["id"].tolist()
    print(f"{args.coleccion}: {len(nuevas)} filas a rellenar")

    ce = pd.read_parquet(CE / args.coleccion / ".tacocat" / "level0.parquet").set_index("id")
    mars = pd.read_csv(CSV_PLUMAS, usecols=["id_loc_image", "plume", "lat", "lon", "case_study"],
                       low_memory=False).set_index("id_loc_image")
    plumas = mars["plume"]

    sel = pd.concat([pd.read_csv(f) for f in RAW.glob(f"seleccion_{args.coleccion}_*.csv")],
                    ignore_index=True)

    filas = []
    for sid in nuevas:
        c = ce.loc[sid]
        fila = {col: None for col in l0_pub.columns}
        fila["id"] = sid
        fila["type"] = "FOLDER"
        fila["plume:geometry"] = plumas.get(sid)
        fila["emission:lat"] = mars["lat"].get(sid)
        fila["emission:lon"] = mars["lon"].get(sid)
        fila["detection:case_study"] = mars["case_study"].get(sid)
        fila["majortom:code"] = c.get("majortom:code")
        fila["geoenrich:elevation"] = c.get("geoenrich:elevation")
        fila["target:sensor"] = c.get("satellite:satellite")
        fila["target:tile"] = c.get("satellite:tile")
        fila["target:sza"] = c.get("satellite:sza")
        fila["target:vza"] = c.get("satellite:vza")
        fila["stac:time_start"] = c.get("stac:time_start")
        fila["stac:crs"] = c.get("stac:crs")
        fila["stac:geotransform"] = c.get("stac:geotransform")
        fila["stac:tensor_shape"] = c.get("stac:tensor_shape")
        for col, src in (("detection:ch4_fluxrate", "methane:ch4_fluxrate"),
                         ("detection:ch4_fluxrate_std", "methane:ch4_fluxrate_std"),
                         ("detection:sector", "methane:sector"),
                         ("detection:offshore", "methane:offshore"),
                         ("detection:wind_source", "methane:wind_source"),
                         ("meteo:wind_u", "methane:wind_u"),
                         ("meteo:wind_v", "methane:wind_v"),
                         ("site:country", "site:country"),
                         ("site:location_name", "site:location_name"),
                         ("split", "split"),
                         ("quality:last_update", "quality:last_update"),
                         ("quality:notified", "quality:notified"),
                         ("quality:observability", "quality:observability"),
                         ("quality:percentage_clear", "quality:percentage_clear")):
            fila[col] = c.get(src)

        # el fondo de MARS: bg0
        fondo = c.get("satellite:background_image_tile")
        fila["bg:tile0"] = fondo
        fila["bg:date0"] = _fecha(fondo)
        fila["bg:sensor0"] = str(fondo)[:4] if isinstance(fondo, str) else None
        if c.get("quality:percentage_clear") is not None:
            fila["target:cloud"] = 1 - float(c["quality:percentage_clear"]) / 100

        # las tres referencias nuevas
        g = sel[sel.taco_id == sid].sort_values("rank")
        for rank, (_, r) in enumerate(g.head(3).iterrows(), 1):
            fila[f"bg:date{rank}"] = _fecha(r["date"])
            fila[f"bg:difference{rank}"] = float(r["difference"])
            fila[f"bg:cloud{rank}"] = float(r["cloud"])
            fila[f"bg:methane{rank}"] = float(r["methane"])
            nombre = str(r["file"]).replace(".tif", "")
            cola = nombre.split("__")[-1]          # LC08_20230125 o S2_20230125
            fila[f"bg:sensor{rank}"] = cola.split("_")[0]

        fila["internal:source_file"] = f"methaneset-{args.coleccion}-finetune_" \
            f"{str(c.get('geoenrich:admin_countries') or 'Unknown').split(';')[0].strip().replace(' ', '_')}.tacozip"
        filas.append(fila)

    nuevo = pd.DataFrame(filas)
    # conservar los internal:* que ya calculó el repaquetado
    viejas = l0.iloc[len(l0_pub):].set_index("id")
    for col in [c for c in l0.columns if c.startswith("internal:")]:
        nuevo[col] = [viejas.loc[s, col] for s in nuevo["id"]]

    final = pd.concat([l0_pub, nuevo[l0_pub.columns]], ignore_index=True)
    for col in l0_pub.columns:
        try:
            if str(l0_pub[col].dtype).startswith("datetime"):
                final[col] = pd.to_datetime(final[col], utc=True).astype(l0_pub[col].dtype)
            else:
                final[col] = final[col].astype(l0_pub[col].dtype)
        except Exception:
            pass
    final.to_parquet(destino, index=False)
    print(f"filas: {len(final)} | con geometria: {final['plume:geometry'].notna().sum()} "
          f"| plumas unicas: {final['plume:geometry'].nunique()}")


if __name__ == "__main__":
    main()
