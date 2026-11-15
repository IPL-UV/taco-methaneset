"""Re-descarga las hojas que salieron vacías o casi vacías (D3).

Solo S2 target/bg0 y las dos referencias L89 con 2,5 % de píxeles válidos. El gránulo se
resuelve por PRODUCT_ID (target, bg0) o por tile+fecha+sensor (referencias), que es el
formato que sí entiende Earth Engine: `<datatake>_<fin>_<tile>`.

Uso:
  python t14_chips.py --taco .../T7/s2 --fam s2 --stage /data/databases/METHANESET_TACOS/_v140/T14
"""

import argparse
import pathlib

import ee
import numpy as np
import pandas as pd
import rasterio
import pyarrow.parquet as pq

import cubexpress
from cubexpress.geo.transform import RasterTransform

BANDAS = {
    "s2": ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"),
    "l89": ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11"),
}
COLECCION = {"s2": "COPERNICUS/S2_HARMONIZED",
             "l89": None}  # Landsat se decide por el prefijo del gránulo


def rt_de(fila) -> RasterTransform:
    gt = np.fromstring(str(fila["stac:geotransform"]).strip("[]"), sep=" ")
    shp = np.fromstring(str(fila["stac:tensor_shape"]).strip("[]"), sep=" ").astype(int)
    return RasterTransform(
        crs=str(fila["stac:crs"]),
        translate_x=float(gt[0]), translate_y=float(gt[3]),
        scale_x=float(gt[1]), scale_y=float(gt[5]),
        width=int(shp[2]), height=int(shp[1]),
        shear_x=float(gt[2]), shear_y=float(gt[4]),
    )


def imagen_eegranulo(gran: str, fam: str):
    if fam == "s2":
        return ee.Image(f"COPERNICUS/S2_HARMONIZED/{gran}")
    sen = gran[:4]
    return ee.Image(f"LANDSAT/{sen}/C02/T1_TOA/{gran}")


def granulo_gee(gran: str, fam: str) -> str:
    """PRODUCT_ID -> id de GEE."""
    if fam == "s2":
        # S2B_MSIL1C_20220618T070629_N0400_R106_T40SBJ_20220618T074913 -> ..._T40SBJ
        parte = "_".join(gran.split("_")[2:])
        return parte
    return gran  # Landsat usa el mismo id en GEE


def candidatos_gee(gran: str, fam: str):
    """Ids de GEE que pueden corresponder al gránulo pedido.

    S2: el PRODUCT_ID de GEE ya no coincide (todo se reprocesó a baseline N0500), así que
    se busca por fecha + tile + hora de toma. Un mismo datatake puede tener varias entradas
    en la colección harmónica y solo una cubre el chip, de ahí que se devuelva una lista.
    Landsat: el LANDSAT_PRODUCT_ID sí coincide.
    """
    if fam != "s2":
        col = ee.ImageCollection(f"LANDSAT/{gran[:4]}/C02/T1_TOA").filter(
            ee.Filter.eq("LANDSAT_PRODUCT_ID", gran))
        return [i["id"].split("/")[-1] for i in col.toList(5).getInfo()]

    partes = gran.split("_")
    if len(partes) < 6:
        return []
    toma, tile = partes[2], partes[5]
    dia = toma[:8]
    fecha = f"{dia[:4]}-{dia[4:6]}-{dia[6:]}"
    col = (ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
           .filterDate(fecha, str(pd.Timestamp(fecha) + pd.Timedelta(days=1))[:10])
           .filter(ee.Filter.eq("MGRS_TILE", tile[1:])))
    ids = [i["id"].split("/")[-1] for i in col.toList(50).getInfo()]
    mismo = [i for i in ids if i.startswith(toma) and i.endswith(f"_{tile}")]
    return mismo or [i for i in ids if i.endswith(f"_{tile}")]


def buscar_gee(gran: str, fam: str, geom=None, banda: str | None = None):
    """Primer id que cubre el chip (los de un mismo datatake pueden estar partidos)."""
    cands = candidatos_gee(gran, fam)
    if not cands:
        return None
    if geom is None or not banda:
        return cands[0]
    mejor, mejor_n = None, 0
    for sid in cands:
        try:
            img = (ee.Image(f"COPERNICUS/S2_HARMONIZED/{sid}") if fam == "s2"
                   else ee.Image(f"LANDSAT/{sid[:4]}/C02/T1_TOA/{sid}"))
            n = img.select(banda).reduceRegion(ee.Reducer.count(), geom, 20).getInfo().get(banda) or 0
        except Exception:  # noqa: BLE001
            n = 0
        if n > mejor_n:
            mejor, mejor_n = sid, n
    return mejor


def geometria(fila, margen: int = 0):
    """Polígono del chip en su CRS, para comprobar cobertura."""
    gt = np.fromstring(str(fila["stac:geotransform"]).strip("[]"), sep=" ")
    shp = np.fromstring(str(fila["stac:tensor_shape"]).strip("[]"), sep=" ").astype(int)
    x0, y0, dx, dy = float(gt[0]), float(gt[3]), float(gt[1]), float(gt[5])
    ancho, alto = int(shp[2]), int(shp[1])
    x1, y1 = x0 + dx * ancho, y0 + dy * alto
    return ee.Geometry.Rectangle([min(x0, x1) - margen, min(y0, y1) - margen,
                                  max(x0, x1) + margen, max(y0, y1) + margen],
                                 proj=str(fila["stac:crs"]), geodesic=False)


def vacia(ruta: pathlib.Path) -> bool:
    with rasterio.open(ruta) as s:
        a = s.read()
    if a.size == 0:
        return True
    return bool(np.all(a == 0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--taco", required=True)
    ap.add_argument("--fam", required=True, choices=["l89", "s2"])
    ap.add_argument("--stage", required=True)
    ap.add_argument("--franja", default=None, help="i/N: solo procesa esa franja de la lista")
    a = ap.parse_args()
    ee.Initialize(project="ee-contrerasnetk")

    taco = pathlib.Path(a.taco)
    stage = pathlib.Path(a.stage)
    stage.mkdir(parents=True, exist_ok=True)
    l0 = pq.read_table(taco / ".tacocat" / "level0.parquet").to_pandas().set_index("id")

    # Hojas vacías: target y bg0 por stats; las referencias por tamaño (<30 KB).
    l1 = pq.read_table(taco / ".tacocat" / "level1.parquet").to_pandas()
    l1["hoja"] = l1["internal:relative_path"].str.split("/").str[-1]
    l1["muestra"] = l1["internal:relative_path"].str.split("/").str[0]

    def es_vacia(st):
        try:
            v = np.asarray(st[0]).ravel()
            return len(v) >= 5 and v[0] == 0 and v[1] == 0 and v[4] >= 99
        except Exception:  # noqa: BLE001
            return False

    malas = l1[l1["geotiff:stats"].map(es_vacia) | ((l1["internal:size"] < 0.03e6)
                                                    & l1["hoja"].isin(["target", "bg0", "bg1", "bg2", "bg3"]))]
    malas = malas[~malas.duplicated(subset=["muestra", "hoja"])]
    print(f"hojas a re-descargar: {len(malas)}", flush=True)

    if a.franja:
        i, total = (int(x) for x in a.franja.split("/"))
        malas = malas.iloc[i::total]
    fallos = []
    for n, (_, r) in enumerate(malas.iterrows(), 1):
        fila = l0.loc[r["muestra"]]
        hoja = r["hoja"]
        if (stage / f"{r['muestra']}__{hoja}.tif").exists() and not vacia(stage / f"{r['muestra']}__{hoja}.tif"):
            print(f"  [{n}/{len(malas)}] {r['muestra'][:8]} {hoja}: ya estaba", flush=True)
            continue
        if hoja == "target":
            gran = str(fila["target:tile"])
        elif hoja == "bg0":
            gran = str(fila["bg:tile0"])
        else:
            # Referencia nuestra: no hay tile en level0; se busca por tile del target,
            # fecha y sensor de la columna.
            k = hoja[-1]
            fecha = str(fila[f"bg:date{k}"])
            sensor = str(fila[f"bg:sensor{k}"])
            mgrs = str(fila["target:tile"]).split("_")[-1].replace(".tif", "")[:6]
            col = (ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
                   .filterDate(fecha, str(pd.Timestamp(fecha) + pd.Timedelta(days=1))[:10])
                   .filter(ee.Filter.eq("MGRS_TILE", mgrs[1:] if mgrs.startswith("T") else mgrs)))
            if sensor.startswith("S2") and len(sensor) == 3:
                col = col.filter(ee.Filter.eq("SPACECRAFT_NAME",
                                              {"S2A": "Sentinel-2A", "S2B": "Sentinel-2B",
                                               "S2C": "Sentinel-2C"}[sensor]))
            if sensor.startswith("LC"):
                col = (ee.ImageCollection(f"LANDSAT/{sensor}/C02/T1_TOA")
                       .filterDate(fecha, str(pd.Timestamp(fecha) + pd.Timedelta(days=1))[:10]))
            n_img = col.size().getInfo()
            if not n_img:
                print(f"  [{n}/{len(malas)}] {r['muestra'][:8]} {hoja}: sin imagen ese día", flush=True)
                fallos.append((r["muestra"], hoja, "sin imagen"))
                continue
            gran = col.first().get("PRODUCT_ID").getInfo() if a.fam == "s2" else \
                col.first().get("LANDSAT_PRODUCT_ID").getInfo()

        try:
            geom = geometria(fila, 30)
            banda = "B5" if a.fam == "l89" else "B4"
            sid = buscar_gee(gran, a.fam, geom, banda)
            if sid is None:
                print(f"  [{n}/{len(malas)}] {r['muestra'][:8]} {hoja}: no existe {gran[:40]}", flush=True)
                fallos.append((r["muestra"], hoja, "no existe"))
                continue
            img = (ee.Image(f"COPERNICUS/S2_HARMONIZED/{sid}") if a.fam == "s2"
                   else ee.Image(f"LANDSAT/{gran[:4]}/C02/T1_TOA/{sid}"))
            out = stage / f"{r['muestra']}__{hoja}.tif"
            req = cubexpress.RequestRow(
                id=f"{r['muestra'][:8]}_{hoja}", raster_transform=rt_de(fila),
                image=img, bands=BANDAS[a.fam])
            cubexpress.express(cubexpress.RequestTable((req,)), str(stage), nworkers=1, overwrite=True)
            hecho = list(stage.glob(f"{r['muestra'][:8]}_{hoja}*"))
            if not hecho:
                print(f"  [{n}/{len(malas)}] {r['muestra'][:8]} {hoja}: express no devolvió fichero", flush=True)
                fallos.append((r["muestra"], hoja, "express vacío"))
                continue
            hecho[0].replace(out)
            if vacia(out):
                print(f"  [{n}/{len(malas)}] {r['muestra'][:8]} {hoja}: sigue vacía", flush=True)
                fallos.append((r["muestra"], hoja, "vacía tras descarga"))
                continue
            with rasterio.open(out) as s:
                print(f"  [{n}/{len(malas)}] {r['muestra'][:8]} {hoja}: ok {s.count} bandas", flush=True)
        except Exception as ex:  # noqa: BLE001
            print(f"  [{n}/{len(malas)}] {r['muestra'][:8]} {hoja}: {type(ex).__name__} {ex}", flush=True)
            fallos.append((r["muestra"], hoja, str(ex)[:60]))

    print(f"fallos: {len(fallos)}")
    for f in fallos:
        print("  ", f[0][:8], f[1], f[2])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
