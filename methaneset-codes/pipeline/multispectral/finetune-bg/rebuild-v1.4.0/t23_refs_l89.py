"""Sustituye las dos referencias L89 que solo cubren el 2,5 % del chip.

La columna `bg:date` apuntaba a una escena del día correcto pero de otra órbita, así
que el chip salía casi vacío. Aquí se busca, para cada ranura, la escena de Landsat que
de verdad cubre el chip en una ventana de ±120 días, con poca nube según el metadato
CLOUD_COVER, se descarga el chip, se recalcula la nube con CloudSEN12 y se añade la fila
a cambios.csv (y a cloud_l89.csv) para que el rebuild la tome como un reemplazo más.

Uso:
  python t23_refs_l89.py --pares 468c9372:1,a025cd0f:3 --out /tmp/opencode/cambios.csv
"""

import argparse
import importlib.util
import pathlib

import ee
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import rasterio
import torch
from cloudsen12_models import cloudsen12
import cubexpress

AQUI = pathlib.Path(__file__).parent
TACO = pathlib.Path("/data/databases/METHANESET_TACOS/_v140/T16/l89")
T11 = pathlib.Path("/data/databases/METHANESET_TACOS/_v140/T11/cloud_l89.csv")
CHIPS = pathlib.Path("/data/databases/METHANESET_TACOS/_v140/T14")
BANDAS = ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11")
IDX_MODELO = [0, 1, 2, 3, 4, 8, 5, 6]


def cargar_t14():
    spec = importlib.util.spec_from_file_location("t14", AQUI / "t14_chips.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def cobertura(ruta: pathlib.Path) -> float:
    with rasterio.open(ruta) as s:
        a = s.read()
    return float(np.isfinite(a).mean())


def nube_l89(ruta: pathlib.Path, modelo) -> float:
    with rasterio.open(ruta) as s:
        x = s.read()
    k = 3
    n = (x.shape[1] // k) * k
    arr = x[:, :n, :n].reshape(x.shape[0], n // k, k, n // k, k).mean(axis=(2, 4)).astype(np.float32)
    pred = np.asarray(modelo.predict(arr[IDX_MODELO])).squeeze()
    return float(np.isin(pred, (1, 2, 3)).mean()) if pred.size else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pares", required=True, help="prefijo_id:slot,prefijo:slot")
    ap.add_argument("--out", default="/tmp/opencode/cambios.csv")
    ap.add_argument("--dias", type=int, default=120)
    ap.add_argument("--cloud-max", type=float, default=10.0)
    a = ap.parse_args()

    t14 = cargar_t14()
    ee.Initialize(project="ee-contrerasnetk")
    modelo = cloudsen12.load_model_by_name(
        "landsat30", device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    l0 = pq.read_table(TACO / ".tacocat" / "level0.parquet").to_pandas().set_index("id")
    cam = pd.read_csv(a.out) if pathlib.Path(a.out).exists() else pd.DataFrame()
    nube = pd.read_csv(T11).set_index("id")

    for par in a.pares.split(","):
        pref, slot = par.split(":")
        slot = int(slot)
        sid = [x for x in l0.index if str(x).startswith(pref)][0]
        fila = l0.loc[sid]
        geom = t14.geometria(fila, 30)
        centro = geom.centroid(1)
        t0 = pd.Timestamp(str(fila["stac:time_start"])[:10])
        d0, d1 = (t0 - pd.Timedelta(days=a.dias)).strftime("%Y-%m-%d"), \
                 (t0 + pd.Timedelta(days=a.dias)).strftime("%Y-%m-%d")

        cands = []
        for sensor in ("LC08", "LC09"):
            col = (ee.ImageCollection(f"LANDSAT/{sensor}/C02/T1_TOA")
                   .filterDate(d0, d1).filter(ee.Filter.bounds(centro)))
            for img in col.toList(200).getInfo():
                p = img["properties"]
                c = p.get("CLOUD_COVER")
                if c is not None and c <= a.cloud_max:
                    gid = img["id"].split("/")[-1]
                    dia = gid.split("_")[2] if len(gid.split("_")) == 3 else gid.split("_")[3]
                    fecha = f"{dia[:4]}-{dia[4:6]}-{dia[6:8]}"
                    cands.append((sensor, gid, fecha, float(c)))
        print(f"{pref}: {len(cands)} escenas candidatas con nube ≤ {a.cloud_max}", flush=True)
        if not cands:
            print("  sin candidatas", flush=True)
            continue
        cands.sort(key=lambda x: (x[3], abs((pd.Timestamp(x[2]) - t0).days)))
        elegido = None
        for sensor, gid, fecha, c in cands[:8]:
            out = CHIPS / f"{sid}__bg{slot}.tif"
            img = ee.Image(f"LANDSAT/{sensor}/C02/T1_TOA/{gid}")
            req = cubexpress.RequestRow(id=f"{sid[:8]}_bg{slot}", raster_transform=t14.rt_de(fila),
                                        image=img, bands=list(BANDAS))
            cubexpress.express(cubexpress.RequestTable((req,)), str(CHIPS), nworkers=1, overwrite=True)
            hecho = list(CHIPS.glob(f"{sid[:8]}_bg{slot}*"))
            if not hecho:
                continue
            hecho[0].replace(out)
            cob = cobertura(out)
            print(f"  {gid} ({fecha}, nube meta {c}) -> cobertura {cob:.3f}", flush=True)
            if cob > 0.97:
                elegido = (sensor, gid, fecha, out, cob)
                break
        if elegido is None:
            print("  ninguna cubre el chip", flush=True)
            continue

        sensor, gid, fecha, out, cob = elegido
        cl = nube_l89(out, modelo)
        print(f"  nube CloudSEN12: {cl:.3f} | chip: {out.name}", flush=True)

        if len(cam) and {"fam", "id", "slot"}.issubset(cam.columns):
            cam = cam[~((cam["fam"] == "l89") & (cam["id"] == sid) & (cam["slot"] == slot))]
        fila_cam = {"fam": "l89", "id": sid, "slot": slot, "ruta": str(out),
                    "sensor": gid[:4], "date": fecha, "cloud": round(cl, 6),
                    "difference": float("nan"), "granule": gid}
        cam = pd.concat([cam, pd.DataFrame([fila_cam])], ignore_index=True)
        nube.loc[sid, f"bg{slot}:cloud"] = cl
        print(f"  añadido a cambios: {sid[:8]} bg{slot} {gid[:4]} {fecha}", flush=True)

    cam.to_csv(a.out, index=False, float_format="%.6f")
    nube.reset_index().to_csv(T11, index=False, float_format="%.6f")
    print("escrito", a.out, "y", T11)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
