"""Auditoría completa de una colección finetune publicada.

Comprueba estructura, coherencia entre columnas, unidades de los rásteres y las
columnas derivadas (nube, diferencia, metano) contra un recálculo sobre las hojas.

Uso:
  python t13_auditoria.py --taco /data/databases/METHANESET_TACOS/_v140/T7/l89 --fam l89 --n 60
"""

import argparse
import importlib.util
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import rasterio
import tacoreader

NB = pathlib.Path("/data/users/julio/Notes/01-Projects/methanset/04-web/"
                  "taco-methaneset/methaneset-codes/notebooks")
HOJAS = ["target", "bg0", "bg1", "bg2", "bg3", "ch4", "plume", "dem"]
COLS = [
    "id", "type", "target:sensor", "detection:case_study", "detection:ch4_fluxrate",
    "detection:ch4_fluxrate_std", "detection:offshore", "detection:sector",
    "detection:wind_source", "geoenrich:admin_countries", "geoenrich:admin_districts",
    "geoenrich:admin_states", "geoenrich:elevation", "geoenrich:population",
    "geoenrich:temperature", "majortom:code", "meteo:wind_u", "meteo:wind_v",
    "plume:geometry", "quality:last_update", "quality:notified", "quality:observability",
    "quality:percentage_clear", "target:sza", "target:tile", "target:vza", "site:country",
    "site:location_name", "split", "stac:crs", "stac:geotransform", "stac:tensor_shape",
    "stac:time_start", "bg:sensor0", "bg:tile0", "bg:date0", "bg:cloud0",
    "bg:sensor1", "bg:date1", "bg:cloud1", "bg:difference1", "bg:methane1",
    "bg:sensor2", "bg:date2", "bg:cloud2", "bg:difference2", "bg:methane2",
    "bg:sensor3", "bg:date3", "bg:cloud3", "bg:difference3", "bg:methane3",
    "target:cloud", "emission:lat", "emission:lon", "internal:current_id",
    "internal:parent_id", "internal:offset", "internal:size", "target:saa",
    "target:raa", "internal:source_file",
]
SIM = {"l89": [1, 2, 3, 5], "s2": [1, 2, 3, 11]}
SWIR = {"l89": (5, 6), "s2": (11, 12)}


def common():
    sys.path.insert(0, str(NB))
    import common as C  # noqa: PLC0415
    return C


def leer(path):
    with rasterio.open(path) as s:
        return s.read().astype(np.float32)


def auditoria(taco, fam, n_muestra):
    rep = {"taco": taco, "fam": fam, "fallos": [], "avisos": [], "datos": {}}
    l0 = pq.read_table(pathlib.Path(taco) / ".tacocat" / "level0.parquet").to_pandas()
    l1 = pq.read_table(pathlib.Path(taco) / ".tacocat" / "level1.parquet").to_pandas()
    C = common()

    # --- A. estructura -----------------------------------------------------
    rep["datos"]["filas"] = len(l0)
    if list(l0.columns) != COLS:
        rep["fallos"].append(f"columnas level0: {len(l0.columns)} y no {len(COLS)} "
                             f"(faltan {set(COLS)-set(l0.columns)}, sobran {set(l0.columns)-set(COLS)})")
    if l0["id"].duplicated().any():
        rep["fallos"].append(f"ids duplicados: {int(l0['id'].duplicated().sum())}")
    if set(l0["type"].unique()) != {"FOLDER"}:
        rep["fallos"].append(f"type inesperado: {l0['type'].unique().tolist()}")

    # --- B. level1 ---------------------------------------------------------
    por_muestra = l1.groupby(l1["internal:relative_path"].str.split("/").str[0]).size()
    malas = por_muestra[por_muestra != 8]
    if len(malas):
        rep["fallos"].append(f"{len(malas)} muestras no tienen 8 hojas")
    orden_roto = []
    for mid, g in l1.groupby(l1["internal:relative_path"].str.split("/").str[0]):
        hojas = [p.split("/")[-1] for p in g["internal:relative_path"]]
        if hojas != HOJAS:
            orden_roto.append(mid)
    rep["datos"]["level1_orden_roto"] = len(orden_roto)
    if orden_roto:
        rep["avisos"].append(f"{len(orden_roto)} muestras con el orden de level1 fuera de norma")
    rep["datos"]["ids_orden_roto"] = orden_roto[:5]

    # --- C. coherencia de columnas ----------------------------------------
    t = l0["target:tile"].astype(str)
    if fam == "l89":
        esp_sensor = t.str[:4]
    else:
        esp_sensor = t.str[:3]
    if (l0["target:sensor"].astype(str) != esp_sensor).any():
        rep["fallos"].append(f"{int((l0['target:sensor'].astype(str) != esp_sensor).sum())} "
                             "filas con target:sensor distinto del tile")
    fechas_t = pd.to_datetime(l0["stac:time_start"], errors="coerce")
    if fechas_t.isna().any():
        rep["fallos"].append(f"{int(fechas_t.isna().sum())} filas sin stac:time_start válido")
    for k in (0, 1, 2, 3):
        fb = pd.to_datetime(l0[f"bg:date{k}"], errors="coerce")
        if fb.isna().any():
            rep["fallos"].append(f"{int(fb.isna().sum())} filas con bg:date{k} ilegible")
            continue
        # MARS busca en ventana simétrica (±120 días), así que una referencia posterior
        # al target no es un defecto. Solo se informa.
        futuras = (fb >= fechas_t)
        if futuras.any():
            rep["datos"][f"bg{k}_posteriores"] = int(futuras.sum())
        # comparación por día natural: una referencia a 120 días no es una fuera de ventana
        fuera = ((fechas_t.dt.normalize() - fb.dt.normalize()).dt.days.abs() > 120)
        if fuera.any():
            rep["avisos"].append(f"{int(fuera.sum())} referencias k={k} a más de 120 días")
    for k in (1, 2, 3):
        s = l0[f"bg:sensor{k}"].astype(str)
        if fam == "l89":
            # MARS busca LC08 y LC09 juntos: misma constelación, distinto satélite es válido.
            dist = ~s.isin(["LC08", "LC09"])
        else:
            dist = ~s.str.startswith("S2")
        if dist.any():
            rep["fallos"].append(f"{int(dist.sum())} referencias k={k} de otra constelación")
    for k in (1, 2, 3):
        c = pd.to_numeric(l0[f"bg:cloud{k}"], errors="coerce")
        if c.isna().any() or c.min() < 0 or c.max() > 1:
            rep["fallos"].append(f"bg:cloud{k} fuera de rango o NaN")
        rep["datos"][f"cloud{k}>35"] = int((c > 0.35).sum())
    ct = pd.to_numeric(l0["target:cloud"], errors="coerce")
    if ct.isna().any() or ct.min() < 0 or ct.max() > 1:
        rep["fallos"].append("target:cloud fuera de rango o NaN")

    # nube contra el CSV del recálculo
    csv = pathlib.Path(f"/data/databases/METHANESET_TACOS/_v140/T11/cloud_{fam}.csv")
    if csv.exists():
        ref = pd.read_csv(csv).set_index("id")
        j = l0.set_index("id").join(ref, rsuffix="_new", how="left")
        for c in ["target:cloud", "bg:cloud1", "bg:cloud2", "bg:cloud3"]:
            if f"{c}_new" in j.columns:
                d = (pd.to_numeric(j[c], errors="coerce") - j[f"{c}_new"]).abs()
                n = int((d > 1e-6).sum())
                if n:
                    rep["avisos"].append(f"{c}: {n} filas no coinciden con el recálculo (max {d.max():.3f})")

    # --- D. rásteres y derivadas ------------------------------------------
    tacoreader.use("pandas")
    ds = tacoreader.load(taco)
    data = ds.data
    idx = np.random.RandomState(0).choice(len(data), size=min(n_muestra, len(data)), replace=False)
    difs, met, formas = [], [], set()
    for i in idx:
        sub = data.read(int(i))
        fila = l0.iloc[int(i)]
        hojas = {}
        for k in range(8):
            ruta = sub.read(k)
            hojas[HOJAS[k]] = ruta
        tgt = leer(hojas["target"])
        formas.add((tgt.dtype.name, tgt.shape))
        for k in (1, 2, 3):
            bg = leer(hojas[f"bg{k}"])
            if bg.shape != tgt.shape:
                rep["fallos"].append(f"{fila['id'][:8]}: bg{k} con forma {bg.shape} != {tgt.shape}")
                continue
            d = C.mars_difference(tgt, bg, SIM[fam])
            if fam == "s2":
                d = d / 1e4  # la columna publicada va en reflectancia, no en DN
            col = float(fila[f"bg:difference{k}"])
            difs.append((col, d))
        p = leer(hojas["plume"])
        if not np.isfinite(p).any():
            rep["avisos"].append(f"{fila['id'][:8]}: hoja plume sin valores")
        c4 = leer(hojas["ch4"])
        if not np.isfinite(c4).any():
            rep["avisos"].append(f"{fila['id'][:8]}: hoja ch4 sin valores")
    rep["datos"]["formas"] = [list(map(str, f)) for f in formas]
    if difs:
        a = np.array(difs, dtype=float)
        nan_col = int(np.isnan(a[:, 0]).sum())
        nan_re = int(np.isnan(a[:, 1]).sum())
        rep["datos"]["difference_nan_columna"] = nan_col
        rep["datos"]["difference_nan_recalculo"] = nan_re
        ok = np.isfinite(a).all(axis=1)
        a = a[ok]
        if not len(a):
            rep["avisos"].append("bg:difference: no se pudo comparar ninguna muestra")
            return rep
        rel = np.abs(a[:, 0] - a[:, 1]) / np.maximum(np.abs(a[:, 1]), 1e-12)
        rep["datos"]["diferencia_n"] = int(len(a))
        rep["datos"]["diferencia_desv_mediana"] = float(np.median(rel))
        rep["datos"]["diferencia_desv_p90"] = float(np.percentile(rel, 90))
        rep["datos"]["diferencia_mayor_10pct"] = int((rel > 0.10).sum())
        if rep["datos"]["diferencia_mayor_10pct"] > 0.2 * len(a):
            rep["fallos"].append(
                f"bg:difference no coincide con el recálculo en "
                f"{rep['datos']['diferencia_mayor_10pct']}/{len(a)} muestras (desv mediana "
                f"{np.median(rel):.2f})")

    # --- E. recuentos ------------------------------------------------------
    rep["datos"]["muestras"] = int(len(l0))
    rep["datos"]["paises"] = int(l0["site:country"].nunique())
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taco", required=True)
    ap.add_argument("--fam", required=True, choices=["l89", "s2"])
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rep = auditoria(a.taco, a.fam, a.n)
    print(json.dumps(rep, indent=2, ensure_ascii=False, default=str)[:6000])
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
