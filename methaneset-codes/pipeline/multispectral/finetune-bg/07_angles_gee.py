"""Step 7 · Table of angles per scene from GEE (separate column, does not touch the tacozip).

Context: `target:sza`/`target:vza` of the finetune/pretraining come from MARS-S2L and
are corrupted in Landsat (LC08/LC09: ~7 % of rows with sza<1 and ~21 % with vza<1 in
the UNEP-IMEO release, which is still the same). Sentinel-2 is clean.

The Landsat granule id can be reconstructed from `target:tile` (e.g.
`LC09_L1TP_190041_20240928_20240928_02_T1` -> LC09, path 190, row 041, 2024-09-28), so
it can be requested from GEE by (path, row, date) without depending on the suffix (RT/T1/T2).

What GEE exposes at scene level: SUN_ELEVATION, SUN_AZIMUTH, ROLL_ANGLE,
NADIR_OFFNADIR, LANDSAT_PRODUCT_ID, centroid. It does NOT expose the view zenith: the
vza of the chip varies linearly with the cross-track position (measured: corr -0.98 with
the easting in one granule, range 3 degrees), so it is a per-pixel quantity (Landsat
angle bands), not a single metadata for the whole image. The sza does behave as
scene metadata: 90 - SUN_ELEVATION matches the valid MARS values within
~0.05 degrees.

Output: `gee_angles/<sensor>_granules_gee.csv` with one row per granule.

Usage:  python 07_angles_gee.py --sensor l89
"""

from __future__ import annotations

import argparse
import io
import pathlib
import re
import sys
import zipfile

import pandas as pd

import ee

ee.Initialize(project="ee-contrerasnetk")

BASE = pathlib.Path(__file__).parent
OUT = BASE / "gee_angles"
ROOT = pathlib.Path("/data/databases/METHANESET_TACOS")

COLLECTIONS = {
    "LC08": ["LANDSAT/LC08/C02/T1_L2", "LANDSAT/LC08/C02/T2_L2",
             "LANDSAT/LC08/C02/T1", "LANDSAT/LC08/C02/T2"],
    "LC09": ["LANDSAT/LC09/C02/T1_L2", "LANDSAT/LC09/C02/T2_L2",
             "LANDSAT/LC09/C02/T1", "LANDSAT/LC09/C02/T2"],
}
PROPS = ("SUN_ELEVATION", "SUN_AZIMUTH", "ROLL_ANGLE", "NADIR_OFFNADIR",
         "DATE_ACQUIRED", "LANDSAT_PRODUCT_ID", "WRS_PATH", "WRS_ROW")


def dataset_granules(ds: str) -> set[str]:
    """Granule ids (target:tile) present in a multispectral dataset."""
    out: set[str] = set()
    for f in sorted((ROOT / ds).glob("*.tacozip")):
        with zipfile.ZipFile(f) as z:
            df = pd.read_parquet(io.BytesIO(z.read("METADATA/level0.parquet")),
                                 columns=["target:tile"])
        out.update(map(str, df["target:tile"].dropna().unique()))
    return out


def parse_id(tile: str):
    m = re.match(r"(LC\d\d)_[A-Z0-9]+_(\d{3})(\d{3})_(\d{8})", tile)
    if not m:
        return None
    return m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)


def gee_scenes(sensor: str, path: int, row: int, d0: str, d1: str) -> list[dict]:
    for coll in COLLECTIONS[sensor]:
        ic = (ee.ImageCollection(coll)
              .filter(ee.Filter.eq("WRS_PATH", path))
              .filter(ee.Filter.eq("WRS_ROW", row))
              .filterDate(d0, d1))
        n = ic.size().getInfo()
        if not n:
            continue
        feats = ic.map(lambda img: ee.Feature(None, {p: img.get(p) for p in PROPS}))
        rows = [f["properties"] for f in feats.getInfo()["features"]]
        for r in rows:
            r["collection"] = coll
        return rows
    return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensor", default="l89", choices=["l89"])
    args = ap.parse_args()

    datasets = ["methaneset-l89-pretraining", "methaneset-l89-finetune"]
    granules: set[str] = set()
    for ds in datasets:
        g = dataset_granules(ds)
        print(f"{ds}: {len(g)} granules")
        granules |= g

    tab = []
    for tile in sorted(granules):
        p = parse_id(tile)
        if p is None:
            print("  unparsed:", tile)
            continue
        sensor, path, row, date = p
        tab.append(dict(granulo=tile, sensor=sensor, path=path, row=row, date=date))
    df = pd.DataFrame(tab)
    df["fecha"] = pd.to_datetime(df["date"], format="%Y%m%d")

    out_rows = []
    for (sensor, path, row), g in df.groupby(["sensor", "path", "row"]):
        d0 = (g["fecha"].min() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        d1 = (g["fecha"].max() + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
        try:
            scenes = gee_scenes(sensor, path, row, d0, d1)
        except Exception as e:  # noqa: BLE001
            print(f"  {sensor} {path}/{row}: error {str(e)[:60]}")
            continue
        for _, r in g.iterrows():
            hit = [e for e in scenes if str(e.get("DATE_ACQUIRED")) == r["fecha"].strftime("%Y-%m-%d")]
            for e in hit:
                out_rows.append(dict(
                    granulo=r["granulo"], sensor=sensor, path=path, row=row,
                    date=r["date"], fecha=r["fecha"].strftime("%Y-%m-%d"),
                    sun_elevation=e.get("SUN_ELEVATION"), sun_azimuth=e.get("SUN_AZIMUTH"),
                    roll_angle=e.get("ROLL_ANGLE"), nadir_offnadir=e.get("NADIR_OFFNADIR"),
                    gee_product_id=e.get("LANDSAT_PRODUCT_ID"), collection=e.get("collection"),
                ))
        print(f"  {sensor} {path}/{row}: {len(g)} granules, {len(scenes)} scenes in GEE")

    res = pd.DataFrame(out_rows)
    if len(res):
        res["sza_gee"] = 90.0 - pd.to_numeric(res["sun_elevation"], errors="coerce")
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / "l89_granules_gee.csv"
    res.to_csv(dest, index=False)
    print(f"\n-> {dest} | rows {len(res)} | granules with GEE {res['granulo'].nunique()} of {df['granulo'].nunique()}")
    missing = df.loc[~df["granulo"].isin(res["granulo"]), "granulo"].tolist()
    if missing:
        print(f"no data in GEE: {len(missing)} granules (e.g. {missing[:3]})")


if __name__ == "__main__":
    main()
