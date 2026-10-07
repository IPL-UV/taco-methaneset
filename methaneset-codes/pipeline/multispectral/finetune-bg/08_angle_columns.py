"""Step 8 · Solar geometry columns for the multispectral sets: target:saa + target:raa.

Does NOT touch sza/vza (that stays as is). Only:
  - extracts the solar azimuth (SAA) per row,
  - computes the angle relative to the wind RAA = SAA - wind_dir.

Where the SAA comes from:
  - L89: `SAA` angle band of `LANDSAT/LC0X/C02/T1` (int16, scale 0.01),
    sampled at the center of each chip. It is per pixel (like sza/vza).
  - S2: scene property `MEAN_SOLAR_AZIMUTH_ANGLE` of COPERNICUS/S2_SR_HARMONIZED
    (or S2_HARMONIZED), per (MGRS_TILE, date).

RAA: computed with the direction the wind BLOWS toward from
`meteo:wind_u`/`meteo:wind_v` (ERA5 convention: u = east, v = north):
    wind_dir_to = atan2(u, v) in degrees (0 = north, increasing toward the east)
    target:raa = (wind_dir_to - saa) mod 360  # bank convention (sun:raa), counterclockwise from the wind
The final convention is being settled with Carlos; it is documented in the COLLECTION.

Resumable cache at `gee_angles/saa_<sensor>.csv`.

Usage:
  python 08_angle_columns.py --sensor l89
  python 08_angle_columns.py --sensor s2
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import io
import math
import pathlib
import re
import zipfile

import numpy as np
import pandas as pd

import ee

ee.Initialize(project="ee-contrerasnetk")

BASE = pathlib.Path(__file__).parent
OUT = BASE / "gee_angles"
ROOT = pathlib.Path("/data/databases/METHANESET_TACOS")

COLL_L89 = {"LC08": "LANDSAT/LC08/C02/T1", "LC09": "LANDSAT/LC09/C02/T1"}
COLL_S2 = ["COPERNICUS/S2_SR_HARMONIZED", "COPERNICUS/S2_HARMONIZED", "COPERNICUS/S2"]

DATASETS = {("l89","pretraining"): "methaneset-l89-pretraining", ("s2","pretraining"): "methaneset-s2-pretraining",
            ("l89","finetune"): "methaneset-l89-finetune", ("s2","finetune"): "methaneset-s2-finetune"}


def read_level0(ds: str, cols: list[str]) -> pd.DataFrame:
    parts = []
    for f in sorted((ROOT / ds).glob("*.tacozip")):
        with zipfile.ZipFile(f) as z:
            parts.append(pd.read_parquet(io.BytesIO(z.read("METADATA/level0.parquet")), columns=cols))
    return pd.concat(parts, ignore_index=True)


def num6(s) -> list[float]:
    return [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", str(s))][:6]


def parser_tile(tile: str):
    m = re.match(r"(LC\d\d)_[A-Z0-9]+_(\d{3})(\d{3})_(\d{8})", str(tile))
    if not m:
        return None
    return m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)


def saa_l89(df: pd.DataFrame, workers: int = 8) -> pd.Series:
    """SAA per chip from the Landsat L1 angle bands."""
    df = df.copy()
    df["_gt"] = df["stac:geotransform"].map(num6)
    df["_cx"] = [g[0] + 100 * g[1] for g in df["_gt"]]
    df["_cy"] = [g[3] - 100 * abs(g[5]) for g in df["_gt"]]
    df["_parsed"] = df["target:tile"].map(parser_tile)

    def by_group(args):
        (sensor, path, row, date), g = args
        try:
            img = (ee.ImageCollection(COLL_L89[sensor])
                   .filter(ee.Filter.eq("WRS_PATH", path))
                   .filter(ee.Filter.eq("WRS_ROW", row))
                   .filterDate(pd.Timestamp(date).strftime("%Y-%m-%d"),
                               (pd.Timestamp(date) + pd.Timedelta(days=2)).strftime("%Y-%m-%d"))
                   .first().select(["SAA"]))
            feats = ee.FeatureCollection([
                ee.Feature(ee.Geometry.Point([cx, cy], proj=str(crs)), {"rid": rid})
                for rid, cx, cy, crs in zip(g.index, g["_cx"], g["_cy"], g["stac:crs"])
            ])
            vals = img.reduceRegions(feats, ee.Reducer.mean(), scale=30).getInfo()["features"]
            return {v["properties"]["rid"]: v["properties"].get("mean") for v in vals}
        except Exception as e:  # noqa: BLE001
            print(f"    error {sensor} {path}/{row} {date}: {str(e)[:50]}")
            return {}

    keys = {}
    for i, p in enumerate(df["_parsed"]):
        if p is None:
            continue
        sensor, path, row, date = p
        keys.setdefault((sensor, path, row, date), []).append(i)
    groups = []
    for (sensor, path, row, date), idx in keys.items():
        groups.append(((sensor, path, row, date), df.loc[idx]))
    out = {}
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(by_group, groups):
            out.update(res)
    saa = pd.Series([out.get(i) for i in df.index], index=df.index, dtype="float64")
    return saa * 0.01  # the band comes in hundredths of a degree


def saa_s2(df: pd.DataFrame, workers: int = 8) -> pd.Series:
    """Scene SAA (MEAN_SOLAR_AZIMUTH_ANGLE) per (MGRS_TILE, date)."""
    df = df.copy()
    tile_col = df["target:tile"].astype(str)
    mgrs = tile_col.str.extract(r"_T(\w{5})_")[0]
    date_str = df["stac:time_start"].astype(str).str[:10]
    keys = pd.DataFrame({"mgrs": mgrs, "fecha": date_str}).dropna().drop_duplicates()
    cache: dict[tuple, float] = {}

    def by_tile(args):
        t, sub = args
        res = {}
        try:
            for coll in COLL_S2:
                ic = ee.ImageCollection(coll).filter(ee.Filter.eq("MGRS_TILE", t))
                feats = ic.map(lambda img: ee.Feature(None, {
                    "f": img.date().format("YYYY-MM-dd"),
                    "saa": img.get("MEAN_SOLAR_AZIMUTH_ANGLE"),
                }))
                for f in feats.getInfo()["features"]:
                    p = f["properties"]
                    if p.get("saa") is not None:
                        res.setdefault((t, p["f"]), p.get("saa"))
        except Exception as e:  # noqa: BLE001
            print(f"    error S2 tile {t}: {str(e)[:50]}")
        return res

    groups = [(t, g) for t, g in keys.groupby("mgrs")]
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(by_tile, groups):
            cache.update(res)
    return pd.Series([cache.get((m, f)) for m, f in zip(mgrs, date_str)], index=df.index, dtype="float64")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensor", required=True, choices=["l89", "s2"])
    ap.add_argument("--split", default="pretraining", choices=["pretraining", "finetune"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="limit to N granules/tiles (test)")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    cols = ["target:tile", "stac:time_start", "stac:crs", "stac:geotransform",
            "meteo:wind_u", "meteo:wind_v"]
    df = read_level0(DATASETS[(args.sensor, args.split)], cols)
    if args.limit:
        df = df[df["target:tile"].isin(df["target:tile"].unique()[:args.limit])].reset_index(drop=True)
    print(f"{args.sensor}: {len(df)} rows, {df['target:tile'].nunique()} granules/tiles")

    cache_f = OUT / f"saa_{args.sensor}_{args.split}.csv"
    if cache_f.exists():
        prev = pd.read_csv(cache_f)
        if len(prev) == len(df):
            print(f"existing cache used: {cache_f}")
            return
    if args.sensor == "l89":
        saa = saa_l89(df, args.workers)
    else:
        saa = saa_s2(df, args.workers)

    u = pd.to_numeric(df["meteo:wind_u"], errors="coerce")
    v = pd.to_numeric(df["meteo:wind_v"], errors="coerce")
    dir_to = (np.degrees(np.arctan2(u, v))) % 360
    raa = (dir_to - saa) % 360
    calm = np.hypot(u, v) < 0.1
    raa[calm | saa.isna()] = np.nan   # bank convention: no wind means no RAA
    res = pd.DataFrame({"target:tile": df["target:tile"], "saa": saa,
                        "wind_dir_to": dir_to, "raa": raa})
    res.to_csv(cache_f, index=False)
    print(f"-> {cache_f} | saa not null {res['saa'].notna().sum()}/{len(res)} "
          f"| raa not null {res['raa'].notna().sum()}")


if __name__ == "__main__":
    main()
