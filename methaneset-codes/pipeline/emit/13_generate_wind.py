"""MODULE 13 of the v2 pipeline: generate wind (NEW layer of the sample).

Decided by Julio (Aug 25): 2-band wind.tif (u10, v10 in m/s) per scene.
Basis in the wind-intra-scene note: within a scene the wind varies ~10x in
speed and rotates ~120 degrees; one value per granule is not enough.

How:
  1. Hourly ERA5-Land (ECMWF/ERA5_LAND/HOURLY) via EE (ee-contrerasnetk):
     u10/v10 at ~9 km nodes over the scene bbox with margin, at the TWO hours
     bracketing the granule timestamp.
  2. LINEAR IN TIME interpolation to the scene minute.
  3. Gather to the sensor grid via location/lat,lon: linear griddata
     (with nearest fill outside the hull, e.g. coasts where ERA5-Land has no
     node).
  4. wind.tif: bands u10, v10 float32, sensor grid NO CRS, tile 128, zstd.

Workers: 3 (limit: EE API). Resumable; argument = limit (test).

Usage:
  nohup /data/users/julio/.conda/envs/deep/bin/python code/v2/13_generate_wind.py \
        > code/v2/13_generate_wind.log 2>&1 &
"""
import datetime
import pathlib
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import ee
import h5py
import numpy as np
import pandas as pd
import rasterio
from scipy.interpolate import griddata

DATA = pathlib.Path(__file__).resolve().parent.parent.parent / "assets" / "data"
RAD_ROOTS = [
    pathlib.Path("/data/databases/MARS-Hyperspectral/EMIT_FULL"),
    pathlib.Path("/data/databases/MARS-Hyperspectral_complement/EMIT_FULL"),
]
OUT_ROOT = pathlib.Path("/data/databases/METHANSET_TACOS/emit")
TS = re.compile(r"(\d{8}T\d{6})")
WORKERS = 3
MARGIN = 0.15
BANDS = ["u_component_of_wind_10m", "v_component_of_wind_10m"]
BASE = dict(driver="GTiff", count=2, dtype="float32", tiled=True,
            blockxsize=128, blockysize=128, compress="zstd", interleave="band")


def era5_grid(region, t_iso):
    img = (ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
           .filterDate(t_iso, (datetime.datetime.fromisoformat(t_iso)
                               + datetime.timedelta(hours=1)).isoformat())
           .select(BANDS))
    g = img.getRegion(region, scale=11132).getInfo()
    df = pd.DataFrame(g[1:], columns=g[0]).dropna(subset=BANDS)
    return df


def interp_to_sensor(df, lat, lon):
    pts = df[["longitude", "latitude"]].values
    out = []
    for b in ("u", "v"):
        vals = df[b].values
        near = griddata(pts, vals, (lon, lat), method="nearest")
        if len(df) >= 4:
            lin = griddata(pts, vals, (lon, lat), method="linear")
            lin[~np.isfinite(lin)] = near[~np.isfinite(lin)]
        else:
            lin = near  # scene almost all sea: pure nearest with whatever is there
        out.append(lin.astype(np.float32))
    return out


def generate_one(nc_path):
    nc_path = pathlib.Path(nc_path)
    scene = nc_path.stem
    out = OUT_ROOT / scene / "wind.tif"
    if out.exists() and out.stat().st_size > 0:
        return scene, "already existed"
    try:
        with h5py.File(nc_path, "r") as h:
            lat = h["location/lat"][:]
            lon = h["location/lon"][:]
        ts = TS.search(scene).group(1)
        t = datetime.datetime.strptime(ts, "%Y%m%dT%H%M%S")
        h0 = t.replace(minute=0, second=0)
        frac = (t - h0).total_seconds() / 3600.0
        region = ee.Geometry.Rectangle([float(lon.min()) - MARGIN,
                                        float(lat.min()) - MARGIN,
                                        float(lon.max()) + MARGIN,
                                        float(lat.max()) + MARGIN])
        d0 = era5_grid(region, h0.isoformat())
        d1 = era5_grid(region, (h0 + datetime.timedelta(hours=1)).isoformat())
        for d in (d0, d1):
            d.rename(columns={BANDS[0]: "u", BANDS[1]: "v"}, inplace=True)
        m = d0.merge(d1, on=["longitude", "latitude"], suffixes=("0", "1"))
        m["u"] = m.u0 * (1 - frac) + m.u1 * frac
        m["v"] = m.v0 * (1 - frac) + m.v1 * frac
        if len(m) < 1:
            return scene, "zero ERA5-Land nodes (scene 100% sea)"
        if len(m) < 4:
            print(f"  warning {scene}: only {len(m)} nodes, pure nearest", flush=True)
        u, v = interp_to_sensor(m, lat, lon)

        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tif.part")
        with rasterio.open(tmp, "w", width=lat.shape[1], height=lat.shape[0],
                           **BASE) as dst:
            dst.write(u, 1); dst.write(v, 2)
            dst.set_band_description(1, "u10_ms_ERA5Land")
            dst.set_band_description(2, "v10_ms_ERA5Land")
        tmp.rename(out)
        return scene, ""
    except Exception as e:
        return scene, f"{type(e).__name__}: {e}"


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    ee.Initialize(project="ee-contrerasnetk")
    cross_dir = sorted((DATA / "cross").iterdir())[-1]
    splits = pd.read_parquet(cross_dir / "splits.parquet")
    idx = {}
    for root in RAD_ROOTS:
        for f in root.glob("*_RAD_*.nc"):
            mm = TS.search(f.name)
            if mm:
                idx.setdefault(mm.group(1).lower(), str(f))
    todo = [idx[ts] for ts in splits.granule_ts if ts in idx]
    if limit:
        todo = todo[:limit]
    print(f"scenes: {len(todo)} · workers {WORKERS}", flush=True)
    t0, done, errs = time.time(), 0, 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(generate_one, t) for t in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            scene, err = fut.result()
            if err and err != "already existed":
                errs += 1
                print(f"  ERROR {scene}: {err}", flush=True)
            elif not err:
                done += 1
            if i % 50 == 0 or i == len(futs):
                print(f"  {i}/{len(todo)}  ({i/(time.time()-t0):.2f} scenes/s)",
                      flush=True)
    print(f"\nwritten {done} · errors {errs} · destination {OUT_ROOT}")


if __name__ == "__main__":
    main()
