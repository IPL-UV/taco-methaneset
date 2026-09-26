"""MODULE 11 of the v2 pipeline: generate latlon + glt.

The two geometry layers, straight from the nc (location group), v1 format
replicated + v2 chunking (tile 128, zstd):

  latlon.tif  sensor grid, NO CRS: band 1 = lat, band 2 = lon
              (float32, descriptions 'lat'/'lon')
  glt.tif     regular ORTHO grid: band 1 = glt_x, band 2 = glt_y (int32,
              nodata 0). It is the ONLY georeferenced layer of the sample:
              CRS and geotransform come from the nc attrs themselves.

Destination: /data/databases/METHANSET_TACOS/emit/<GRANULE>/
Resumable; numeric argument = scene limit (test).

Usage:
  nohup /data/users/julio/.conda/envs/deep/bin/python code/v2/11_generate_latlon_glt.py \
        > code/v2/11_generate_latlon_glt.log 2>&1 &
"""
import pathlib
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine

DATA = pathlib.Path(__file__).resolve().parent.parent.parent / "assets" / "data"
RAD_ROOTS = [
    pathlib.Path("/data/databases/MARS-Hyperspectral/EMIT_FULL"),
    pathlib.Path("/data/databases/MARS-Hyperspectral_complement/EMIT_FULL"),
]
OUT_ROOT = pathlib.Path("/data/databases/METHANSET_TACOS/emit")
TS = re.compile(r"(\d{8}T\d{6})")
WORKERS = 8
BASE = dict(driver="GTiff", tiled=True, blockxsize=128, blockysize=128,
            compress="zstd", interleave="band")


def write2(path, b1, b2, desc, **extra):
    tmp = path.with_suffix(".tif.part")
    p = dict(BASE, width=b1.shape[1], height=b1.shape[0], count=2,
             dtype=str(b1.dtype), **extra)
    with rasterio.open(tmp, "w", **p) as dst:
        dst.write(b1, 1); dst.write(b2, 2)
        dst.set_band_description(1, desc[0])
        dst.set_band_description(2, desc[1])
    tmp.rename(path)


def generate_one(nc_path):
    nc_path = pathlib.Path(nc_path)
    scene = nc_path.stem
    out_dir = OUT_ROOT / scene
    f_ll, f_glt = out_dir / "latlon.tif", out_dir / "glt.tif"
    if f_ll.exists() and f_glt.exists():
        return scene, "already existed"
    try:
        with h5py.File(nc_path, "r") as h:
            lat = h["location/lat"][:].astype(np.float32)
            lon = h["location/lon"][:].astype(np.float32)
            gx = h["location/glt_x"][:].astype(np.int32)
            gy = h["location/glt_y"][:].astype(np.int32)
            gt = h.attrs["geotransform"]
            wkt = h.attrs["spatial_ref"]
            wkt = wkt.decode() if isinstance(wkt, bytes) else str(wkt)
        out_dir.mkdir(parents=True, exist_ok=True)
        write2(f_ll, lat, lon, ("lat", "lon"))
        transform = Affine(gt[1], gt[2], gt[0], gt[4], gt[5], gt[3])
        write2(f_glt, gx, gy, ("glt_x", "glt_y"),
               crs=rasterio.crs.CRS.from_wkt(wkt), transform=transform, nodata=0)
        return scene, ""
    except Exception as e:
        return scene, f"{type(e).__name__}: {e}"


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    cross_dir = sorted((DATA / "cross").iterdir())[-1]
    splits = pd.read_parquet(cross_dir / "splits.parquet")
    idx = {}
    for root in RAD_ROOTS:
        for f in root.glob("*_RAD_*.nc"):
            m = TS.search(f.name)
            if m:
                idx.setdefault(m.group(1).lower(), str(f))
    todo = [idx[ts] for ts in splits.granule_ts if ts in idx]
    if limit:
        todo = todo[:limit]
    print(f"scenes: {len(todo)}", flush=True)
    t0, done, errs = time.time(), 0, 0
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(generate_one, t) for t in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            scene, err = fut.result()
            if err and err != "already existed":
                errs += 1
                print(f"  ERROR {scene}: {err}", flush=True)
            elif not err:
                done += 1
            if i % 100 == 0 or i == len(futs):
                print(f"  {i}/{len(todo)}  ({i/(time.time()-t0):.2f} scenes/s)",
                      flush=True)
    print(f"\nwritten {done} · errors {errs} · destination {OUT_ROOT}")


if __name__ == "__main__":
    main()
