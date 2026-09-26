"""MODULE 12 of the v2 pipeline: generate elevation.

DEM decided by Julio: COPERNICUS/DEM/GLO30_2024_1 via Earth Engine with the
ee-contrerasnetk project. Mechanics inherited from v1's 61 with two fixes:
updated EE project and WITHOUT the intermediate float16 rounding.

  1. computePixels of the DEM on the scene ortho grid (bounds and resolution
     come from the nc geotransform itself + glt dimensions), with buffer.
  2. Bilinear gather to the sensor grid via location/lat,lon
     (map_coordinates order=1), as everywhere in v2.
  3. elevation.tif: 1 float32 band, sensor grid NO CRS, tile 128, zstd.

Workers: 3 (the limit is the EE API, not the disk).
Resumable; numeric argument = scene limit (test).

Usage:
  nohup /data/users/julio/.conda/envs/deep/bin/python code/v2/12_generate_elevation.py \
        > code/v2/12_generate_elevation.log 2>&1 &
"""
import io
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
from rasterio.transform import Affine
from scipy.ndimage import map_coordinates

DATA = pathlib.Path(__file__).resolve().parent.parent.parent / "assets" / "data"
RAD_ROOTS = [
    pathlib.Path("/data/databases/MARS-Hyperspectral/EMIT_FULL"),
    pathlib.Path("/data/databases/MARS-Hyperspectral_complement/EMIT_FULL"),
]
OUT_ROOT = pathlib.Path("/data/databases/METHANSET_TACOS/emit")
TS = re.compile(r"(\d{8}T\d{6})")
BUFFER_PX = 8
WORKERS = 3
BASE = dict(driver="GTiff", count=1, dtype="float32", tiled=True,
            blockxsize=128, blockysize=128, compress="zstd", interleave="band")

DEM = None  # built after ee.Initialize


def generate_one(nc_path):
    nc_path = pathlib.Path(nc_path)
    scene = nc_path.stem
    out = OUT_ROOT / scene / "elevation.tif"
    if out.exists() and out.stat().st_size > 0:
        return scene, "already existed"
    try:
        with h5py.File(nc_path, "r") as h:
            gt = h.attrs["geotransform"]
            oh, ow = h["location/glt_x"].shape
            lat = h["location/lat"][:]
            lon = h["location/lon"][:]
        res_x, res_y = float(gt[1]), float(-gt[5])
        bminx = float(gt[0]) - BUFFER_PX * res_x
        bmaxy = float(gt[3]) + BUFFER_PX * res_y
        width, height = ow + 2 * BUFFER_PX, oh + 2 * BUFFER_PX

        # computePixels has a 48 MB cap per request: request horizontal
        # strips and stack them (long scenes exceeded it)
        # EE counts ~5 bytes/pixel (float32 + mask); margin with 6
        max_rows = max(1, (45 * 1024 * 1024) // (width * 6))
        parts = []
        for r0 in range(0, height, max_rows):
            rows = min(max_rows, height - r0)
            req = {"expression": DEM, "fileFormat": "GEO_TIFF",
                   "bandIds": ["DEM"],
                   "grid": {"dimensions": {"width": width, "height": rows},
                            "affineTransform": {
                                "scaleX": res_x, "shearX": 0,
                                "translateX": bminx, "shearY": 0,
                                "scaleY": -res_y,
                                "translateY": bmaxy - r0 * res_y},
                            "crsCode": "EPSG:4326"}}
            tiff = ee.data.computePixels(req)
            with rasterio.open(io.BytesIO(tiff)) as src:
                parts.append(src.read(1).astype(np.float32))
        dem = np.vstack(parts)

        inv = ~Affine(res_x, 0, bminx, 0, -res_y, bmaxy)
        cols, rows = inv * (lon, lat)
        elev = map_coordinates(dem, [rows, cols], order=1, mode="nearest",
                               cval=0.0).astype(np.float32)

        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tif.part")
        with rasterio.open(tmp, "w", width=lat.shape[1], height=lat.shape[0],
                           **BASE) as dst:
            dst.write(elev, 1)
            dst.set_band_description(1, "elevation_m_GLO30_2024_1")
        tmp.rename(out)
        return scene, ""
    except Exception as e:
        return scene, f"{type(e).__name__}: {e}"


def main():
    global DEM
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    ee.Initialize(project="ee-contrerasnetk")
    DEM = (ee.ImageCollection("COPERNICUS/DEM/GLO30_2024_1")
           .select("DEM").mosaic().unmask(0))

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
    print(f"scenes: {len(todo)} · workers {WORKERS} (limit: EE API)", flush=True)

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
