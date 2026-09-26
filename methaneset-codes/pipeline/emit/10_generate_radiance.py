"""MODULE 10 of the v2 pipeline: generate radiance.

First physical layer of the v2 sample. For each of the 721 granules of the
split (700 with plume + 21 free), reads the RAD nc and writes radiance.tif:

  - sensor grid, NO CRS (per-pixel georeferencing goes in latlon.tif)
  - float32, 285 bands
  - NODATA read carefully: fill values (<= -9990) or non-finite values are
    written as -9999.0 and the GeoTIFF nodata tag is declared
  - this module leaves the "raw" radiance (GTiff zstd); the final format
    (COG + DISCARD_LSB per SNR zones, tile 512, Cesar's) is set by
    module 10b (10b_apply_lsb.py), which ALWAYS runs after this one

Destination (Julio's decision): /data/databases/METHANSET_TACOS/emit/<GRANULE>/radiance.tif

Resumable (skips scenes already written). TEST: passing a number as argument
limits to that many scenes (e.g. `... 10_generate_radiance.py 3`).

Usage (long, nohup + twin log):
  cd 01-Projects/methanset
  nohup /data/users/julio/.conda/envs/deep/bin/python code/v2/10_generate_radiance.py \
        > code/v2/10_generate_radiance.log 2>&1 &
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

DATA = pathlib.Path(__file__).resolve().parent.parent.parent / "assets" / "data"
RAD_ROOTS = [
    pathlib.Path("/data/databases/MARS-Hyperspectral/EMIT_FULL"),
    pathlib.Path("/data/databases/MARS-Hyperspectral_complement/EMIT_FULL"),
]
OUT_ROOT = pathlib.Path("/data/databases/METHANSET_TACOS/emit")
TS = re.compile(r"(\d{8}T\d{6})")
FILL = -9990.0
NODATA = -9999.0
ROWS = 128
WORKERS = 6


def generate_one(args):
    ts, nc_path = args
    nc_path = pathlib.Path(nc_path)
    scene = nc_path.stem
    out_dir = OUT_ROOT / scene
    out = out_dir / "radiance.tif"
    if out.exists() and out.stat().st_size > 0:
        return scene, -1.0, "already existed"
    try:
        with h5py.File(nc_path, "r") as h:
            r = h["radiance"]
            nrow, ncol, nband = r.shape
            cube = np.empty((nrow, ncol, nband), dtype=np.float32)
            for i in range(0, nrow, ROWS):
                cube[i:i + ROWS] = r[i:i + ROWS]
        bad = (cube <= FILL) | ~np.isfinite(cube)
        cube[bad] = NODATA
        pct = float(bad[:, :, nband // 2].mean() * 100)

        out_dir.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tif.part")
        profile = dict(driver="GTiff", width=ncol, height=nrow, count=nband,
                       dtype="float32", nodata=NODATA, tiled=True,
                       blockxsize=128, blockysize=128, compress="zstd",
                       interleave="band")
        with rasterio.open(tmp, "w", **profile) as dst:
            for b in range(nband):
                dst.write(cube[:, :, b], b + 1)
        tmp.rename(out)
        return scene, pct, ""
    except Exception as e:
        return scene, np.nan, f"{type(e).__name__}: {e}"


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
    todo = [(ts, idx[ts]) for ts in splits.granule_ts if ts in idx]
    missing = [ts for ts in splits.granule_ts if ts not in idx]
    if limit:
        todo = todo[:limit]
    print(f"granules in the split: {len(splits)} · with RAD: {len(todo)} · "
          f"without RAD: {len(missing)} {missing[:3]}", flush=True)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    t0, done, errs = time.time(), 0, 0
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(generate_one, t) for t in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            scene, pct, err = fut.result()
            if err and err != "already existed":
                errs += 1
                print(f"  ERROR {scene}: {err}", flush=True)
            elif err == "already existed":
                pass
            else:
                done += 1
                if pct > 0:
                    print(f"  {scene}: nodata {pct:.1f}%", flush=True)
            if i % 25 == 0 or i == len(futs):
                rate = i / (time.time() - t0)
                print(f"  {i}/{len(todo)}  ({rate:.2f} scenes/s)", flush=True)
    print(f"\nwritten {done} · errors {errs} · destination {OUT_ROOT}")


if __name__ == "__main__":
    main()
