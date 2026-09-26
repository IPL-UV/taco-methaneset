"""MODULE 14 of the v2 pipeline: generate retrievals (mf, rmf, mag1c).

Adaptation of v1's 26: Julio's methanex engine on GPU (RTX A5500, float64),
SWIR 2122-2488 nm, three recipes per scene:
  mf    basic matched filter (no albedo correction)
  rmf   with albedo correction, one pass
  mag1c iterative sparse (30 iterations)

v2 changes with respect to v1:
  - output in METHANSET_TACOS/emit/<GRANULE>/ next to the rest of the sample
  - v2 profile: float32, tile 128, zstd, sensor grid NO CRS
  - careful nodata: radiance fill pixels are forced to -9999 in the three
    outputs and the nodata tag is declared

Sequential GPU (1 scene at a time). Resumable; argument = limit (test).

Usage:
  nohup /data/users/julio/.conda/envs/deep/bin/python code/v2/14_generate_retrievals.py \
        > code/v2/14_generate_retrievals.log 2>&1 &
"""
import pathlib
import re
import sys
import time

import h5py
import numpy as np
import pandas as pd
import rasterio
import torch

sys.path.insert(0, "/data/users/julio/methanset")
from methanex import MethaneRetrieval, MFConfig, RMFConfig, MAG1CConfig  # noqa: E402

DATA = pathlib.Path(__file__).resolve().parent.parent.parent / "assets" / "data"
RAD_ROOTS = [
    pathlib.Path("/data/databases/MARS-Hyperspectral/EMIT_FULL"),
    pathlib.Path("/data/databases/MARS-Hyperspectral_complement/EMIT_FULL"),
]
OUT_ROOT = pathlib.Path("/data/databases/METHANSET_TACOS/emit")
TS = re.compile(r"(\d{8}T\d{6})")
METHODS = {"mf": MFConfig, "rmf": RMFConfig, "mag1c": MAG1CConfig}
BATCH_SIZE = 64
WL_MIN, WL_MAX = 2122, 2488
FILL, NODATA = -9990.0, -9999.0
BASE = dict(driver="GTiff", count=1, dtype="float32", nodata=NODATA,
            tiled=True, blockxsize=128, blockysize=128, compress="zstd",
            interleave="band")


def load_swir(nc_path):
    with h5py.File(nc_path, "r") as h:
        wl = h["sensor_band_parameters/wavelengths"][:]
        sel = np.where((wl >= WL_MIN) & (wl <= WL_MAX))[0]
        rad = h["radiance"][:, :, sel[0]:sel[-1] + 1]
    return rad


def write_layer(path, arr, badmask, desc):
    arr = arr.astype(np.float32)
    arr[badmask] = NODATA
    tmp = path.with_suffix(".tif.part")
    with rasterio.open(tmp, "w", width=arr.shape[1], height=arr.shape[0],
                       **BASE) as dst:
        dst.write(arr, 1)
        dst.set_band_description(1, desc)
    tmp.rename(path)


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

    engine = MethaneRetrieval(device="cuda", dtype=torch.float64)
    print(f"engine ready · template bands: {int(engine.band_mask.sum())} · "
          f"{torch.cuda.get_device_name(0)}", flush=True)

    t0, done, errs = time.time(), 0, 0
    for i, nc in enumerate(todo, 1):
        nc = pathlib.Path(nc)
        scene = nc.stem
        scene_dir = OUT_ROOT / scene
        scene_dir.mkdir(parents=True, exist_ok=True)
        if all((scene_dir / f"{m}.tif").exists() for m in METHODS):
            continue
        try:
            rad = load_swir(nc)
            bad = (rad[:, :, rad.shape[2] // 2] <= FILL)
            rad_t = torch.from_numpy(rad)
            for m, cfg_cls in METHODS.items():
                out = scene_dir / f"{m}.tif"
                if out.exists():
                    continue
                res = engine.retrieve(rad_t, cfg_cls(batch_size=BATCH_SIZE),
                                      display_pbar=False)
                write_layer(out, res.mf.cpu().numpy(), bad, f"{m}_ppm_m")
            done += 1
        except Exception as e:
            errs += 1
            print(f"  ERROR {scene}: {type(e).__name__}: {e}", flush=True)
            torch.cuda.empty_cache()
        if i % 25 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)}  ({i/(time.time()-t0):.2f} scenes/s)",
                  flush=True)
    print(f"\nprocessed {done} · errors {errs} · destination {OUT_ROOT}")


if __name__ == "__main__":
    main()
