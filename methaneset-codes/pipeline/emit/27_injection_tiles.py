"""MODULE 27 of the v2 pipeline: compute injectable tiles for each methaneset-emit granule.

Injectable means: no plume (IMEO or CM dilated 5 px), no cloud (OmniCloudMask), and no nodata.
Outputs:
  - long parquet (audit): id, size, tile, row_off, col_off, width, height,
    valid, cloud_frac, plume_frac, nodata_frac
  - level0 columns (--apply): injection:windows_128, injection:windows_256,
    injection:n_tiles_128, injection:n_tiles_256

Resumable: caches per scene in _tiles_cache/<id>.parquet.

Usage:
  python 27_injection_tiles.py --run        # compute (resumable)
  python 27_injection_tiles.py --apply      # write level0
"""
from __future__ import annotations

import argparse
import io
import json
import pathlib
import time
import zipfile

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from scipy.ndimage import binary_dilation

import omnicloudmask
from omnicloudmask.download_models import get_models
from omnicloudmask.model_utils import load_model_from_weights
import torch

_MODELS = None
def models():
    """Load OmniCloudMask models ONCE (otherwise they reload per scene)."""
    global _MODELS
    if _MODELS is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _MODELS = [load_model_from_weights(
            model_name=m["timm_model_name"], weights_path=m["Path"],
            model_library=m["model_library"], device=dev)
            for m in get_models()]
    return _MODELS

ROOT = pathlib.Path("/data/databases/METHANESET_TACOS/methaneset-emit")
LEVEL0 = ROOT / "METADATA" / "level0.parquet"
CACHE = pathlib.Path("/data/databases/METHANESET_TACOS/_tiles_cache")
LONG = pathlib.Path("/data/databases/METHANESET_TACOS/_injection_tiles.parquet")
SIZES = (128, 256)
RELAX = 5
NODATA = -9999.0

def band_index(nm):
    return int(round((nm - 381.0) / ((2493.0 - 381.0) / 284.0))) + 1

IDX = [band_index(665), band_index(560), band_index(865)]
E0 = np.array([1550.0, 1900.0, 1000.0])
DIL = np.ones((11, 11), dtype=bool)


def grid(h, w, size):
    rows = list(range(0, h - size + 1, size))
    cols = list(range(0, w - size + 1, size))
    return [(r, c) for r in rows for c in cols]


def frac(mask, r, c, size):
    return float(mask[r:r + size, c:c + size].mean())


def scene_tiles(sid, sza, d):
    ddir = ROOT / "DATA" / sid
    with rasterio.open(ddir / "plume_imeo.tif") as a, rasterio.open(ddir / "plume_cm.tif") as b:
        plume = (a.read(1) > 0) | (b.read(1) > 0)
    plume = binary_dilation(plume, structure=DIL)
    with rasterio.open(ddir / "radiance.tif") as s:
        st = s.read(IDX).astype("float32")
        nod = (st[0] == NODATA)
    st = np.where(st == NODATA, 0.0, st) * 10.0
    rho = np.empty_like(st)
    for k in range(3):
        rho[k] = np.clip(np.pi * st[k] * d ** 2 / (E0[k] * np.cos(np.radians(sza))), 0, 1.5)
    cloud = np.squeeze(omnicloudmask.predict_from_array(
        rho, custom_models=models(), inference_device="cuda", batch_size=1)) > 0
    h, w = plume.shape
    rows = []
    for size in SIZES:
        for (r, c) in grid(h, w, size):
            pf, cf, nf = frac(plume, r, c, size), frac(cloud, r, c, size), frac(nod, r, c, size)
            rows.append(dict(id=sid, size=size, tile=f"r{r//size:02d}c{c//size:02d}",
                             row_off=r, col_off=c, width=size, height=size,
                             valid=bool(pf == 0 and cf == 0 and nf == 0),
                             cloud_frac=round(cf, 4), plume_frac=round(pf, 4), nodata_frac=round(nf, 4)))
    return rows


def run():
    CACHE.mkdir(exist_ok=True)
    df = pd.read_parquet(LEVEL0)
    ids = df["id"].tolist()
    t0 = time.time()
    for i, sid in enumerate(ids, 1):
        f = CACHE / f"{sid}.parquet"
        if f.exists():
            continue
        row = df[df["id"] == sid].iloc[0]
        sza = float(row["target:sza"]); d = float(row["sensor:earth_sun_distance"])
        try:
            rows = scene_tiles(sid, sza, d)
            pd.DataFrame(rows).to_parquet(f, index=False)
        except Exception as e:
            print(f"  ERROR {sid}: {e}", flush=True)
            continue
        if i % 20 == 0 or i == len(ids):
            done = len(list(CACHE.glob("*.parquet")))
            print(f"[{i}/{len(ids)}] {done} en cache | {time.time()-t0:.0f}s", flush=True)
    # merge
    parts = [pd.read_parquet(p) for p in sorted(CACHE.glob("*.parquet"))]
    long = pd.concat(parts, ignore_index=True)
    long.to_parquet(LONG, index=False)
    print("largo:", LONG, long.shape, "| validas 128:", int(((long["size"] == 128) & long["valid"]).sum()),
          "| 256:", int(((long["size"] == 256) & long["valid"]).sum()), flush=True)


def apply_level0():
    long = pd.read_parquet(LONG)
    df = pd.read_parquet(LEVEL0)
    def build(size):
        sub = long[(long["size"] == size) & long["valid"]]
        return (sub.groupby("id")
                .apply(lambda g: [{"col_off": int(c), "row_off": int(r), "width": size, "height": size}
                                  for r, c in zip(g["row_off"], g["col_off"])])
                .to_dict())
    for size in SIZES:
        m = build(size)
        df[f"injection:windows_{size}"] = df["id"].map(m).apply(lambda x: x if isinstance(x, list) else [])
        df[f"injection:n_tiles_{size}"] = df[f"injection:windows_{size}"].apply(len)
    df.to_parquet(LEVEL0, index=False)
    print("level0 actualizado:", df.shape)
    print(df[["injection:n_tiles_128", "injection:n_tiles_256"]].describe().round(1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    if a.run: run()
    if a.apply: apply_level0()
