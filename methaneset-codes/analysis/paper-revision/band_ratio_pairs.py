#!/usr/bin/env python3
"""Plume contrast for every band pair (Figure 8 of the paper), from the published TACOs.

One pass per sample computes both normalisations at once.

  raw   -> |mean_p - mean_b| / sd_b of the raw quotient  t_i/t_j - r_i/r_j   (a_k = 1)
  mars  -> the same on the MARS-S2L MBMP: each band ratio divided by its own
           scene mean without plumes, target over reference (to_mbmp, ratio_bands)

Work is split by sample and the final aggregate is a median, so the result does not
depend on the order and matches a serial run.

Writes one JSON per normalisation into $METHANESET_WORK/band_ratio/:

  band_ratio_matrices_raw.json    raw quotient        t_i/t_j - r_i/r_j
  band_ratio_matrices_mars.json   MARS-S2L per-scene normalisation

Usage:
  python band_ratio_pairs.py                                  # both sensors, ~4 min at 22 procs
  SENSORS=s2 MAX_SAMPLES=40 WORKERS=8 python band_ratio_pairs.py
"""
import json
import multiprocessing as mp
import os
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
WORK = pathlib.Path(os.environ.get("METHANESET_WORK", HERE / "work"))
OUT = WORK / "band_ratio"
OUT.mkdir(parents=True, exist_ok=True)

WORKERS = int(os.environ.get("WORKERS", "20"))
CHUNK = int(os.environ.get("CHUNK", "4"))
MAX_SAMPLES = int(os.environ.get("MAX_SAMPLES", "0")) or None
ONLY = os.environ.get("SENSORS", "").strip()

SENSORS = {
    "s2": {
        "bands": ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08",
                  "B8A", "B09", "B10", "B11", "B12"],
        "path": "/data/databases/METHANESET_TACOS/methaneset-s2-finetune",
        "label": "Sentinel-2",
    },
    "l89": {
        "bands": ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08",
                  "B09", "B10", "B11"],
        "path": "/data/databases/METHANESET_TACOS/methaneset-l89-finetune",
        "label": "Landsat 8/9",
    },
}


def sample_task(job):
    """Return (raw, mars) as flat n*n vectors, NaN where the pair does not apply."""
    target_p, bg_p, plume_p, n = job
    flat = np.full(n * n, np.nan, dtype=np.float64)
    flat_m = np.full(n * n, np.nan, dtype=np.float64)
    try:
        import rasterio

        with rasterio.open(target_p) as src:
            target = src.read().astype(np.float32)
        with rasterio.open(bg_p) as src:
            reference = src.read().astype(np.float32)
        with rasterio.open(plume_p) as src:
            mask = src.read(1) > 0
        if mask.sum() < 10:
            return None
        plume_px = mask.ravel()
        bg_px = ~plume_px
        t = target.reshape(n, -1)
        r = reference.reshape(n, -1)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                k = i * n + j
                safe = (np.abs(t[j]) > 1e-9) & (np.abs(r[j]) > 1e-9)

                mbmp = np.full(t.shape[1], np.nan, dtype=np.float32)
                mbmp[safe] = t[i, safe] / t[j, safe] - r[i, safe] / r[j, safe]
                vp = np.isfinite(mbmp) & plume_px
                vb = np.isfinite(mbmp) & bg_px
                if vp.sum() >= 5 and vb.sum() >= 20:
                    sd = np.nanstd(mbmp[vb])
                    if sd >= 1e-9:
                        flat[k] = abs(np.nanmean(mbmp[vp]) - np.nanmean(mbmp[vb])) / sd

                ti = np.full(t.shape[1], np.nan, dtype=np.float32)
                ri = np.full(t.shape[1], np.nan, dtype=np.float32)
                ti[safe] = t[i, safe] / t[j, safe]
                ri[safe] = r[i, safe] / r[j, safe]
                nt, nr = np.nanmean(ti[bg_px]), np.nanmean(ri[bg_px])
                if not (np.isfinite(nt) and np.isfinite(nr)) or nt == 0 or nr == 0:
                    continue
                mb = (ti / nt) / (ri / nr) - 1.0
                vp2 = np.isfinite(mb) & plume_px
                vb2 = np.isfinite(mb) & bg_px
                if vp2.sum() >= 5 and vb2.sum() >= 20:
                    sd2 = np.nanstd(mb[vb2])
                    if sd2 >= 1e-9:
                        flat_m[k] = abs(np.nanmean(mb[vp2]) - np.nanmean(mb[vb2])) / sd2
    except Exception:
        return None
    return flat, flat_m


def load_samples(ds_path):
    import tacoreader

    tacoreader.use("pandas")
    level0 = tacoreader.load(ds_path).flatten()
    pivot = level0.pivot_table(
        index="l0:id", columns="l1:id", values="gdal_vsi", aggfunc="first",
    ).reset_index()
    return pivot


def run(key, cfg, pool):
    n = len(cfg["bands"])
    print(f"\n=== {cfg['label']}: reading catalog", flush=True)
    samples = load_samples(cfg["path"])
    if MAX_SAMPLES is not None:
        samples = samples.iloc[:MAX_SAMPLES]
    jobs = [(row["target"], row["bg0"], row["plume"], n)
            for _, row in samples.iterrows()]
    print(f"    {len(jobs)} samples, {WORKERS} workers", flush=True)

    raw_acc, mars_acc = [], []
    t0 = time.time()
    for done, res in enumerate(pool.imap_unordered(sample_task, jobs, chunksize=CHUNK), 1):
        if res is not None:
            raw_acc.append(res[0])
            mars_acc.append(res[1])
        if done % 200 == 0 or done == len(jobs):
            el = time.time() - t0
            eta = el / done * (len(jobs) - done)
            print(f"    {done}/{len(jobs)}  {el:6.0f}s  eta {eta:4.0f}s", flush=True)

    mats = {}
    for name, acc in (("raw", raw_acc), ("mars", mars_acc)):
        stack = np.vstack(acc) if acc else np.full((1, n * n), np.nan)
        med = np.nanmedian(stack, axis=0)
        mat = med.reshape(n, n)
        mats[name] = mat
        print(f"    {name}: {len(acc)} usable samples", flush=True)
    return mats


def main():
    keys = [ONLY] if ONLY else list(SENSORS)
    print(f"workers={WORKERS} chunk={CHUNK} max_samples={MAX_SAMPLES} sensors={keys}")
    ctx = mp.get_context("fork")
    todo = {}
    with ctx.Pool(WORKERS) as pool:
        for key in keys:
            todo[key] = run(key, SENSORS[key], pool)

    # One file per normalisation, with whichever sensors were computed.
    for name in ("raw", "mars"):
        blob = {}
        for key, mats in todo.items():
            mat = mats[name]
            blob[key] = np.where(np.isnan(mat), None, mat).tolist()
        path = OUT / f"band_ratio_matrices_{name}.json"
        path.write_text(json.dumps(blob))
        print(f"{name} -> {path.name} ({', '.join(blob)})", flush=True)
    print("\ndone")


if __name__ == "__main__":
    sys.exit(main())
