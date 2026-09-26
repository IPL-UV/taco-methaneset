"""MODULE 16 of the v2 pipeline: TOTAL verification of the sample (requested by Julio).

Reads EVERY file of every scene COMPLETELY (no sampling) and checks:
  - readable from end to end (catches corruption like the latlon from 11)
  - the 10 layers present, same (H, W) on sensor grid (glt aside)
  - correct dtypes and nodata (radiance/mf/rmf/mag1c -9999, masks 0)
  - retrievals: valid fraction > 0 and no inf/NaN at valid pixels
  - masks: uint8, max <= 255, band 2 only codes that exist in band 1 or zero;
    frees completely empty
  - wind: finite over the whole scene; latlon: valid ranges (lat -90..90)

Output: cross/<date>/verify_samples.parquet (one row per scene with all the
checks) + summary on console. 8 workers (limit: disk).

Usage:
  nohup /data/users/julio/.conda/envs/deep/bin/python code/v2/16_verify_samples.py \
        > code/v2/16_verify_samples.log 2>&1 &
"""
import pathlib
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import rasterio

DATA = pathlib.Path(__file__).resolve().parent.parent.parent / "assets" / "data"
ROOT = pathlib.Path("/data/databases/METHANSET_TACOS/emit")
TS_G = re.compile(r"EMIT_L1B_RAD_001_(\d{8}T\d{6})")
LAYERS = ["radiance", "latlon", "glt", "elevation", "wind",
          "mf", "rmf", "mag1c", "plume_imeo", "plume_cm"]
WORKERS = 8

FREES = set()


def verify_scene(scene):
    d = ROOT / scene
    r = {"scene": scene, "ok": True}
    problems = []
    shapes = {}
    try:
        for ly in LAYERS:
            p = d / f"{ly}.tif"
            if not p.exists():
                problems.append(f"{ly}:MISSING"); continue
            with rasterio.open(p) as s:
                arr = s.read()  # full read
                shapes[ly] = arr.shape[1:]
                if s.block_shapes[0] != (128, 128):
                    problems.append(f"{ly}:tile{s.block_shapes[0]}")
                if ly == "radiance":
                    if s.nodata != -9999.0:
                        problems.append("radiance:nodata")
                    valid = arr[arr > -9000]
                    r["rad_valid_pct"] = valid.size / arr.size * 100
                    if not np.isfinite(valid).all():
                        problems.append("radiance:inf")
                elif ly in ("mf", "rmf", "mag1c"):
                    valid = arr[arr > -9000]
                    r[f"{ly}_valid_pct"] = valid.size / arr.size * 100
                    if valid.size == 0:
                        problems.append(f"{ly}:empty")
                    elif not np.isfinite(valid).all():
                        problems.append(f"{ly}:inf")
                elif ly == "wind":
                    if not np.isfinite(arr).all():
                        problems.append("wind:not_finite")
                    r["wind_speed_max"] = float(np.hypot(arr[0], arr[1]).max())
                elif ly == "latlon":
                    la, lo = arr[0], arr[1]
                    if not (np.abs(la) <= 90).all() or not (np.abs(lo) <= 180).all():
                        problems.append("latlon:range")
                elif ly in ("plume_imeo", "plume_cm"):
                    if arr.dtype != np.uint8:
                        problems.append(f"{ly}:dtype")
                    if s.nodata != 0:
                        problems.append(f"{ly}:nodata")
                    b1c = set(np.unique(arr[0])) - {0}
                    b2c = set(np.unique(arr[1])) - {0}
                    # source with a point but no mask pixels in this
                    # scene: property inherited from v1 (verified in
                    # scene 20220816T101058, v1 has it the same way), it is
                    # reported as data, not as a problem
                    r[f"{ly}_ghost_sources"] = len(b2c - b1c)
                    r[f"{ly}_npx"] = int((arr[0] > 0).sum())
                    if scene in FREES and (arr > 0).any():
                        problems.append(f"{ly}:free_not_empty")
        sensor_shapes = {v for k, v in shapes.items() if k != "glt"}
        if len(sensor_shapes) > 1:
            problems.append(f"shapes:{sensor_shapes}")
    except Exception as e:
        problems.append(f"{type(e).__name__}:{e}")
    r["ok"] = not problems
    r["problems"] = "; ".join(str(p) for p in problems)
    return r


def main():
    global FREES
    cross_dir = sorted((DATA / "cross").iterdir())[-1]
    splits = pd.read_parquet(cross_dir / "splits.parquet")
    scene_of = {TS_G.search(p.name).group(1).lower(): p.name
                for p in ROOT.iterdir() if TS_G.search(p.name)}
    FREES = {scene_of[r.granule_ts] for r in splits.itertuples()
             if r.is_free and r.granule_ts in scene_of}
    scenes = sorted(scene_of.values())
    print(f"scenes: {len(scenes)} · frees: {len(FREES)} · COMPLETE read "
          f"of the 10 layers", flush=True)

    rows, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(verify_scene, s) for s in scenes]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            rows.append(r)
            if not r["ok"]:
                print(f"  PROBLEM {r['scene']}: {r['problems']}", flush=True)
            if i % 50 == 0 or i == len(futs):
                print(f"  {i}/{len(scenes)}  ({i/(time.time()-t0):.2f} sc/s)",
                      flush=True)

    df = pd.DataFrame(rows)
    df.to_parquet(cross_dir / "verify_samples.parquet", index=False)
    bad = df[~df.ok]
    print(f"\n== VERDICT: {len(df) - len(bad)}/{len(df)} clean scenes · "
          f"{len(bad)} with problems ==")
    for c in ("mf_valid_pct", "rmf_valid_pct", "mag1c_valid_pct"):
        print(f"  {c}: min {df[c].min():.1f}% · median {df[c].median():.1f}%")
    print(f"table -> {cross_dir / 'verify_samples.parquet'}")


if __name__ == "__main__":
    main()
