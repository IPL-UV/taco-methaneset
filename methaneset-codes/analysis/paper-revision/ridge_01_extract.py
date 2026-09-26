"""Extract sampled pixels from the MethaneSET TACO finetune datasets.

For each sensor (s2 = 13 bands, l89 = 11 bands):
  - index every tacozip (level0 splits, level1 leaf offsets inside the zip)
  - draw up to CAP_PER_SPLIT samples per split (all if fewer; fixed RNG)
  - per sample, read target + plume, keep valid pixels (all bands finite),
    and draw up to MAX_PER_CLASS plume pixels and MAX_PER_CLASS background
    pixels for each of the SEEDS (<= 2000 px/sample/seed)
  - save one npz per seed with X (float32, raw DN/reflectance), y, sid
  - save a CSV with the sampled samples and their per-seed pixel counts

Run: /data/users/julio/.conda/envs/deep/bin/python 01_extract.py
"""

import glob
import io
import os
import zipfile

import numpy as np
import pandas as pd
import rasterio

BASES = {
    "s2": "/data/databases/METHANESET_TACOS/methaneset-s2-finetune",
    "l89": "/data/databases/METHANESET_TACOS/methaneset-l89-finetune",
}
OUT = "/tmp/opencode/ridge_bands"
CAP_PER_SPLIT = 500          # max samples per split
MAX_PER_CLASS = 1000         # max plume / background pixels per sample
SEEDS = [0, 1, 2]
SAMPLE_RNG_SEED = 12345      # fixed sample subset across seeds
LEAVES = ["target", "bg0", "bg1", "bg2", "bg3", "ch4", "plume", "dem"]


def build_index(base):
    """One row per sample: uuid, country, split, target/plume vsisubfile."""
    rows = []
    for path in sorted(glob.glob(os.path.join(base, "*.tacozip"))):
        z = zipfile.ZipFile(path)
        l0 = pd.read_parquet(io.BytesIO(z.read("METADATA/level0.parquet")))
        l1 = pd.read_parquet(io.BytesIO(z.read("METADATA/level1.parquet")))
        l1["uuid"] = l1["internal:relative_path"].str.split("/").str[0]
        l1["leaf"] = l1["internal:relative_path"].str.split("/").str[-1]
        z.close()
        country = os.path.basename(path).replace(".tacozip", "")
        offsets = {}
        for uuid, grp in l1.groupby("uuid", sort=False):
            offsets[uuid] = {
                row["leaf"]: (
                    int(row["internal:offset"]),
                    int(row["internal:size"]),
                )
                for _, row in grp.iterrows()
            }
        for _, s in l0.iterrows():
            off = offsets.get(s["id"])
            if off is None or "target" not in off or "plume" not in off:
                continue
            t_off, t_size = off["target"]
            p_off, p_size = off["plume"]
            rows.append(
                {
                    "uuid": s["id"],
                    "country": country,
                    "split": s["split"],
                    "target_vsi": f"/vsisubfile/{t_off}_{t_size},{path}",
                    "plume_vsi": f"/vsisubfile/{p_off}_{p_size},{path}",
                }
            )
    return pd.DataFrame(rows)


def draw_indices(idx, n, rng):
    if n <= 0 or len(idx) == 0:
        return idx[:0]
    if n >= len(idx):
        return rng.permutation(idx)
    return rng.choice(idx, size=n, replace=False)


def sample_pixels(target, mask, sample_idx, seed):
    """Balanced draw: up to MAX_PER_CLASS per class, <= 2000 total."""
    valid = np.isfinite(target).all(axis=0).ravel()
    plume = (mask.ravel() > 0) & valid
    idx_p = np.flatnonzero(plume)
    idx_b = np.flatnonzero(~plume & valid)
    rng = np.random.default_rng([sample_idx, seed])
    n_p = min(len(idx_p), MAX_PER_CLASS)
    n_b = min(len(idx_b), MAX_PER_CLASS)
    total = n_p + n_b
    if total < 2 * MAX_PER_CLASS:
        if n_p < MAX_PER_CLASS:
            n_b = min(len(idx_b), n_b + (2 * MAX_PER_CLASS - total))
        elif n_b < MAX_PER_CLASS:
            n_p = min(len(idx_p), n_p + (2 * MAX_PER_CLASS - total))
    take = np.concatenate([draw_indices(idx_p, n_p, rng), draw_indices(idx_b, n_b, rng)])
    y = np.zeros(len(take), dtype=np.uint8)
    y[:n_p] = 1
    return take, y, len(idx_p), len(idx_b)


def main():
    for sensor, base in BASES.items():
        print(f"\n=== {sensor}: indexing {base}")
        idx = build_index(base)
        print(idx["split"].value_counts().to_dict(), "total", len(idx))

        selected = []
        rng = np.random.default_rng(SAMPLE_RNG_SEED)
        for split in ["train", "validation", "test"]:
            sub = idx[idx["split"] == split]
            if len(sub) == 0:
                continue
            take = min(CAP_PER_SPLIT, len(sub))
            sel = sub.sample(n=take, random_state=int(rng.integers(2**31)))
            selected.append(sel)
            print(f"  {split}: {len(sub)} available -> {take} selected")
        sel = pd.concat(selected).reset_index(drop=True)

        accum = {s: {"X": [], "y": [], "sid": []} for s in SEEDS}
        meta = []
        for i, row in sel.iterrows():
            with rasterio.open(row["target_vsi"]) as src:
                target = src.read().astype(np.float32)
            with rasterio.open(row["plume_vsi"]) as src:
                mask = src.read(1)
            n_valid = int(np.isfinite(target).all(axis=0).sum())
            n_plume_valid = int(((mask.ravel() > 0) & np.isfinite(target).all(axis=0).ravel()).sum())
            Xf = target.reshape(target.shape[0], -1).T
            info = {
                "sensor": sensor,
                "uuid": row["uuid"],
                "country": row["country"],
                "split": row["split"],
                "sample_idx": i,
                "n_valid_px": n_valid,
                "n_plume_px": n_plume_valid,
                "n_bg_px": n_valid - n_plume_valid,
            }
            for seed in SEEDS:
                take, y, n_p, n_b = sample_pixels(target, mask, i, seed)
                if len(take) == 0:
                    accum[seed]["X"].append(np.zeros((0, Xf.shape[1]), np.float32))
                    accum[seed]["y"].append(np.zeros(0, np.uint8))
                    accum[seed]["sid"].append(np.zeros(0, np.int32))
                else:
                    accum[seed]["X"].append(Xf[take])
                    accum[seed]["y"].append(y)
                    accum[seed]["sid"].append(np.full(len(take), i, np.int32))
                info[f"n_sampled_s{seed}"] = len(take)
                info[f"n_plume_sampled_s{seed}"] = int(y.sum())
            meta.append(info)
            if (i + 1) % 100 == 0:
                print(f"  read {i + 1}/{len(sel)}")

        meta = pd.DataFrame(meta)
        meta.to_csv(os.path.join(OUT, f"samples_{sensor}.csv"), index=False)
        print(meta.groupby("split")[["n_valid_px", "n_plume_px"]].describe().round(1).to_string())
        for seed in SEEDS:
            X = np.concatenate(accum[seed]["X"])
            y = np.concatenate(accum[seed]["y"])
            sid = np.concatenate(accum[seed]["sid"])
            out = os.path.join(OUT, f"pix_{sensor}_s{seed}.npz")
            np.savez_compressed(out, X=X, y=y, sid=sid)
            print(f"  seed {seed}: X {X.shape} y mean {y.mean():.3f} -> {out}")


if __name__ == "__main__":
    main()
