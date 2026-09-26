"""Extract multi-reference pixels for the ridge comparison (MethaneSET review).

The goal is a controlled comparison of ridge features built from FOUR
background references (bg0..bg3, provided per finetune sample) against the
previous single-reference experiment in /tmp/opencode/ridge_bands.

Comparability with the previous experiment:
  - the previous samples_<sensor>.csv is the authoritative sample list
    (same uuid, split and sample_idx order -> same RNG seed per sample);
  - pixel draws follow the same balanced procedure and the same
    np.random.default_rng([sample_idx, seed]) as 01_extract.py, so the
    drawn pixels are identical as long as target/plume are unchanged;
  - every drawn pixel with finite target is kept, as in the previous
    experiment. Background validity is stored per reference so each
    feature set can define its own mask (never silently drops pixels).

For every kept pixel, stores:
  - Xt    : target values, (n, B) float32
  - R     : target / bg_k - 1 for k = 0..3, (n, 4, B) float32,
            NaN where the background is invalid (non-finite or zero)
  - bg_ok : per-reference validity, (n, 4) uint8
  - y     : plume label, sid : sample index

Run: /data/users/julio/.conda/envs/deep/bin/python 01_extract_multiref.py
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
PREV = "/tmp/opencode/ridge_bands"
OUT = "/tmp/opencode/ridge_multiref"
MAX_PER_CLASS = 1000
SEEDS = [0, 1, 2]
LEAVES = ["target", "bg0", "bg1", "bg2", "bg3", "plume"]


def build_index(base):
    """One row per sample: uuid, country, split and a vsisubfile per leaf."""
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
                row["leaf"]: (int(row["internal:offset"]), int(row["internal:size"]))
                for _, row in grp.iterrows()
            }
        for _, s in l0.iterrows():
            off = offsets.get(s["id"])
            if off is None or any(leaf not in off for leaf in LEAVES):
                continue
            rec = {"uuid": s["id"], "country": country, "split": s["split"]}
            for leaf in LEAVES:
                o, sz = off[leaf]
                rec[f"vsi_{leaf}"] = f"/vsisubfile/{o}_{sz},{path}"
            rows.append(rec)
    return pd.DataFrame(rows)


def draw_indices(idx, n, rng):
    if n <= 0 or len(idx) == 0:
        return idx[:0]
    if n >= len(idx):
        return rng.permutation(idx)
    return rng.choice(idx, size=n, replace=False)


def sample_pixels(target, mask, sample_idx, seed):
    """Same balanced draw as the single-reference experiment (01_extract.py)."""
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
    return take, y


def main():
    for sensor, base in BASES.items():
        prev = pd.read_csv(os.path.join(PREV, f"samples_{sensor}.csv"))
        print(f"\n=== {sensor}: previous sample list {len(prev)} samples")
        idx = build_index(base)
        merged = prev.merge(idx, on=["uuid", "split"], how="left", suffixes=("_prev", ""))
        missing = merged["vsi_target"].isna().sum()
        print(f"  indexed {len(idx)} samples in TACO; missing from index: {missing}")
        merged = merged.sort_values("sample_idx").reset_index(drop=True)
        merged.to_csv(os.path.join(OUT, f"samples_multiref_{sensor}.csv"), index=False)

        accum = {s: {"Xt": [], "R": [], "ok": [], "y": [], "sid": []} for s in SEEDS}
        meta = []
        for i, row in merged.iterrows():
            with rasterio.open(row["vsi_target"]) as src:
                target = src.read().astype(np.float32)
            bgs = []
            for k in range(4):
                with rasterio.open(row[f"vsi_bg{k}"]) as src:
                    bgs.append(src.read().astype(np.float32))
            with rasterio.open(row["vsi_plume"]) as src:
                mask = src.read(1)
            B = target.shape[0]
            tf = target.reshape(B, -1).T
            bf = np.stack([bg.reshape(B, -1).T for bg in bgs])  # (4, npix, B)
            info = {
                "sensor": sensor,
                "uuid": row["uuid"],
                "country": row["country"],
                "split": row["split"],
                "sample_idx": i,
            }
            for seed in SEEDS:
                take, y = sample_pixels(target, mask, i, seed)
                if len(take) == 0:
                    accum[seed]["Xt"].append(np.zeros((0, B), np.float32))
                    accum[seed]["R"].append(np.zeros((0, 4, B), np.float32))
                    accum[seed]["ok"].append(np.zeros((0, 4), np.uint8))
                    accum[seed]["y"].append(np.zeros(0, np.uint8))
                    accum[seed]["sid"].append(np.zeros(0, np.int32))
                    info[f"n_sampled_s{seed}"] = 0
                    info[f"n_bg0_ok_s{seed}"] = 0
                    info[f"n_bg4_ok_s{seed}"] = 0
                    continue
                t = tf[take]
                b = bf[:, take, :]  # (4, n, B)
                bg_ok = np.isfinite(b).all(axis=2) & (b != 0).all(axis=2)  # (4, n)
                with np.errstate(divide="ignore", invalid="ignore"):
                    R = (t[None, :, :] / b - 1.0).transpose(1, 0, 2)  # (n, 4, B)
                R[~bg_ok.T] = np.nan
                accum[seed]["Xt"].append(t)
                accum[seed]["R"].append(R)
                accum[seed]["ok"].append(bg_ok.T.astype(np.uint8))
                accum[seed]["y"].append(y)
                accum[seed]["sid"].append(np.full(len(t), i, np.int32))
                info[f"n_sampled_s{seed}"] = int(len(t))
                info[f"n_bg0_ok_s{seed}"] = int(bg_ok[0].sum())
                info[f"n_bg4_ok_s{seed}"] = int(bg_ok.all(axis=0).sum())
            meta.append(info)
            if (i + 1) % 100 == 0:
                print(f"  read {i + 1}/{len(merged)}")

        meta = pd.DataFrame(meta)
        meta.to_csv(os.path.join(OUT, f"multiref_meta_{sensor}.csv"), index=False)
        for seed in SEEDS:
            print(
                f"  seed {seed}: px target-valid {meta[f'n_sampled_s{seed}'].sum()}, "
                f"bg0-valid {meta[f'n_bg0_ok_s{seed}'].sum()}, "
                f"bg4-valid {meta[f'n_bg4_ok_s{seed}'].sum()}"
            )
        for seed in SEEDS:
            Xt = np.concatenate(accum[seed]["Xt"])
            R = np.concatenate(accum[seed]["R"])
            ok = np.concatenate(accum[seed]["ok"])
            y = np.concatenate(accum[seed]["y"])
            sid = np.concatenate(accum[seed]["sid"])
            out = os.path.join(OUT, f"pix_multiref_{sensor}_s{seed}.npz")
            np.savez_compressed(out, Xt=Xt, R=R, bg_ok=ok, y=y, sid=sid)
            print(f"  seed {seed}: Xt {Xt.shape} R {R.shape} y mean {y.mean():.3f} -> {out}")


if __name__ == "__main__":
    main()
