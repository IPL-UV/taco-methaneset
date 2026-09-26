"""Ridge on multi-reference features: does using bg0..bg3 help over the
single-reference (target only) baseline of /tmp/opencode/ridge_bands?

Inputs: pix_multiref_<sensor>_s<seed>.npz + samples_multiref_<sensor>.csv
from 01_extract_multiref.py.

Feature sets (one feature per band unless said otherwise):
  - target        : target bands only (the previous experiment's features),
                    on exactly the same pixels -> reproduction check
  - target_common : target bands only, restricted to the bg4-valid pixel set
                    (isolates the pixel-set effect from the feature effect)
  - r0            : target / bg0 - 1 (single reference)
  - rmean         : target / mean(bg0..bg3) - 1 (multi-ref, same width as r0)
  - r4            : target / bg_k - 1, k = 0..3 concatenated (4 features/band)
  - r4t           : r4 + target values (5 features/band)

Band subsets: 'mars' (S2 B11,B12 / L89 B06,B07) and 'all'.
Normalizations: 'raw' and 'zscore' (per sample, over the pixels the feature
set uses). Same protocol as before: C chosen by validation AP, refit on
train, test AUC/AP + per-sample AUC, 3 seeds.

Run: /data/users/julio/.conda/envs/deep/bin/python 02_ridge_multiref.py
"""

import os

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

OUT = "/tmp/opencode/ridge_multiref"
SENSORS = {"s2": 13, "l89": 11}
MARS_BANDS = {"s2": [11, 12], "l89": [5, 6]}
C_GRID = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]
SEEDS = [0, 1, 2]


def normalize(X, sid, n_samples, kind, mask):
    """Per-sample normalization over the masked pixels only."""
    if kind == "raw":
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if kind != "zscore":
        raise ValueError(kind)
    Xz = np.zeros_like(X)
    s = sid[mask]
    for b in range(X.shape[1]):
        v = X[mask, b].astype(np.float64)
        counts = np.bincount(s, minlength=n_samples).astype(np.float64)
        counts[counts == 0] = 1.0
        mean = np.bincount(s, weights=v, minlength=n_samples) / counts
        var = np.bincount(s, weights=v**2, minlength=n_samples) / counts - mean**2
        std = np.sqrt(np.maximum(var, 0))
        std[std < 1e-12] = 1.0
        mean = np.where(np.isfinite(mean), mean, 0.0)
        std = np.where(np.isfinite(std), std, 1.0)
        col = np.nan_to_num(X[:, b], nan=0.0, posinf=0.0, neginf=0.0)
        Xz[:, b] = ((col - mean[sid]) / std[sid]).astype(np.float32)
    return Xz


def per_sample_auc(y, p, sid):
    vals = []
    for s in np.unique(sid):
        m = sid == s
        if 0 < y[m].sum() < m.sum():
            vals.append(roc_auc_score(y[m], p[m]))
    return float(np.mean(vals)), float(np.std(vals)), len(vals)


def build_feature_sets(Xt, R, bg_ok):
    """Return {name: (X, mask, band_of_col)}; NaN in ratios where bg invalid."""
    n, B = Xt.shape
    n_all = np.ones(n, dtype=bool)
    mask4 = bg_ok.all(axis=1)
    bands = np.arange(B)
    with np.errstate(divide="ignore", invalid="ignore"):
        bg_mean = np.nanmean(Xt[:, None, :] / (1.0 + R), axis=1)
        rmean = Xt / bg_mean - 1.0
    rmean_ok = mask4 & np.isfinite(bg_mean).all(axis=1) & (bg_mean > 0).all(axis=1)
    rmean[~rmean_ok] = np.nan
    r4 = R.reshape(n, 4 * B)
    return {
        "target": (Xt, n_all, np.tile(bands, 1)),
        "target_common": (Xt, mask4, np.tile(bands, 1)),
        "r0": (R[:, 0, :], bg_ok[:, 0].astype(bool), np.tile(bands, 1)),
        "rmean": (rmean, rmean_ok, np.tile(bands, 1)),
        "r4": (r4, mask4, np.tile(bands, 4)),
        "r4t": (np.concatenate([r4, Xt], axis=1), mask4, np.concatenate([np.tile(bands, 4), bands])),
    }


def main():
    rows = []
    for sensor, n_bands in SENSORS.items():
        meta = pd.read_csv(os.path.join(OUT, f"samples_multiref_{sensor}.csv"))
        split_of = meta["split"].to_numpy()
        n_samples = len(meta)
        for seed in SEEDS:
            d = np.load(os.path.join(OUT, f"pix_multiref_{sensor}_s{seed}.npz"))
            Xt, R, bg_ok = d["Xt"], d["R"], d["bg_ok"].astype(bool)
            y, sid = d["y"].astype(np.int8), d["sid"]
            pix_split = split_of[sid]
            tr_s, va_s, te_s = pix_split == "train", pix_split == "validation", pix_split == "test"
            fsets = build_feature_sets(Xt, R, bg_ok)
            for fname, (X, mask, band_of_col) in fsets.items():
                for bname, bands in {"mars": MARS_BANDS[sensor], "all": range(n_bands)}.items():
                    cols = np.flatnonzero(np.isin(band_of_col, list(bands)))
                    Xb = X[:, cols]
                    tr, va, te = tr_s & mask, va_s & mask, te_s & mask
                    for norm in ["raw", "zscore"]:
                        Xn = normalize(Xb, sid, n_samples, norm, mask)
                        best = None
                        for c in C_GRID:
                            clf = RidgeClassifier(alpha=1.0 / c, solver="cholesky")
                            clf.fit(Xn[tr], y[tr])
                            pv = clf.decision_function(Xn[va])
                            ap = average_precision_score(y[va], pv)
                            if best is None or ap > best[0]:
                                best = (ap, c)
                        _, c = best
                        clf = RidgeClassifier(alpha=1.0 / c, solver="cholesky")
                        clf.fit(Xn[tr], y[tr])
                        pt = clf.decision_function(Xn[te])
                        auc_ps, auc_ps_sd, n_mixed = per_sample_auc(y[te], pt, sid[te])
                        rows.append(
                            {
                                "sensor": sensor,
                                "seed": seed,
                                "normalization": norm,
                                "features": fname,
                                "bands": bname,
                                "n_features": len(cols),
                                "C": c,
                                "val_AP": round(best[0], 4),
                                "test_AUC": round(roc_auc_score(y[te], pt), 4),
                                "test_AP": round(average_precision_score(y[te], pt), 4),
                                "test_AUC_per_sample_mean": round(auc_ps, 4),
                                "test_AUC_per_sample_sd": round(auc_ps_sd, 4),
                                "n_test_samples_mixed": n_mixed,
                                "n_train_px": int(tr.sum()),
                                "n_val_px": int(va.sum()),
                                "n_test_px": int(te.sum()),
                                "test_pos_frac": round(float(y[te].mean()), 4),
                            }
                        )
                        print(
                            f"{sensor} s{seed} {fname:13s} {bname:4s} {norm:7s} "
                            f"nf={len(cols):2d} C={c:<6g} valAP={best[0]:.4f} "
                            f"testAUC={rows[-1]['test_AUC']:.4f} testAP={rows[-1]['test_AP']:.4f} "
                            f"psAUC={auc_ps:.4f}"
                        )
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "results_multiref_long.csv"), index=False)

    agg = (
        res[res["normalization"] == "zscore"]
        .groupby(["sensor", "bands", "features"])[["test_AUC", "test_AP", "test_AUC_per_sample_mean", "n_test_px"]]
        .agg(["mean", "std"])
        .round(4)
    )
    agg.to_csv(os.path.join(OUT, "results_multiref_summary_zscore.csv"))
    print("\n=== z-score: mean +/- sd over seeds")
    print(agg.to_string())

    aggr = (
        res[res["normalization"] == "raw"]
        .groupby(["sensor", "bands", "features"])[["test_AUC", "test_AP", "test_AUC_per_sample_mean"]]
        .agg(["mean", "std"])
        .round(4)
    )
    aggr.to_csv(os.path.join(OUT, "results_multiref_summary_raw.csv"))
    print("\n=== raw: mean +/- sd over seeds")
    print(aggr.to_string())


if __name__ == "__main__":
    main()
