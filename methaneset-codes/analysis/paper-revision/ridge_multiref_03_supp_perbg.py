"""Supplementary check for the multi-reference ridge comparison:

  - r_k : ratio target/bg_k - 1 for each background separately (k=0..3),
          to see whether bg0 (used as 'the' single reference) is
          representative and whether 4 concatenated refs beat any single one;
  - rmean_ratio : mean over k of (target/bg_k - 1), one feature per band.

Same protocol as 02_ridge_multiref.py: z-score/raw, C by validation AP,
test AUC/AP, 3 seeds.

Run: /data/users/julio/.conda/envs/deep/bin/python 03_supp_perbg.py
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
    if kind == "raw":
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
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
            fsets = {f"r{k}": (R[:, k, :], bg_ok[:, k]) for k in range(4)}
            mask4 = bg_ok.all(axis=1)
            fsets["rmean_ratio"] = (np.nanmean(R, axis=1), mask4)
            for fname, (X, mask) in fsets.items():
                for bname, bands in {"mars": MARS_BANDS[sensor], "all": range(n_bands)}.items():
                    Xb = X[:, list(bands)]
                    tr, va, te = tr_s & mask, va_s & mask, te_s & mask
                    for norm in ["raw", "zscore"]:
                        Xn = normalize(Xb, sid, n_samples, norm, mask)
                        best = None
                        for c in C_GRID:
                            clf = RidgeClassifier(alpha=1.0 / c, solver="cholesky")
                            clf.fit(Xn[tr], y[tr])
                            ap = average_precision_score(y[va], clf.decision_function(Xn[va]))
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
                                "n_features": len(list(bands)),
                                "C": c,
                                "val_AP": round(best[0], 4),
                                "test_AUC": round(roc_auc_score(y[te], pt), 4),
                                "test_AP": round(average_precision_score(y[te], pt), 4),
                                "test_AUC_per_sample_mean": round(auc_ps, 4),
                                "n_test_samples_mixed": n_mixed,
                                "n_test_px": int(te.sum()),
                            }
                        )
                        print(
                            f"{sensor} s{seed} {fname:11s} {bname:4s} {norm:7s} C={c:<6g} "
                            f"testAUC={rows[-1]['test_AUC']:.4f} testAP={rows[-1]['test_AP']:.4f} "
                            f"psAUC={auc_ps:.4f}"
                        )
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "supp_perbg_long.csv"), index=False)
    agg = (
        res[res["normalization"] == "zscore"]
        .groupby(["sensor", "bands", "features"])[["test_AUC", "test_AP", "test_AUC_per_sample_mean"]]
        .agg(["mean", "std"])
        .round(4)
    )
    agg.to_csv(os.path.join(OUT, "supp_perbg_summary_zscore.csv"))
    print("\n=== z-score: mean +/- sd over seeds")
    print(agg.to_string())


if __name__ == "__main__":
    main()
