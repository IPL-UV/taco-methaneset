"""Ridge regression (sklearn RidgeClassifier, closed-form cholesky):
MARS-S2L bands vs all bands. Scores are decision values (AUC/AP only,
no calibration needed).

Inputs: pix_<sensor>_s<seed>.npz + samples_<sensor>.csv from 01_extract.py.
Protocol per sensor and seed:
  - train on the train split, pick C by validation average precision (AP),
    refit on train with the picked C, evaluate on test.
Normalization variants (per sample, over the sampled pixels):
  - raw    : stored values (S2 = uint16 DN ~1e4*reflectance, L89 = reflectance/TIR)
  - zscore : (x - mean_band) / std_band
  - visratio: x / mean(visible bands B02-B04)
Feature sets:
  - mars : S2 -> B11,B12 ; L89 -> B06,B07 (0-based 11,12 / 5,6)
  - all  : every band available in the target

Run: /data/users/julio/.conda/envs/deep/bin/python 02_ridge.py
"""

import os

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

OUT = "/tmp/opencode/ridge_bands"
SENSORS = {"s2": 13, "l89": 11}
MARS_BANDS = {"s2": [11, 12], "l89": [5, 6]}
VISIBLE = [1, 2, 3]
C_GRID = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]
SEEDS = [0, 1, 2]


def normalize(X, sid, n_samples, kind):
    if kind == "raw":
        return X
    counts = np.bincount(sid, minlength=n_samples).astype(np.float64)
    counts[counts == 0] = 1.0
    if kind == "zscore":
        Xz = np.empty_like(X)
        for b in range(X.shape[1]):
            col = X[:, b].astype(np.float64)
            mean = np.bincount(sid, weights=col, minlength=n_samples) / counts
            var = np.bincount(sid, weights=col**2, minlength=n_samples) / counts - mean**2
            std = np.sqrt(np.maximum(var, 0))
            std[std < 1e-12] = 1.0
            Xz[:, b] = ((col - mean[sid]) / std[sid]).astype(np.float32)
        return Xz
    if kind == "visratio":
        vis = X[:, VISIBLE].astype(np.float64).mean(axis=1)
        denom = np.bincount(sid, weights=vis, minlength=n_samples) / counts
        denom[denom <= 0] = 1.0
        return (X / denom[sid, None]).astype(np.float32)
    raise ValueError(kind)


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
        meta = pd.read_csv(os.path.join(OUT, f"samples_{sensor}.csv"))
        split_of = meta["split"].to_numpy()
        n_samples = len(meta)
        for seed in SEEDS:
            d = np.load(os.path.join(OUT, f"pix_{sensor}_s{seed}.npz"))
            X, y, sid = d["X"], d["y"].astype(np.int8), d["sid"]
            pix_split = split_of[sid]
            tr, va, te = pix_split == "train", pix_split == "validation", pix_split == "test"
            print(f"{sensor} seed {seed}: train {tr.sum()} val {va.sum()} test {te.sum()} px"
                  f" | positives train {y[tr].mean():.3f} val {y[va].mean():.3f} test {y[te].mean():.3f}")
            for norm in ["raw", "zscore", "visratio"]:
                Xn = normalize(X, sid, n_samples, norm)
                for feats, cols in {
                    "mars": MARS_BANDS[sensor],
                    "all": list(range(n_bands)),
                }.items():
                    Xf = Xn[:, cols]
                    best = None
                    for c in C_GRID:
                        clf = RidgeClassifier(alpha=1.0 / c, solver="cholesky")
                        clf.fit(Xf[tr], y[tr])
                        pv = clf.decision_function(Xf[va])
                        ap = average_precision_score(y[va], pv)
                        auc = roc_auc_score(y[va], pv)
                        if best is None or ap > best[0]:
                            best = (ap, auc, c)
                    _, val_auc, c = best
                    clf = RidgeClassifier(alpha=1.0 / c, solver="cholesky")
                    clf.fit(Xf[tr], y[tr])
                    pt = clf.decision_function(Xf[te])
                    auc_ps, auc_ps_sd, n_mixed = per_sample_auc(y[te], pt, sid[te])
                    rows.append(
                        {
                            "sensor": sensor,
                            "seed": seed,
                            "normalization": norm,
                            "features": feats,
                            "n_features": len(cols),
                            "C": c,
                            "val_AUC": round(auc, 4),
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
                    print(f"  {norm:8s} {feats:4s} C={c:<6g} valAP={best[0]:.4f} "
                          f"testAUC={rows[-1]['test_AUC']:.4f} testAP={rows[-1]['test_AP']:.4f} "
                          f"psAUC={auc_ps:.4f}")
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "results_long.csv"), index=False)

    agg = (
        res.groupby(["sensor", "normalization", "features"])[["test_AUC", "test_AP", "test_AUC_per_sample_mean"]]
        .agg(["mean", "std"])
        .round(4)
    )
    agg.to_csv(os.path.join(OUT, "results_summary.csv"))
    print("\n=== mean +/- sd over seeds")
    print(agg.to_string())


if __name__ == "__main__":
    main()
