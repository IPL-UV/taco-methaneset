"""Paired per-sample comparison: 4 refs (r4) vs bg0 only (r0), z-score.

For each sensor, seed and band subset, fits both models with the protocol of
02_ridge_multiref.py and compares their per-sample test AUC on the SAME test
pixels. Reports the paired mean difference, its sd over samples, the fraction
of samples where r4 wins and a paired t statistic (descriptive, not a formal
inference: samples are not independent across scenes).

Run: /data/users/julio/.conda/envs/deep/bin/python 04_paired_test.py
"""

import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

OUT = "/tmp/opencode/ridge_multiref"
SENSORS = {"s2": 13, "l89": 11}
MARS_BANDS = {"s2": [11, 12], "l89": [5, 6]}
C_GRID = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]
SEEDS = [0, 1, 2]


def normalize(X, sid, n_samples, mask):
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


def fit_predict(Xb, y, tr, va, te, sid, n_samples, mask):
    Xn = normalize(Xb, sid, n_samples, mask)
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
    return clf.decision_function(Xn[te]), c


def per_sample_aucs(y, p, sid):
    out = {}
    for s in np.unique(sid):
        m = sid == s
        if 0 < y[m].sum() < m.sum():
            out[s] = roc_auc_score(y[m], p[m])
    return out


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
            mask4 = bg_ok.all(axis=1)
            tr_s, va_s, te_s = pix_split == "train", pix_split == "validation", pix_split == "test"
            for bands in [MARS_BANDS[sensor], list(range(n_bands))]:
                bname = "mars" if bands == MARS_BANDS[sensor] else "all"
                res = {}
                for fname, X, mask in [
                    ("r0", R[:, 0, :], bg_ok[:, 0]),
                    ("r4", R.reshape(len(y), -1), mask4),
                ]:
                    if fname == "r4":
                        B = Xt.shape[1]
                        cols = [k * B + b for k in range(4) for b in bands]
                    else:
                        cols = list(bands)
                    Xb = X[:, cols]
                    tr, va, te = tr_s & mask, va_s & mask, te_s & mask
                    pt, c = fit_predict(Xb, y, tr, va, te, sid, n_samples, mask)
                    res[fname] = per_sample_aucs(y[te], pt, sid[te])
                common = sorted(set(res["r0"]) & set(res["r4"]))
                a0 = np.array([res["r0"][s] for s in common])
                a4 = np.array([res["r4"][s] for s in common])
                diff = a4 - a0
                t, p = stats.ttest_rel(a4, a0)
                rows.append(
                    {
                        "sensor": sensor,
                        "seed": seed,
                        "bands": bname,
                        "n_test_samples": len(common),
                        "psAUC_r0_mean": round(a0.mean(), 4),
                        "psAUC_r4_mean": round(a4.mean(), 4),
                        "paired_diff_mean": round(diff.mean(), 4),
                        "paired_diff_sd": round(diff.std(ddof=1), 4),
                        "frac_samples_r4_better": round(float((diff > 0).mean()), 3),
                        "t_stat": round(float(t), 2),
                        "p_value": float(p),
                    }
                )
                print(rows[-1])
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "paired_r4_vs_r0.csv"), index=False)
    print("\n", df.to_string(index=False))


if __name__ == "__main__":
    main()
