"""Does a nonlinear model change the "extra bands / extra backgrounds" conclusions?

Benchmark for the MethaneSET review, reusing the exact protocol and pixels of
$METHANESET_WORK/ridge_multiref (which itself reuses $METHANESET_WORK/ridge_bands):

  - input : pix_multiref_<sensor>_s<seed>.npz (Xt, R, bg_ok, y, sid)
            + samples_multiref_<sensor>.csv (split per scene)
  - features: target/bg - 1 with per-sample z-score, reference sets:
        refs=1 : r0   = target/bg0 - 1                (width B)
        refs=4 : r4   = target/bg_k - 1, k=0..3 concat (width 4B)
    band subsets: mars (S2 B11,B12; L89 B06,B07) and all.
  - split : predefined train/validation/test from the TACO metadata
  - models: RidgeClassifier (baseline, C by validation AP)
            HistGradientBoostingClassifier (fixed modest hyperparameters,
            early stopping on the official validation AP)
  - seeds : 0, 1, 2 (same pixel draws as the ridge experiments)

Metrics: test AUC-ROC and AP (pixel-level, grouped), plus mean per-scene AUC
and paired per-scene comparisons (r4 vs r0, all vs mars).

Run: python bench_01_nonlin_vs_ridge.py
     [--scenes N] [--max-iter N] [--sensors s2,l89] [--seeds 0,1,2] [--tag full]

Outputs (in $METHANESET_WORK/bench_nonlin):
  results_bench_long_<tag>.csv     one row per model x config x sensor x seed
  results_bench_summary_<tag>.csv  mean +/- sd over seeds
  paired_scene_<tag>.csv           paired per-scene AUC differences + t-test
  bench_<tag>.log                  stdout log
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

WORK = os.path.abspath(os.environ.get(
    "METHANESET_WORK", os.path.join(os.path.dirname(os.path.abspath(__file__)), "work")))
SRC = os.path.join(WORK, "ridge_multiref")
OUT = os.path.join(WORK, "bench_nonlin")
os.makedirs(OUT, exist_ok=True)
SENSORS = {"s2": 13, "l89": 11}
MARS_BANDS = {"s2": [11, 12], "l89": [5, 6]}
C_GRID = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]

# Fixed, modest HGB hyperparameters (no search).
HGB_PARAMS = dict(
    loss="log_loss",
    learning_rate=0.1,
    max_iter=300,
    max_leaf_nodes=31,
    max_depth=None,
    min_samples_leaf=20,
    l2_regularization=0.0,
    max_bins=255,
    early_stopping=False,  # early stopping is done manually on the validation split
    random_state=None,     # set per run to the pixel seed
)
HGB_CHECK = 10   # evaluate validation AP every N iterations
HGB_PATIENCE = 3  # stop after PATIENCE * CHECK iterations without AP improvement
HGB_MIN_DELTA = 1e-4


def normalize(X, sid, n_samples, mask, kind="zscore"):
    """Per-sample z-score over the masked pixels. Same as 02_ridge_multiref.py."""
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


def per_sample_scores(y, score, sid):
    """Per-scene AUC on scenes that have both classes. Returns dict sid -> AUC."""
    out = {}
    for s in np.unique(sid):
        m = sid == s
        if 0 < y[m].sum() < m.sum():
            out[int(s)] = float(roc_auc_score(y[m], score[m]))
    return out


def build_config(Xt, R, bg_ok, refs, band_cols):
    """Return (X, mask) for a reference set ('r0'/'r4') restricted to band_cols."""
    n, B = Xt.shape
    if refs == 1:
        X = R[:, 0, :]
        mask = bg_ok[:, 0].astype(bool)
        band_of_col = np.arange(B)
    else:
        X = R.reshape(n, 4 * B)
        mask = bg_ok.all(axis=1)
        band_of_col = np.tile(np.arange(B), 4)
    cols = np.flatnonzero(np.isin(band_of_col, band_cols))
    return X[:, cols], mask


def run_ridge(Xn, y, sid, tr, va, te, n_samples):
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
    return dict(val_AP=float(best[0]), C=c, test_score=pt)


def run_hgb(Xn, y, sid, tr, va, te, seed, max_iter):
    params = dict(HGB_PARAMS)
    params["max_iter"] = max_iter
    params["random_state"] = seed
    hg = HistGradientBoostingClassifier(**params)
    hg.fit(Xn[tr], y[tr])

    # Manual early stopping on the official validation split (raw scores are a
    # monotone transform of the probabilities, so AUC/AP are unaffected).
    best_ap, best_stage, no_improve, last_stage = -np.inf, None, 0, 0
    for it, raw in enumerate(hg.staged_decision_function(Xn[va]), start=1):
        last_stage = it
        if it % HGB_CHECK == 0 or it == 1:
            ap = average_precision_score(y[va], raw)
            if ap > best_ap + HGB_MIN_DELTA:
                best_ap, best_stage, no_improve = float(ap), it, 0
            else:
                no_improve += HGB_CHECK
                if no_improve >= HGB_PATIENCE * HGB_CHECK:
                    break
    if best_stage is None:
        best_stage = last_stage
    pt = None
    for it, raw in enumerate(hg.staged_decision_function(Xn[te]), start=1):
        if it == best_stage:
            pt = raw
            break
    return dict(val_AP=best_ap, C=np.nan, n_iter=best_stage, n_iter_fit=last_stage, test_score=pt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=0, help="max scenes per split (0 = all)")
    ap.add_argument("--max-iter", type=int, default=HGB_PARAMS["max_iter"])
    ap.add_argument("--sensors", type=str, default="s2,l89")
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--tag", type=str, default="full")
    args = ap.parse_args()

    sensors = [s for s in args.sensors.split(",") if s]
    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    os.makedirs(OUT, exist_ok=True)
    rows, t0 = [], time.time()
    scene_auc = {}  # (sensor, seed, model, refs, bands) -> {scene: auc}

    for sensor in sensors:
        n_bands = SENSORS[sensor]
        meta = pd.read_csv(os.path.join(SRC, f"samples_multiref_{sensor}.csv"))
        split_of = meta["split"].to_numpy()
        n_samples = len(meta)
        scene_sample_idx = meta["sample_idx"].to_numpy()
        if args.scenes:
            keep_idx = (
                meta.sort_values("sample_idx")
                .groupby("split", group_keys=False)
                .head(args.scenes)["sample_idx"]
                .to_numpy()
            )
            keep_sid = np.flatnonzero(np.isin(scene_sample_idx, keep_idx))
        else:
            keep_sid = None
        for seed in seeds:
            d = np.load(os.path.join(SRC, f"pix_multiref_{sensor}_s{seed}.npz"))
            Xt, R, bg_ok = d["Xt"], d["R"], d["bg_ok"].astype(bool)
            y, sid = d["y"].astype(np.int8), d["sid"]
            if keep_sid is not None:
                m = np.isin(sid, keep_sid)
                Xt, R, bg_ok, y, sid = Xt[m], R[m], bg_ok[m], y[m], sid[m]
            pix_split = split_of[sid]
            tr_s, va_s, te_s = pix_split == "train", pix_split == "validation", pix_split == "test"
            print(
                f"\n=== {sensor} seed {seed}: {len(y)} px "
                f"(train {tr_s.sum()}, val {va_s.sum()}, test {te_s.sum()}, pos {y.mean():.3f})"
            )
            for refs in (1, 4):
                for bands_name, band_cols in {
                    "mars": MARS_BANDS[sensor],
                    "all": list(range(n_bands)),
                }.items():
                    X, mask = build_config(Xt, R, bg_ok, refs, band_cols)
                    Xn = normalize(X, sid, n_samples, mask)
                    tr, va, te = tr_s & mask, va_s & mask, te_s & mask
                    print(
                        f"  refs={refs} bands={bands_name:4s} nf={X.shape[1]:2d} "
                        f"px tr/va/te = {tr.sum()}/{va.sum()}/{te.sum()}"
                    )
                    for model in ("ridge", "hgb"):
                        t1 = time.perf_counter()
                        if model == "ridge":
                            r = run_ridge(Xn, y, sid, tr, va, te, n_samples)
                        else:
                            r = run_hgb(Xn, y, sid, tr, va, te, seed, args.max_iter)
                        dt = time.perf_counter() - t1
                        pt = r["test_score"]
                        auc = float(roc_auc_score(y[te], pt))
                        apx = float(average_precision_score(y[te], pt))
                        ps = per_sample_scores(y[te], pt, sid[te])
                        scene_auc[(sensor, seed, model, refs, bands_name)] = ps
                        rows.append(
                            {
                                "sensor": sensor,
                                "seed": seed,
                                "model": model,
                                "refs": refs,
                                "bands": bands_name,
                                "n_features": X.shape[1],
                                "val_AP": round(r["val_AP"], 4),
                                "C": r["C"],
                                "hgb_best_iter": r.get("n_iter", np.nan),
                                "hgb_fit_iter": r.get("n_iter_fit", np.nan),
                                "test_AUC": round(auc, 4),
                                "test_AP": round(apx, 4),
                                "test_AUC_per_scene_mean": round(float(np.mean(list(ps.values()))), 4)
                                if ps
                                else np.nan,
                                "test_AUC_per_scene_sd": round(float(np.std(list(ps.values()))), 4)
                                if ps
                                else np.nan,
                                "n_test_scenes": len(ps),
                                "n_train_px": int(tr.sum()),
                                "n_val_px": int(va.sum()),
                                "n_test_px": int(te.sum()),
                                "test_pos_frac": round(float(y[te].mean()), 4),
                                "seconds": round(dt, 1),
                            }
                        )
                        extra = f"it={r['n_iter']}" if model == "hgb" else f"C={r['C']:g}"
                        print(
                            f"    {model:5s} valAP={r['val_AP']:.4f} "
                            f"testAUC={auc:.4f} testAP={apx:.4f} "
                            f"psAUC={rows[-1]['test_AUC_per_scene_mean']:.4f} "
                            f"({dt:.1f}s, {extra})"
                        )
            del d, Xt, R, bg_ok

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, f"results_bench_long_{args.tag}.csv"), index=False)
    agg = (
        res.groupby(["sensor", "model", "refs", "bands"])[
            ["test_AUC", "test_AP", "test_AUC_per_scene_mean", "val_AP", "seconds"]
        ]
        .agg(["mean", "std"])
        .round(4)
    )
    agg.to_csv(os.path.join(OUT, f"results_bench_summary_{args.tag}.csv"))
    print("\n=== mean +/- sd over seeds")
    print(agg.to_string())

    paired = []
    for sensor in sensors:
        for seed in seeds:
            for model in ("ridge", "hgb"):
                for bands_name in ("mars", "all"):
                    a = scene_auc.get((sensor, seed, model, 4, bands_name), {})
                    b = scene_auc.get((sensor, seed, model, 1, bands_name), {})
                    common = sorted(set(a) & set(b))
                    if len(common) > 2:
                        da = np.array([a[s] for s in common])
                        db = np.array([b[s] for s in common])
                        diff = da - db
                        t, p = stats.ttest_rel(da, db)
                        paired.append(
                            dict(
                                sensor=sensor, seed=seed, model=model, bands=bands_name,
                                comparison="r4_minus_r0", n_scenes=len(common),
                                mean_diff=round(float(diff.mean()), 4),
                                sd_diff=round(float(diff.std(ddof=1)), 4),
                                frac_positive=round(float((diff > 0).mean()), 3),
                                p_value=round(float(p), 4),
                            )
                        )
                for refs in (1, 4):
                    a = scene_auc.get((sensor, seed, model, refs, "all"), {})
                    b = scene_auc.get((sensor, seed, model, refs, "mars"), {})
                    common = sorted(set(a) & set(b))
                    if len(common) > 2:
                        da = np.array([a[s] for s in common])
                        db = np.array([b[s] for s in common])
                        diff = da - db
                        t, p = stats.ttest_rel(da, db)
                        paired.append(
                            dict(
                                sensor=sensor, seed=seed, model=model, bands=f"refs{refs}",
                                comparison="all_minus_mars", n_scenes=len(common),
                                mean_diff=round(float(diff.mean()), 4),
                                sd_diff=round(float(diff.std(ddof=1)), 4),
                                frac_positive=round(float((diff > 0).mean()), 3),
                                p_value=round(float(p), 4),
                            )
                        )
    if paired:
        pd.DataFrame(paired).to_csv(os.path.join(OUT, f"paired_scene_{args.tag}.csv"), index=False)
    print(f"\ntotal {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
