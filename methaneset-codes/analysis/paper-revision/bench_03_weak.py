"""Weak-plume extension of the nonlinear benchmark (HGB vs ridge).

Same pixels/split/protocol as bench_01_nonlin_vs_ridge.py, with a new
evaluation regime: plumes with detection:ch4_fluxrate < threshold (1000 kg/h,
plus 2000 kg/h as sensitivity). Training is unchanged: full train scenes, full
validation for C (ridge) and early stopping (HGB); only the TEST pixels are
restricted to weak-plume scenes ("subgroup" mode of $METHANESET_WORK/bench_weak).

Ridge is re-run in the same mode so the comparison against the previously
measured numbers is apples to apples; the HGB is the same fixed configuration
as bench_01 (documented there).

Run: python bench_03_weak.py
     [--sensors s2,l89] [--seeds 0,1,2] [--tag weakfull]

Outputs: results_weak_nonlin_long_<tag>.csv, scene_auc_weak_<tag>.csv,
weak_scene_counts_<tag>.csv, flux_check_<tag>.csv
"""

import argparse
import glob
import importlib.util
import io
import os
import time
import zipfile

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("bench01", os.path.join(HERE, "bench_01_nonlin_vs_ridge.py"))
b1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b1)

WORK = os.path.abspath(os.environ.get(
    "METHANESET_WORK", os.path.join(os.path.dirname(os.path.abspath(__file__)), "work")))
SRC = os.path.join(WORK, "ridge_multiref")
WEAKREF = os.path.join(WORK, "bench_weak")
OUT = os.path.join(WORK, "bench_nonlin")
os.makedirs(OUT, exist_ok=True)
DATA = os.environ.get("METHANESET_DATA", "")
BASES = {
    "s2": os.path.join(DATA, "methaneset-s2-finetune"),
    "l89": os.path.join(DATA, "methaneset-l89-finetune"),
}
THRESHOLDS = [("lt1000", 1000.0), ("lt2000", 2000.0)]
SUBSETS = ["full", "lt1000", "lt2000"]


def read_flux(sensor):
    """Scene uuid -> detection:ch4_fluxrate (kg/h) from level0 of each tacozip."""
    frames = []
    for path in sorted(glob.glob(os.path.join(BASES[sensor], "*.tacozip"))):
        z = zipfile.ZipFile(path)
        l0 = pd.read_parquet(io.BytesIO(z.read("METADATA/level0.parquet")))
        z.close()
        flux_col = [c for c in l0.columns if "flux" in c.lower()][0]
        frames.append(l0[["id", "split", flux_col]].rename(columns={"id": "uuid", flux_col: "flux"}))
    f = pd.concat(frames, ignore_index=True)
    f["flux"] = pd.to_numeric(f["flux"], errors="coerce")
    return f


def fit_ridge_full(Xn, y, tr, va):
    best = None
    for c in b1.C_GRID:
        clf = RidgeClassifier(alpha=1.0 / c, solver="cholesky")
        clf.fit(Xn[tr], y[tr])
        ap = average_precision_score(y[va], clf.decision_function(Xn[va]))
        if best is None or ap > best[0]:
            best = (ap, c)
    _, c = best
    clf = RidgeClassifier(alpha=1.0 / c, solver="cholesky")
    clf.fit(Xn[tr], y[tr])
    return dict(val_AP=float(best[0]), C=c, model=clf)


def fit_hgb_full(Xn, y, tr, va, seed, max_iter):
    params = dict(b1.HGB_PARAMS)
    params["max_iter"] = max_iter
    params["random_state"] = seed
    hg = HistGradientBoostingClassifier(**params)
    hg.fit(Xn[tr], y[tr])
    best_ap, best_stage, no_improve, last = -np.inf, None, 0, 0
    for it, raw in enumerate(hg.staged_decision_function(Xn[va]), start=1):
        last = it
        if it % b1.HGB_CHECK == 0 or it == 1:
            ap = average_precision_score(y[va], raw)
            if ap > best_ap + b1.HGB_MIN_DELTA:
                best_ap, best_stage, no_improve = float(ap), it, 0
            else:
                no_improve += b1.HGB_CHECK
                if no_improve >= b1.HGB_PATIENCE * b1.HGB_CHECK:
                    break
    if best_stage is None:
        best_stage = last
    return dict(val_AP=best_ap, n_iter=best_stage, n_iter_fit=last, model=hg)


def hgb_scores_at(r, Xn, mask):
    for it, raw in enumerate(r["model"].staged_decision_function(Xn[mask]), start=1):
        if it == r["n_iter"]:
            return raw
    raise RuntimeError("stage not reached")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensors", type=str, default="s2,l89")
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--max-iter", type=int, default=b1.HGB_PARAMS["max_iter"])
    ap.add_argument("--tag", type=str, default="weakfull")
    args = ap.parse_args()
    sensors = [s for s in args.sensors.split(",") if s]
    seeds = [int(s) for s in args.seeds.split(",") if s != ""]

    rows, scene_rows, count_rows, check_rows = [], [], [], []
    t0 = time.time()
    for sensor in sensors:
        meta = pd.read_csv(os.path.join(SRC, f"samples_multiref_{sensor}.csv"))
        flux_df = read_flux(sensor)
        meta = meta.merge(flux_df[["uuid", "flux"]], on="uuid", how="left")
        assert meta["flux"].notna().all(), f"{sensor}: missing flux for sampled scenes"
        split_of = meta["split"].to_numpy()
        flux = meta["flux"].to_numpy()
        n_samples = len(meta)

        # cross-check against the previous weak benchmark (read-only)
        prev = pd.read_csv(os.path.join(WEAKREF, f"sampled_flux_{sensor}.csv"))
        m = meta[["uuid", "flux"]].merge(prev[["uuid", "flux"]], on="uuid", suffixes=("", "_prev"))
        ok = np.allclose(m["flux"].to_numpy(), m["flux_prev"].to_numpy(), rtol=1e-6, atol=1e-3)
        check_rows.append(dict(sensor=sensor, n=len(m), flux_match=bool(ok),
                               max_abs_diff=float(np.max(np.abs(m["flux"] - m["flux_prev"])))))
        for tname, thr in THRESHOLDS:
            w = flux < thr
            for sp in ("train", "validation", "test"):
                count_rows.append(dict(sensor=sensor, threshold=tname, split=sp,
                                       n_scenes=int(((split_of == sp) & w).sum()),
                                       flux_min=round(float(flux[(split_of == sp) & w].min()), 1) if ((split_of == sp) & w).any() else np.nan,
                                       flux_max=round(float(flux[(split_of == sp) & w].max()), 1) if ((split_of == sp) & w).any() else np.nan))
            print(f"\n{sensor} {tname}: weak scenes by split "
                  f"{pd.Series(split_of[w]).value_counts().to_dict()} "
                  f"(flux {flux[w].min():.0f}-{flux[w].max():.0f} kg/h)")

        for seed in seeds:
            d = np.load(os.path.join(SRC, f"pix_multiref_{sensor}_s{seed}.npz"))
            Xt, R, bg_ok = d["Xt"], d["R"], d["bg_ok"].astype(bool)
            y, sid = d["y"].astype(np.int8), d["sid"]
            pix_split = split_of[sid]
            pix_flux = flux[sid]
            tr_s, va_s, te_s = pix_split == "train", pix_split == "validation", pix_split == "test"
            sub_masks = {"full": np.ones(len(y), dtype=bool)}
            sub_masks.update({t: pix_flux < thr for t, thr in THRESHOLDS})
            print(f"\n=== {sensor} seed {seed}: full px {len(y)}, weak px "
                  f"{ {t: int(m.sum()) for t, m in sub_masks.items() if t != 'full'} }")
            for refs in (1, 4):
                for bands_name, band_cols in {"mars": b1.MARS_BANDS[sensor], "all": list(range(b1.SENSORS[sensor]))}.items():
                    X, mask = b1.build_config(Xt, R, bg_ok, refs, band_cols)
                    Xn = b1.normalize(X, sid, n_samples, mask)
                    tr, va = tr_s & mask, va_s & mask
                    models = {}
                    t1 = time.perf_counter()
                    models["ridge"] = fit_ridge_full(Xn, y, tr, va)
                    dt_r = time.perf_counter() - t1
                    t1 = time.perf_counter()
                    models["hgb"] = fit_hgb_full(Xn, y, tr, va, seed, args.max_iter)
                    dt_h = time.perf_counter() - t1
                    for model in ("ridge", "hgb"):
                        r = models[model]
                        dt = dt_r if model == "ridge" else dt_h
                        for sname in SUBSETS:
                            te = te_s & mask & sub_masks[sname]
                            if te.sum() == 0 or y[te].sum() == 0 or y[te].sum() == te.sum():
                                print(f"  SKIP {sensor} s{seed} {sname} {model} refs{refs} {bands_name}: unusable")
                                continue
                            if model == "ridge":
                                pt = r["model"].decision_function(Xn[te])
                            else:
                                pt = hgb_scores_at(r, Xn, te)
                            auc = float(roc_auc_score(y[te], pt))
                            apx = float(average_precision_score(y[te], pt))
                            ps = b1.per_sample_scores(y[te], pt, sid[te])
                            for s_, a_ in ps.items():
                                scene_rows.append(dict(sensor=sensor, seed=seed, model=model, refs=refs,
                                                       bands=bands_name, subset=sname, sid=s_, auc=a_,
                                                       uuid=meta["uuid"].iloc[s_], flux=float(flux[s_])))
                            rows.append(dict(
                                sensor=sensor, seed=seed, model=model, refs=refs, bands=bands_name,
                                subset=sname, n_features=X.shape[1], val_AP=round(r["val_AP"], 4),
                                C=r.get("C", np.nan), hgb_best_iter=r.get("n_iter", np.nan),
                                hgb_fit_iter=r.get("n_iter_fit", np.nan),
                                test_AUC=round(auc, 4), test_AP=round(apx, 4),
                                per_scene_AUC_mean=round(float(np.mean(list(ps.values()))), 4) if ps else np.nan,
                                per_scene_AUC_sd=round(float(np.std(list(ps.values()))), 4) if ps else np.nan,
                                n_test_scenes=len(ps), n_train_px=int(tr.sum()), n_val_px=int(va.sum()),
                                n_test_px=int(te.sum()), test_pos_frac=round(float(y[te].mean()), 4),
                                fit_seconds=round(dt, 1)))
                            if sname == "lt1000":
                                extra = f"it={r['n_iter']}" if model == "hgb" else f"C={r['C']:g}"
                                print(f"    {model:5s} refs{refs} {bands_name:4s} lt1000: "
                                      f"AUC={auc:.4f} AP={apx:.4f} scenes={len(ps)} "
                                      f"px={int(te.sum())} pos={y[te].mean():.3f} ({extra})")
            del d, Xt, R, bg_ok

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, f"results_weak_nonlin_long_{args.tag}.csv"), index=False)
    pd.DataFrame(scene_rows).to_csv(os.path.join(OUT, f"scene_auc_weak_{args.tag}.csv"), index=False)
    pd.DataFrame(count_rows).to_csv(os.path.join(OUT, f"weak_scene_counts_{args.tag}.csv"), index=False)
    chk = pd.DataFrame(check_rows)
    chk.to_csv(os.path.join(OUT, f"flux_check_{args.tag}.csv"), index=False)
    print("\n=== flux check vs bench_weak/sampled_flux_*.csv")
    print(chk.to_string(index=False))
    print(f"\ntotal {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
