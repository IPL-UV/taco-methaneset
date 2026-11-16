"""Audit control 1: r0 restricted to the 4-ref-valid pixel set (r0_common).

Reviewer's confounder: the r4 - r0 gain mixes the reference effect with the
pixel-set effect (r4 can only use pixels where all 4 backgrounds are valid,
which in L89 drops ~2 % of pixels and may change the test composition).
This script reproduces refs=1 on exactly the bg_ok.all(axis=1) pixels, with
the z-score computed on that same mask, for ridge and HGB, per sensor. Same
scene split, pixels, seeds and models as bench_01; evaluation on the full
test and on the weak (<1000/<2000 kg/h) subsets for the bootstrap analysis.

Outputs: results_r0common_long_<tag>.csv, scene_auc_r0common_<tag>.csv
Run: python bench_05_r0common.py --sensors s2 --tag s2c
"""

import argparse
import importlib.util
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("bench01", os.path.join(HERE, "bench_01_nonlin_vs_ridge.py"))
b1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b1)
spec3 = importlib.util.spec_from_file_location("bench03", os.path.join(HERE, "bench_03_weak.py"))
b3 = importlib.util.module_from_spec(spec3)
spec3.loader.exec_module(b3)

WORK = os.path.abspath(os.environ.get(
    "METHANESET_WORK", os.path.join(os.path.dirname(os.path.abspath(__file__)), "work")))
SRC = os.path.join(WORK, "ridge_multiref")
OUT = os.path.join(WORK, "bench_nonlin")
os.makedirs(OUT, exist_ok=True)


def build_r0_common(Xt, R, bg_ok, band_cols):
    """refs=1 features on the bg_ok.all pixel set (same width as r0)."""
    B = Xt.shape[1]
    mask = bg_ok.all(axis=1)
    band_of_col = np.arange(B)
    cols = np.flatnonzero(np.isin(band_of_col, band_cols))
    return R[:, 0, :][:, cols], mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensors", type=str, default="s2,l89")
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--tag", type=str, default="r0common")
    args = ap.parse_args()
    sensors = [s for s in args.sensors.split(",") if s]
    seeds = [int(s) for s in args.seeds.split(",") if s != ""]

    rows, scene_rows = [], []
    t0 = time.time()
    for sensor in sensors:
        meta = pd.read_csv(os.path.join(SRC, f"samples_multiref_{sensor}.csv"))
        flux_df = b3.read_flux(sensor)
        meta = meta.merge(flux_df[["uuid", "flux"]], on="uuid", how="left")
        assert meta["flux"].notna().all()
        split_of = meta["split"].to_numpy()
        flux = meta["flux"].to_numpy()
        n_samples = len(meta)
        for seed in seeds:
            d = np.load(os.path.join(SRC, f"pix_multiref_{sensor}_s{seed}.npz"))
            Xt, R, bg_ok = d["Xt"], d["R"], d["bg_ok"].astype(bool)
            y, sid = d["y"].astype(np.int8), d["sid"]
            sub_masks = {"full": np.ones(len(y), dtype=bool)}
            sub_masks.update({t: flux[sid] < thr for t, thr in b3.THRESHOLDS})
            pix_split = split_of[sid]
            tr_s, va_s, te_s = pix_split == "train", pix_split == "validation", pix_split == "test"
            for bands_name, band_cols in {"mars": b1.MARS_BANDS[sensor], "all": list(range(b1.SENSORS[sensor]))}.items():
                X, mask = build_r0_common(Xt, R, bg_ok, band_cols)
                Xn = b1.normalize(X, sid, n_samples, mask)
                tr, va = tr_s & mask, va_s & mask
                t1 = time.perf_counter()
                rr = b3.fit_ridge_full(Xn, y, tr, va)
                dt_r = time.perf_counter() - t1
                t1 = time.perf_counter()
                rh = b3.fit_hgb_full(Xn, y, tr, va, seed, b1.HGB_PARAMS["max_iter"])
                dt_h = time.perf_counter() - t1
                for model, r, dt in (("ridge", rr, dt_r), ("hgb", rh, dt_h)):
                    for sname in ("full", "lt1000", "lt2000"):
                        te = te_s & mask & sub_masks[sname]
                        if te.sum() == 0 or y[te].sum() == 0 or y[te].sum() == te.sum():
                            continue
                        if model == "ridge":
                            pt = r["model"].decision_function(Xn[te])
                        else:
                            pt = b3.hgb_scores_at(r, Xn, te)
                        auc = float(roc_auc_score(y[te], pt))
                        apx = float(average_precision_score(y[te], pt))
                        ps = b1.per_sample_scores(y[te], pt, sid[te])
                        for s_, a_ in ps.items():
                            scene_rows.append(dict(sensor=sensor, seed=seed, model=model, bands=bands_name,
                                                   subset=sname, sid=s_, auc=a_, uuid=meta["uuid"].iloc[s_],
                                                   flux=float(flux[s_])))
                        rows.append(dict(sensor=sensor, seed=seed, model=model, bands=bands_name, refs="1_common",
                                         subset=sname, n_features=X.shape[1], val_AP=round(r["val_AP"], 4),
                                         C=r.get("C", np.nan), hgb_best_iter=r.get("n_iter", np.nan),
                                         hgb_fit_iter=r.get("n_iter_fit", np.nan),
                                         test_AUC=round(auc, 4), test_AP=round(apx, 4),
                                         per_scene_AUC_mean=round(float(np.mean(list(ps.values()))), 4) if ps else np.nan,
                                         n_test_scenes=len(ps), n_train_px=int(tr.sum()), n_val_px=int(va.sum()),
                                         n_test_px=int(te.sum()), test_pos_frac=round(float(y[te].mean()), 4),
                                         fit_seconds=round(dt, 1)))
                        if sname == "full":
                            print(f"  {sensor} s{seed} r0_common {bands_name:4s} {model:5s} "
                                  f"full: AUC={auc:.4f} AP={apx:.4f} px={int(te.sum())}")
            del d, Xt, R, bg_ok
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"results_r0common_long_{args.tag}.csv"), index=False)
    pd.DataFrame(scene_rows).to_csv(os.path.join(OUT, f"scene_auc_r0common_{args.tag}.csv"), index=False)
    print(f"total {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
