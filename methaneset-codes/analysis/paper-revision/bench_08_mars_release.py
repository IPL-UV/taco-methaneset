"""Extension 1: does the conclusion depend on what "MARS bands" means?

The original MARS-S2L release distributes only the six bands common to
Sentinel-2 and Landsat (blue, green, red, NIR, SWIR1, SWIR2; main.tex line
138: "only the six bands common to both sensors"). The previous benchmark used
the two detection SWIR bands only. This script compares three band sets:

  - swir2: S2 B11,B12 (0-based 11,12); L89 B6,B7 (0-based 5,6)  [previous MARS]
  - mars6: S2 B02,B03,B04,B08,B11,B12 (1,2,3,7,11,12);
           L89 B2,B3,B4,B5,B6,B7 (1,2,3,4,5,6)                   [release bands]
  - all:   all 13 S2 / 11 L89 bands

with refs = 1 (bg0) and 4 (bg0..bg3), ridge and HGB, scene split, 3 seeds.
Band names verified in the GeoTIFF descriptions of the TACOs (see RESULTS.md).

Outputs: results_marsrelease_long_<tag>.csv, scene_auc_marsrelease_<tag>.csv
Run: deep python bench_08_mars_release.py --sensors s2 --tag s2mr
"""

import argparse
import importlib.util
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))


def load_mod(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


b1 = load_mod("bench_01_nonlin_vs_ridge")
b3 = load_mod("bench_03_weak")

WORK = os.path.abspath(os.environ.get(
    "METHANESET_WORK", os.path.join(os.path.dirname(os.path.abspath(__file__)), "work")))
SRC = os.path.join(WORK, "ridge_multiref")
OUT = os.path.join(WORK, "bench_nonlin")
os.makedirs(OUT, exist_ok=True)
BAND_SETS = {
    "s2": {
        "swir2": [11, 12],
        "mars6": [1, 2, 3, 7, 11, 12],
        "all": list(range(13)),
    },
    "l89": {
        "swir2": [5, 6],
        "mars6": [1, 2, 3, 4, 5, 6],
        "all": list(range(11)),
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensors", type=str, default="s2,l89")
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--tag", type=str, default="marsrelease")
    args = ap.parse_args()
    sensors = [s for s in args.sensors.split(",") if s]
    seeds = [int(s) for s in args.seeds.split(",") if s != ""]

    rows, scene_rows = [], []
    t0 = time.time()
    for sensor in sensors:
        meta = pd.read_csv(os.path.join(SRC, f"samples_multiref_{sensor}.csv"))
        split_of = meta["split"].to_numpy()
        n_samples = len(meta)
        for seed in seeds:
            d = np.load(os.path.join(SRC, f"pix_multiref_{sensor}_s{seed}.npz"))
            Xt, R, bg_ok = d["Xt"], d["R"], d["bg_ok"].astype(bool)
            y, sid = d["y"].astype(np.int8), d["sid"]
            pix_split = split_of[sid]
            tr_s, va_s, te_s = pix_split == "train", pix_split == "validation", pix_split == "test"
            for refs in (1, 4):
                for bname, band_cols in BAND_SETS[sensor].items():
                    X, mask = b1.build_config(Xt, R, bg_ok, refs, band_cols)
                    Xn = b1.normalize(X, sid, n_samples, mask)
                    tr, va, te = tr_s & mask, va_s & mask, te_s & mask
                    t1 = time.perf_counter()
                    rr = b3.fit_ridge_full(Xn, y, tr, va)
                    dt_r = time.perf_counter() - t1
                    t1 = time.perf_counter()
                    rh = b3.fit_hgb_full(Xn, y, tr, va, seed, b1.HGB_PARAMS["max_iter"])
                    dt_h = time.perf_counter() - t1
                    for model, r, dt in (("ridge", rr, dt_r), ("hgb", rh, dt_h)):
                        pt = r["model"].decision_function(Xn[te]) if model == "ridge" else b3.hgb_scores_at(r, Xn, te)
                        auc = float(roc_auc_score(y[te], pt))
                        apx = float(average_precision_score(y[te], pt))
                        ps = b1.per_sample_scores(y[te], pt, sid[te])
                        for s_, a_ in ps.items():
                            scene_rows.append(dict(sensor=sensor, seed=seed, model=model, refs=refs,
                                                   band_set=bname, sid=s_, auc=a_, uuid=meta["uuid"].iloc[s_]))
                        rows.append(dict(sensor=sensor, seed=seed, model=model, refs=refs, band_set=bname,
                                         n_features=X.shape[1], val_AP=round(r["val_AP"], 4),
                                         C=r.get("C", np.nan), hgb_best_iter=r.get("n_iter", np.nan),
                                         test_AUC=round(auc, 4), test_AP=round(apx, 4),
                                         per_scene_AUC_mean=round(float(np.mean(list(ps.values()))), 4) if ps else np.nan,
                                         n_test_scenes=len(ps), n_test_px=int(te.sum()),
                                         test_pos_frac=round(float(y[te].mean()), 4), fit_seconds=round(dt, 1)))
                        print(f"  {sensor} s{seed} refs{refs} {bname:5s} {model:5s} "
                              f"AUC={auc:.4f} AP={apx:.4f} nf={X.shape[1]}")
            del d, Xt, R, bg_ok
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"results_marsrelease_long_{args.tag}.csv"), index=False)
    pd.DataFrame(scene_rows).to_csv(os.path.join(OUT, f"scene_auc_marsrelease_{args.tag}.csv"), index=False)
    print(f"total {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
