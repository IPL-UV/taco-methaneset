"""Per-scene paired bootstrap for the six-band comparison (review).

Reads the per-scene AUC table written by bench_08_mars_release.py and the
site of every finetune sample, then bootstraps the mean per-scene delta with
clusters by site, the same estimator used in bench_07_bootstrap.py.
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get("METHANESET_WORK", os.path.join(HERE, "work"))
OUT = os.path.join(WORK, "bench_nonlin")
SRC = os.path.join(WORK, "ridge_multiref")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


b7 = load("bench07", os.path.join(HERE, "bench_07_bootstrap.py"))
b6 = load("bench06", os.path.join(HERE, "bench_06_sitesplit.py"))

tag = sys.argv[1] if len(sys.argv) > 1 else "mr"
sc = pd.read_csv(os.path.join(OUT, f"scene_auc_marsrelease_{tag}.csv"))
rows = []
for sensor in ("s2", "l89"):
    sub = sc[(sc.sensor == sensor) & (sc.model == "hgb")]
    for refs in (1, 4):
        s = sub[sub.refs == refs]
        a = s[s.band_set == "all"].groupby("sid")["auc"].mean()
        b = s[s.band_set == "mars6"].groupby("sid")["auc"].mean()
        common = a.index.intersection(b.index)
        if len(common) == 0:
            print(f"{sensor} refs{refs}: no rows, skipped")
            continue
        delta = a.loc[common].to_numpy() - b.loc[common].to_numpy()
        meta = b7.scene_sites(sensor)
        site = np.array([str(meta["site"].iloc[int(i)]) for i in common])
        res = b7.boot_ci(delta, site)
        res.update(sensor=sensor, refs=refs,
                   auc_all=round(float(a.loc[common].mean()), 4),
                   auc_mars6=round(float(b.loc[common].mean()), 4))
        rows.append(res)
        print(f"{sensor} refs{refs}: mars6 {res['auc_mars6']:.4f} -> all {res['auc_all']:.4f}  "
              f"delta {res['mean_delta']:+.4f} [{res['ci_lo']:+.4f}, {res['ci_hi']:+.4f}]  "
              f"scenes {res['n_scenes']} sites {res['n_sites']}")

pd.DataFrame(rows).to_csv(os.path.join(OUT, f"bootstrap_marsrelease_{tag}.csv"), index=False)
