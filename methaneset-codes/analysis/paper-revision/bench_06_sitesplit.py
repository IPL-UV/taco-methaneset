"""Audit control 2: site-disjoint evaluation (site:location_name).

Reviewer's confounder: the scene-level split shares sites between train and
test (70 % of S2 test scenes and 60 % of L89 test scenes have a train scene
from the same site), so "extra bands/backgrounds help" could be site
memorization. Here the train/validation/test split is made by SITE: greedy
60/20/20 assignment of sites (by number of sampled scenes, fixed seed), all
scenes of a site in the same split. Train is unchanged in size but test sites
are never seen in training.

Configs: r0, r0_common, r4 x (mars, all) for ridge and HGB, per sensor,
3 pixel seeds, same models/C/early-stopping rules as bench_01 but on the
site-based validation.

Outputs: site_split_<tag>.csv, results_sitesplit_long_<tag>.csv,
scene_auc_sitesplit_<tag>.csv
Run: python bench_06_sitesplit.py --sensors s2 --tag s2site
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
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("bench01", os.path.join(HERE, "bench_01_nonlin_vs_ridge.py"))
b1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b1)
spec3 = importlib.util.spec_from_file_location("bench03", os.path.join(HERE, "bench_03_weak.py"))
b3 = importlib.util.module_from_spec(spec3)
spec3.loader.exec_module(b3)
spec5 = importlib.util.spec_from_file_location("bench05", os.path.join(HERE, "bench_05_r0common.py"))
b5 = importlib.util.module_from_spec(spec5)
spec5.loader.exec_module(b5)

WORK = os.path.abspath(os.environ.get(
    "METHANESET_WORK", os.path.join(os.path.dirname(os.path.abspath(__file__)), "work")))
SRC = os.path.join(WORK, "ridge_multiref")
OUT = os.path.join(WORK, "bench_nonlin")
os.makedirs(OUT, exist_ok=True)
FRACS = {"train": 0.6, "validation": 0.2, "test": 0.2}
SPLIT_SEED = 12345


def read_l0(sensor):
    frames = []
    paths = sorted(glob.glob(os.path.join(b3.BASES[sensor], "*.tacozip")))
    if not paths:
        raise SystemExit(f"no .tacozip under {b3.BASES[sensor]!r}; set METHANESET_DATA")
    for path in paths:
        z = zipfile.ZipFile(path)
        l0 = pd.read_parquet(io.BytesIO(z.read("METADATA/level0.parquet")))
        z.close()
        frames.append(l0[["id", "site:location_name", "detection:ch4_fluxrate"]].rename(
            columns={"id": "uuid", "site:location_name": "site", "detection:ch4_fluxrate": "flux"}))
    f = pd.concat(frames, ignore_index=True)
    f["site"] = f["site"].astype(str)
    f["flux"] = pd.to_numeric(f["flux"], errors="coerce")
    return f


def assign_sites(sizes, seed):
    """Greedy 60/20/20 by scene count; largest sites first, random tie-break."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"size": sizes, "tie": rng.random(len(sizes))})
    order = df.sort_values(["size", "tie"], ascending=[False, True]).index
    total = sizes.sum()
    cur = {k: 0 for k in FRACS}
    out = {}
    for site in order:
        k = max(FRACS, key=lambda s: FRACS[s] * total - cur[s])
        out[site] = k
        cur[k] += sizes[site]
    return pd.Series(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensors", type=str, default="s2,l89")
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--tag", type=str, default="sitesplit")
    args = ap.parse_args()
    sensors = [s for s in args.sensors.split(",") if s]
    seeds = [int(s) for s in args.seeds.split(",") if s != ""]

    rows, scene_rows, split_rows = [], [], []
    t0 = time.time()
    for sensor in sensors:
        meta = pd.read_csv(os.path.join(SRC, f"samples_multiref_{sensor}.csv"))
        l0 = read_l0(sensor)
        meta = meta.merge(l0[["uuid", "site", "flux"]], on="uuid", how="left")
        assert meta["site"].notna().all() and meta["flux"].notna().all()
        sizes = meta.groupby("site").size()
        assign = assign_sites(sizes, SPLIT_SEED)
        meta["site_split"] = meta["site"].map(assign)
        assert meta["site_split"].notna().all()
        print(f"\n=== {sensor}: {meta['site'].nunique()} sites, "
              f"{meta.groupby('site_split').size().to_dict()} scenes by site-split")
        for site, g in meta.groupby("site"):
            split_rows.append(dict(sensor=sensor, site=site, site_split=assign[site], n_scenes=len(g),
                                   n_scenes_lt1000=int((g["flux"] < 1000).sum()),
                                   flux_min=round(float(g["flux"].min()), 1), flux_max=round(float(g["flux"].max()), 1)))
        split_of = meta["site_split"].to_numpy()
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
            print(f"  seed {seed}: px tr/va/te = {tr_s.sum()}/{va_s.sum()}/{te_s.sum()}")
            for fname in ("r0", "r0_common", "r4"):
                for bands_name, band_cols in {"mars": b1.MARS_BANDS[sensor],
                                              "all": list(range(b1.SENSORS[sensor]))}.items():
                    if fname == "r0_common":
                        X, mask = b5.build_r0_common(Xt, R, bg_ok, band_cols)
                    else:
                        X, mask = b1.build_config(Xt, R, bg_ok, 1 if fname == "r0" else 4, band_cols)
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
                                scene_rows.append(dict(sensor=sensor, seed=seed, model=model, features=fname,
                                                       bands=bands_name, subset=sname, sid=s_, auc=a_,
                                                       uuid=meta["uuid"].iloc[s_], site=meta["site"].iloc[s_],
                                                       site_split=assign[meta["site"].iloc[s_]], flux=float(flux[s_])))
                            rows.append(dict(sensor=sensor, seed=seed, model=model, features=fname, bands=bands_name,
                                             subset=sname, n_features=X.shape[1], val_AP=round(r["val_AP"], 4),
                                             C=r.get("C", np.nan), hgb_best_iter=r.get("n_iter", np.nan),
                                             hgb_fit_iter=r.get("n_iter_fit", np.nan),
                                             test_AUC=round(auc, 4), test_AP=round(apx, 4),
                                             per_scene_AUC_mean=round(float(np.mean(list(ps.values()))), 4) if ps else np.nan,
                                             n_test_scenes=len(ps), n_train_px=int(tr.sum()), n_val_px=int(va.sum()),
                                             n_test_px=int(te.sum()), test_pos_frac=round(float(y[te].mean()), 4),
                                             fit_seconds=round(dt, 1)))
                            if sname == "full":
                                print(f"    {sensor} s{seed} {fname:9s} {bands_name:4s} {model:5s} "
                                      f"site-test: AUC={auc:.4f} AP={apx:.4f} scenes={len(ps)}")
            del d, Xt, R, bg_ok
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"results_sitesplit_long_{args.tag}.csv"), index=False)
    pd.DataFrame(scene_rows).to_csv(os.path.join(OUT, f"scene_auc_sitesplit_{args.tag}.csv"), index=False)
    pd.DataFrame(split_rows).to_csv(os.path.join(OUT, f"site_split_{args.tag}.csv"), index=False)
    print(f"total {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
