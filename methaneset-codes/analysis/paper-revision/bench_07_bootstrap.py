"""Audit control 3: cluster bootstrap by site instead of per-seed t-tests.

The old paired t-tests used the three pixel seeds as replicates, but the seeds
resample pixels within the SAME scenes, so they are not independent. This
script replaces that inference with a cluster bootstrap over sites
(site:location_name): for each comparison it takes the per-scene AUC delta
(mean over the 3 seeds), resamples SITES with replacement B = 2000 times and
reports the percentile 95 % CI of the mean per-scene delta and the fraction of
bootstrap means above 0.

It also produces the seen-vs-unseen-site decomposition inside the original
scene split (test scenes whose site appears in the train split vs not) and the
r0_common decomposition: r4 - r0 (total), r0_common - r0 (pixel-set effect),
r4 - r0_common (reference effect).

Declared comparisons (no multiplicity correction beyond declaration):
  - family "main" (scene split, full test): 2 models x 2 sensors x 4 gaps = 16
  - family "weak1000" (same, test < 1000 kg/h): 16
  - family "sitedisjoint" (same, site split, unseen test sites): 16
  - family "seen_sites"/"unseen_sites" (scene split, full test): 32 (2 labels)
The paper claim rests on the 8 HGB gaps of "main" (primary); the rest are
secondary and reported with CIs only.

Outputs: bootstrap_main.csv, bootstrap_weak1000.csv, bootstrap_sitedisjoint.csv,
bootstrap_sitenovelty.csv, bootstrap_r0common.csv, bootstrap_primary.csv
"""

import importlib.util
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
spec6 = importlib.util.spec_from_file_location("bench06", os.path.join(HERE, "bench_06_sitesplit.py"))
b6 = importlib.util.module_from_spec(spec6)
spec6.loader.exec_module(b6)

WORK = os.path.abspath(os.environ.get(
    "METHANESET_WORK", os.path.join(os.path.dirname(os.path.abspath(__file__)), "work")))
SRC = os.path.join(WORK, "ridge_multiref")
OUT = os.path.join(WORK, "bench_nonlin")
os.makedirs(OUT, exist_ok=True)
B = 2000
BOOT_SEED = 20260926

GAPS = [
    ("all - mars", "r4|all", "r4|mars"),
    ("all - mars", "r0|all", "r0|mars"),
    ("4refs - 1ref", "r4|all", "r0|all"),
    ("4refs - 1ref", "r4|mars", "r0|mars"),
]


def boot_ci(delta, site, n_boot=B, seed=BOOT_SEED):
    """Cluster bootstrap over sites of the mean per-scene delta."""
    delta = np.asarray(delta, dtype=float)
    site = np.asarray(site)
    sites = np.unique(site)
    clusters = [np.flatnonzero(site == s) for s in sites]
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(clusters), len(clusters))
        idx = np.concatenate([clusters[i] for i in pick])
        draws[b] = delta[idx].mean()
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return dict(mean_delta=round(float(delta.mean()), 4), ci_lo=round(float(lo), 4),
                ci_hi=round(float(hi), 4), frac_gt0=round(float((draws > 0).mean()), 3),
                n_scenes=len(delta), n_sites=len(sites))


def load_scene_frames():
    frames = []
    for t in ("s2weak", "l89weak"):
        path = os.path.join(OUT, f"scene_auc_weak_{t}.csv")
        if not os.path.exists(path):
            print(f"  skip weak-plume scenes: {os.path.basename(path)} not found")
            continue
        d = pd.read_csv(path)
        d["features"] = np.where(d["refs"].astype(int) == 1, "r0", "r4")
        d["source"] = "scene"
        frames.append(d)
    for t in ("s2c", "l89c"):
        path = os.path.join(OUT, f"scene_auc_r0common_{t}.csv")
        if not os.path.exists(path):
            print(f"  skip single-reference control: {os.path.basename(path)} not found")
            continue
        d = pd.read_csv(path)
        d["features"] = "r0_common"
        d["source"] = "scene"
        frames.append(d)
    for t in ("s2site", "l89site"):
        path = os.path.join(OUT, f"scene_auc_sitesplit_{t}.csv")
        if not os.path.exists(path):
            print(f"  skip site split: {os.path.basename(path)} not found")
            continue
        d = pd.read_csv(path)
        d["source"] = "site"
        frames.append(d)
    sc = pd.concat(frames, ignore_index=True)
    sc["config"] = sc["features"] + "|" + sc["bands"]
    return sc


def scene_sites(sensor):
    """sid -> site for the scene-split scene list (sids are meta row indices)."""
    meta = pd.read_csv(os.path.join(SRC, f"samples_multiref_{sensor}.csv"))
    l0 = b6.read_l0(sensor)
    meta = meta.merge(l0[["uuid", "site"]], on="uuid", how="left")
    assert meta["site"].notna().all()
    tr_sites = set(meta.loc[meta["split"] == "train", "site"])
    meta["site_seen"] = meta["site"].isin(tr_sites)
    return meta


def delta_arrays(sc, sensor, subset, model, cfg_a, cfg_b, source):
    """Per-scene AUC delta (mean over seeds) and the scene ids; site codes for source='site'."""
    sub = sc[(sc.sensor == sensor) & (sc["subset"] == subset) & (sc.model == model) & (sc["source"] == source)]
    a = sub[sub.config == cfg_a].groupby("sid")["auc"].mean()
    b = sub[sub.config == cfg_b].groupby("sid")["auc"].mean()
    common = a.index.intersection(b.index)
    if len(common) < 3:
        return None
    delta = a.loc[common].to_numpy() - b.loc[common].to_numpy()
    sites = None
    if source == "site":
        sites = sub.drop_duplicates("sid").set_index("sid")["site"].loc[common].to_numpy()
    return delta, np.array(common), sites


def main():
    sc = load_scene_frames()
    site_of, seen_of = {}, {}
    for s in ("s2", "l89"):
        m = scene_sites(s)
        site_of[s] = {int(i): str(m["site"].iloc[i]) for i in m.index}
        seen_of[s] = {int(i): bool(m["site_seen"].iloc[i]) for i in m.index}

    out = []
    for sensor in ("s2", "l89"):
        for model in ("ridge", "hgb"):
            for gname, cfg_a, cfg_b in GAPS:
                for subset, family, source in (("full", "main", "scene"), ("lt1000", "weak1000", "scene"),
                                               ("full", "sitedisjoint", "site")):
                    da = delta_arrays(sc, sensor, subset, model, cfg_a, cfg_b, source)
                    if da is None:
                        continue
                    d, sids, sites_codes = da
                    site = sites_codes if source == "site" else np.array([site_of[sensor][int(s)] for s in sids])
                    res = boot_ci(d, site)
                    res.update(family=family, sensor=sensor, model=model, gap=gname,
                               cfg_a=cfg_a, cfg_b=cfg_b)
                    out.append(res)

    # seen vs unseen sites within the scene split (full test)
    for sensor in ("s2", "l89"):
        for model in ("ridge", "hgb"):
            for gname, cfg_a, cfg_b in GAPS:
                da = delta_arrays(sc, sensor, "full", model, cfg_a, cfg_b, "scene")
                if da is None:
                    continue
                d, sids, _ = da
                site = np.array([site_of[sensor][int(s)] for s in sids])
                seen = np.array([seen_of[sensor][int(s)] for s in sids])
                for label, mask in (("seen_sites", seen), ("unseen_sites", ~seen)):
                    if mask.sum() < 5:
                        continue
                    res = boot_ci(d[mask], site[mask])
                    res.update(family=label, sensor=sensor, model=model, gap=gname,
                               cfg_a=cfg_a, cfg_b=cfg_b)
                    out.append(res)

    if not out:
        print("no comparison could be built: run bench_05 and bench_06 first "
              "(tags s2c/l89c and s2site/l89site)")
        return
    res_df = pd.DataFrame(out)
    for fam, fname in [("main", "bootstrap_main.csv"), ("weak1000", "bootstrap_weak1000.csv"),
                       ("sitedisjoint", "bootstrap_sitedisjoint.csv")]:
        res_df[res_df.family == fam].to_csv(os.path.join(OUT, fname), index=False)
    res_df[res_df.family.str.contains("sites")].to_csv(os.path.join(OUT, "bootstrap_sitenovelty.csv"), index=False)
    print("=== cluster bootstrap (site), B =", B)
    print(res_df[["family", "sensor", "model", "gap", "mean_delta", "ci_lo", "ci_hi", "frac_gt0",
                  "n_scenes", "n_sites"]].to_string(index=False))

    # r0_common decomposition (scene split, full test and weak)
    dec = []
    for sensor in ("s2", "l89"):
        for model in ("ridge", "hgb"):
            for bands in ("all", "mars"):
                for subset in ("full", "lt1000"):
                    for label, cfg_a, cfg_b in (
                        ("total r4-r0", f"r4|{bands}", f"r0|{bands}"),
                        ("pixels r0c-r0", f"r0_common|{bands}", f"r0|{bands}"),
                        ("refs r4-r0c", f"r4|{bands}", f"r0_common|{bands}"),
                    ):
                        da = delta_arrays(sc, sensor, subset, model, cfg_a, cfg_b, "scene")
                        if da is None:
                            continue
                        d, sids, _ = da
                        site = np.array([site_of[sensor][int(s)] for s in sids])
                        res = boot_ci(d, site)
                        res.update(sensor=sensor, model=model, bands=bands, subset=subset, term=label,
                                   cfg_a=cfg_a, cfg_b=cfg_b)
                        dec.append(res)
    dec_df = pd.DataFrame(dec)
    dec_df.to_csv(os.path.join(OUT, "bootstrap_r0common.csv"), index=False)
    if len(dec_df):
        print("\n=== r0_common decomposition (scene split)")
        print(dec_df[["subset", "sensor", "model", "bands", "term", "mean_delta", "ci_lo", "ci_hi",
                      "frac_gt0", "n_scenes", "n_sites"]].to_string(index=False))

    # declared primary comparisons: HGB, main family, all 8
    prim = res_df[(res_df.family == "main") & (res_df.model == "hgb")]
    prim.to_csv(os.path.join(OUT, "bootstrap_primary.csv"), index=False)


if __name__ == "__main__":
    main()
