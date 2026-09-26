"""MODULE 07 of the v2 pipeline: the PROPOSED split (train/val/test 70/15/15).

Philosophy (Julio): everything is published; the split is a GRANULE-level
proposal. Hard rule: one same source (source_name) never sits in two splits.
Since a granule can have several sources and a source several granules,
CONNECTED COMPONENTS are built (granules linked by shared sources) and a whole
component is assigned to one split.

Assignment: components from largest to smallest, each one to the split whose
"need" is greatest, where need = size deficit + deficit in the marginals
(flux_bin, year, region, sector) that the component contributes.

The 25 free granules of v1 (IMEO test pool) are added at the end, distributed
17/4/4 balancing region.

Output: cross/<date>/splits.parquet (granule_ts, split, is_free)
        + verification: marginals per split and KS of flux per plume.

Usage:
  /data/users/julio/.conda/envs/deep/bin/python code/v2/07_propose_splits.py \
        > code/v2/07_propose_splits.log 2>&1
"""
import pathlib
import re

import numpy as np
import pandas as pd
from scipy import stats

DATA = pathlib.Path(__file__).resolve().parent.parent.parent / "assets" / "data"
TS = re.compile(r"EMIT_L1B_RAD_001_(\d{8}T\d{6})")
FRAC = {"train": 0.70, "val": 0.15, "test": 0.15}
AXES = ["flux_bin", "year", "region", "sector"]
TACO_L0 = pathlib.Path("/data/databases/METHANE_DATASETS_TACOv2/methaneset-emit/"
                       "methaneset-emit/METADATA/level0.parquet")


def components(sel, im):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    for ts, grp in im.groupby("granule_ts"):
        g = f"g:{ts}"
        for s in grp.source_name.dropna().unique():
            union(g, f"s:{s}")
    comp = {}
    for ts in sel.granule_ts:
        comp[ts] = find(f"g:{ts}")
    return pd.Series(comp, name="comp")


def main():
    cross_dir = sorted((DATA / "cross").iterdir())[-1]
    sel = pd.read_parquet(cross_dir / "selection.parquet")
    imeo_dir = sorted((DATA / "imeo").iterdir())[-1]
    im = pd.read_csv(imeo_dir / "unep_methanedata_detected_plumes.csv")
    im = im[im.satellite.str.contains("EMIT", na=False)].copy()
    im["granule_ts"] = im.tile.str.extract(TS)[0].str.lower()
    im = im[im.granule_ts.isin(set(sel.granule_ts))]

    sel = sel.set_index("granule_ts")
    sel["comp"] = components(sel.reset_index(), im)
    n_comp = sel.comp.nunique()
    sizes = sel.groupby("comp").size().sort_values(ascending=False)
    print(f"scenes: {len(sel)} · source-connected components: {n_comp} "
          f"· largest: {sizes.iloc[0]} scenes")

    targets = {sp: {ax: sel[ax].value_counts() * f for ax in AXES}
               for sp, f in FRAC.items()}
    size_t = {sp: len(sel) * f for sp, f in FRAC.items()}
    # PLUME-level targets (so that the flux KS closes per split)
    pl_t = {sp: {"n_weak": sel.n_weak.sum() * f, "n_strong": sel.n_strong.sum() * f}
            for sp, f in FRAC.items()}
    counts = {sp: {ax: {} for ax in AXES} for sp in FRAC}
    pl_c = {sp: {"n_weak": 0, "n_strong": 0} for sp in FRAC}
    size_c = {sp: 0 for sp in FRAC}
    assign = {}

    for comp_id in sizes.index:
        rows = sel[sel.comp == comp_id]

        def need(sp):
            n = (size_t[sp] - size_c[sp]) / size_t[sp]
            for ax in AXES:
                for cat, k in rows[ax].value_counts().items():
                    tgt = targets[sp][ax].get(cat, 0)
                    cur = counts[sp][ax].get(cat, 0)
                    if tgt > 0:
                        n += max((tgt - cur) / tgt, 0) * k / len(rows)
            # plume-level term, double weight: it is the one that closes the KS
            for key, col in (("n_weak", rows.n_weak), ("n_strong", rows.n_strong)):
                tgt = pl_t[sp][key]
                if tgt > 0 and col.sum() > 0:
                    n += 2.0 * max((tgt - pl_c[sp][key]) / tgt, 0)
            return n

        best = max(FRAC, key=need)
        assign[comp_id] = best
        size_c[best] += len(rows)
        pl_c[best]["n_weak"] += int(rows.n_weak.sum())
        pl_c[best]["n_strong"] += int(rows.n_strong.sum())
        for ax in AXES:
            for cat, k in rows[ax].value_counts().items():
                counts[best][ax][cat] = counts[best][ax].get(cat, 0) + k

    sel["split"] = sel.comp.map(assign)
    sel["is_free"] = False

    # ---- verification ----
    print("\n== size per split ==")
    print(sel.split.value_counts().to_string())
    for ax in AXES:
        tab = sel.groupby(["split", ax], observed=True).size().unstack(fill_value=0)
        print(f"\n{ax} (% per split):")
        print((tab.div(tab.sum(1), axis=0) * 100).round(0).astype(int).to_string())

    flux = im.merge(sel[["split"]], left_on="granule_ts", right_index=True)
    glob = flux.ch4_fluxrate.dropna()
    print("\n== KS of flux per plume, split vs global ==")
    for sp in FRAC:
        s = flux[flux.split == sp].ch4_fluxrate.dropna()
        ks, p = stats.ks_2samp(s, glob)
        w = (s < 1000).mean() * 100
        print(f"  {sp:5s}: n={len(s):4d} plumes · weak {w:.0f}% · KS={ks:.03f} (p={p:.2f})")

    # no-leak sources (by construction; verify)
    src_split = flux.groupby("source_name").split.nunique()
    print(f"\nsources in more than one split: {(src_split > 1).sum()} (must be 0)")

    # ---- v1 frees (with exclusions decided on Aug 26) ----
    # 3 with nodata in the radiance (25%/82%/93% valid) and 1 "stale
    # negative" (20240817t095623 gained an IMEO plume in the fresh snapshot).
    FREE_EXCL = {"20240621t095845", "20240821t095244", "20240228t074038",
                 "20240817t095623"}
    l0 = pd.read_parquet(TACO_L0, columns=["id", "detection:n_imeo", "detection:n_cm"])
    frees = l0[(l0["detection:n_imeo"] == 0) & (l0["detection:n_cm"] == 0)].id.tolist()
    ts_free = [TS.search(i).group(1).lower() for i in frees]
    ts_free = [t for t in ts_free if t not in FREE_EXCL]
    fr = pd.DataFrame({"granule_ts": ts_free})
    fr["split"] = (["train"] * 13 + ["val"] * 4 + ["test"] * 4)[:len(fr)]
    fr["is_free"] = True
    print(f"\nv1 frees added: {len(fr)} (13 train / 4 val / 4 test) · "
          f"excluded {len(FREE_EXCL)}")

    out = pd.concat([sel.reset_index()[["granule_ts", "split", "is_free"]], fr])
    out.to_parquet(cross_dir / "splits.parquet", index=False)
    print(f"total with frees: {len(out)} -> {cross_dir / 'splits.parquet'}")


if __name__ == "__main__":
    main()
