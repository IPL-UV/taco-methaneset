"""MODULE 06 of the v2 pipeline: the balanced selector (marginals, not Cartesian).

Decision (Aug 25, v2 of the selector after rethinking it with Julio): N = 700
in WEAK-MAX mode: all 428 has_weak scenes enter (the universe only has 864
weak plumes and none is left out); the remaining 272 are chosen among the
strong_only ones, balancing year/region/sector and preferring scenes with few
plumes (so as not to dilute the weak share). It achieves ~36% weak plumes
(universe: 23%) and 61% of scenes with weak. The previous version (scene 50/50,
29% weak plumes) is kept as selection_scene5050.parquet.
The balance is of MARGINALS: each scene is chosen by minimizing the
simultaneous deficit of four target distributions, without requiring every
exact combination to be filled (the 122-cell Cartesian fragments, see
module 05):

  flux    : 350 has_weak / 350 strong_only  (physical limit: 428 has_weak)
  year    : 2022 best-effort (all that fit), rest uniform
  region  : uniform across the 6 majors; Sub-Sahara/Other best-effort
  sector  : Oil&Gas ~45% / Waste ~45% / Coal ~10% (the real universe proportion)

Greedy: at each step the scene whose contribution reduces the total deficit
most enters (sum of normalized deficits of its categories). Ties: larger
n_plumes.

Output: cross/<date>/selection.parquet (the 700 with all their axes)
        + summary of achieved marginals vs target on console.

Usage:
  /data/users/julio/.conda/envs/deep/bin/python code/v2/06_select_balanced.py \
        > code/v2/06_select_balanced.log 2>&1
"""
import pathlib

import numpy as np
import pandas as pd

DATA = pathlib.Path(__file__).resolve().parent.parent.parent / "assets" / "data"
N = 700

FIX_REGION = {
    "Iran (Islamic Republic of)": "Middle East",
    "Venezuela (Bolivarian Republic of)": "Latin America",
    "Bolivia (Plurinational State of)": "Latin America",
    "Syrian Arab Republic": "Middle East",
    "Russian Federation": "Central Asia",
    "Türkiye": "Middle East", "Turkey": "Middle East",
}


def build_targets(df):
    t = {}
    n_hw = int((df.flux_bin == "has_weak").sum())
    t["flux_bin"] = {"has_weak": n_hw, "strong_only": N - n_hw}
    avail22 = int((df.year == "2022").sum())
    rest = (N - avail22) / 3
    t["year"] = {"2022": avail22, "2023": rest, "2024": rest, "2025": rest}
    majors = ["North America", "South/East Asia", "North Africa",
              "Central Asia", "Middle East", "Latin America"]
    minor = {"Sub-Saharan Africa": int((df.region == "Sub-Saharan Africa").sum()),
             "Other": int(N * 0.06)}
    per = (N - sum(minor.values())) / len(majors)
    t["region"] = {**{m: per for m in majors}, **minor}
    t["sector"] = {"Oil and Gas": N * 0.45, "Waste": N * 0.45,
                   "Coal": N * 0.09, "Other": N * 0.01}
    return t


def main():
    cross_dir = sorted((DATA / "cross").iterdir())[-1]
    df = pd.read_parquet(cross_dir / "candidates.parquet")
    df["region"] = df.apply(
        lambda r: FIX_REGION.get(r.country, r.region), axis=1)
    print(f"clean candidates: {len(df):,}")

    targets = build_targets(df)
    axes = list(targets)
    counts = {ax: {k: 0 for k in targets[ax]} for ax in axes}

    def deficit_gain(row):
        gain = 0.0
        for ax in axes:
            cat = row[ax]
            tgt = targets[ax].get(cat, 0)
            if tgt <= 0:
                continue
            d = (tgt - counts[ax][cat]) / tgt
            gain += max(d, 0)
        return gain

    # WEAK-MAX: force all has_weak and spread the rest among strong_only
    forced = df[df.flux_bin == "has_weak"]
    chosen = [row for _, row in forced.iterrows()]
    for ax in axes:
        for _, row in forced.iterrows():
            if row[ax] in counts[ax]:
                counts[ax][row[ax]] += 1
    pool = df[df.flux_bin == "strong_only"].copy()
    print(f"forced has_weak: {len(forced)} · strong_only left to pick: {N - len(forced)}")
    for step in range(N - len(forced)):
        gains = pool.apply(deficit_gain, axis=1)
        top = gains[gains == gains.max()].index
        # tie: FEWER strong plumes, so as not to dilute the weak share
        pick = pool.loc[top].sort_values("n_plumes", ascending=True).index[0]
        row = pool.loc[pick]
        chosen.append(row)
        for ax in axes:
            if row[ax] in counts[ax]:
                counts[ax][row[ax]] += 1
        pool = pool.drop(index=pick)
        if (step + 1) % 100 == 0:
            print(f"  {step+1}/{N}", flush=True)

    sel = pd.DataFrame(chosen)
    out = cross_dir / "selection.parquet"
    sel.to_parquet(out, index=False)

    print(f"\n== achieved marginals (target in parentheses) ==")
    for ax in axes:
        print(f"\n{ax}:")
        for k, v in sorted(counts[ax].items(), key=lambda x: -x[1]):
            print(f"   {k:22s} {v:4d}  ({targets[ax].get(k, 0):.0f})")
    print(f"\ntotal plumes in the selection: {int(sel.n_plumes.sum()):,} · "
          f"weak: {int(sel.n_weak.sum()):,} · strong: {int(sel.n_strong.sum()):,}")
    print(f"countries: {sel.country.nunique()} · distinct sources: "
          f"{int(sel.n_sources.sum()):,}")
    print(f"\nselection -> {out}")


if __name__ == "__main__":
    main()
