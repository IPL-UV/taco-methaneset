"""MODULE 23: where does each institution's wind come from, and does it show?

Julio's questions (Aug 27): "what source does IMEO have? check whether what
IMEO says about the wind is correctly computed... what is the source of the
winds in CM? otherwise with what criterion could we make a choice".

WHAT THE DOCUMENTATION SAYS (verified Aug 27, 2026):
  - IMEO / MARS: 10 m wind from **ERA5-Land** (~9 km, hourly); GEOS-FP for
    offshore platforms.  (MARS data dictionary + IMEO technical doc)
  - Carbon Mapper: 10 m wind from **HRRR 3 km, 60 min** INSIDE the US and
    **ECMWF IFS 9 km** outside; forecast version in the quick-look.
    (Carbon Mapper emissions monitoring system, AMT 18, 6933, 2025)

TESTABLE PREDICTION: our layer is ERA5-Land, the SAME family as IMEO.
Hence our wind should resemble IMEO's more than CM's, and the
difference with CM should be LARGER in the US (where CM uses HRRR at 3 km, another
model and another resolution) than outside (where CM uses IFS at 9 km, comparable).
If that holds, the choice criterion stops being an opinion.

Also: how many 1:1 pairs are "on air" because the matched CM plume
has a twin? (new column detection:cm_dup_list).

Outputs: console + assets/images/wind-sources.png
Usage:
  /data/users/julio/.conda/envs/deep/bin/python code/v2/23_wind_sources.py \
        > code/v2/23_wind_sources.log 2>&1
"""
import json
import pathlib
import shutil
import sys

import numpy as np
import pandas as pd
from scipy import stats

_theme = pathlib.Path(os.environ.get("METHANESET_THEME", "/data/users/julio/Notes/01-Projects/methanset/assets/slides/theme"))
if _theme.exists():
    sys.path.insert(0, str(_theme))
import figstyle  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(os.environ.get("METHANESET_ROOT", "/data/users/julio/Notes/01-Projects/methanset"))
IMG = ROOT / "assets" / "images"
PUB = ROOT / "assets" / "slides" / "public"
CROSS = ROOT / "assets" / "data" / "cross" / "2026-08-25"
LEVEL0 = CROSS / "metadata_level0.parquet"
USA = "United States of America"


def main():
    md = pd.read_parquet(LEVEL0)

    # ---------- 1. our ERA5-Land vs each institution's wind ----------
    rows = []
    for r in md.to_dict("records"):
        ours = r["meteo:wind_speed"]
        if not ours or np.isnan(ours):
            continue
        for cat, col in (("imeo", "meteo:imeo_wind"), ("cm", "meteo:cm_wind")):
            for name, w in (json.loads(r[col]) or {}).items():
                if not w or not w.get("speed"):
                    continue
                rows.append({"granule": r["id"], "catalog": cat, "plume": name,
                             "theirs": w["speed"], "ours": ours,
                             "usa": r["site:country"] == USA})
    d = pd.DataFrame(rows)
    print("== 1. our ERA5-Land layer vs the wind each catalog publishes ==")
    print(f"{len(d)} plumes with the catalog's own wind\n")
    print(f"{'catalog':10s} {'scope':10s} {'n':>6s} {'median |dif|':>14s} "
          f"{'Pearson r':>10s} {'within 1 m/s':>13s}")
    summary = {}
    for cat in ("imeo", "cm"):
        for scope, sel in (("global", slice(None)), ("US", True), ("outside", False)):
            g = d[d.catalog == cat]
            if scope != "global":
                g = g[g.usa == sel]
            if len(g) < 20:
                continue
            dif = (g.theirs - g.ours).abs()
            r_ = stats.pearsonr(g.theirs, g.ours)[0]
            summary[(cat, scope)] = (len(g), dif.median(), r_, (dif <= 1).mean())
            print(f"{cat:10s} {scope:10s} {len(g):6d} {dif.median():14.2f} "
                  f"{r_:10.2f} {(dif<=1).mean():12.0%}")

    print("\nreading: our layer is ERA5-Land, the same family IMEO uses;")
    print("CM uses HRRR 3 km inside the US and IFS 9 km outside.")
    if ("cm", "US") in summary and ("cm", "outside") in summary:
        a, b = summary[("cm", "US")], summary[("cm", "outside")]
        print(f"-> CM deviates from our ERA5-Land more in the US "
              f"({a[1]:.2f} m/s) than outside ({b[1]:.2f} m/s): "
              f"{'CONSISTENT' if a[1] > b[1] else 'NOT consistent'} with its use of "
              f"HRRR at 3 km there.")

    # ---------- 2. 1:1 pairs whose CM plume has a twin ----------
    print("\n== 2. 1:1 pairs 'on air': the matched CM plume has a twin ==")
    tot = susp = 0
    per_scene = []
    for r in md.to_dict("records"):
        pairs = json.loads(r["match:pairs"])
        if not pairs:
            continue
        dups = json.loads(r.get("detection:cm_dup_list") or "[]")
        twins = set()
        for a, b, _ in dups:
            twins.add(a)
            twins.add(b)
        n = sum(1 for p in pairs if p["cm"] in twins)
        tot += len(pairs)
        susp += n
        if n:
            per_scene.append({"id": r["id"], "n_pairs": len(pairs),
                              "n_pairs_with_twin": n})
    print(f"total pairs: {tot} · with the CM plume flagged as twin: "
          f"{susp} ({susp/tot:.0%}) in {len(per_scene)} scenes")
    print("these are the pairs where CM's flux and area are a lower bound: "
          "part of its plume is archived under the other identity.")
    pd.DataFrame(per_scene).to_parquet(CROSS / "pairs_with_twin.parquet",
                                       index=False)

    # ---------- figure ----------
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), constrained_layout=True)
    ax = axes[0]
    for cat, col in (("imeo", figstyle.PALETTE["teal"]),
                     ("cm", figstyle.PALETTE["orange"])):
        g = d[d.catalog == cat]
        ax.plot(g.ours, g.theirs, ".", ms=3.5, alpha=0.28, color=col,
                label=f"{cat.upper()} (n={len(g):,})")
    lim = [0.05, max(d.theirs.max(), d.ours.max()) * 1.1]
    ax.plot(lim, lim, "-", color="#94a3b8", lw=1.2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("our ERA5-Land layer, scene mean [m/s]")
    ax.set_ylabel("wind published by the catalog [m/s]")
    ax.set_title("our reanalysis layer vs each institution's wind", fontsize=11.5)
    ax.legend(fontsize=9, markerscale=3)

    ax = axes[1]
    labels, vals, cols = [], [], []
    for cat, col in (("imeo", figstyle.PALETTE["teal"]),
                     ("cm", figstyle.PALETTE["orange"])):
        for scope in ("US", "outside"):
            if (cat, scope) in summary:
                labels.append(f"{cat.upper()}\n{'in US' if scope=='US' else 'outside US'}")
                vals.append(summary[(cat, scope)][1])
                cols.append(col)
    ax.bar(labels, vals, color=cols, width=0.6)
    for x, v in enumerate(vals):
        ax.text(x, v, f"{v:.2f}", ha="center", va="bottom", fontsize=11,
                fontweight="bold")
    ax.set_ylabel("median |their wind - our ERA5-Land|  [m/s]")
    ax.set_title("IMEO uses ERA5-Land like us; CM uses HRRR 3 km inside the US\n"
                 "and IFS 9 km outside, and it shows", fontsize=11.5)
    figstyle.save(fig, IMG / "wind-sources.png")
    shutil.copy2(IMG / "wind-sources.png", PUB / "wind-sources.png")
    print("\nfigure -> wind-sources.png")


if __name__ == "__main__":
    main()
