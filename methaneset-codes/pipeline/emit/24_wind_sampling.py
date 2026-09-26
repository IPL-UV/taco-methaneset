"""MODULE 24: HOW does IMEO compute its wind from ERA5? (Julio's question)

Julio (Aug 27): "you did not check whether IMEO's per-plume wind looks like
or how it is computed from ERA5. Is it averaged over the pixels it touches? or
how exactly".

Module 23 compared the published wind against the SCENE MEAN of our
layer: that measures whether they agree, not HOW they sample it. An EMIT scene measures
~76x74 km and ERA5-Land has ~9 km cells, so within a scene there are
several distinct cells: the sampling mode can be distinguished.

Three ways of getting ONE number from our wind.tif per plume are tested:
  (a) at the EMISSION POINT PIXEL (mask band 2)
  (b) averaged over the PLUME PIXELS (band 1)
  (c) averaged over the WHOLE SCENE  (what module 23 did)
and we look at which one best reproduces the wind each institution publishes. The
winner is, with good probability, the way they sample it.

Beware of what CANNOT be separated: our layer interpolates in time to the
minute of the scene and they might use the nearest hour; that leaves an
irreducible residual.

Outputs: console + assets/images/wind-sampling.png
Usage:
  /data/users/julio/.conda/envs/deep/bin/python code/v2/24_wind_sampling.py \
        > code/v2/24_wind_sampling.log 2>&1
"""
import json
import pathlib
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import rasterio
from scipy import stats

_theme = pathlib.Path(os.environ.get("METHANESET_THEME", "/data/users/julio/Notes/01-Projects/methanset/assets/slides/theme"))
if _theme.exists():
    sys.path.insert(0, str(_theme))
import figstyle  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(os.environ.get("METHANESET_ROOT", "/data/users/julio/Notes/01-Projects/methanset"))
IMG = ROOT / "assets" / "images"
PUB = ROOT / "assets" / "slides" / "public"
TACO = pathlib.Path("/data/databases/METHANSET_TACOS/methaneset-emit")
DATA = TACO / "DATA"


def scene_rows(row):
    gid = row["id"]
    iw = json.loads(row["meteo:imeo_wind"] or "{}")
    cw = json.loads(row["meteo:cm_wind"] or "{}")
    if not iw and not cw:
        return []
    icod = json.loads(row["detection:imeo_cod"] or "{}")
    ccod = json.loads(row["detection:cm_cod"] or "{}")
    try:
        with rasterio.open(DATA / gid / "wind.tif") as s:
            u, v = s.read(1), s.read(2)
    except Exception:
        return []
    spd = np.hypot(u, v)
    scene_mean = float(spd.mean())
    out = []
    for cat, wd, cod, fname in (("imeo", iw, icod, "plume_imeo.tif"),
                                ("cm", cw, ccod, "plume_cm.tif")):
        if not wd:
            continue
        with rasterio.open(DATA / gid / fname) as s:
            m1, m2 = s.read(1), s.read(2)
        for name, w in wd.items():
            if not w or not w.get("speed"):
                continue
            codes = cod.get(name) or []
            if not codes:
                continue
            at_pt = np.nan
            ys, xs = np.nonzero(m2 == codes[0])
            if len(ys):
                at_pt = float(spd[ys[0], xs[0]])
            mask = np.isin(m1, codes)
            over_plume = float(spd[mask].mean()) if mask.any() else np.nan
            out.append({"granule": gid, "catalog": cat, "plume": name,
                        "theirs": float(w["speed"]), "at_point": at_pt,
                        "over_plume": over_plume, "scene_mean": scene_mean})
    return out


def main():
    md = pd.read_parquet(TACO / "METADATA" / "level0.parquet")
    md = md[~md["selection:is_free"]]
    rows = md.to_dict("records")
    print(f"scenes: {len(rows)}")
    res = []
    with ProcessPoolExecutor(8) as ex:
        for i, part in enumerate(ex.map(scene_rows, rows, chunksize=8), 1):
            res.extend(part)
            if i % 200 == 0:
                print(f"  {i}/{len(rows)}")
    d = pd.DataFrame(res).dropna(subset=["at_point", "over_plume"])
    print(f"\nplumes with published wind and our own layer: {len(d)}")

    print(f"\n{'catalog':9s} {'sampling':12s} {'n':>6s} {'median |dif|':>14s} "
          f"{'median bias':>14s} {'r':>6s} {'<=0.5 m/s':>10s}")
    best = {}
    for cat in ("imeo", "cm"):
        g = d[d.catalog == cat]
        for how, col in (("emission point", "at_point"),
                         ("plume mean", "over_plume"),
                         ("scene mean", "scene_mean")):
            dif = (g[col] - g.theirs)
            r_ = stats.pearsonr(g[col], g.theirs)[0]
            print(f"{cat:9s} {how:12s} {len(g):6d} {dif.abs().median():14.3f} "
                  f"{dif.median():+14.3f} {r_:6.2f} {(dif.abs()<=0.5).mean():9.0%}")
            k = (cat, how)
            best[k] = (dif.abs().median(), r_)
        win = min([k for k in best if k[0] == cat], key=lambda k: best[k][0])
        print(f"  -> for {cat.upper()} the winner is: **{win[1]}** "
              f"(|dif| {best[win][0]:.3f} m/s, r={best[win][1]:.2f})")

    # how much does the wind vary WITHIN a scene? if it does not vary, the test
    # cannot tell anything apart and that must be said.
    var = d.groupby("granule").agg(pt=("at_point", "mean"),
                                   sc=("scene_mean", "mean"))
    spread = (var.pt - var.sc).abs()
    print(f"\ntest contrast: |point - scene mean| median "
          f"{spread.median():.2f} m/s · p90 {spread.quantile(0.9):.2f} m/s")
    print("(if this were ~0 the test would not distinguish anything; that is not the case)")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), constrained_layout=True)
    for ax, cat, col in ((axes[0], "imeo", figstyle.PALETTE["teal"]),
                         (axes[1], "cm", figstyle.PALETTE["orange"])):
        g = d[d.catalog == cat]
        labels, vals = [], []
        for how, c in (("at the\nemission point", "at_point"),
                       ("averaged over\nthe plume", "over_plume"),
                       ("averaged over\nthe whole scene", "scene_mean")):
            labels.append(how)
            vals.append((g[c] - g.theirs).abs().median())
        bars = ax.bar(labels, vals, color=col, width=0.6)
        bars[int(np.argmin(vals))].set_color(figstyle.PALETTE["slate"])
        for x, vv in enumerate(vals):
            ax.text(x, vv, f"{vv:.2f}", ha="center", va="bottom",
                    fontsize=11, fontweight="bold")
        ax.set_ylabel("median |our layer - their published wind|  [m/s]")
        ax.set_title(f"{cat.upper()} · n={len(g):,}", fontsize=12)
        ax.tick_params(labelsize=9)
    fig.suptitle("how each institution samples its wind: the sampling that "
                 "reproduces their number best", fontsize=12.5)
    figstyle.save(fig, IMG / "wind-sampling.png")
    shutil.copy2(IMG / "wind-sampling.png", PUB / "wind-sampling.png")
    print("\nfigure -> wind-sampling.png")


if __name__ == "__main__":
    main()
