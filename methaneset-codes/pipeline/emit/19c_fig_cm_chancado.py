"""MODULE 19c: figure of the CM overlap (two overlapping plumes, new slide).

Julio's idea: show TWO CM plume records of the SAME scene,
attributed to DIFFERENT SOURCES, drawn with transparency: they overlap
(high IoU), it is the same plume with two identities. "CM bug that stays
there: that is why IMEO is the reference" (for sectors and as the flux baseline).

Geometries: plume-outline.geojson from the CM STAC items (public,
equivalence with the alpha of plume.tif verified on Aug 26, IoU 1.000).
Candidates: manifest + sources from the 2026-08-25 snapshot (the workshop
tifs were already deleted; the STAC avoids re-downloading them).

Output: assets/images/cm-chancado.png (+ copy to assets/slides/public/)
Usage:
  /data/users/julio/.conda/envs/deep/bin/python code/v2/19c_fig_cm_chancado.py \
        > code/v2/19c_fig_cm_chancado.log 2>&1
"""
import itertools
import pathlib
import shutil
import sys

import numpy as np
import pandas as pd
import requests
from shapely import make_valid
from shapely.geometry import shape

_theme = pathlib.Path(os.environ.get("METHANESET_THEME", "/data/users/julio/Notes/01-Projects/methanset/assets/slides/theme"))
if _theme.exists():
    sys.path.insert(0, str(_theme))
import figstyle  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(os.environ.get("METHANESET_ROOT", "/data/users/julio/Notes/01-Projects/methanset"))
IMG = ROOT / "assets" / "images"
PUB = ROOT / "assets" / "slides" / "public"
CM_DIR = ROOT / "assets" / "data" / "carbonmapper" / "2026-08-25"
STAC = "https://api.carbonmapper.org/api/v1/stac/search"

MAX_GRANULES = 60  # how many candidate pairs to test against the STAC


def outline(plume_id, session):
    """plume-outline.geojson from this plume's STAC item (key WITH extension).

    Fallbacks for old items: plume.geojson (mfa-v1).
    """
    r = session.post(STAC, json={"ids": [plume_id], "limit": 10}, timeout=60)
    r.raise_for_status()
    feats = r.json().get("features", [])
    feats.sort(key=lambda it: it.get("collection", ""), reverse=True)
    for it in feats:
        assets = it.get("assets", {})
        asset = assets.get("plume-outline.geojson") or assets.get("plume.geojson")
        if asset:
            g = session.get(asset["href"], timeout=60)
            g.raise_for_status()
            gj = g.json()
            geoms = [make_valid(shape(f["geometry"])) for f in gj.get("features", [gj])]
            from shapely.ops import unary_union
            return unary_union(geoms)
    return None


def main():
    man = pd.read_parquet(CM_DIR / "manifest.parquet")
    src = pd.read_parquet(CM_DIR / "sources.parquet")
    man = man.merge(src[["plume_id", "source_name"]], on="plume_id", how="left")
    man = man.dropna(subset=["source_name"])

    # candidates: same scene, different sources, points within < 2 km (~0.02 deg)
    cands = []
    for ts, grp in man.groupby("granule_ts"):
        if grp.source_name.nunique() < 2:
            continue
        for a, b in itertools.combinations(grp.itertuples(), 2):
            if a.source_name == b.source_name:
                continue
            d = np.hypot(a.lon - b.lon, a.lat - b.lat)
            if d < 0.02:
                cands.append((d, ts, a, b))
    cands.sort(key=lambda t: t[0])
    print(f"candidate pairs (different sources, points <0.02 deg): {len(cands)}")

    ses = requests.Session()
    # we prefer a VISIBLE pair (high IoU but < 1: two shifted silhouettes
    # can be seen); if there are only perfect clones, the best clone is used.
    best, best_vis = None, None
    for d, ts, a, b in cands[:MAX_GRANULES]:
        ga, gb = outline(a.plume_id, ses), outline(b.plume_id, ses)
        if ga is None or gb is None or not ga.intersects(gb):
            continue
        iou = ga.intersection(gb).area / ga.union(gb).area
        print(f"{a.plume_id} vs {b.plume_id} · dist {d:.4f} deg · IoU {iou:.2f}")
        if best is None or iou > best[0]:
            best = (iou, ts, a, b, ga, gb)
        if 0.5 <= iou <= 0.97 and (best_vis is None or iou > best_vis[0]):
            best_vis = (iou, ts, a, b, ga, gb)
        if best_vis is not None and best_vis[0] >= 0.8:
            break
    best = best_vis or best
    if best is None:
        sys.exit("no overlapping pair found: increase MAX_GRANULES")

    iou, ts, a, b, ga, gb = best
    print(f"\nCHOSEN: {a.plume_id} ({a.source_name}) vs "
          f"{b.plume_id} ({b.source_name}) · IoU {iou:.2f} · scene {ts}")

    # the bottom one (orange) has denser fill and a solid edge; the top one
    # (teal) is VERY transparent with a dashed edge: both silhouettes show.
    # Legend OUTSIDE the image (bottom), Julio's request.
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    for g, col, ls, alpha, lw in ((ga, figstyle.PALETTE["orange"], "-", 0.55, 2.0),
                                  (gb, figstyle.PALETTE["teal"], "--", 0.25, 2.0)):
        for gg in getattr(g, "geoms", [g]):
            x, y = gg.exterior.xy
            ax.fill(x, y, alpha=alpha, fc=col, ec="none")
            ax.plot(x, y, color=col, lw=lw, ls=ls)
    ax.plot([a.lon, b.lon], [a.lat, b.lat], "o", ms=7, mfc="#fde047",
            mec="#a16207", mew=1.4, ls="none")
    handles = [
        Line2D([], [], color=figstyle.PALETTE["orange"], lw=3,
               label=f"{a.plume_id} · source at {a.lon:.4f}, {a.lat:.4f}"),
        Line2D([], [], color=figstyle.PALETTE["teal"], lw=3, ls="--",
               label=f"{b.plume_id} · source at {b.lon:.4f}, {b.lat:.4f}"),
        Line2D([], [], marker="o", ls="none", mfc="#fde047", mec="#a16207",
               ms=8, label="reported source points, 300 m apart"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=1, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    ax.set_title(f"Two CM plume records, two sources, one plume · IoU {iou:.2f}\n"
                 f"EMIT scene {ts}", fontsize=12)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_aspect("equal")
    fig.subplots_adjust(bottom=0.24)
    figstyle.save(fig, IMG / "cm-chancado.png")
    shutil.copy2(IMG / "cm-chancado.png", PUB / "cm-chancado.png")
    print("figure -> cm-chancado.png (images and slides/public)")


if __name__ == "__main__":
    main()
