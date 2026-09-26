"""The Turkmenistan plume (TKM_S_037): HF mask versus the portal polygon,
on top of the scene's real mf as a base.

It is the only loose match (IoU 0.12) of the 14 inheritances from script 41.
Output: assets/images/tkm-iou-portal-vs-hf.png
"""
import pathlib
import sys

import geopandas as gpd
import numpy as np
import rasterio
from shapely import make_valid

sys.path.insert(0, "/data/users/julio/Notes/01-Projects/methanset/assets/slides/theme")
import figstyle  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

SCENE = "EMIT_L1B_RAD_001_20240817T082107_2423006_021"
DATA = pathlib.Path("/data/databases/METHANE_DATASETS_TACOv2/methaneset-emit/"
                    "methaneset-emit/DATA") / SCENE
NEW = pathlib.Path("/data/users/julio/methanset/data")
OUT = pathlib.Path("/data/users/julio/Notes/01-Projects/methanset/assets/images/"
                   "tkm-iou-portal-vs-hf.png")
MARGIN = 0.012  # degrees around the union of both polygons


def main():
    hf = gpd.read_parquet(NEW / "carbonmapper_new/imeo_polygons_new.parquet")
    hf = hf[hf.scene_key == "20240817t082107"].iloc[0]
    g_hf = make_valid(hf.geometry)

    plumes = gpd.read_file(NEW / "unep_methanedata_detected_plumes.geojson")
    cand = plumes[plumes.geometry.intersects(g_hf)].copy()
    inter = cand.geometry.apply(lambda g: make_valid(g).intersection(g_hf).area)
    union = cand.geometry.apply(lambda g: make_valid(g).union(g_hf).area)
    cand["iou"] = inter / union
    best = cand.sort_values("iou", ascending=False).iloc[0]
    g_portal = make_valid(best.geometry)
    iou = best.iou

    srcs = gpd.read_file(NEW / "unep_methanedata_detected_sources.geojson")
    src = srcs[srcs.source_name == hf.matched_source].iloc[0]

    minx = min(g_hf.bounds[0], g_portal.bounds[0]) - MARGIN
    miny = min(g_hf.bounds[1], g_portal.bounds[1]) - MARGIN
    maxx = max(g_hf.bounds[2], g_portal.bounds[2]) + MARGIN
    maxy = max(g_hf.bounds[3], g_portal.bounds[3]) + MARGIN

    with rasterio.open(DATA / "latlon.tif") as s:
        lat, lon = s.read(1), s.read(2)
    with rasterio.open(DATA / "mf.tif") as s:
        mf = s.read(1)
    sel = (lon >= minx) & (lon <= maxx) & (lat >= miny) & (lat <= maxy) & (mf > -9000)

    figstyle.apply()
    fig, ax = plt.subplots(figsize=(7.6, 6.2), constrained_layout=True)
    v = mf[sel]
    vmax = np.percentile(v, 99.5)
    ax.scatter(lon[sel], lat[sel], c=np.clip(v, 0, vmax), s=4, marker="s",
               cmap="Greys", vmax=vmax, rasterized=True)

    for geom, color, label in [(g_hf, figstyle.PALETTE["teal"], "HF mask (our label)"),
                               (g_portal, figstyle.PALETTE["orange"], "portal polygon")]:
        polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
        first = True
        for p in polys:
            x, y = p.exterior.xy
            ax.plot(x, y, color=color, lw=2.2, label=label if first else None)
            ax.fill(x, y, color=color, alpha=0.14)
            first = False

    ax.plot(src.geometry.x, src.geometry.y, marker="*", ms=17, mec="#a16207",
            mfc=figstyle.PALETTE["yellow"], ls="none",
            label=f"source {hf.matched_source}")
    ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy)
    ax.set_aspect(1 / np.cos(np.radians((miny + maxy) / 2)))
    ax.set_xlabel("longitude [deg]"); ax.set_ylabel("latitude [deg]")
    ax.grid(False)
    ax.legend(loc="upper left", fontsize=10)
    ax.text(0.98, 0.02, f"IoU = {iou:.2f}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=13, fontweight="bold",
            color=figstyle.PALETTE["red"])
    figstyle.save(fig, OUT)
    print("IoU recomputed:", round(float(iou), 3), "->", OUT)


if __name__ == "__main__":
    main()
