"""MODULE 19: the four CM-IMEO analyses requested by Julio (Aug 26).

  1. SECTOR AGREEMENT: for matched pairs, cross table
     IMEO sector (names) vs CM sector (mapped IPCC code). Do they agree?
  2. FLUX vs PIXEL SIZE, BOTH catalogs: now that the CM masks
     exist, this pending item is unblocked. Pixels per source from band 1
     (base code of each source) and flux from the dicts.
  3. WHEN DOES CM HAVE NO FLUX?: profile of CM sources without emission_auto
     (are they the small ones?): median pixels with vs without flux, and by sector.
  4. HOW MUCH DO CM PLUMES OVERLAP?: overlap between CM plume records
     within the same scene (pre-dissolve), same source and different
     sources: IoU distribution and % with high overlap.

Outputs: console + assets/images/flux-vs-pixels-ambos.png
                  + assets/images/cm-solapes.png
Usage:
  /data/users/julio/.conda/envs/deep/bin/python code/v2/19_cm_imeo_analyses.py \
        > code/v2/19_cm_imeo_analyses.log 2>&1
"""
import json
import pathlib
import re
import sys
from collections import Counter
from itertools import combinations

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes as rio_shapes
from scipy import stats
from shapely import make_valid
from shapely.geometry import shape
from shapely.ops import unary_union

_theme = pathlib.Path(os.environ.get("METHANESET_THEME", "/data/users/julio/Notes/01-Projects/methanset/assets/slides/theme"))
if _theme.exists():
    sys.path.insert(0, str(_theme))
import figstyle  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

DATA = pathlib.Path(__file__).resolve().parent.parent.parent / "assets" / "data"
IMG = pathlib.Path(__file__).resolve().parent.parent.parent / "assets" / "images"
ROOT = pathlib.Path("/data/databases/METHANSET_TACOS/emit")
CM_DIR = pathlib.Path("/data/databases/METHANSET_TACOS/cm_plume_tifs")
TS_G = re.compile(r"EMIT_L1B_RAD_001_(\d{8}T\d{6})")
def cm_sector_name(code):
    """CM IPCC map -> IMEO vocabulary, prefix-robust (1B1a, etc.)."""
    c = (code or "").strip()
    for pref, name in (("1B1", "Coal"), ("1B2", "Oil and Gas"),
                       ("1A", "Energy"), ("6", "Waste"), ("4B", "Livestock")):
        if c.startswith(pref):
            return name
    return "Other" if c else "Unknown"


def px_by_code(mask_path):
    with rasterio.open(mask_path) as s:
        b = s.read(1)
    vals, counts = np.unique(b[b > 0], return_counts=True)
    return dict(zip(vals.astype(int), counts.astype(int)))


def vec(path):
    try:
        with rasterio.open(path) as src:
            if src.count < 4:
                return None
            a = src.read(4)
            m = (a == 255).astype(np.uint8)
            if m.sum() == 0:
                return None
            polys = [shape(g) for g, v in rio_shapes(
                m, mask=m.astype(bool), transform=src.transform) if v == 1]
            g = unary_union(polys)
            return make_valid(gpd.GeoSeries([g], crs=src.crs).to_crs(4326).iloc[0])
    except Exception:
        return None


def main():
    md = pd.read_parquet(sorted((DATA / "cross").iterdir())[-1] / "metadata_level0.parquet")
    md = md[~md["selection:is_free"]]
    im_csv = pd.read_csv(sorted((DATA / "imeo").iterdir())[-1] /
                         "unep_methanedata_detected_plumes.csv")
    im_csv = im_csv[im_csv.satellite.str.contains("EMIT", na=False)]
    src_sector = im_csv.groupby("source_name").sector.first().to_dict()
    man = pd.read_parquet(CM_DIR / "manifest.parquet").merge(
        pd.read_parquet(CM_DIR / "sources.parquet"), on="plume_id", how="left")
    man["source_name"] = man.source_name.fillna(man.plume_id)

    # ---------- 1. sector agreement on the matched pairs ----------
    cross = Counter()
    for d in md.to_dict("records"):
        pairs = json.loads(d.get("match:pairs") or "[]")
        cm_secs = {}
        pc = man[man.granule_ts == TS_G.search(d["id"]).group(1).lower()]
        for pid, grp_sec in pc.groupby("source_name").sector.first().items():
            cm_secs[pid] = cm_sector_name(grp_sec)
        # fallback: source without a row (plume_id as name)
        for p in pairs:
            si = src_sector.get(p["imeo"], "Unknown")
            sc = cm_secs.get(p["cm"], "Unknown")
            cross[(si, sc)] += 1
    tab = pd.Series(cross).unstack(fill_value=0)
    print("== 1. IMEO SECTOR (rows) vs CM SECTOR (columns), on matched pairs ==")
    print(tab.to_string())
    total = tab.values.sum()
    agree = sum(tab.loc[i, c] for i in tab.index for c in tab.columns if i == c)
    print(f"diagonal agreement: {agree}/{total} ({agree/total:.0%})\n")

    # ---------- 2. flux vs pixels, both catalogs ----------
    rows_i, rows_c = [], []
    cm_px_flux = []
    for d in md.to_dict("records"):
        scene_dir = ROOT / d["id"]
        cod_i = json.loads(d["detection:imeo_cod"] or "{}")
        cod_c = json.loads(d["detection:cm_cod"] or "{}")
        fl_i = json.loads(d["detection:imeo_flux"] or "{}")
        fl_c = json.loads(d["detection:cm_flux"] or "{}")
        px_i = px_by_code(scene_dir / "plume_imeo.tif")
        px_c = px_by_code(scene_dir / "plume_cm.tif")
        for srcn, codes in cod_i.items():
            f, p = fl_i.get(srcn), px_i.get(codes[0], 0)
            if f and p:
                rows_i.append((p, f))
        for srcn, codes in cod_c.items():
            f, p = fl_c.get(srcn), px_c.get(codes[0], 0)
            if p:
                cm_px_flux.append((p, f))
            if f and p:
                rows_c.append((p, f))
    for name, rows in (("IMEO", rows_i), ("CM", rows_c)):
        a = np.array(rows, float)
        rho, _ = stats.spearmanr(np.log10(a[:, 0]), np.log10(a[:, 1]))
        print(f"== 2. {name}: {len(a)} sources with flux and pixels · "
              f"Spearman log-log r={rho:.2f} · median px "
              f"{np.median(a[:, 0]):.0f} · median flux {np.median(a[:, 1]):.0f}")

    figstyle.apply()
    fig, axes = plt.subplots(1, 2, figsize=figstyle.FIGSIZE_WIDE,
                             constrained_layout=True, sharey=True)
    for ax, (name, rows, color) in zip(axes, [
            ("IMEO", rows_i, figstyle.PALETTE["teal"]),
            ("Carbon Mapper", rows_c, figstyle.PALETTE["orange"])]):
        a = np.array(rows, float)
        rho, _ = stats.spearmanr(np.log10(a[:, 0]), np.log10(a[:, 1]))
        ax.scatter(a[:, 0], a[:, 1], s=10, alpha=0.4, color=color,
                   edgecolors="none", rasterized=True)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("mask size [EMIT pixels]")
        ax.set_title(f"{name} (n={len(a):,} · Spearman r={rho:.2f})", fontsize=12)
    axes[0].set_ylabel("flux rate [kg/h]")
    figstyle.save(fig, IMG / "flux-vs-pixels-ambos.png")

    # ---------- 3. profile of CM sources without flux ----------
    a = [(p, f) for p, f in cm_px_flux]
    con = np.array([p for p, f in a if f])
    sin = np.array([p for p, f in a if not f])
    print(f"\n== 3. CM without flux: {len(sin)} of {len(a)} sources "
          f"({len(sin)/len(a):.0%}) ==")
    print(f"median pixels WITH flux: {np.median(con):.0f} · "
          f"WITHOUT flux: {np.median(sin):.0f}")
    u, p_ = stats.mannwhitneyu(con, sin)
    print(f"Mann-Whitney p={p_:.2e} -> {'those without flux ARE smaller' if np.median(sin)<np.median(con) and p_<0.01 else 'no clear difference'}")

    # ---------- 4. overlaps between CM plume records ----------
    print("\n== 4. overlaps between CM PLUMES of the same scene (pre-dissolve) ==")
    man2 = man
    ious_same, ious_diff = [], []
    n_pl = 0
    for ts, grp in man2.groupby("granule_ts"):
        geoms = {}
        for r in grp.itertuples():
            g = vec(CM_DIR / f"{r.plume_id}.tif")
            if g is not None:
                geoms[r.plume_id] = (g, r.source_name)
        n_pl += len(geoms)
        for (pa, (ga, sa)), (pb, (gb, sb)) in combinations(geoms.items(), 2):
            if not ga.intersects(gb):
                continue
            it = ga.intersection(gb).area
            un = ga.union(gb).area
            iou = it / un if un else 0
            (ious_same if sa == sb else ious_diff).append(iou)
    for name, arr in (("same source", ious_same), ("different sources", ious_diff)):
        a = np.array(arr)
        if len(a):
            print(f"{name}: {len(a)} touching pairs · median IoU "
                  f"{np.median(a):.2f} · >0.5: {(a>0.5).sum()} · >0.9: {(a>0.9).sum()}")
        else:
            print(f"{name}: 0 pairs")
    print(f"CM plumes analyzed: {n_pl}")

    fig, ax = plt.subplots(figsize=figstyle.FIGSIZE_HALF, constrained_layout=True)
    ax.hist([ious_same, ious_diff], bins=20, stacked=False,
            color=[figstyle.PALETTE["orange"], figstyle.PALETTE["slate"]],
            label=[f"same source ({len(ious_same)})",
                   f"different sources ({len(ious_diff)})"])
    ax.set_xlabel("IoU between CM plume records (same scene)")
    ax.set_ylabel("overlapping pairs")
    ax.legend(fontsize=9)
    figstyle.save(fig, IMG / "cm-solapes.png")
    print("\nfigures -> flux-vs-pixels-ambos.png · cm-solapes.png")


if __name__ == "__main__":
    main()
