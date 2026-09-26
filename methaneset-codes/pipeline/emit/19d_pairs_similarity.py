"""MODULE 19d: how similar are the matched IMEO-CM plumes?

Julio's request (Aug 26): "the plumes that are 'the same' according to the
Hungarian analysis, how similar are they in flux, area, IoU, distance between
emitters?". Pair-a-pair analysis on the PUBLISHED dataset: everything comes
from the TACO tifs themselves (masks + latlon) and the level0 table: it is
reproducible by any user of the dataset.

Per matched pair (base code shared in plume_imeo/plume_cm):
  - flux_imeo vs flux_cm (kg/h, from the level0 dicts)
  - area_imeo vs area_cm (pixels of their codes in each file)
  - iou_px (intersection/union pixel by pixel between both files)
  - dist_m (haversine between the emitter pixels of band 2, via latlon)

Outputs: assets/data/cross/2026-08-25/pairs_similarity.parquet
         assets/images/pairs-similarity.png (4 panels, + copy to slides)
Usage:
  /data/users/julio/.conda/envs/deep/bin/python code/v2/19d_pairs_similarity.py \
        > code/v2/19d_pairs_similarity.log 2>&1
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
CROSS = ROOT / "assets" / "data" / "cross" / "2026-08-25"
DATA = pathlib.Path("/data/databases/METHANSET_TACOS/methaneset-emit/DATA")
IMG = ROOT / "assets" / "images"
PUB = ROOT / "assets" / "slides" / "public"
LEVEL0 = pathlib.Path("/data/databases/METHANSET_TACOS/methaneset-emit/METADATA/level0.parquet")


def haversine_m(lo1, la1, lo2, la2):
    r = 6371000.0
    p1, p2 = np.radians(la1), np.radians(la2)
    dp, dl = np.radians(la2 - la1), np.radians(lo2 - lo1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def scene_pairs(row):
    gid, pairs = row["id"], json.loads(row["match:pairs"])
    if not pairs:
        return []
    icod = json.loads(row["detection:imeo_cod"])
    ccod = json.loads(row["detection:cm_cod"])
    iflx = json.loads(row["detection:imeo_flux"])
    cflx = json.loads(row["detection:cm_flux"])
    d = DATA / gid
    with rasterio.open(d / "plume_imeo.tif") as s:
        mi, si = s.read(1), s.read(2)
    with rasterio.open(d / "plume_cm.tif") as s:
        mc, sc = s.read(1), s.read(2)
    with rasterio.open(d / "latlon.tif") as s:
        lat, lon = s.read(1), s.read(2)
    out = []
    for pr in pairs:
        ci, cc = icod.get(pr["imeo"], []), ccod.get(pr["cm"], [])
        base = sorted(set(ci) & set(cc))
        if not base:
            continue
        a = np.isin(mi, ci)
        b = np.isin(mc, cc)
        inter = int((a & b).sum())
        union = int((a | b).sum())
        # emitter: band 2 pixel with the base code
        pi = np.argwhere(si == base[0])
        pc = np.argwhere(sc == base[0])
        dist = np.nan
        if len(pi) and len(pc):
            (yi, xi), (yc, xc) = pi[0], pc[0]
            dist = float(haversine_m(lon[yi, xi], lat[yi, xi],
                                     lon[yc, xc], lat[yc, xc]))
        out.append({
            "granule": gid, "imeo_id": pr["imeo"], "cm_id": pr["cm"],
            "flux_imeo": iflx.get(pr["imeo"]), "flux_cm": cflx.get(pr["cm"]),
            "area_imeo_px": int(a.sum()), "area_cm_px": int(b.sum()),
            "iou_px": inter / union if union else 0.0, "dist_m": dist,
        })
    return out


def main():
    md = pd.read_parquet(LEVEL0)
    md = md[md["match:n_pairs"] > 0]
    rows = md.to_dict("records")
    print(f"scenes with pairs: {len(rows)}")
    res = []
    with ProcessPoolExecutor(8) as ex:
        for i, part in enumerate(ex.map(scene_pairs, rows, chunksize=8), 1):
            res.extend(part)
            if i % 100 == 0:
                print(f"  {i}/{len(rows)}")
    df = pd.DataFrame(res)
    df.to_parquet(CROSS / "pairs_similarity.parquet")
    print(f"\npairs with shared code: {len(df)}")

    both = df.dropna(subset=["flux_imeo", "flux_cm"])
    both = both[(both.flux_imeo > 0) & (both.flux_cm > 0)]
    ratio = both.flux_cm / both.flux_imeo
    rho_f, _ = stats.spearmanr(both.flux_imeo, both.flux_cm)
    rho_a, _ = stats.spearmanr(df.area_imeo_px, df.area_cm_px)
    in2 = ((ratio >= 0.5) & (ratio <= 2)).mean()
    print(f"flux: {len(both)} pairs with both fluxes · Spearman {rho_f:.2f} · "
          f"median CM/IMEO {ratio.median():.2f} · within factor 2: {in2:.0%}")
    print(f"area: Spearman {rho_a:.2f} · median IMEO {df.area_imeo_px.median():.0f} px "
          f"vs CM {df.area_cm_px.median():.0f} px")
    print(f"iou_px: median {df.iou_px.median():.2f} · >=0.5: {(df.iou_px>=0.5).mean():.0%}")
    dd = df.dist_m.dropna()
    print(f"emitter distance: median {dd.median():.0f} m · <=500 m: {(dd<=500).mean():.0%} "
          f"· <=1 km: {(dd<=1000).mean():.0%}")

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4), constrained_layout=True)
    ax = axes[0]
    ax.loglog(both.flux_imeo, both.flux_cm, ".", ms=4, alpha=0.35,
              color=figstyle.PALETTE["teal"])
    lim = [both.flux_imeo.min() * 0.7, both.flux_imeo.max() * 1.4]
    ax.plot(lim, lim, "-", color="#94a3b8", lw=1.2)
    ax.fill_between(lim, [l / 2 for l in lim], [l * 2 for l in lim],
                    color="#94a3b8", alpha=0.15, label="factor 2 band")
    ax.set_xlabel("IMEO flux [kg/h]")
    ax.set_ylabel("CM flux [kg/h]")
    ax.set_title(f"flux · Spearman {rho_f:.2f} · {in2:.0%} within x2", fontsize=11)
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.loglog(df.area_imeo_px, df.area_cm_px, ".", ms=4, alpha=0.35,
              color=figstyle.PALETTE["orange"])
    lim = [df.area_imeo_px.min() * 0.7, df.area_imeo_px.max() * 1.4]
    ax.plot(lim, lim, "-", color="#94a3b8", lw=1.2)
    ax.set_xlabel("IMEO mask [px]")
    ax.set_ylabel("CM mask [px]")
    ax.set_title(f"area · Spearman {rho_a:.2f}", fontsize=11)
    ax = axes[2]
    ax.hist(df.iou_px, bins=25, color=figstyle.PALETTE["teal"])
    ax.axvline(df.iou_px.median(), color=figstyle.PALETTE["orange"], lw=2,
               label=f"median {df.iou_px.median():.2f}")
    ax.set_xlabel("pixel IoU of the pair")
    ax.set_ylabel("pairs")
    ax.set_title("mask agreement", fontsize=11)
    ax.legend(fontsize=8)
    ax = axes[3]
    ax.hist(np.clip(dd, 0, 3000), bins=30, color=figstyle.PALETTE["slate"])
    ax.axvline(dd.median(), color=figstyle.PALETTE["orange"], lw=2,
               label=f"median {dd.median():.0f} m")
    ax.set_xlabel("emitter distance [m, clipped at 3 km]")
    ax.set_ylabel("pairs")
    ax.set_title("source point agreement", fontsize=11)
    ax.legend(fontsize=8)
    fig.suptitle(f"the {len(df)} matched IMEO-CM pairs of the published dataset, "
                 "compared attribute by attribute", fontsize=12.5)
    figstyle.save(fig, IMG / "pairs-similarity.png")
    shutil.copy2(IMG / "pairs-similarity.png", PUB / "pairs-similarity.png")
    print("figure -> pairs-similarity.png (images and slides/public)")


if __name__ == "__main__":
    main()
