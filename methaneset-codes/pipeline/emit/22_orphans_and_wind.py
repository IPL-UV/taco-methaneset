"""MODULE 22: two questions from Julio (Aug 27) that support the 1:1 matching.

  A. "The CM overlap bug: an overlapping one will probably give me a
      pair with CM. How is it supported?"
     -> CM orphans are CLASSIFIED: how many are leftover twins of
        a record that DID match (the overlap), and how many are genuine
        detections that IMEO did not see? Without that number, "orphan" is ambiguous.
        Twin criterion: overlap with an already matched CM of the same scene
        with IoU >= 0.5 or containment >= 0.9 (the same matching threshold).

  B. Why do the fluxes of a pair disagree? Each institution inverts with ITS
     wind (module 20). The flux ratio is compared against the wind ratio
     over the matched pairs: if the flux is ~proportional to the
     wind, both ratios must move together.

Outputs: assets/data/cross/2026-08-25/orphan_kinds.parquet
         assets/images/orphan-kinds.png · assets/images/flux-vs-wind.png
Usage:
  /data/users/julio/.conda/envs/deep/bin/python code/v2/22_orphans_and_wind.py \
        > code/v2/22_orphans_and_wind.log 2>&1
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
CROSS = ROOT / "assets" / "data" / "cross" / "2026-08-25"
TACO = pathlib.Path("/data/databases/METHANSET_TACOS/methaneset-emit")
DATA = TACO / "DATA"
IOU_TWIN, CONT_TWIN = 0.5, 0.9


def classify_scene(row):
    """Each CM orphan: twin of a matched one, or genuine detection?"""
    orph = json.loads(row["match:orphans"])
    if not orph["cm"]:
        return []
    pairs = json.loads(row["match:pairs"])
    ccod = json.loads(row["detection:cm_cod"])
    matched = [p["cm"] for p in pairs]
    if not matched:
        return [{"granule": row["id"], "cm": n, "kind": "genuine",
                 "best_iou": 0.0, "best_cont": 0.0} for n in orph["cm"]]
    with rasterio.open(DATA / row["id"] / "plume_cm.tif") as s:
        mc = s.read(1)
    masks = {n: np.isin(mc, c) for n, c in ccod.items()}
    out = []
    for n in orph["cm"]:
        a = masks.get(n)
        if a is None or not a.any():
            continue
        best_iou = best_cont = 0.0
        for m in matched:
            b = masks.get(m)
            if b is None or not b.any():
                continue
            inter = int((a & b).sum())
            if not inter:
                continue
            best_iou = max(best_iou, inter / int((a | b).sum()))
            best_cont = max(best_cont, inter / int(a.sum()))
        kind = ("twin" if (best_iou >= IOU_TWIN or best_cont >= CONT_TWIN)
                else ("partial" if best_iou > 0 else "genuine"))
        out.append({"granule": row["id"], "cm": n, "kind": kind,
                    "best_iou": round(best_iou, 3),
                    "best_cont": round(best_cont, 3)})
    return out


def part_a(md):
    rows = md[md["match:n_orph_cm"] > 0].to_dict("records")
    print(f"== A. CM orphans: {len(rows)} scenes with at least one ==")
    res = []
    with ProcessPoolExecutor(8) as ex:
        for part in ex.map(classify_scene, rows, chunksize=8):
            res.extend(part)
    df = pd.DataFrame(res)
    df.to_parquet(CROSS / "orphan_kinds.parquet", index=False)
    n = len(df)
    share = df.kind.value_counts()
    print(f"total CM orphans: {n}")
    for k in ["twin", "partial", "genuine"]:
        c = int(share.get(k, 0))
        print(f"  {k:8s}: {c:4d}  ({c/n:.0%})")
    print(f"  -> median IoU of the twins: "
          f"{df[df.kind=='twin'].best_iou.median():.2f}")

    fig, ax = plt.subplots(figsize=(7.6, 4.4), constrained_layout=True)
    order = ["twin", "partial", "genuine"]
    labels = [f"leftover twin of a\nmatched CM record\n({int(share.get('twin',0))})",
              f"partial overlap\nwith a matched one\n({int(share.get('partial',0))})",
              f"genuine CM-only\ndetection\n({int(share.get('genuine',0))})"]
    vals = [int(share.get(k, 0)) for k in order]
    cols = [figstyle.PALETTE["orange"], "#fbbf24", figstyle.PALETTE["teal"]]
    ax.bar(labels, vals, color=cols, width=0.62)
    for x, v in enumerate(vals):
        ax.text(x, v, f"{v/n:.0%}", ha="center", va="bottom",
                fontsize=12, fontweight="bold")
    ax.set_ylabel("CM orphan records")
    ax.set_title(f"What a Carbon Mapper orphan really is  (n={n})", fontsize=12)
    ax.tick_params(labelsize=9)
    figstyle.save(fig, IMG / "orphan-kinds.png")
    shutil.copy2(IMG / "orphan-kinds.png", PUB / "orphan-kinds.png")
    print("  -> orphan-kinds.png")
    return df


def part_b(md):
    """Is the flux disagreement a wind disagreement?"""
    print("\n== B. flux vs wind on the matched pairs ==")
    rows = []
    for r in md.to_dict("records"):
        pairs = json.loads(r["match:pairs"])
        if not pairs:
            continue
        iflux, cflux = json.loads(r["detection:imeo_flux"]), json.loads(r["detection:cm_flux"])
        iw, cw = json.loads(r["meteo:imeo_wind"]), json.loads(r["meteo:cm_wind"])
        for p in pairs:
            fi, fc = iflux.get(p["imeo"]), cflux.get(p["cm"])
            wi, wc = iw.get(p["imeo"]), cw.get(p["cm"])
            if not fi or not fc or not wi or not wc:
                continue
            if wi["speed"] <= 0 or wc["speed"] <= 0:
                continue
            rows.append({"flux_ratio": fc / fi,
                         "wind_ratio": wc["speed"] / wi["speed"],
                         "wi": wi["speed"], "wc": wc["speed"]})
    df = pd.DataFrame(rows)
    rho, p_ = stats.spearmanr(df.wind_ratio, df.flux_ratio)
    print(f"pairs with flux and wind from both: {len(df)}")
    print(f"Spearman(wind ratio, flux ratio) = {rho:.2f} (p={p_:.1e})")
    same = df[(df.wind_ratio > 0.9) & (df.wind_ratio < 1.1)]
    print(f"when the winds agree (+-10%, n={len(same)}): "
          f"median flux ratio {same.flux_ratio.median():.2f} · "
          f"within x2: {((same.flux_ratio>0.5)&(same.flux_ratio<2)).mean():.0%}")
    print(f"in the rest (n={len(df)-len(same)}): within x2: "
          f"{((df.drop(same.index).flux_ratio>0.5)&(df.drop(same.index).flux_ratio<2)).mean():.0%}")

    fig, ax = plt.subplots(figsize=(7.2, 5.4), constrained_layout=True)
    ax.loglog(df.wind_ratio, df.flux_ratio, ".", ms=4.5, alpha=0.4,
              color=figstyle.PALETTE["teal"])
    lim = [df.wind_ratio.min() * 0.8, df.wind_ratio.max() * 1.2]
    ax.plot(lim, lim, "-", color=figstyle.PALETTE["orange"], lw=1.8,
            label="flux ratio = wind ratio")
    ax.axhline(1, color="#94a3b8", lw=0.8, ls=":")
    ax.axvline(1, color="#94a3b8", lw=0.8, ls=":")
    ax.set_xlabel("wind speed ratio  CM / IMEO")
    ax.set_ylabel("flux ratio  CM / IMEO")
    ax.set_title(f"the flux disagreement follows the wind disagreement\n"
                 f"Spearman {rho:.2f} over {len(df):,} matched pairs", fontsize=12)
    ax.legend(fontsize=9)
    figstyle.save(fig, IMG / "flux-vs-wind.png")
    shutil.copy2(IMG / "flux-vs-wind.png", PUB / "flux-vs-wind.png")
    print("  -> flux-vs-wind.png")


def main():
    md = pd.read_parquet(TACO / "METADATA" / "level0.parquet")
    part_a(md)
    part_b(md)


if __name__ == "__main__":
    main()
