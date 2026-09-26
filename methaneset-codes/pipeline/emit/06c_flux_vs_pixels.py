"""MODULE 06c: flux vs area IN EMIT PIXELS of the plumes of the selection.

Julio's question: how many pixels does a plume measure according to its flux?
It serves to (a) size patches and per-instance metrics and (b) verify that the
flux-size relation holds in the selection.

Area of the portal polygon (equal-area) converted to EMIT pixels:
1 EMIT pixel ~ 60 x 60 m = 3,600 m2.

Output: assets/images/flux-vs-pixels.png + per-bin statistics on console.

Usage:
  /data/users/julio/.conda/envs/deep/bin/python code/v2/06c_flux_vs_pixels.py \
        > code/v2/06c_flux_vs_pixels.log 2>&1
"""
import pathlib
import re
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import stats

_theme = pathlib.Path(os.environ.get("METHANESET_THEME", "/data/users/julio/Notes/01-Projects/methanset/assets/slides/theme"))
if _theme.exists():
    sys.path.insert(0, str(_theme))
import figstyle  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

DATA = pathlib.Path(__file__).resolve().parent.parent.parent / "assets" / "data"
OUT = (pathlib.Path(__file__).resolve().parent.parent.parent
       / "assets" / "images" / "flux-vs-pixels.png")
TS = re.compile(r"EMIT_L1B_RAD_001_(\d{8}T\d{6})")
PX_M2 = 60.0 * 60.0
WEAK = 1000.0


def main():
    cross_dir = sorted((DATA / "cross").iterdir())[-1]
    sel_ts = set(pd.read_parquet(cross_dir / "selection.parquet").granule_ts)

    imeo_dir = sorted((DATA / "imeo").iterdir())[-1]
    pl = gpd.read_file(imeo_dir / "unep_methanedata_detected_plumes.geojson")
    pl = pl[pl.satellite.str.contains("EMIT", na=False)].copy()
    pl["granule_ts"] = pl.tile.str.extract(TS)[0].str.lower()
    pl = pl[pl.granule_ts.isin(sel_ts) & (pl.ch4_fluxrate > 0)]
    pl["px"] = pl.geometry.to_crs("EPSG:6933").area / PX_M2
    pl = pl[pl.px > 0]
    pl["bin"] = np.where(pl.ch4_fluxrate < WEAK, "weak", "strong")
    print(f"plumes of the selection with polygon and flux: {len(pl):,}")

    r, p = stats.spearmanr(np.log10(pl.ch4_fluxrate), np.log10(pl.px))
    print(f"Spearman log-log flux vs pixels: r={r:.2f} (p={p:.1e})")
    for b in ("weak", "strong"):
        s = pl[pl.bin == b].px
        print(f"{b:7s}: n={len(s):,} · pixels p25/median/p75 = "
              f"{s.quantile(.25):.0f} / {s.median():.0f} / {s.quantile(.75):.0f}"
              f" · min {s.min():.0f} · max {s.max():.0f}")
    for size in (64, 128, 256):
        frac = (pl.px <= size * size * 0.25).mean()
        print(f"plumes that fit in a {size}x{size} patch (<=25% of the patch): "
              f"{frac:.0%}")

    figstyle.apply()
    fig, ax = plt.subplots(figsize=figstyle.FIGSIZE_WIDE, constrained_layout=True)
    colors = {"weak": figstyle.PALETTE["red"], "strong": figstyle.PALETTE["teal"]}
    for b in ("strong", "weak"):
        s = pl[pl.bin == b]
        ax.scatter(s.px, s.ch4_fluxrate, s=12, alpha=0.4, edgecolors="none",
                   color=colors[b], label=f"{b} (n={len(s):,})", rasterized=True)
    med_w = pl[pl.bin == "weak"].px.median()
    med_s = pl[pl.bin == "strong"].px.median()
    ax.axhline(WEAK, color=figstyle.PALETTE["slate"], lw=1.2, ls="--")
    ax.axvline(med_w, color=colors["weak"], lw=1.2, ls=":")
    ax.axvline(med_s, color=colors["teal"] if False else colors["strong"],
               lw=1.2, ls=":")
    ax.annotate(f"median weak\n{med_w:.0f} px", (med_w, 0.03),
                xycoords=("data", "axes fraction"), fontsize=9,
                color=colors["weak"], ha="center")
    ax.annotate(f"median strong\n{med_s:.0f} px", (med_s, 0.03),
                xycoords=("data", "axes fraction"), fontsize=9,
                color=colors["strong"], ha="center")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("plume size [EMIT pixels, 60 m]")
    ax.set_ylabel("flux rate [kg/h]")
    ax.legend(loc="upper left", fontsize=10)
    ax.text(0.98, 0.03, f"Spearman r = {r:.2f} (log-log)",
            transform=ax.transAxes, ha="right", fontsize=11)
    figstyle.save(fig, OUT)
    print("figure ->", OUT)


if __name__ == "__main__":
    main()
