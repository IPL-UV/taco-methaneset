"""MODULE 06d: the selection on the map, AT PLUME LEVEL (Julio's correction:
each plume has its own sector and its own flux; coloring the granule by a
dominant sector hid the mixtures).

Panel 1: each PLUME of the selection colored by its sector.
Panel 2: each PLUME colored by its flux class (weak / strong).

Output: assets/images/seleccion-mapa.png

Usage:
  /data/users/julio/.conda/envs/deep/bin/python code/v2/06d_map_selection.py \
        > code/v2/06d_map_selection.log 2>&1
"""
import pathlib
import re
import sys

import pandas as pd

_theme = pathlib.Path(os.environ.get("METHANESET_THEME", "/data/users/julio/Notes/01-Projects/methanset/assets/slides/theme"))
if _theme.exists():
    sys.path.insert(0, str(_theme))
import figstyle  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import cartopy.crs as ccrs  # noqa: E402
import cartopy.feature as cfeature  # noqa: E402

DATA = pathlib.Path(__file__).resolve().parent.parent.parent / "assets" / "data"
OUT = (pathlib.Path(__file__).resolve().parent.parent.parent
       / "assets" / "images" / "seleccion-mapa.png")

TS = re.compile(r"EMIT_L1B_RAD_001_(\d{8}T\d{6})")
WEAK = 1000.0
SECTOR_C = {"Oil and Gas": figstyle.PALETTE["orange"],
            "Waste": figstyle.PALETTE["teal"],
            "Coal": figstyle.PALETTE["ink"],
            "Other": figstyle.PALETTE["slate"]}
FLUX_C = {"weak": figstyle.PALETTE["red"],
          "strong": figstyle.PALETTE["teal"]}


def base(ax):
    ax.add_feature(cfeature.LAND, facecolor="#eef2f6", edgecolor="none")
    ax.add_feature(cfeature.COASTLINE, lw=0.4, edgecolor="#94a3b8")
    ax.set_global()
    ax.spines["geo"].set_visible(False)


def main():
    cross_dir = sorted((DATA / "cross").iterdir())[-1]
    sel_ts = set(pd.read_parquet(cross_dir / "selection.parquet").granule_ts)
    imeo_dir = sorted((DATA / "imeo").iterdir())[-1]
    im = pd.read_csv(imeo_dir / "unep_methanedata_detected_plumes.csv")
    im = im[im.satellite.str.contains("EMIT", na=False)].copy()
    im["granule_ts"] = im.tile.str.extract(TS)[0].str.lower()
    pl = im[im.granule_ts.isin(sel_ts) & (im.ch4_fluxrate > 0)].copy()
    pl["flux_class"] = pl.ch4_fluxrate.apply(
        lambda f: "weak" if f < WEAK else "strong")
    print(f"plumes of the selection: {len(pl)}")
    print(pl.sector.value_counts().to_string())

    figstyle.apply()
    fig, axes = plt.subplots(2, 1, figsize=(9.6, 8.6),
                             subplot_kw={"projection": ccrs.Robinson()},
                             constrained_layout=True)
    for ax, (col, cmap, title) in zip(axes, [
            ("sector", SECTOR_C, "every plume, colored by its sector"),
            ("flux_class", FLUX_C, "every plume, colored by its flux class")]):
        base(ax)
        for cat, color in cmap.items():
            s = pl[pl[col] == cat]
            if not len(s):
                continue
            ax.scatter(s.lon, s.lat, s=11, color=color, alpha=0.7,
                       edgecolors="none", transform=ccrs.PlateCarree(),
                       label=f"{cat} ({len(s):,} plumes)")
        ax.set_title(title, fontsize=12)
        ax.legend(loc="lower left", fontsize=8.5, frameon=True,
                  facecolor="white", framealpha=0.85)
    figstyle.save(fig, OUT)
    print("figure ->", OUT)


if __name__ == "__main__":
    main()
