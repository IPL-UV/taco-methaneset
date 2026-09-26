"""Flux-area relation for the EMIT plumes in the IMEO portal, and where the
Turkmenistan case (TKM_S_037) falls with its two geometries (portal vs HF mask).

Julio's question: given the reported ch4_fluxrate, which area makes sense,
the small one from the HF or the large one from the portal?

Output: assets/images/flux-vs-area-tkm.png + numbers on the console.
"""
import pathlib
import sys

import geopandas as gpd
import numpy as np
from scipy import stats
from shapely import make_valid

sys.path.insert(0, "/data/users/julio/Notes/01-Projects/methanset/assets/slides/theme")
import figstyle  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

DATA = pathlib.Path("/data/users/julio/methanset/data")
OUT = pathlib.Path("/data/users/julio/Notes/01-Projects/methanset/assets/images/"
                   "flux-vs-area-tkm.png")
EQ_AREA = "EPSG:6933"


def main():
    pl = gpd.read_file(DATA / "unep_methanedata_detected_plumes.geojson")
    emit = pl[pl.satellite.str.contains("EMIT", na=False)].copy()
    emit = emit[emit.ch4_fluxrate > 0]
    emit["area_km2"] = emit.geometry.to_crs(EQ_AREA).area / 1e6
    emit = emit[emit.area_km2 > 0]
    print(f"portal EMIT plumes with flux and polygon: {len(emit)}")

    r, p = stats.spearmanr(np.log10(emit.ch4_fluxrate), np.log10(emit.area_km2))
    print(f"correlation (Spearman, log-log) flux vs area: r={r:.2f} (p={p:.1e})")

    # the TKM case: HF mask and best portal candidate
    hf = gpd.read_parquet(DATA / "carbonmapper_new/imeo_polygons_new.parquet")
    hf = hf[hf.scene_key == "20240817t082107"].iloc[0]
    g_hf = make_valid(hf.geometry)
    cand = emit[emit.geometry.intersects(g_hf)].copy()
    inter = cand.geometry.apply(lambda g: make_valid(g).intersection(g_hf).area)
    union = cand.geometry.apply(lambda g: make_valid(g).union(g_hf).area)
    best = cand.loc[(inter / union).idxmax()]
    flux = best.ch4_fluxrate
    a_portal = best.area_km2
    a_hf = gpd.GeoSeries([g_hf], crs="EPSG:4326").to_crs(EQ_AREA).area.iloc[0] / 1e6
    print(f"TKM: flux={flux:.0f} kg/h  portal area={a_portal:.2f} km2  "
          f"HF area={a_hf:.2f} km2  (ratio {a_portal/a_hf:.1f}x)")

    # plumes of similar flux (half to double): where the two areas fall
    similar = emit[(emit.ch4_fluxrate >= flux / 2) & (emit.ch4_fluxrate <= flux * 2)]
    pct_portal = stats.percentileofscore(similar.area_km2, a_portal)
    pct_hf = stats.percentileofscore(similar.area_km2, a_hf)
    print(f"among {len(similar)} plumes of similar flux: the portal area falls "
          f"at percentile {pct_portal:.0f}, the HF one at {pct_hf:.0f}")

    figstyle.apply()
    fig, ax = plt.subplots(figsize=figstyle.FIGSIZE_WIDE, constrained_layout=True)
    ax.scatter(emit.area_km2, emit.ch4_fluxrate, s=9, alpha=0.25,
               color=figstyle.PALETTE["slate"], edgecolors="none",
               label=f"EMIT plumes in the portal (n={len(emit):,})", rasterized=True)
    ax.plot([a_hf, a_portal], [flux, flux], color=figstyle.PALETTE["red"],
            lw=1.4, ls=":", zorder=3)
    ax.plot(a_hf, flux, marker="*", ms=17, ls="none", mec="#0f766e",
            mfc=figstyle.PALETTE["teal"], zorder=4,
            label=f"TKM with HF area ({a_hf:.2f} km$^2$, pct {pct_hf:.0f})")
    ax.plot(a_portal, flux, marker="*", ms=17, ls="none", mec="#9a3412",
            mfc=figstyle.PALETTE["orange"], zorder=4,
            label=f"TKM with portal area ({a_portal:.2f} km$^2$, pct {pct_portal:.0f})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("plume polygon area [km$^2$]")
    ax.set_ylabel("ch4_fluxrate [kg/h]")
    ax.legend(loc="lower right", fontsize=10)
    ax.text(0.02, 0.97, f"Spearman r = {r:.2f} (log-log)",
            transform=ax.transAxes, va="top", fontsize=11,
            color=figstyle.PALETTE["ink"])
    figstyle.save(fig, OUT)
    print("figure ->", OUT)


if __name__ == "__main__":
    main()
