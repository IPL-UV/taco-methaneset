"""MODULE 06b: before/after figure of the balanced selection.

Left panel: ECDF of flux per PLUME (log axis) of four populations:
  - IMEO universe, all sensors (the world according to IMEO)
  - IMEO universe, EMIT only (what EMIT can see)
  - CM universe, all CH4 (the world according to CM, aircraft included)
  - the plumes of the 700-scene SELECTION
Right panel: % of scenes with any weak plume, clean universe vs selection.

Output: assets/images/seleccion-antes-despues.png (+ manual copy to slides).

Usage:
  /data/users/julio/.conda/envs/deep/bin/python code/v2/06b_fig_before_after.py \
        > code/v2/06b_fig_before_after.log 2>&1
"""
import pathlib
import re
import sys

import numpy as np
import pandas as pd

_theme = pathlib.Path(os.environ.get("METHANESET_THEME", "/data/users/julio/Notes/01-Projects/methanset/assets/slides/theme"))
if _theme.exists():
    sys.path.insert(0, str(_theme))
import figstyle  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

DATA = pathlib.Path(__file__).resolve().parent.parent.parent / "assets" / "data"
OUT = (pathlib.Path(__file__).resolve().parent.parent.parent
       / "assets" / "images" / "seleccion-antes-despues.png")
TS = re.compile(r"EMIT_L1B_RAD_001_(\d{8}T\d{6})")
WEAK = 1000.0


def ecdf(vals):
    v = np.sort(vals[vals > 0])
    return v, np.arange(1, len(v) + 1) / len(v)


def main():
    cross_dir = sorted((DATA / "cross").iterdir())[-1]
    sel = pd.read_parquet(cross_dir / "selection.parquet")
    cand = pd.read_parquet(cross_dir / "candidates.parquet")
    sel_ts = set(sel.granule_ts)

    imeo_dir = sorted((DATA / "imeo").iterdir())[-1]
    im = pd.read_csv(imeo_dir / "unep_methanedata_detected_plumes.csv")
    im_emit = im[im.satellite.str.contains("EMIT", na=False)].copy()
    im_emit["granule_ts"] = im_emit.tile.str.extract(TS)[0].str.lower()

    cm_dir = sorted((DATA / "carbonmapper").iterdir())[-1]
    cm = pd.read_parquet(cm_dir / "plumes.parquet")
    cm = cm[(cm.gas == "CH4") & (cm.emission_auto > 0)]

    cm_emit = cm[cm.instrument == "emi"]
    pops = [
        ("IMEO · EMIT", im_emit.ch4_fluxrate.dropna().values,
         figstyle.PALETTE["slate"], "-"),
        ("CM · EMIT", cm_emit.emission_auto.values,
         figstyle.PALETTE["orange"], "-"),
        ("MethaneSET v2", im_emit[im_emit.granule_ts.isin(sel_ts)]
         .ch4_fluxrate.dropna().values, figstyle.PALETTE["teal"], "-"),
    ]

    figstyle.apply()
    fig, (ax, axp, ax2) = plt.subplots(1, 3, figsize=(11.5, 5.4),
                                       constrained_layout=True,
                                       gridspec_kw={"width_ratios": [2.0, 1, 1]})
    for name, vals, color, ls in pops:
        x, y = ecdf(np.asarray(vals, float))
        ax.plot(x, y, color=color, ls=ls, lw=2.4,
                label=f"{name} (n={len(vals):,})")
    ax.axvspan(1, WEAK, color=figstyle.PALETTE["red"], alpha=0.06)
    ax.axvline(WEAK, color=figstyle.PALETTE["red"], lw=1.6, ls="--")
    ax.text(WEAK * 0.85, 0.97, "weak", fontsize=11, ha="right", va="top",
            color=figstyle.PALETTE["red"], fontweight="bold")
    ax.text(WEAK * 1.2, 0.97, "strong", fontsize=11, va="top",
            color=figstyle.PALETTE["ink"], fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlabel("plume flux rate [kg/h]")
    ax.set_ylabel("cumulative share of plumes")
    ax.legend(fontsize=9, loc="upper left")

    # middle panel: % of weak PLUMES per population
    colors3 = [figstyle.PALETTE["slate"], figstyle.PALETTE["orange"],
               figstyle.PALETTE["teal"]]
    pw = [float((np.asarray(v) < WEAK).mean() * 100) for _, v, *_ in pops]
    bars = axp.bar(["IMEO\nEMIT", "CM\nEMIT", "v2"], pw, color=colors3, width=0.6)
    for b in bars:
        axp.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2,
                 f"{b.get_height():.0f}%", ha="center", fontweight="bold")
    axp.set_ylabel("weak share of PLUMES [%]")
    axp.set_ylim(0, 70)
    axp.set_title("v2 keeps ALL 864 weak plumes\n(catalogs arrive ~1:3)", fontsize=10)

    # right panel: % of SCENES with any weak
    cm_scene = cm_emit.assign(
        ts=cm_emit.plume_id.str.extract(r"^emi(\d{8}t\d{6})")[0])
    cm_sc = cm_scene.groupby("ts").emission_auto.min()
    sc = [
        ("IMEO\nscenes", (cand.n_weak > 0).mean() * 100,
         figstyle.PALETTE["slate"]),
        ("CM\nscenes", float((cm_sc < WEAK).mean() * 100),
         figstyle.PALETTE["orange"]),
        ("v2", (sel.n_weak > 0).mean() * 100,
         figstyle.PALETTE["teal"]),
    ]
    bars = ax2.bar([s[0] for s in sc], [s[1] for s in sc],
                   color=[s[2] for s in sc], width=0.6)
    for b in bars:
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2,
                 f"{b.get_height():.0f}%", ha="center", fontweight="bold")
    ax2.set_ylabel("scenes with a weak plume [%]")
    ax2.set_ylim(0, 70)
    ax2.set_title("and more than doubles\nweak-scene exposure", fontsize=10)
    before, after = sc[0][1], sc[2][1]

    figstyle.save(fig, OUT)
    print("populations:", {n: len(v) for n, v, *_ in pops})
    print(f"scenes with weak: universe {before:.1f}% -> selection {after:.1f}%")
    print("figure ->", OUT)


if __name__ == "__main__":
    main()
