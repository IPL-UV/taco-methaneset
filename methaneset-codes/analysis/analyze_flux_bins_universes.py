"""The two universes (IMEO portal and Carbon Mapper) split by the bins
with literature support:

  weak < 1,000 kg/h  (STARCOP; ~90% POD for EMIT at 3 m/s)
  strong 1,000 to 10,000
  very strong > 10,000  (scale of TROPOMI super-emitter monitoring)

Counts n and % per bin for: IMEO all, IMEO EMIT only, CM all (CH4),
CM EMIT only. Two-panel figure for notes and slides.

Output: assets/images/flux-bins-universos.png + console.
"""
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/data/users/julio/Notes/01-Projects/methanset/assets/slides/theme")
import figstyle  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

B = pathlib.Path("/data/users/julio/Notes/01-Projects/methanset/assets/data")
OUT = pathlib.Path("/data/users/julio/Notes/01-Projects/methanset/assets/images/"
                   "flux-bins-universos.png")
WEAK, VSTRONG = 1000.0, 10000.0
BIN_LBL = ["weak\n<1,000", "strong\n1,000-10,000", "very strong\n>10,000"]


def bins_of(flux):
    flux = flux[flux > 0]
    n = len(flux)
    c = [int((flux < WEAK).sum()),
         int(((flux >= WEAK) & (flux <= VSTRONG)).sum()),
         int((flux > VSTRONG).sum())]
    return n, c, flux


def report(name, flux):
    n, c, f = bins_of(flux)
    pct = [x / n for x in c]
    print(f"{name}: n={n:,}  median={f.median():,.0f} kg/h")
    for lbl, x, p in zip(["weak", "strong", "vstrong"], c, pct):
        print(f"   {lbl:8s} {x:7,d}  ({p:.1%})")
    return c, pct, f


def main():
    im = pd.read_csv(B / "imeo/2026-08-22/unep_methanedata_detected_plumes.csv")
    cm = pd.read_parquet(B / "carbonmapper/2026-08-22/plumes.parquet")
    cm = cm[cm.gas == "CH4"]

    sets = {
        "IMEO · all instruments": im.ch4_fluxrate,
        "IMEO · EMIT only": im[im.satellite.str.contains("EMIT", na=False)].ch4_fluxrate,
        "Carbon Mapper · all CH4": cm.emission_auto,
        "Carbon Mapper · EMIT only": cm[cm.instrument == "emi"].emission_auto,
    }
    stats = {k: report(k, v.dropna()) for k, v in sets.items()}

    figstyle.apply()
    fig, axes = plt.subplots(1, 2, figsize=figstyle.FIGSIZE_WIDE,
                             constrained_layout=True, sharey=True)
    pal = [figstyle.PALETTE["teal"], figstyle.PALETTE["orange"], figstyle.PALETTE["red"]]
    panels = [("IMEO portal", "IMEO · all instruments", "IMEO · EMIT only"),
              ("Carbon Mapper", "Carbon Mapper · all CH4", "Carbon Mapper · EMIT only")]
    x = np.arange(3)
    for ax, (title, k_all, k_emit) in zip(axes, panels):
        p_all = stats[k_all][1]
        p_emit = stats[k_emit][1]
        b1 = ax.bar(x - 0.2, [p * 100 for p in p_all], width=0.38,
                    color=pal, alpha=0.45, label="all instruments")
        b2 = ax.bar(x + 0.2, [p * 100 for p in p_emit], width=0.38,
                    color=pal, label="EMIT only")
        for bars in (b1, b2):
            for r in bars:
                ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 1,
                        f"{r.get_height():.0f}%", ha="center", fontsize=10)
        n_all = sum(stats[k_all][0]); n_emit = sum(stats[k_emit][0])
        ax.set_title(f"{title}  (all n={n_all:,} · EMIT n={n_emit:,})", fontsize=12)
        ax.set_xticks(x); ax.set_xticklabels(BIN_LBL, fontsize=10)
        ax.set_ylim(0, 100)
        handles = [plt.Rectangle((0, 0), 1, 1, color="#94a3b8", alpha=0.45),
                   plt.Rectangle((0, 0), 1, 1, color="#94a3b8")]
        ax.legend(handles, ["all instruments", "EMIT only"], fontsize=9,
                  loc="upper right")
    axes[0].set_ylabel("share of catalog plumes [%]")
    figstyle.save(fig, OUT)
    print("\nfigure ->", OUT)


if __name__ == "__main__":
    main()
