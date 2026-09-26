"""Step 7b · Visual sheet to review the bank by eye, reading the generated GeoTIFFs.

Top row: the same plume with the lowest sun (SZA 70.8) in four directions; the solar footprint
must fall on the side opposite the sun. Bottom row: four different plumes at nadir.
Everything in metres relative to the source, read from the transform of each GeoTIFF.
Usage: python 07b_inspection_figure.py DIR   ->  DIR/inspeccion.png
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rasterio

D = Path(sys.argv[1])
t = pd.read_parquet(D / "tabla.parquet")


def pinta(ax, r, titulo):
    with rasterio.open(D / r["path"]) as s:
        x = s.read(1); b = s.bounds
    ax.imshow(np.log10(np.clip(x, 0.1, None)), cmap="magma", vmin=-1, vmax=3.3,
              extent=[b.left, b.right, b.bottom, b.top])
    ax.plot(0, 0, "+", ms=14, mew=2.2, color="#22d3ee")
    ax.set_title(titulo, fontsize=8.6)
    ax.tick_params(labelsize=7)
    ax.set_xlabel("m downwind", fontsize=7.5)


fig, axs = plt.subplots(2, 4, figsize=(17, 8.2))
uid = sorted(t.loc[(t["methane:emitter"] == "p5") & (t["methane:wind_speed"] == 7.0), "methane:plume_uid"].unique())[0]
p = t[(t["methane:plume_uid"] == uid) & (t["sun:sza"] == t["sun:sza"].max())]
for ax, (raa, lado) in zip(axs[0], [(0, "sun downwind: footprint backwards"),
                                    (90, "sun to the north: footprint to the south"),
                                    (180, "sun upwind: footprint forwards"),
                                    (270, "sun to the south: footprint to the north")]):
    r = p.iloc[(p["sun:raa"] - raa).abs().argmin()]
    pinta(ax, r, f"{uid}\nSZA {r['sun:sza']:.1f}, RAA {r['sun:raa']:.1f} · {lado}")
    ang = np.radians(r["sun:raa"])
    ax.annotate("sun", xy=(0, 0), xytext=(900 * np.cos(ang), 900 * np.sin(ang)), color="#facc15",
                fontsize=9, fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="<-", color="#facc15", lw=1.6))
n0 = t[t["sun:sza"] == 0]
ejemplos = [("p1", 2.0), ("a5", 5.0), ("a6", 2.0), ("p9", 10.0)]
for ax, (em, ws) in zip(axs[1], ejemplos):
    c = n0[n0["methane:emitter"] == em]
    r = c.iloc[(c["methane:wind_speed"] - ws).abs().argmin()]
    pinta(ax, r, f"{r['methane:plume_uid']} · nadir\npeak {r['methane:peak_ppb']:.0f} ppb, "
                 f"edge {r['edge:downwind_ppb']:.0f} ppb, mean height {r['methane:mean_height_m']:.0f} m")
fig.suptitle("methaneset-bank v3 · real GeoTIFFs, log10 ppb at 3 t/h · cross = source at (0, 0) from the transform",
             fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig(D / "inspeccion.png", dpi=110)
print("ok", D / "inspeccion.png")
