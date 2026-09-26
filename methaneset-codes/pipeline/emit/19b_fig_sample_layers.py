"""MODULE 19b: figure of the 10 layers of one TACO sample (new slide).

A real scene from the published dataset, 2x5 panel, one panel per file:
radiance (RGB), latlon (lat), glt (glt_x), elevation, wind (u/v speed),
mf, rmf, mag1c, plume_imeo (band 1), plume_cm (band 1).

Scene: EMIT_L1B_RAD_001_20220827T060753_2223904_013 (16 IMEO / 14 CM,
Oil and Gas): the most plume-populated among the weak ones.

Output: assets/images/sample-layers.png (+ copy to assets/slides/public/)
Usage:
  /data/users/julio/.conda/envs/deep/bin/python code/v2/19b_fig_sample_layers.py \
        > code/v2/19b_fig_sample_layers.log 2>&1
"""
import pathlib
import shutil
import sys

import numpy as np
import rasterio

_theme = pathlib.Path(os.environ.get("METHANESET_THEME", "/data/users/julio/Notes/01-Projects/methanset/assets/slides/theme"))
if _theme.exists():
    sys.path.insert(0, str(_theme))
import figstyle  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

ROOT = pathlib.Path(os.environ.get("METHANESET_ROOT", "/data/users/julio/Notes/01-Projects/methanset"))
IMG = ROOT / "assets" / "images"
PUB = ROOT / "assets" / "slides" / "public"
SCENE = "EMIT_L1B_RAD_001_20220827T060753_2223904_013"
DATA = pathlib.Path("/data/databases/METHANSET_TACOS/methaneset-emit/DATA") / SCENE

# EMIT: 285 bands, ~381-2493 nm, ~7.4 nm step
RGB_NM = (640.0, 550.0, 460.0)


def band_for(nm):
    return int(round((nm - 381.0) / 7.41)) + 1


def stretch(a, lo=2, hi=98):
    a = a.astype("float32")
    v = a[np.isfinite(a)]
    p1, p2 = np.percentile(v, [lo, hi])
    return np.clip((a - p1) / (p2 - p1 + 1e-9), 0, 1)


def read(name, band=1):
    with rasterio.open(DATA / f"{name}.tif") as src:
        a = src.read(band).astype("float32")
        nod = src.nodata
    if nod is not None:
        a[a == nod] = np.nan
    return a


def main():
    print(f"scene: {SCENE}")
    # radiance RGB
    with rasterio.open(DATA / "radiance.tif") as src:
        rgb = np.stack([src.read(band_for(nm)).astype("float32") for nm in RGB_NM], -1)
    rgb[rgb <= -9990] = np.nan
    rgb = np.dstack([stretch(rgb[..., i]) for i in range(3)])

    lat = read("latlon", 1)
    glt = read("glt", 1)
    elev = read("elevation", 1)
    with rasterio.open(DATA / "wind.tif") as src:
        u, v = src.read(1), src.read(2)
    speed = np.hypot(u, v)
    mf = read("mf")
    rmf = read("rmf")
    mag = read("mag1c")
    with rasterio.open(DATA / "plume_imeo.tif") as src:
        mi = src.read(1).astype("float32")
    with rasterio.open(DATA / "plume_cm.tif") as src:
        mc = src.read(1).astype("float32")

    panels = [
        ("radiance.tif · 285 bands", rgb, None, None),
        ("latlon.tif · lat, lon", lat, "viridis", "latitude [deg]"),
        ("glt.tif · glt_x, glt_y", np.where(glt == 0, np.nan, glt), "cividis", "ortho column"),
        ("elevation.tif · GLO30", elev, "terrain", "elevation [m]"),
        ("wind.tif · u10, v10", speed, "PuBuGn", "wind speed [m/s]"),
        ("mf.tif · matched filter", np.clip(mf, 0, np.nanpercentile(mf, 99.5)), "magma", "ppm·m"),
        ("rmf.tif", np.clip(rmf, 0, np.nanpercentile(rmf, 99.5)), "magma", "ppm·m"),
        ("mag1c.tif", np.clip(mag, 0, np.nanpercentile(mag, 99.5)), "magma", "ppm·m"),
        ("plume_imeo.tif · instance codes", mi, None, None),
        ("plume_cm.tif · instance codes", mc, None, None),
    ]

    fig, axes = plt.subplots(2, 5, figsize=(16, 7.4), constrained_layout=True)
    for ax, (title, a, cmap, cblab) in zip(axes.flat, panels):
        if a.ndim == 3:
            ax.imshow(a)
        elif title.startswith("plume"):
            codes = np.unique(a[a > 0]).astype(int)
            base = plt.get_cmap("tab20")(np.linspace(0, 1, max(len(codes), 1)))
            lut = {c: base[i] for i, c in enumerate(codes)}
            img = np.zeros(a.shape + (4,), "float32")
            img[..., :3] = 0.94
            img[..., 3] = 1.0
            for c, col in lut.items():
                img[a == c] = col
            ax.imshow(img)
            ax.set_xlabel(f"{len(codes)} instance codes", fontsize=8)
        else:
            im = ax.imshow(a, cmap=cmap)
            cb = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
            cb.set_label(cblab, fontsize=7)
            cb.ax.tick_params(labelsize=7)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{SCENE} · the 10 layers of one published sample", fontsize=13)
    figstyle.save(fig, IMG / "sample-layers.png")
    shutil.copy2(IMG / "sample-layers.png", PUB / "sample-layers.png")
    print("figure -> sample-layers.png (images and slides/public)")


if __name__ == "__main__":
    main()
