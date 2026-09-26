"""Figure for the holes bug: original mask (GLT) versus the fixed one.

Zoom around the plume of a real scene. The original came out "moth-eaten"
when taking the georeferenced mask to sensor coordinates via GLT: sensor
pixels with no match in the geographic grid are left unpainted.

Output: assets/images/bug-huecos-glt.png (note figure, not slide)
"""
import pathlib
import sys

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from scipy import ndimage

sys.path.insert(0, "/data/users/julio/Notes/01-Projects/methanset/assets/slides/theme")
import figstyle  # noqa: E402

S = pathlib.Path("/data/databases/METHANE_DATASETS_HYPER_BAK/"
                 "EMIT_L1B_RAD_001_20220810T064957_2222205_033")
OUT = pathlib.Path(__file__).resolve().parent.parent / "assets" / "images" / "bug-huecos-glt.png"
PAD = 26


def leer(nombre):
    with rasterio.open(S / nombre) as src:
        return src.read(1) > 0


def main():
    bak, actual = leer("plume_imeo_bak.tif"), leer("plume_imeo.tif")

    figstyle.apply()
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.4))
    for ax, (m, titulo) in zip(axes, [
        (bak, "original: reprojection via GLT"),
        (actual, "published: closing 3x3 + fill holes"),
    ]):
        # independent zoom per mask: between versions the plume also
        # changes position (possible orientation flip, pending with Cesar)
        ys, xs = np.where(m)
        crop = m[ys.min()-PAD:ys.max()+PAD, xs.min()-PAD:xs.max()+PAD]
        holes = int(ndimage.binary_fill_holes(crop).sum() - crop.sum())
        ax.imshow(crop, cmap="gray_r", interpolation="nearest")
        ax.set_title(f"{titulo}\n{int(crop.sum())} px · {holes} px of internal hole",
                     fontsize=11, pad=8)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    figstyle.save(fig, OUT)
    print("saved to", OUT)


if __name__ == "__main__":
    main()
