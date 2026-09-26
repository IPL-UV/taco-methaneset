"""How much did the ranking of the 503 change because of the holes bug?

Repeats Cesar's criterion (plume_pct = % of plume pixels per granule,
top-K per catalog, union) on the ORIGINAL masks with holes (_bak)
and on the FIXED ones, and compares the selected sets.

Usage:  python analyze_ranking_holes.py [topk]   (default 350)
"""
import pathlib
import sys
import warnings

import numpy as np
import rasterio

warnings.filterwarnings("ignore")
BAK = pathlib.Path("/data/databases/METHANE_DATASETS_HYPER_BAK")


def pct(path):
    with rasterio.open(path) as src:
        a = src.read(1)
    return np.count_nonzero(a) / a.size * 100


def main(topk=350):
    filas = []
    for d in sorted(BAK.iterdir()):
        req = ["plume_imeo_bak.tif", "plume_imeo.tif", "plume_cm_bak.tif", "plume_cm.tif"]
        if not all((d / f).exists() for f in req):
            continue
        filas.append((d.name, *[pct(d / f) for f in req]))
    n = len(filas)
    ids = [f[0] for f in filas]
    col = lambda i: np.array([f[i] for f in filas])
    im_bak, im_fix, cm_bak, cm_fix = col(1), col(2), col(3), col(4)

    def top(v):
        return set(np.array(ids)[np.argsort(-v)[:topk]])

    sel_bak = top(im_bak) | top(cm_bak)
    sel_fix = top(im_fix) | top(cm_fix)
    print(f"scenes compared: {n}   topk={topk}")
    perd = lambda b, f: np.nanmean(1 - b / np.where(f > 0, f, np.nan)) * 100
    print(f"mean pixel loss from holes: IMEO {perd(im_bak, im_fix):.1f}%   CM {perd(cm_bak, cm_fix):.1f}%")
    print(f"top-IMEO changes in {len(top(im_bak) ^ top(im_fix))//2} granules, "
          f"top-CM changes in {len(top(cm_bak) ^ top(cm_fix))//2}")
    print(f"final selection (union): {len(sel_bak)} with holes vs {len(sel_fix)} fixed, "
          f"{len(sel_bak ^ sel_fix)//2} different granules")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 350)
