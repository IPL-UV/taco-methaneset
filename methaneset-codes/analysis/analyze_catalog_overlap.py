"""REAL overlap between the Carbon Mapper and IMEO catalogs, methane only.

Beware the confusion that motivates this script: IMEO's detection_institution
field does NOT measure overlap. It measures who reported the row. Two catalogs
can contain the SAME physical plume, each detected on its own, and the field
does not reflect it. Overlap has to be measured by crossing scene and position.

Clean case: EMIT. Both catalogs process the same granules, so they can be
crossed by scene_key (YYYYMMDDtHHMMSS of the granule) and then plumes paired
by distance.

IMPORTANT: pairs by distance are CANDIDATES, not verification. Under 2 km
there may be two different sources or the same plume displaced. Confirming it
is the same plume requires crossing masks (IoU) and inheriting the source,
which is exactly the matching in the MethaneSET pipeline. The verified number
from the paper: 1,592 consensus granules (both systems with at least one
verified plume), of which the 503 in the dataset are selected.

Usage:  python analyze_catalog_overlap.py [date]   (default 2026-08-22)
"""
import pathlib
import re
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent / "assets" / "data"
UMBRAL_KM = 2.0


def scene_key(text):
    m = re.search(r"(\d{8}[Tt]\d{6})", str(text))
    return m.group(1).lower() if m else None


def dist_km(lat1, lon1, lat2, lon2):
    """Flat approximation, good enough for thresholds of a few km."""
    ky = 111.32
    kx = ky * np.cos(np.radians((lat1 + lat2) / 2))
    return np.hypot((lat1 - lat2) * ky, (lon1 - lon2) * kx)


def main(fecha="2026-08-22"):
    im = pd.read_csv(ROOT / "imeo" / fecha / "unep_methanedata_detected_plumes.csv")
    cm = pd.read_parquet(ROOT / "carbonmapper" / fecha / "plumes.parquet")
    cm = cm[cm.gas == "CH4"].copy()                      # methane only
    im_e = im[im.satellite.str.startswith("EMIT")].copy()
    cm_e = cm[cm.instrument == "emi"].copy()
    im_e["key"] = im_e.tile.map(scene_key)
    cm_e["key"] = cm_e.plume_id.map(scene_key)

    esc_im, esc_cm = set(im_e.key.dropna()), set(cm_e.key.dropna())
    comp = esc_im & esc_cm
    print(f"EMIT scenes   IMEO {len(esc_im)}   CM {len(esc_cm)}   shared {len(comp)}")
    print(f"  IMEO only {len(esc_im - esc_cm)}   CM only {len(esc_cm - esc_im)}")

    # pair plumes within shared scenes, greedy by distance
    pares = 0
    im_comp = im_e[im_e.key.isin(comp)]
    cm_comp = cm_e[cm_e.key.isin(comp)].dropna(subset=["lat", "lon"])
    for k, g_im in im_comp.groupby("key"):
        g_cm = cm_comp[cm_comp.key == k]
        if g_cm.empty:
            continue
        d = dist_km(g_im.lat.values[:, None], g_im.lon.values[:, None],
                    g_cm.lat.values[None, :], g_cm.lon.values[None, :])
        while d.size and d.min() < UMBRAL_KM:
            i, j = np.unravel_index(d.argmin(), d.shape)
            pares += 1
            d = np.delete(np.delete(d, i, 0), j, 1)

    n_im, n_cm = len(im_comp), len(cm_comp)
    print(f"\nEMIT plumes in shared scenes   IMEO {n_im}   CM {n_cm}")
    print(f"CANDIDATE pairs by proximity (< {UMBRAL_KM} km, unverified)   {pares}")
    print(f"  {100*pares/n_im:.0f}% of IMEO plumes has a partner in CM")
    print(f"  {100*pares/n_cm:.0f}% of CM plumes has a partner in IMEO")
    print(f"\nEMIT plumes total   IMEO {len(im_e)}   CM {len(cm_e)}")
    print(f"context: detection_institution='Carbon Mapper' in IMEO = "
          f"{(im.detection_institution.str.contains('Carbon', na=False)).sum()} rows "
          f"(attribution, NOT overlap)")


if __name__ == "__main__":
    main(*sys.argv[1:])
