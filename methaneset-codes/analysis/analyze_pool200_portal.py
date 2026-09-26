"""Crosses the 200 granules of the IMEO full-tile test against its public portal.

Julio's question: does the portal (CSV/GeoJSON) also publish the granules
observed WITHOUT a plume? Measured answer (22 Aug 2026): NO.

  - 100/100 plume granules from the pool appear in the portal.
  - Only 6/100 free ones appear, and 5 of those 6 are plumes INSERTED AFTER
    the test cutoff (2025-2026): negatives expired by the movement of the
    catalog. The sixth has a plume since Apr 2024 (odd, ask about it).
  - tile_background does not cross (it is for the multispectral pairs).

Conclusion: free granules live ONLY in the HF full-tile test.

Usage:  python analyze_pool200_portal.py [snapshot_date]
"""
import pathlib
import re
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
FULLTILES = pathlib.Path("/data/users/julio/methanset/data/fulltiles_test_v4c.csv")


def key(t):
    m = re.search(r"(\d{8}[Tt]\d{6})", str(t))
    return m.group(1).lower() if m else None


def main(fecha="2026-08-22"):
    ft = pd.read_csv(FULLTILES)
    portal = pd.read_csv(ROOT / "assets/data/imeo" / fecha /
                         "unep_methanedata_detected_plumes.csv",
                         parse_dates=["insert_date"])
    ft["key"] = ft.tile.map(key)
    portal["key"] = portal.tile.map(key)
    pk = set(portal.key.dropna())

    plume, free = ft[ft.num_plumes > 0], ft[ft.num_plumes == 0]
    print(f"with plume in portal: {plume.key.isin(pk).sum()}/{len(plume)}")
    print(f"free in portal     : {free.key.isin(pk).sum()}/{len(free)}")

    raros = portal[portal.key.isin(set(free.key))]
    if len(raros):
        print("\nfree with plumes in today's portal (expired negatives):")
        print(raros.groupby("key").agg(n=("id_plume", "count"),
                                       primera=("insert_date", "min")).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:])
