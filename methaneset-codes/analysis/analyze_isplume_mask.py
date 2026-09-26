"""Crosses the isplume flag against the real mask in the local IMEO tiles.

Result (22 Aug 2026, 1,005 tiles in /data/databases/MARS-Hyperspectral-old):
             with pixels   empty
  true (404)     404          0
  false (591)     83        508

The trap is in the false ones: 83 (14%) contain labeled plume pixels.
Using isplume as a negative contaminates 14% of the negatives.

Usage:  python analyze_isplume_mask.py
"""
import json
import pathlib

import rasterio

BANCO = pathlib.Path("/data/databases/MARS-Hyperspectral-old/EMIT")


def main():
    stats = {"true_mask": 0, "true_empty": 0, "false_mask": 0, "false_empty": 0}
    for tile in sorted(BANCO.iterdir()):
        info_f, mask_f = tile / "info.json", tile / "plumemask.tif"
        if not (info_f.exists() and mask_f.exists()):
            continue
        info = json.loads(info_f.read_text())
        with rasterio.open(mask_f) as src:
            con_pluma = bool((src.read(1) > 0).any())
        clave = ("true" if info.get("isplume") else "false") + ("_mask" if con_pluma else "_empty")
        stats[clave] += 1
    print(f"{'':14s}{'with pixels':>12s}{'empty':>8s}")
    print(f"{'true':14s}{stats['true_mask']:12d}{stats['true_empty']:8d}")
    print(f"{'false':14s}{stats['false_mask']:12d}{stats['false_empty']:8d}")
    n_false = stats["false_mask"] + stats["false_empty"]
    print(f"\nfalse with gas inside: {stats['false_mask']}/{n_false} "
          f"({100*stats['false_mask']/n_false:.0f}% of the 'negatives')")


if __name__ == "__main__":
    main()
