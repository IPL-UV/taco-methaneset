"""Measures the MethaneSET multispectral chips (Sentinel-2 and Landsat-8/9).

Answers: how big they are, at what resolution, how many bands, and how much
ground they cover compared with the IMEO hyperspectral tile.

Usage:  python measure_multispectral_chips.py
"""
import glob
import pathlib

import rasterio

BASE = pathlib.Path("/data/databases/METHANE_DATASETS_MULTISPECTRAL")
CASOS = [("Sentinel-2", "S2/methaneset-s2-finetune"),
         ("Landsat-8", "LC08/methaneset-l89-finetune"),
         ("Landsat-9", "LC09/methaneset-l89-finetune")]
TILE_IMEO_PX, TILE_IMEO_M = 256, 60


def main():
    for nombre, sub in CASOS:
        carpetas = sorted((BASE / sub).iterdir())
        muestra = glob.glob(str(carpetas[0] / "*.tif"))[0]
        with rasterio.open(muestra) as src:
            lado_m = src.width * src.res[0]
            print(f"{nombre}")
            print(f"   folders (one per plume)  : {len(carpetas)}")
            print(f"   size                     : {src.width} x {src.height} px")
            print(f"   resolution               : {src.res[0]:.0f} m")
            print(f"   ground side              : {lado_m/1000:.1f} km")
            print(f"   bands                    : {src.count} {src.descriptions}")
            print(f"   dtype / nodata           : {src.dtypes[0]} / {src.nodata}")
            print(f"   backgrounds per plume    : {len(glob.glob(str(carpetas[0]/'*.tif')))}")
            print()
    lado = TILE_IMEO_PX * TILE_IMEO_M / 1000
    print(f"Reference, IMEO hyperspectral tile: "
          f"{TILE_IMEO_PX} px at {TILE_IMEO_M} m = {lado:.1f} km per side")


if __name__ == "__main__":
    main()
