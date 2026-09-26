"""Inventory of the Hugging Face UNEP-IMEO/MARS-Hyperspectral repo WITHOUT downloading it.

The repo ships a 7.8 MB .gitattributes with 70,067 lines: it is the LFS manifest
and lists ALL files in the repo, downloaded or not. Parsing it yields the
complete inventory (11,959 tiles) without fetching a single GeoTIFF.

Usage:  python inventory_hf_manifest.py
"""
import collections
import pathlib

MANIFEST = pathlib.Path("/data/databases/MARS-Hyperspectral-old/.gitattributes")
CARPETAS = ["EMIT", "PRISMA", "EnMAP",
            "EMIT_fulltiles", "PRISMA_fulltiles", "EnMAP_fulltiles"]


def lfs_paths(manifest=MANIFEST):
    lineas = manifest.read_text().splitlines()
    return [l.split(" filter=")[0].strip() for l in lineas if " filter=" in l and "/" in l]


def main():
    paths = lfs_paths()
    print(f"{len(paths)} files listed in the manifest\n")
    total = 0
    for top in CARPETAS:
        sub = [p for p in paths if p.startswith(top + "/")]
        tiles = {p.split("/")[1] for p in sub}
        ficheros = collections.Counter(p.split("/")[-1] for p in sub)
        # split csv files hang directly from the folder, they are not tiles
        splits = [f for f in ficheros if f.endswith(".csv")]
        n_tiles = len(tiles) - len(splits)
        total += n_tiles
        print(f"{top:18s} {n_tiles:6d} tiles")
        for k, v in ficheros.most_common():
            print(f"      {k:52s} {v}")
        print()
    print(f"TOTAL {total} tiles")


if __name__ == "__main__":
    main()
