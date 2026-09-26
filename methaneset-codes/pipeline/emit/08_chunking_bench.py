"""MODULE 08 of the v2 pipeline: GeoTIFF chunking benchmark for the dataloader.

Julio's question: which internal tile is best (128 vs 256 vs 512) to read
patches fast, and is saving the full scene tiled equivalent to saving loose
128 tiles?

Method: the real radiance of one scene (285 bands, 1280x1242, float32) is
rewritten with different blocksizes (same zstd and interleave as the current
TACO) and the reading of random 128 and 256 patches is measured (all bands,
which is what the model eats). 30 patches per combination, OS cache dodged
using distinct random offsets.

Output: console table (file size + ms per patch).

Usage:
  /data/users/julio/.conda/envs/deep/bin/python code/v2/08_chunking_bench.py \
        > code/v2/08_chunking_bench.log 2>&1
"""
import pathlib
import time

import numpy as np
import rasterio
from rasterio.windows import Window

SRC = pathlib.Path("/data/databases/METHANE_DATASETS_TACOv2/methaneset-emit/"
                   "methaneset-emit/DATA/EMIT_L1B_RAD_001_20220810T064957_2222205_033/"
                   "radiance.tif")
TMP = pathlib.Path("/tmp/claude-1088/-data-users-julio-Notes/"
                   "eea8c626-cf41-49f7-96d4-e4925a9f8415/scratchpad/chunkbench")
BLOCKS = [64, 128, 256, 512]
PATCHES = [128, 256]
N = 30
RNG = np.random.default_rng(7)


def rewrite(data, profile, block):
    out = TMP / f"radiance_b{block}.tif"
    if out.exists():
        return out
    p = dict(profile)
    p.update(tiled=True, blockxsize=block, blockysize=block,
             compress="zstd", interleave="band")
    with rasterio.open(out, "w", **p) as dst:
        dst.write(data)
    return out


def bench(path, patch):
    with rasterio.open(path) as src:
        h, w = src.shape
        t0 = time.time()
        for _ in range(N):
            r = int(RNG.integers(0, h - patch))
            c = int(RNG.integers(0, w - patch))
            _ = src.read(window=Window(c, r, patch, patch))
        return (time.time() - t0) / N * 1000


def main():
    TMP.mkdir(parents=True, exist_ok=True)
    with rasterio.open(SRC) as src:
        data = src.read()
        profile = src.profile
    print(f"scene: {data.shape} float32 · {SRC.stat().st_size/1e6:.0f} MB "
          f"(zstd, current 512 tile)\n")
    print(f"{'tile':>6} {'file MB':>11}" +
          "".join(f"  patch {p} [ms]" for p in PATCHES))
    for b in BLOCKS:
        f = rewrite(data, profile, b)
        row = f"{b:>6} {f.stat().st_size/1e6:>11.0f}"
        for p in PATCHES:
            row += f"  {bench(f, p):>13.1f}"
        print(row, flush=True)
    print("\nreference: read the WHOLE SCENE from the tile-512 file: ", end="")
    with rasterio.open(SRC) as src:
        t0 = time.time(); _ = src.read()
    print(f"{(time.time()-t0)*1000:.0f} ms")

    # ---- part 2 (Julio's question): PRE-CUT 128 chips as
    # loose files vs a 128 window on the scene chunked to 128 ----
    print("\n== loose chips vs chunked scene (30 reads of 128) ==")
    chip_dir = TMP / "chips"
    chip_dir.mkdir(exist_ok=True)
    p = dict(profile)
    p.update(tiled=True, blockxsize=128, blockysize=128, compress="zstd",
             interleave="band", width=128, height=128)
    offs = [(int(RNG.integers(0, data.shape[1] - 128)),
             int(RNG.integers(0, data.shape[2] - 128))) for _ in range(N)]
    total_chip_mb = 0
    for i, (r, c) in enumerate(offs):
        f = chip_dir / f"chip_{i:02d}.tif"
        if not f.exists():
            with rasterio.open(f, "w", **p) as dst:
                dst.write(data[:, r:r + 128, c:c + 128])
        total_chip_mb += f.stat().st_size / 1e6
    t0 = time.time()
    for i in range(N):
        with rasterio.open(chip_dir / f"chip_{i:02d}.tif") as src:
            _ = src.read()
    ms_chip = (time.time() - t0) / N * 1000
    ms_window = bench(TMP / "radiance_b128.tif", 128)
    print(f"  loose chip (open+read file): {ms_chip:.1f} ms · "
          f"{total_chip_mb/N:.1f} MB per chip")
    print(f"  128 window on tile-128 scene: {ms_window:.1f} ms")
    print("  (the window pays for neighboring blocks when not aligned; "
          "the chip pays for opening one file per patch and freezes the grid)")


if __name__ == "__main__":
    main()
