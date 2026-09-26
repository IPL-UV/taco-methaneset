"""MODULE 10b of the v2 pipeline: leaves the radiance as COG + DISCARD_LSB (Cesar's zones).

Final format (the one described by the paper and the README):
  COG (LAYOUT=COG), ZSTD, PREDICTOR=2, BLOCKSIZE=128, INTERLEAVE=TILE
  + DISCARD_LSB per SNR zones: (381,1000,8) (1000,1340,0) (1340,1450,16)
  (1450,1800,0) (1800,1960,16) (1960,2494,0). lsb=0 -> lossless.

Why 128: benchmark from module 08 (patch 128: 70 ms; patch 256: 102 ms;
full scene 4.3 s; size +1.5% versus 512). Matches the other 9 layers of the
sample and the README.

Two steps: GTiff zstd+LSB -> COG CreateCopy (direct COG with DISCARD_LSB
is pathologically slow). Resumable: skips scenes already COG 128 with low
bits at zero; verifies structure and dimensions before replacing (atomic).

Usage:
  nohup /data/users/julio/.conda/envs/deep/bin/python code/v2/10b_apply_lsb.py \
        > code/v2/10b_apply_lsb.out 2>&1 &
"""
import argparse
import pathlib
import time
from multiprocessing import Pool

import h5py
import numpy as np
from osgeo import gdal

gdal.UseExceptions()

ROOT = pathlib.Path("/data/databases/METHANESET_TACOS/methaneset-emit/DATA")
NC = pathlib.Path("/data/databases/MARS-Hyperspectral/EMIT_FULL/"
                  "EMIT_L1B_RAD_001_20220810T064957_2222205_033.nc")
ZONES = [(381, 1000, 8), (1000, 1340, 0), (1340, 1450, 16),
         (1450, 1800, 0), (1800, 1960, 16), (1960, 2494, 0)]
INTER_OPTS = ["TILED=YES", "BLOCKXSIZE=128", "BLOCKYSIZE=128", "COMPRESS=ZSTD",
              "PREDICTOR=2", "BIGTIFF=YES", "NUM_THREADS=ALL_CPUS"]
COG_OPTS = ["COMPRESS=ZSTD", "PREDICTOR=2", "BLOCKSIZE=128", "OVERVIEWS=NONE",
            "INTERLEAVE=TILE", "BIGTIFF=YES"]
LOG = pathlib.Path(__file__).with_suffix(".log")


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    LOG.open("a").write(line + "\n")


def lsb_string():
    with h5py.File(NC) as h:
        wl = h["sensor_band_parameters/wavelengths"][:].ravel()
    bits = []
    for w in wl:
        for a, b, l in ZONES:
            if a <= w < b:
                bits.append(str(l))
                break
        else:
            bits.append("0")
    return ",".join(bits)


def estado_cog(p):
    """(ok, dims): ok if it is COG with tile 128 and low bits of band 1 at zero."""
    ds = gdal.Open(str(p))
    if ds is None:
        return False, (0, 0)
    md = ds.GetMetadata("IMAGE_STRUCTURE")
    dims = (ds.RasterXSize, ds.RasterYSize)
    b = ds.GetRasterBand(1)
    block = b.GetBlockSize()
    arr = b.ReadAsArray()
    ds = None
    v = arr.astype(np.float32).view(np.uint32)
    ok = (md.get("LAYOUT") == "COG" and block == [128, 128]
          and float((v & 255 == 0).mean()) > 0.99)
    return ok, dims


def reescribir(p, lsb):
    inter = pathlib.Path(str(p) + ".inter.tif")
    tmp = pathlib.Path(str(p) + ".tmp.tif")
    ds = gdal.Open(str(p))
    gdal.Translate(str(inter), ds, options=gdal.TranslateOptions(
        creationOptions=INTER_OPTS + [f"DISCARD_LSB={lsb}"]))
    ds = None
    dsi = gdal.Open(str(inter))
    gdal.GetDriverByName("COG").CreateCopy(str(tmp), dsi, options=COG_OPTS)
    dsi = None
    inter.unlink()
    return tmp


def procesar(p):
    try:
        ok, dims_orig = estado_cog(p)
        if ok:
            return p.parent.name, "skipped"
        tmp = reescribir(p, lsb_string())
        ok2, dims = estado_cog(tmp)
        if not ok2 or dims != dims_orig:
            tmp.unlink(missing_ok=True)
            return p.parent.name, f"verification failed (cog={ok2}, dims={dims} vs {dims_orig})"
        tmp.replace(p)
        return p.parent.name, "done"
    except Exception as e:  # noqa: BLE001
        return p.parent.name, f"failed {type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    escenas = sorted(p / "radiance.tif" for p in ROOT.iterdir()
                     if (p / "radiance.tif").exists())
    log(f"{len(escenas)} scenes | workers {a.workers} | COG+128+LSB format")
    hechas = saltadas = fallos = 0
    with Pool(a.workers) as pool:
        for i, (nombre, estado) in enumerate(pool.imap_unordered(procesar, escenas), 1):
            if estado == "done":
                hechas += 1
            elif estado == "skipped":
                saltadas += 1
            else:
                fallos += 1
                log(f"[{estado}] {nombre}")
            if i % 25 == 0:
                log(f"  {i}/{len(escenas)} | done {hechas} | skipped {saltadas} | failed {fallos}")
    log(f"END: done {hechas} | skipped {saltadas} | failed {fallos}")


if __name__ == "__main__":
    main()
