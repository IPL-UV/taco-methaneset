"""Step 6 · Recompute the `ch4` leaf of the S2/L89 finetune sets with the new MARS code.

The new MARS code (marss2l, January 2026, IMEO-620) keeps negative ΔXCH4 values; the
published datasets were built with the old code (`clip(..., 0, None)`) and the negatives
collapsed to 0, which is also the nodata value. This script recomputes the leaf from
target + bg0 + plume and rebuilds each tacozip changing ONLY the ch4: every other leaf
stays byte-identical, and the internal tables (level0, level1, __meta__) only get new
offsets/sizes (they shift) plus the ch4 stats/header. The GeoTIFF description changes
from DeltaCH4(ppm) to DeltaCH4(ppb).

Requires the MARS retrieval code and the band-integrated LUT. Both are public:
  - https://github.com/UNEP-IMEO-MARS/marss2l  (LGPL-3.0)
  - integrated_transmittances.json (in this dataset repo root, and shipped with marss2l)
A vendored copy used for the October 2026 run lives in the vault at
`code/multispectral-finetune-bg/ch4fix/` (marss2l modules + marshsi stubs + the LUT).

Usage:
  python 06_recompute_ch4.py --sensor s2 --out DIR [--only COUNTRY] [--limit N]
"""

from __future__ import annotations

import argparse
import io
import os
import pathlib
import shutil
import struct
import sys
import tempfile
import zipfile

BASE = pathlib.Path(__file__).parent
sys.path.insert(0, str(BASE / "ch4fix"))

try:
    import pyproj

    os.environ.setdefault("PROJ_DATA", pyproj.datadir.get_data_dir())
except Exception:
    pass

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio
import tacotiff
import tacozip
from osgeo import gdal
from marss2l.mars_sentinel2.transmittance_to_ch4 import (
    TransmittanceCH4InterpolationFromDict,
    compute_xch4_retrieval,
)

gdal.UseExceptions()

SENSORS = {
    "s2": {
        "id": "methaneset-s2-finetune",
        "root": pathlib.Path("/data/databases/METHANESET_TACOS/methaneset-s2-finetune"),
        "b11": 11,
        "b12": 12,
    },
    "l89": {
        "id": "methaneset-l89-finetune",
        "root": pathlib.Path("/data/databases/METHANESET_TACOS/methaneset-l89-finetune"),
        "b11": 5,
        "b12": 6,
    },
}

COG_OPTS = ["COMPRESS=ZSTD", "PREDICTOR=2", "BLOCKSIZE=224", "BIGTIFF=IF_SAFER",
            "OVERVIEWS=NONE", "INTERLEAVE=TILE"]
PERCENTILES = [25, 50, 75, 95]

LUT = BASE / "ch4fix" / "integrated_transmittances.json"
if not LUT.exists():
    LUT = pathlib.Path("integrated_transmittances.json")
INTERP = TransmittanceCH4InterpolationFromDict(str(LUT))
INTERP.allow_negative_ch4 = True


def read_array(data: bytes):
    with rasterio.open(io.BytesIO(data)) as src:
        return src.read().astype("float32"), src.transform, src.crs


def compute_ch4(zin: zipfile.ZipFile, sid: str, row, cfg) -> tuple[np.ndarray, object, object]:
    target, transform, crs = read_array(zin.read(f"DATA/{sid}/target"))
    bg0, _, _ = read_array(zin.read(f"DATA/{sid}/bg0"))
    with rasterio.open(io.BytesIO(zin.read(f"DATA/{sid}/plume"))) as src:
        plume = src.read(1).astype(bool)
    ch4 = compute_xch4_retrieval(
        target,
        bg0,
        offshore=bool(row["detection:offshore"]),
        satellite=str(row["target:sensor"]),
        sza=float(row["target:sza"]),
        vza=float(row["target:vza"]),
        b11_index=cfg["b11"],
        b12_index=cfg["b12"],
        label=plume,
        corregister=False,
        transmittance_interpolator=INTERP,
    )
    ch4 = ch4.values if hasattr(ch4, "values") else ch4
    return ch4.astype("float64"), transform, crs


def write_cog(ch4: np.ndarray, transform, crs, path: pathlib.Path) -> None:
    tmp = path.with_suffix(".plain.tif")
    with rasterio.open(
        tmp, "w", driver="GTiff", height=ch4.shape[0], width=ch4.shape[1], count=1,
        dtype="float64", crs=crs, transform=transform, nodata=0,
    ) as dst:
        dst.write(ch4, 1)
        dst.set_band_description(1, "DeltaCH4(ppb)")
    gdal.Translate(str(path), str(tmp), format="COG", creationOptions=COG_OPTS)
    tmp.unlink()


def stats_and_header(path: pathlib.Path):
    ds = gdal.Open(str(path))
    band = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    min_v, max_v, mean_v, std_v = band.GetStatistics(True, True)
    arr = band.ReadAsArray()
    valid_pct = float((arr != nodata).mean() * 100.0) if nodata is not None else 100.0
    hist = band.GetHistogram(min_v, max_v, 100)
    cum = np.cumsum(hist)
    total = cum[-1] if cum[-1] > 0 else 1
    width = (max_v - min_v) / 100.0
    pcts = []
    for p in PERCENTILES:
        idx = int(min(np.searchsorted(cum, (p / 100.0) * total), 99))
        pcts.append(float(min_v + (idx + 0.5) * width))
    stats = [np.array([min_v, max_v, mean_v, std_v, valid_pct, *pcts], dtype="float32")]
    header = tacotiff.metadata_from_tiff(str(path))
    ds = None
    return stats, header


def write_entry(zout: zipfile.ZipFile, name: str, data: bytes) -> tuple[int, int]:
    before = zout.fp.tell()
    zout.writestr(name, data)
    zout.fp.flush()
    zout.fp.seek(before)
    head = zout.fp.read(30)
    nlen, elen = struct.unpack("<HH", head[26:30])
    zout.fp.seek(0, 2)
    return before + 30 + nlen + elen, len(data)


def rebuild_zip(src_zip: pathlib.Path, dst_zip: pathlib.Path, cfg: dict, limit: int | None):
    zin = zipfile.ZipFile(src_zip)
    l0 = pq.read_table(io.BytesIO(zin.read("METADATA/level0.parquet")))
    l1 = pq.read_table(io.BytesIO(zin.read("METADATA/level1.parquet")))
    d0, d1 = l0.to_pydict(), l1.to_pydict()
    sample_order = d0["id"]
    if limit:
        sample_order = sample_order[:limit]

    tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="ch4_"))
    ch4_info = {}
    for i, sid in enumerate(sample_order):
        row = {k: d0[k][i] for k in ("detection:offshore", "target:sensor", "target:sza", "target:vza")}
        arr, transform, crs = compute_ch4(zin, sid, row, cfg)
        cog = tmpdir / f"{sid}.tif"
        write_cog(arr, transform, crs, cog)
        stats, header = stats_and_header(cog)
        ch4_info[sid] = {"data": cog.read_bytes(), "stats": stats, "header": header}

    dst_zip.parent.mkdir(parents=True, exist_ok=True)
    zout = zipfile.ZipFile(dst_zip, "w", zipfile.ZIP_STORED)
    offsets: dict[str, tuple[int, int]] = {}
    for name in zin.namelist():
        if name == "TACO_HEADER":
            write_entry(zout, name, b"\x00" * 116)
        elif name.endswith("/ch4"):
            sid = name.split("/")[1]
            offsets[name] = write_entry(zout, name, ch4_info[sid]["data"])
        elif name.endswith("/__meta__"):
            sid = name.split("/")[1]
            meta = pq.read_table(io.BytesIO(zin.read(name)))
            dm = meta.to_pydict()
            for j, leaf in enumerate(dm["id"]):
                leaf_name = f"DATA/{sid}/{leaf}"
                off, size = offsets[leaf_name]
                dm["internal:offset"][j] = off
                dm["internal:size"][j] = size
                if leaf == "ch4":
                    dm["geotiff:stats"][j] = ch4_info[sid]["stats"]
                    dm["taco:header"][j] = ch4_info[sid]["header"]
            buf = io.BytesIO()
            pq.write_table(pa.table(dm, schema=meta.schema), buf, compression="zstd")
            offsets[name] = write_entry(zout, name, buf.getvalue())
        elif name == "METADATA/level0.parquet":
            for i, sid in enumerate(sample_order):
                off, size = offsets[f"DATA/{sid}/__meta__"]
                d0["internal:offset"][i] = off
                d0["internal:size"][i] = size
            buf = io.BytesIO()
            pq.write_table(pa.table(d0, schema=l0.schema), buf, compression="zstd")
            offsets[name] = write_entry(zout, name, buf.getvalue())
        elif name == "METADATA/level1.parquet":
            for j, sid in enumerate(d1["internal:parent_id"]):
                sid = sample_order[sid] if isinstance(sid, int) else sid
                leaf = d1["id"][j]
                off, size = offsets[f"DATA/{sid}/{leaf}"]
                d1["internal:offset"][j] = off
                d1["internal:size"][j] = size
                if leaf == "ch4":
                    d1["geotiff:stats"][j] = ch4_info[sid]["stats"]
                    d1["taco:header"][j] = ch4_info[sid]["header"]
            buf = io.BytesIO()
            pq.write_table(pa.table(d1, schema=l1.schema), buf, compression="zstd")
            offsets[name] = write_entry(zout, name, buf.getvalue())
        else:
            offsets[name] = write_entry(zout, name, zin.read(name))
    zout.close()
    zin.close()
    tacozip.update_header(
        str(dst_zip),
        [offsets["METADATA/level0.parquet"], offsets["METADATA/level1.parquet"], offsets["COLLECTION.json"]],
    )
    shutil.rmtree(tmpdir, ignore_errors=True)

    l0n = pa.table(d0, schema=l0.schema)
    l1n = pa.table(d1, schema=l1.schema)
    return l0n, l1n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensor", choices=["s2", "l89"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=None, help="process a single country")
    ap.add_argument("--limit", type=int, default=None, help="only N samples per zip (test)")
    args = ap.parse_args()

    cfg = SENSORS[args.sensor]
    root = cfg["root"]
    out = pathlib.Path(args.out)
    (out / ".tacocat").mkdir(parents=True, exist_ok=True)

    m0 = pq.read_table(root / ".tacocat" / "level0.parquet")
    order = list(pd.unique(m0.column("internal:source_file").to_pandas()))

    updated0, updated1 = [], []
    for src_name in order:
        if args.only and args.only not in src_name:
            continue
        src_zip = root / src_name
        dst_zip = out / src_name
        n0, n1 = rebuild_zip(src_zip, dst_zip, cfg, args.limit)
        updated0.append(n0)
        updated1.append(n1)
        print(f"[ok] {src_name}  {n0.num_rows} samples", flush=True)

    if args.only:
        print("single country: merged tables not written")
        return

    merged0 = pa.concat_tables(updated0)
    merged1 = pa.concat_tables(updated1)
    pq.write_table(merged0, out / ".tacocat" / "level0.parquet", compression="zstd")
    pq.write_table(merged1, out / ".tacocat" / "level1.parquet", compression="zstd")
    shutil.copy(root / ".tacocat" / "COLLECTION.json", out / ".tacocat" / "COLLECTION.json")
    print(f"merged tables: {merged0.num_rows} samples, {merged1.num_rows} leaves -> {out}")


if __name__ == "__main__":
    main()
