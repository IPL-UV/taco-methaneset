"""MODULE 04 of the v2 pipeline: nodata scan over the consensus universe.

Julio's criterion: only images WITHOUT nodata enter TACO v2. This module
measures the nodata of each candidate so the filter can be applied with
numbers.

How: EMIT nodata is per whole pixel (the entire spectrum at -9999), so ONE
band is enough to measure it EXACTLY. Important performance detail:
the EMIT nc files come WITHOUT chunking or compression (contiguous layout),
and asking h5py for "one band" generates millions of scattered mini-reads
(10 s/scene and the disk thrashing). The fast way is to read BLOCKS of whole
rows (sequential) and slice the band in memory: ~2 s/scene.

Parallel: 8 workers (more does not help: the limit is disk bandwidth,
not cores). ~180 MB of RAM per worker.

Output: assets/data/cross/<date>/nodata.parquet + summary on console.

Usage (long, nohup + twin log):
  cd 01-Projects/methanset
  nohup /data/users/julio/.conda/envs/deep/bin/python code/v2/04_nodata_scan.py \
        > code/v2/04_nodata_scan.log 2>&1 &
"""
import pathlib
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
import numpy as np
import pandas as pd

DATA = pathlib.Path(__file__).resolve().parent.parent.parent / "assets" / "data"
RAD_ROOTS = [
    pathlib.Path("/data/databases/MARS-Hyperspectral/EMIT_FULL"),
    pathlib.Path("/data/databases/MARS-Hyperspectral_complement/EMIT_FULL"),
]
TS = re.compile(r"(\d{8}T\d{6})")
WORKERS = 8
BAND = 100
FILL = -9990.0
ROWS = 128  # rows per sequential read block


def scan_one(args):
    ts, path = args
    try:
        bad = tot = 0
        with h5py.File(path, "r") as h:
            r = h["radiance"]
            for i in range(0, r.shape[0], ROWS):
                block = r[i:i + ROWS, :, :]        # sequential read
                band = block[:, :, BAND]
                bad += int(((band <= FILL) | ~np.isfinite(band)).sum())
                tot += band.size
        return ts, bad / tot * 100, ""
    except Exception as e:
        return ts, np.nan, f"{type(e).__name__}: {e}"


def main():
    cross_dir = sorted((DATA / "cross").iterdir())[-1]
    g = pd.read_parquet(cross_dir / "granules.parquet")
    cons = g[g.consensus].copy()

    idx = {}
    for root in RAD_ROOTS:
        for f in root.glob("*_RAD_*.nc"):
            m = TS.search(f.name)
            if m:
                idx.setdefault(m.group(1).lower(), str(f))
    cons["rad_path"] = cons.granule_ts.map(idx)
    todo = cons[cons.rad_path.notna()][["granule_ts", "rad_path"]]
    print(f"consensus: {len(cons):,} · with RAD on disk: {len(todo):,} · "
          f"workers: {WORKERS}", flush=True)

    rows, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(scan_one, t)
                for t in todo.itertuples(index=False, name=None)]
        for i, fut in enumerate(as_completed(futs), 1):
            ts, pct, err = fut.result()
            rows.append({"granule_ts": ts, "nodata_pct": pct, "error": err})
            if err:
                print(f"  ERROR {ts}: {err}", flush=True)
            if i % 100 == 0 or i == len(futs):
                rate = i / (time.time() - t0)
                print(f"  {i}/{len(futs)}  ({rate:.1f} scenes/s)", flush=True)

    df = pd.DataFrame(rows).merge(todo, on="granule_ts")
    out = cross_dir / "nodata.parquet"
    df.to_parquet(out, index=False)

    ok = df[df.error == ""]
    print(f"\n== nodata summary ({len(ok):,} scanned, "
          f"{(df.error != '').sum()} errors) ==")
    for th, lbl in [(0.0, "== 0% (clean)"), (5.0, "<= 5%"), (25.0, "<= 25%")]:
        n = (ok.nodata_pct <= th).sum()
        print(f"  {lbl:16s} {n:5,d}  ({n/len(ok):.1%})")
    print(f"  > 25%            {(ok.nodata_pct > 25).sum():5,d}")
    print(f"  median: {ok.nodata_pct.median():.2f}%  ·  max: {ok.nodata_pct.max():.1f}%")
    print(f"\ntable -> {out}  ·  {time.time()-t0:.0f}s total")


if __name__ == "__main__":
    main()
