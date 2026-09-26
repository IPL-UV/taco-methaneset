"""Scans all-null references in the already extracted pretraining files.

Usage: python scan_ref_nulls.py s2|l89
"""
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import rasterio

WORK = Path("/data/databases/_pre_work_julio")


def scan_sample(sample_dir: Path):
    p = sample_dir / "reference"
    if not p.exists():
        return sample_dir.name, None
    try:
        with rasterio.open(p) as r:
            a = r.read()
            nod = r.nodata
            if nod is not None and (a == nod).all():
                return sample_dir.name, True
    except Exception:
        return sample_dir.name, None
    return sample_dir.name, False


def main():
    which = sys.argv[1]
    ds = "methaneset-s2-pretraining" if which == "s2" else "methaneset-l89-pretraining"
    ex = WORK / ds / "extract"
    dirs = [d for zdir in sorted(ex.iterdir()) if zdir.is_dir() for d in (zdir / "DATA").iterdir() if d.is_dir()]
    print(f"{ds}: {len(dirs)} samples to scan", flush=True)
    nulos, done = [], 0
    with ProcessPoolExecutor(max_workers=24) as pool:
        futs = {pool.submit(scan_sample, d): d for d in dirs}
        for f in as_completed(futs):
            sid, es_nulo = f.result()
            done += 1
            if es_nulo:
                nulos.append(sid)
            if done % 10000 == 0:
                print(f"  {done}/{len(dirs)} | nulls: {len(nulos)}", flush=True)
    out = Path(f"/tmp/opencode/excluir_ref_{ds}.txt")
    out.write_text("\n".join(sorted(nulos)) + "\n")
    print(f"{ds}: all-null references: {len(nulos)} -> {out}")


if __name__ == "__main__":
    main()
