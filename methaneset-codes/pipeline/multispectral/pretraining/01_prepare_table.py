"""Extracts the pretraining tacozips and prepares the table (renames + exclusions) and paths.

Usage: python 01_prepare_table.py s2|l89
"""
import sys
import zipfile
from pathlib import Path

import pandas as pd

BASE = Path("/data/databases/METHANE_DATASETS_TACOv2")
WORK = Path("/data/databases/_pre_work_julio")

REN = {
    "satellite:platform": "target:sensor",
    "satellite:tile": "target:tile",
    "satellite:sza": "target:sza",
    "satellite:vza": "target:vza",
    "satellite:background_tile": "bg:tile0",
}
DROP = ["stac:centroid", "stac:time_end", "stac:time_middle"]
HOJAS = ["target", "reference", "dem"]


def run(ds: str) -> None:
    ex_dir = WORK / ds / "extract"
    raw = WORK / ds / "raw"
    ex_dir.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)

    zips = sorted((BASE / ds).glob("*.tacozip"))
    tablas, rows = [], []
    for i, zp in enumerate(zips, 1):
        d = ex_dir / zp.stem
        if not d.exists():
            with zipfile.ZipFile(zp) as z:
                z.extractall(d)
        l0 = pd.read_parquet(d / "METADATA" / "level0.parquet")
        l0 = l0[[c for c in l0.columns if not c.startswith("internal:") and c != "type"]]
        tablas.append(l0)
        for sid in sorted(x.name for x in (d / "DATA").iterdir() if x.is_dir()):
            r = {"id": sid}
            for h in HOJAS:
                p = d / "DATA" / sid / h
                r[h] = str(p) if p.exists() else ""
            rows.append(r)
        print(f"{ds} [{i}/{len(zips)}] {zp.name}", flush=True)

    tabla = pd.concat(tablas, ignore_index=True)
    excl = set()
    for tag in ("pre", "ref"):
        f = Path(f"/tmp/opencode/excluir_{tag}_{ds}.txt")
        if f.exists():
            s = set(f.read_text().split())
            print(f"{ds}: exclusions {tag}: {len(s)}")
            excl |= s
    if excl:
        antes = len(tabla)
        tabla = tabla[~tabla["id"].isin(excl)].reset_index(drop=True)
        rows = [r for r in rows if r["id"] not in excl]
        print(f"{ds}: excluded {antes - len(tabla)} samples")

    tabla = tabla.rename(columns=REN)
    tabla = tabla[[c for c in tabla.columns if c not in DROP]]
    tabla.to_csv(raw / "tabla_propuesta.csv", index=False)
    pd.DataFrame(rows).to_csv(raw / "paths_cesar.csv", index=False)
    print(f"{ds}: table {tabla.shape} | paths {len(rows)}")
    print(f"{ds}: columns {list(tabla.columns)}")


if __name__ == "__main__":
    which = sys.argv[1]
    run("methaneset-s2-pretraining" if which == "s2" else "methaneset-l89-pretraining")
