"""Step 9 · Reads the TACO back with tacoreader and compares it with the raw table. The raw can only
be deleted once everything comes out ALL OK.

Usage (deep environment, tacoreader 2.x): python 09_verify_taco.py [--muestra N]
"""
import argparse
import numpy as np
import pandas as pd
import rasterio
import tacoreader
from banco import config as C
from banco.esquema import COLUMNAS

ap = argparse.ArgumentParser()
ap.add_argument("--muestra", type=int, default=500)
a = ap.parse_args()
fallos = 0


def check(cond, msg):
    global fallos
    fallos += 0 if cond else 1
    print(("  OK    " if cond else "  FAIL  ") + msg)


t = pd.read_parquet(C.RAW / "tabla.parquet").set_index("id")
ds = tacoreader.load(str(C.FINAL / ".tacocat"))
d = ds.data.to_arrow().to_pandas()
print(f"TACO: {ds.id} v{getattr(ds, 'version', '?')} · {d.shape[0]} rows")
check(len(d) == len(t), f"rows: {len(d)} (table {len(t)})")
check(d["id"].is_unique and set(d["id"]) == set(t.index), "same ids as the table")
faltan = [c for c in COLUMNAS[2:] if c not in d.columns]
check(not faltan, f"all schema columns ({len(COLUMNAS) - 2}) {faltan or ''}")
check(list(d.columns[:len(COLUMNAS)]) == ["id", "type"] + COLUMNAS[2:],
      "columns in schema order: id, type, methane, sun, edge, array, qa")
d = d.set_index("id").loc[t.index]
num = [c for c in COLUMNAS[2:] if not pd.api.types.is_string_dtype(t[c])]
dif = max(float(np.nanmax(np.abs(d[c].astype(float).values - t[c].astype(float).values))) for c in num)
check(dif == 0, f"identical numeric values (maximum difference {dif})")
txt = all((d[c].values == t[c].values).all() for c in COLUMNAS[2:] if pd.api.types.is_string_dtype(t[c]))
check(txt, "identical texts")

print(f"sample of {a.muestra} GeoTIFFs read from inside the TACO")
vsi = next(c for c in d.columns if "vsi" in c.lower())
mal = 0
for i in np.random.default_rng(0).choice(len(t), a.muestra, replace=False):
    idx = t.index[i]
    with rasterio.open(d.loc[idx, vsi]) as s:
        x = s.read(1).astype("float64")
    with rasterio.open(C.RAW / t.loc[idx, "path"]) as s:
        y = s.read(1).astype("float64")
    mal += not (x.shape == y.shape and np.array_equal(x, y))
check(mal == 0, f"GeoTIFFs different from the raw: {mal}")
for f in ("README.md", "index.html"):
    check((C.FINAL / f).exists(), f"{f} generated")
print("ALL OK: the raw can be deleted" if fallos == 0 else f"{fallos} FAILURES: do not delete the raw")
