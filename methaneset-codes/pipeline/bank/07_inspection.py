"""Step 7 · Automatic checks of the table and the GeoTIFFs, before looking at anything by hand.

Usage: python 07_inspection.py DIR [--muestra N]
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from banco import config as C
from banco.esquema import COLUMNAS
from banco.geometria import rejilla

ap = argparse.ArgumentParser()
ap.add_argument("dir", type=Path)
ap.add_argument("--muestra", type=int, default=300)
a = ap.parse_args()
t = pd.read_parquet(a.dir / "tabla.parquet")
sel = pd.read_csv(C.TRABAJO / "seleccion.csv")
crit = pd.read_csv(C.TRABAJO / "criterios.csv")
nod = pd.DataFrame(rejilla(), columns=["anillo", "sza", "raa"])
fallos = 0


def check(cond, msg):
    global fallos
    fallos += 0 if cond else 1
    print(("  OK    " if cond else "  FAIL  ") + msg)


print("counts")
check(list(t.columns) == COLUMNAS, f"columns in schema order ({t.shape[1]})")
n_pl = t["methane:plume_uid"].nunique()
check(n_pl == len(sel), f"physical plumes: {n_pl} (selected {len(sel)})")
check(len(t) == len(sel) * len(nod), f"rows: {len(t)} (expected {len(sel) * len(nod)})")
check(t.groupby("methane:plume_uid").size().eq(len(nod)).all(), f"{len(nod)} rows per plume")
check(t["id"].is_unique, "unique ids")
check(not t.isna().any().any(), f"no gaps ({int(t.isna().sum().sum())} empty cells)")
print(t.groupby("methane:emitter")["methane:plume_uid"].nunique().to_string())

print("consistency with the criteria")
m = t.drop_duplicates("methane:plume_uid").copy()
m["plume"] = m["methane:emitter"].str[1:].astype(int)
m = m.merge(crit, left_on=["methane:sim_type", "methane:wind_speed", "plume", "methane:snapshot_index"],
            right_on=["tipo", "ws", "plume", "snap"])
for col_t, col_c in [("methane:peak_ppb", "pico"), ("edge:downwind_ppb", "edge"), ("edge:upwind_ppb", "entrada"),
                     ("edge:downwind_slope", "slope"), ("edge:downwind_ref_ppb", "interior_min")]:
    err = np.nanmax(np.abs(m[col_t] - m[col_c]) / np.maximum(np.abs(m[col_c]), 1e-9))
    check(err < 1e-3, f"{col_t} same as in the criteria (maximum relative error {err:.1e})")
check((m["edge:downwind_slope"] < 0).all() and (m["edge:downwind_ppb"] < m["edge:downwind_ref_ppb"]).all(), "all fade out")
check((m["edge:upwind_frac"] < C.ENTRADA_FRAC).all(), "all with a clean upwind edge (west)")
check((m["edge:top_frac"] < C.LATERAL_FRAC).all() and (m["edge:bottom_frac"] < C.LATERAL_FRAC).all(),
      "all with clean north and south edges")
err = np.nanmax(np.abs(m["qa:foreign_mass_frac"] - m["fuera_frac"]))
check(err < 1e-6, f"qa:foreign_mass_frac same as in the criteria (maximum difference {err:.1e})")
check((m["qa:foreign_mass_frac"] < C.FUERA_MAX).all(), f"all with less than {100 * C.FUERA_MAX:.1f}% of foreign methane")
err = np.nanmax(np.abs(m["edge:downwind_width_m"] - m["ancho_borde_ruido"]))
check(err < 1e-6, f"edge:downwind_width_m same as in the criteria (maximum difference {err:.1e})")
check((m["edge:downwind_width_m"] <= C.ANCHO_BORDE_MAX).all(), f"all leave with {C.ANCHO_BORDE_MAX:.0f} m of width or less")

print("quality")
perd = 1 - t["qa:ime_after"] / t["qa:ime_before"]
check(perd.max() <= C.PRESUPUESTO_MASA * (1 + 1e-9), f"zeros: maximum loss {100 * perd.max():.3f}%, median {100 * perd.median():.3f}%")
g = ["array:grow_upwind_px", "array:grow_downwind_px", "array:grow_top_px", "array:grow_bottom_px"]
check((t.loc[t["sun:sza"] == 0, g] <= 0).all().all(), "with the sun overhead the image never grows")
check((t["array:grow_downwind_px"] >= 0).all(), "never cropped on the downwind side (the plume reaches the edge)")
check(t["qa:threshold_rel"].max() <= C.UMBRAL_CEROS * (1 + 1e-9), "threshold never above 1e-4 of the maximum")
check(t["sun:sza"].round(6).isin(nod.sza.round(6)).all() and t["sun:raa"].between(0, 360, inclusive="left").all(), "angles inside the grid")

print(f"sample of {a.muestra} GeoTIFFs")
rng = np.random.default_rng(0)
mal = 0
with rasterio.Env():
    for i in rng.choice(len(t), min(a.muestra, len(t)), replace=False):
        r = t.iloc[i]
        with rasterio.open(a.dir / r["path"]) as src:
            x = src.read(1).astype("float64")
            fx, fy = src.transform * (r["array:source_col"] + 0.5, r["array:source_row"] + 0.5)
            ok = (x.shape == (r["array:height"], r["array:width"]) and src.tags().get("units") == "ppb"
                  and src.tags().get("plume_uid") == r["methane:plume_uid"]
                  and abs(x.sum() - r["qa:ime_after"]) <= 1e-4 * r["qa:ime_after"]
                  and abs(x.max() - r["array:peak_ppb"]) <= 1e-4 * r["array:peak_ppb"]
                  and abs(fx) < 1e-6 and abs(fy) < 1e-6 and (x >= 0).all())
            mal += not ok
check(mal == 0, f"GeoTIFFs that do not match their row: {mal}")
tam = sum(p.stat().st_size for p in (a.dir / "tifs").rglob("*.tif"))
print(f"on-disk size of the GeoTIFFs: {tam / 1e9:.2f} GB")
print("ALL OK" if fallos == 0 else f"{fallos} FAILURES")
