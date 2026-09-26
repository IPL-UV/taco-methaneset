"""Step 3 · Automatic checks of the table and the GeoTIFFs of the raw of the les bank.

Usage: python 03_inspection.py
"""
import numpy as np
import pandas as pd
import rasterio
from bankles import config as C
from bankles.esquema import COLUMNAS

fallos = 0


def check(cond, msg):
    global fallos
    fallos += 0 if cond else 1
    print(("  OK    " if cond else "  FAIL  ") + msg)


t = pd.read_parquet(C.RAW / "tabla.parquet")
proy = pd.read_csv(C.PROYECTADO)
base = {(ti, float(w), p, s) for ti in C.TIPOS for w in C.WS for p in C.PLUMAS for s in C.SNAPS}
extra = {(r.tipo, float(r.ws), int(r.plume), int(r.snap)) for r in proy.itertuples()}
n_esp = len(base | extra)
print(f"table: {len(t)} rows, {t['id'].nunique()} ids ({len(base)} from the 5-min set + {len(extra) - len(base & extra)} from the projected bank)")
check(len(t) == n_esp, f"rows: {len(t)} (expected {n_esp})")
check(list(t.columns) == COLUMNAS, "columns in the order of bankles/esquema.py")
check(t["id"].is_unique, "unique ids")
check(t["id"].tolist() == t["path"].str.split("/").str[-1].str.replace(".tif", "", regex=False).tolist(),
      "path = id")
check(set(zip(t["methane:sim_type"], t["methane:emitter"])) ==
      {(ti, C.emisor(ti, p)) for ti in C.TIPOS for p in C.PLUMAS}, "all 18 emitters present")
check(set(t["methane:wind_speed"]) == set(C.WS), "all 11 wind speeds present")

L = t["grid:level_heights_m"].apply(len)
check((L == C.NZ).all(), f"level_heights has {C.NZ} values in every row")
check(t["grid:level_heights_m"].apply(lambda h: bool(np.all(np.diff(h) > 0))).all(),
      "level_heights strictly increasing")
check((t["methane:peak_ppb"] > 0).all(), "positive peak in all")
check((t["qa:zeros_frac"] > 0).all(), "there are cells exactly zero")
check((t["methane:ime_ppb_px"] > 0).all(), "positive IME")

esperado = (t["criteria:fades_out"] & t["criteria:ent_ok"] & t["criteria:lat_ok"] & t["criteria:fuera_ok"]
            & t["criteria:ancho_ok"])
check((t["criteria:pasa"] == esperado).all(), "criteria:pasa = the five flags")
check((t["criteria:ancho_ok"] == (t["edge:downwind_width_m"] <= C.ANCHO_BORDE_MAX)).all(),
      f"ancho_ok = downwind width <= {C.ANCHO_BORDE_MAX:.0f} m")
check((t["criteria:fuera_ok"] == (t["edge:foreign_mass_frac"] < C.FUERA_MAX)).all(),
      f"fuera_ok = foreign methane < {C.FUERA_MAX}")
check((t["edge:touches_edge"] == (t["edge:max_frac"] > C.TOCA_BORDE_FRAC)).all(),
      "touches_edge = edge:max_frac > 0.05")

faltan = [C.RAW / p for p in t["path"] if not (C.RAW / p).exists()]
check(not faltan, f"all GeoTIFFs exist ({len(t) - len(faltan)}/{len(t)})")
mal = []
z_mal = 0
for p in (C.RAW / t["path"]).sample(min(200, len(t)), random_state=0):
    with rasterio.open(p) as s:
        if (s.count, s.height, s.width) != (C.NZ, C.NY, C.NX):
            mal.append(p.name)
        else:
            # the mean height of the table is measured on the interior; crop the relaxation band.
            # It serves to catch a band flip: if they are reversed, the mean height comes out at the top
            row = t[t["path"] == str(p.relative_to(C.RAW))].iloc[0]
            a = s.read().astype("float64")[:, C.RELAX:C.NY - C.RELAX, C.RELAX:C.NX - C.RELAX]
            mk = np.clip(a, 0, None).sum(axis=(1, 2))
            zg = np.array(row["grid:level_heights_m"])
            if mk.sum() > 0 and abs((mk * zg).sum() / mk.sum() - row["plume:mean_height_m"]) > 1:
                z_mal += 1
check(not mal, f"200 GeoTIFFs with 49 bands of 90 x 120 (bad: {mal})")
check(z_mal == 0, f"table mean height rebuilt from the TIF (bad: {z_mal})")

print(f"volumes passing the five criteria: {int(t['criteria:pasa'].sum())} of {len(t)}")
print(f"volumes touching an edge (more than 5% of the peak): {int(t['edge:touches_edge'].sum())}")
print("ALL OK" if fallos == 0 else f"{fallos} FAILURES")
