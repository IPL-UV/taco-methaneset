"""Step 1 · Criteria per snapshot: fades out, clean upwind edge, clean lateral edges and little
foreign methane (fuera-todos.csv, from 01-injection-notes/assets/code/fuera-todos.py).

Input: perfiles-todos.npz (maximum of each column of the column map, 90 rows x 120 columns,
in ppb at 3 t/h, for the 11,880 valid snapshots of the 18 emitters).
Output: trabajo/criterios.csv with, per snapshot, slope, edge, interior minimum, upwind edge,
peak and whether it passes.
"""
import numpy as np
import pandas as pd
from core import config as C

z = np.load(C.PROFILES)
px, pico = z["px"].astype("float64"), z["pico"].astype("float64")
W = C.WINDOW
X = np.arange(W)
Lg = np.log(np.clip(px[:, C.EXIT_COL + 1 - W:C.EXIT_COL + 1], 1e-3, None))
slope = ((X - X.mean()) * (Lg - Lg.mean(axis=1, keepdims=True))).sum(1) / ((X - X.mean()) ** 2).sum()
edge = px[:, C.EXIT_COL]
mmin = px[:, C.INTERIOR[0]:C.INTERIOR[1]].min(axis=1)
entrada = px[:, C.ENTRY_COL]
d = pd.DataFrame(dict(tipo=z["tipo"], ws=z["ws"], plume=z["plume"], snap=z["snap"],
                      minuto=z["snap"] / 2, pico=pico, edge=edge, interior_min=mmin,
                      entrada=entrada, slope=slope))
d["apaga"] = (d.slope < 0) & (d.edge < d.interior_min)
d["ent_ok"] = d.entrada < C.ENTRY_FRAC * d.pico
lat = pd.read_csv(C.SIDES)[["tipo", "ws", "plume", "snap", "norte_max", "sur_max"]]
d = d.merge(lat, on=["tipo", "ws", "plume", "snap"], how="left")
assert d.norte_max.notna().all(), "some snapshot is missing its lateral edges"
d["lat_ok"] = (d.norte_max < C.LATERAL_FRAC * d.pico) & (d.sur_max < C.LATERAL_FRAC * d.pico)
fu = pd.read_csv(C.OUTSIDE)[["tipo", "ws", "plume", "snap", "fuera_frac"]]
d = d.merge(fu, on=["tipo", "ws", "plume", "snap"], how="left")
assert len(d) == len(fu), "some snapshots are missing from fuera-todos.csv"
d["fuera_ok"] = d.fuera_frac < C.OUTSIDE_MAX          # NaN (no upwind strip) does not pass
an = pd.read_csv(C.WIDTHS)[["tipo", "ws", "plume", "snap", "ancho_max_ruido", "ancho_borde_ruido"]]
d = d.merge(an, on=["tipo", "ws", "plume", "snap"], how="left")
assert d.ancho_borde_ruido.notna().all(), "some snapshot is missing its widths"
d["ancho_ok"] = d.ancho_borde_ruido <= C.EDGE_WIDTH_MAX   # leaves the domain narrower than one EMIT pixel
d["pasa"] = d.apaga & d.ent_ok & d.lat_ok & d.fuera_ok & d.ancho_ok
C.WORK.mkdir(exist_ok=True)
d.to_csv(C.WORK / "criterios.csv", index=False)
print(f"{len(d)} snapshots · {d.pasa.sum()} pass ({100 * d.pasa.mean():.1f}%)")
viejo = C.PROFILES.parent / "criterios-snapshots.csv"
if viejo.exists():
    v = pd.read_csv(viejo)
    m = d.merge(v[["tipo", "ws", "plume", "snap", "pasa"]], on=["tipo", "ws", "plume", "snap"], suffixes=("", "_viejo"))
    print(f"fade-out and upwind edge match criterios-snapshots.csv for {100 * ((m.apaga & m.ent_ok) == m.pasa_viejo).mean():.2f}% of {len(m)}")
    print(f"the lateral criterion removes {int((d.apaga & d.ent_ok & ~d.lat_ok).sum())} snapshots that passed the other two")
print(f"the foreign methane (< {100 * C.OUTSIDE_MAX:.1f}% of the mass) removes {int((d.apaga & d.ent_ok & d.lat_ok & ~d.fuera_ok).sum())} more snapshots")
print(f"the edge width (<= {C.EDGE_WIDTH_MAX:.0f} m) removes {int((d.apaga & d.ent_ok & d.lat_ok & d.fuera_ok & ~d.ancho_ok).sum())} more snapshots")
