"""Step 4 · Projection tests before generating anything.

1. Sources read from solve_em.F.
2. Direction of the shift with a synthetic blob: the solar footprint falls on the side opposite
   the sun and at a distance k*DZ*tan(sza).
3. With sza = 0 the map is the sum over the column.
4. Mass conservation in a real plume at every sun position.
5. The crop loses nothing that is not zero and the zeros stay under the 0.1% budget.
"""
import math
import sys
import time
import numpy as np
from banco import config as C
from banco.geometria import fuentes, centro, rejilla
from banco.proyeccion import proyectar, recortar

ok = True
def check(cond, msg):
    global ok
    ok &= bool(cond)
    print(("  OK   " if cond else "  FAIL ") + msg)

print("1 · sources")
F = fuentes()
check(len(F) == 18, f"18 sources read ({len(F)})")
for k in sorted(F):
    r, c, h, w = centro(F[k]); print(f"     {k[0]:5s} {k[1]}: rows {F[k][0]}-{F[k][1]}, columns {F[k][2]}-{F[k][3]}, centre ({r}, {c}), {w}x{h} px")

print("2 · direction of the shift")
vol = np.zeros((C.NZ, 80, 110)); k0 = 10; vol[k0, 40, 50] = 1.0
for raa, esperado in [(0, (40, 40)), (90, (30, 50)), (180, (40, 60)), (270, (50, 50))]:
    m, f0, c0 = proyectar(vol, 45.0, raa, a_v=0.0)        # solar path only
    rr, cc = np.indices(m.shape); tot = m.sum()
    fila, col = (rr * m).sum() / tot + f0, (cc * m).sum() / tot + c0
    check(abs(fila - esperado[0]) < 1e-6 and abs(col - esperado[1]) < 1e-6,
          f"sun at RAA {raa:3d}: footprint at row {fila:.2f}, col {col:.2f} (expected {esperado})")

sys.path.insert(0, C.LOADER)
from methanebank.loader import load_single_simulation
t0 = time.time()
n, sf = load_single_simulation(C.LES, "multi", 7.0, [5], [93])
full = np.asarray(n[(5, 93)], dtype="float64") / (C.N_TOTAL_AIR * C.PIXEL_AREA) * 1e9 * sf[5]
vol = full[:, C.RELAX:-C.RELAX, C.RELAX:-C.RELAX]
print(f"   real plume loaded in {time.time() - t0:.0f} s · levels with something: "
      f"{sum(np.any(vol[k] != 0) for k in range(C.NZ))} of {C.NZ}")

print("3 · sza = 0 is the column")
m, f0, c0 = proyectar(vol, 0.0, 0.0)
col = vol.sum(0)
check(np.allclose(m[-f0:-f0 + col.shape[0], -c0:-c0 + col.shape[1]], col), "sza 0 = sum over levels")

NODOS = rejilla()
print(f"4 and 5 · mass at the {len(NODOS)} positions")
tot = vol.sum(); peor_masa = 0.0; peor_ceros = 0.0; t0 = time.time()
for anillo, sza, raa in NODOS:
    m, f0, c0 = proyectar(vol, sza, raa)
    peor_masa = max(peor_masa, abs(m.sum() - tot) / tot)
    arr, f, c, qa = recortar(m, f0, c0)
    peor_ceros = max(peor_ceros, 1 - qa["qa:ime_after"] / qa["qa:ime_before"])
    fuera = (m >= qa["qa:threshold_ppb"]).sum() - (arr > 0).sum()
    if fuera:
        check(False, f"the crop lost {fuera} pixels at sza {sza:.1f} raa {raa:.1f}")
dt = time.time() - t0
check(peor_masa < 1e-9, f"mass conserved at the {len(NODOS)} positions: maximum error {peor_masa:.1e}")
check(peor_ceros <= C.PRESUPUESTO_MASA * (1 + 1e-9), f"the zeros cost at most {100 * peor_ceros:.3f}% of the mass (budget {100 * C.PRESUPUESTO_MASA:.1f}%)")
print(f"   {len(NODOS)} projections in {dt:.1f} s ({1000 * dt / len(NODOS):.0f} ms each)")
print("ALL OK" if ok else "FAILURES FOUND")
