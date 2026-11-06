"""Step 4 · Projection tests before generating anything.

1. Sources read from solve_em.F.
2. Direction of the shift with a synthetic blob: the solar footprint falls on the side opposite
   the sun and at a distance k*DZ*tan(sza).
3. With sza = 0 the map is the sum over the column.
4. Mass conservation in a real plume at every sun position.
5. The crop loses nothing that is not zero and the zeros stay under the 0.1% budget.
6. VZA = 0 reproduces the original maps bit for bit.
7. The view path shifts towards VAA + 180 and conserves mass.
"""
import math
import sys
import time
import numpy as np
from core import config as C
from core.geometry import fuentes, centro, rejilla
from core.projection import proyectar, recortar

ok = True
def check(cond, msg):
    global ok
    ok &= bool(cond)
    print(("  OK   " if cond else "  FAIL ") + msg)


def proyectar_nadir(vol, sza, raa):
    """Pre-VZA implementation kept as a bitwise regression oracle."""
    nz, ny, nx = vol.shape
    activos = [k for k in range(nz) if np.any(vol[k] != 0)]
    s = np.array([k * C.DZ * math.tan(math.radians(sza)) / C.PX for k in activos])
    ang = math.radians(raa + 180.0)
    dx, dy = s * math.cos(ang), s * math.sin(ang)
    x_lo, x_hi = min(0, math.floor(dx.min())), max(0, math.floor(dx.max()) + 1)
    y_lo, y_hi = min(0, math.floor(dy.min())), max(0, math.floor(dy.max()) + 1)
    H, W = ny + y_hi - y_lo, nx + x_hi - x_lo
    sol = np.zeros((H, W))
    nad = np.zeros((H, W))
    for k, ddx, ddy in zip(activos, dx, dy):
        L = vol[k]
        ix, iy = math.floor(ddx), math.floor(ddy)
        fx, fy = ddx - ix, ddy - iy
        r, c = iy - y_lo, ix - x_lo
        sol[r:r + ny, c:c + nx] += (1 - fx) * (1 - fy) * L
        sol[r:r + ny, c + 1:c + 1 + nx] += fx * (1 - fy) * L
        sol[r + 1:r + 1 + ny, c:c + nx] += (1 - fx) * fy * L
        sol[r + 1:r + 1 + ny, c + 1:c + 1 + nx] += fx * fy * L
        nad[-y_lo:-y_lo + ny, -x_lo:-x_lo + nx] += L
    a_s = 1.0 / math.cos(math.radians(sza))
    a_v = C.A_V
    return (a_s * sol + a_v * nad) / (a_s + a_v), y_lo, x_lo


def centroide_masa(m, f0, c0):
    """Mass centroid in the vol frame, in pixels."""
    rr, cc = np.indices(m.shape)
    tot = m.sum()
    return (rr * m).sum() / tot + f0, (cc * m).sum() / tot + c0


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

print("2b · direction and size of the view shift")
m0, f00, c00 = proyectar(vol, 0.0, 0.0)
fila0, col0 = centroide_masa(m0, f00, c00)
for vza in (8.42, 16.0):
    m, f0, c0 = proyectar(vol, 0.0, 0.0, vza=vza, vaa=0.0)
    fila, col = centroide_masa(m, f0, c0)
    recorrido = k0 * C.DZ * math.tan(math.radians(vza))
    esperado = recorrido / (1 + math.cos(math.radians(vza)))
    observado = (col0 - col) * C.PX
    check(abs(fila - fila0) < 1e-6 and abs(observado - esperado) < 1e-6,
          f"VZA {vza:5.2f}, VAA 0: centroid moved {observado:.6f} m towards VAA + 180 "
          f"(expected {esperado:.6f} m)")
    print(f"     view path {recorrido:.6f} m = {recorrido / C.PX:.6f} px; "
          f"combined centroid {observado:.6f} m = {observado / C.PX:.6f} px")

sys.path.insert(0, C.LOADER)
from methanebank.loader import load_single_simulation
t0 = time.time()
n, sf = load_single_simulation(C.LES, "multi", 7.0, [5], [93])
full = np.asarray(n[(5, 93)], dtype="float64") / (C.N_TOTAL_AIR * C.PIXEL_AREA) * 1e9 * sf[5]
vol = full[:, C.RELAX:-C.RELAX, C.RELAX:-C.RELAX]
print(f"   real plume loaded in {time.time() - t0:.0f} s · levels with something: "
      f"{sum(np.any(vol[k] != 0) for k in range(C.NZ))} of {C.NZ}")

print("3 · sza = 0 is the column and vza = 0 is bit-identical")
m, f0, c0 = proyectar(vol, 0.0, 0.0, vza=0.0, vaa=123.0)
col = vol.sum(0)
check(np.allclose(m[-f0:-f0 + col.shape[0], -c0:-c0 + col.shape[1]], col), "sza 0 = sum over levels")
m_ref, f_ref, c_ref = proyectar_nadir(vol, 0.0, 0.0)
check(np.array_equal(m, m_ref) and (f0, c0) == (f_ref, c_ref),
      "vza 0 reproduces the original map bit for bit, regardless of vaa")

NODOS = rejilla()
tot = vol.sum()
m, f0, c0 = proyectar(vol, 45.0, 30.0, vza=8.42, vaa=240.0)
error_vza = abs(m.sum() - tot) / tot
check(error_vza < 1e-9, f"mass conserved with VZA 8.42 and VAA 240: relative error {error_vza:.1e}")

print(f"4 and 5 · mass and bitwise compatibility at the {len(NODOS)} positions")
peor_masa = 0.0; peor_ceros = 0.0; bit_exact = True; t0 = time.time()
for anillo, sza, raa in NODOS:
    m, f0, c0 = proyectar(vol, sza, raa)
    m_ref, f_ref, c_ref = proyectar_nadir(vol, sza, raa)
    bit_exact &= np.array_equal(m, m_ref) and (f0, c0) == (f_ref, c_ref)
    peor_masa = max(peor_masa, abs(m.sum() - tot) / tot)
    arr, f, c, qa = recortar(m, f0, c0)
    peor_ceros = max(peor_ceros, 1 - qa["qa:ime_after"] / qa["qa:ime_before"])
    fuera = (m >= qa["qa:threshold_ppb"]).sum() - (arr > 0).sum()
    if fuera:
        check(False, f"the crop lost {fuera} pixels at sza {sza:.1f} raa {raa:.1f}")
dt = time.time() - t0
check(bit_exact, f"vza 0 is bit-identical at all {len(NODOS)} solar positions")
check(peor_masa < 1e-9, f"mass conserved at the {len(NODOS)} positions: maximum error {peor_masa:.1e}")
check(peor_ceros <= C.PRESUPUESTO_MASA * (1 + 1e-9), f"the zeros cost at most {100 * peor_ceros:.3f}% of the mass (budget {100 * C.PRESUPUESTO_MASA:.1f}%)")
print(f"   {len(NODOS)} projection pairs in {dt:.1f} s ({1000 * dt / len(NODOS):.0f} ms each)")
print("ALL OK" if ok else "FAILURES FOUND")
