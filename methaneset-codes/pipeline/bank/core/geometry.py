"""Emission sources (read from Gorroño's WRF code) and grid of sun positions."""
from __future__ import annotations
import math
import re
import numpy as np
from . import config as C

_PAT = re.compile(
    r"DO j = (\d+)\*\(jde \+ jds\)/90,\s*(\d+)\*\(jde \+ jds\)/90,\s*1\s*\n"
    r"\s*DO i = (\d+)\*\(ide \+ ids\)/120,\s*(\d+)\*\(ide \+ ids\)/120,\s*1\s*\n"
    r".*?tracer\(i, 1, j, P_plume(\d)\) = 1\.", re.S)


def fuentes() -> dict:
    """{(tipo, plume): (row0, row1, col0, col1)} inclusive, 0-based.

    In solve_em.F: j = n*(jde+jds)/90 and i = n*(ide+ids)/120 with Fortran integer arithmetic,
    jde+jds = 91+1 and ide+ids = 121+1, and 1-based indices."""
    out = {}
    for tipo, sub in (("multi", "multisource"), ("area", "areasource")):
        txt = (C.CONF_WRF / sub / "solve_em.F").read_text(errors="replace")
        for j0, j1, i0, i1, p in _PAT.findall(txt):
            fj = lambda n: int(n) * (C.NY + 2) // 90
            fi = lambda n: int(n) * (C.NX + 2) // 120
            out[(tipo, int(p))] = (fj(j0) - 1, fj(j1) - 1, fi(i0) - 1, fi(i1) - 1)
    return out


def centro(caja) -> tuple[float, float, int, int]:
    """(centre_row, centre_col, height_px, width_px) of a source box."""
    r0, r1, c0, c1 = caja
    return 0.5 * (r0 + r1), 0.5 * (c0 + c1), r1 - r0 + 1, c1 - c0 + 1


def rejilla() -> list[tuple[int, float, float]]:
    """855 sun positions (ring, sza, raa).

    Two neighbouring positions move the plume exactly TOL metres over the ground at a height
    Z_REF: rings at tan(sza) = n*TOL/Z_REF and, in each ring, azimuth step 2*asin(TOL/2d)
    with d = Z_REF*tan(sza)."""
    dt = C.TOL / C.Z_REF
    tans = np.arange(0, math.tan(math.radians(C.SZA_TOPE)) + dt, dt)
    nodos = []
    for anillo, t in enumerate(tans):
        if anillo == 0:
            nodos.append((0, 0.0, 0.0))
            continue
        sza = math.degrees(math.atan(t))
        paso = math.degrees(2 * math.asin(min(1.0, C.TOL / (2 * C.Z_REF * t))))
        n = max(1, round(360 / paso))
        nodos += [(anillo, sza, i * 360.0 / n) for i in range(n)]
    return nodos
