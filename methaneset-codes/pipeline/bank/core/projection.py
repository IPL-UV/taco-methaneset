"""Projection of a plume volume to a sun position, cropping and saving."""
from __future__ import annotations
import math
import numpy as np
from . import config as C


def proyectar(vol: np.ndarray, sza: float, raa: float, a_v: float = C.A_V):
    """Column map seen by a nadir sensor with the sun at (sza, raa).

    vol: (nz, ny, nx) contribution of each level to the column, in ppb, without the relaxation band.
    Each level is shifted k*DZ*tan(sza) towards raa + 180 (the side opposite the sun) with bilinear
    splitting, which conserves mass exactly; the view path (VZA = 0) is not shifted.
    They are combined with the path weights: (A_s*sol + A_v*nadir) / (A_s + A_v).

    Returns (mapa, fila0, col0): the canvas contains all the mass, and (fila0, col0) is the
    position of its [0, 0] pixel in the vol frame."""
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
    return (a_s * sol + a_v * nad) / (a_s + a_v), y_lo, x_lo


def umbral(mapa: np.ndarray) -> float:
    """The highest threshold that fits the mass budget, without exceeding UMBRAL_CEROS*maximum.

    The positive values are sorted and accumulated from the smallest: those that fit in
    PRESUPUESTO_MASA of the positive mass can go to zero."""
    v = np.sort(mapa[mapa > 0])
    cs = np.cumsum(v)
    i = np.searchsorted(cs, C.PRESUPUESTO_MASA * cs[-1], side="right")
    por_presupuesto = v[i] if i < len(v) else v[-1]
    return float(min(C.UMBRAL_CEROS * v[-1], por_presupuesto))


def recortar(mapa: np.ndarray, fila0: int, col0: int):
    """Sets to zero whatever falls below the threshold (v < u, so also the negatives) and crops to
    the box with values. Returns (array, fila, col, qa) with (fila, col) of array[0, 0] in the vol
    frame."""
    pico = float(mapa.max())
    thr = umbral(mapa)
    antes = float(mapa[mapa > 0].sum())
    m = np.where(mapa >= thr, mapa, 0.0)
    filas, cols = np.flatnonzero(m.any(axis=1)), np.flatnonzero(m.any(axis=0))
    r0, r1, c0, c1 = filas[0], filas[-1], cols[0], cols[-1]
    arr = m[r0:r1 + 1, c0:c1 + 1]
    qa = {"qa:threshold_ppb": thr, "qa:threshold_rel": thr / pico, "array:peak_ppb": pico,
          "qa:ime_before": antes, "qa:ime_after": float(arr.sum())}
    return arr, fila0 + r0, col0 + c0, qa
