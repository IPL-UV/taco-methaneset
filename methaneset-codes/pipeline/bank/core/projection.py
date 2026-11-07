"""Projection of a plume volume to a sun and view geometry, cropping and saving."""
from __future__ import annotations
import math
import numpy as np
from . import config as C


def project(vol: np.ndarray, sza: float, raa: float, vza: float = 0.0, vaa: float = 0.0,
              a_v: float | None = None):
    """Column map seen at (vza, vaa) with the sun at (sza, raa).

    vol: (nz, ny, nx) contribution of each level to the column, in ppb, without the relaxation band.
    Solar and view paths shift each level by k*DZ*tan(angle) towards azimuth + 180, with bilinear
    splitting that conserves mass. Their weights are A_s = 1/cos(sza) and A_v = 1/cos(vza), unless
    a_v is given. At vza = 0 the view path keeps the original unshifted accumulation.

    Returns (colmap, row0, col0): the canvas contains all the mass, and (row0, col0) is the
    position of its [0, 0] pixel in the vol frame."""
    if a_v is None:
        a_v = 1.0 / math.cos(math.radians(vza))
    nz, ny, nx = vol.shape
    active = [k for k in range(nz) if np.any(vol[k] != 0)]
    s = np.array([k * C.DZ * math.tan(math.radians(sza)) / C.PX for k in active])
    ang = math.radians(raa + 180.0)
    dx, dy = s * math.cos(ang), s * math.sin(ang)
    if vza == 0.0:
        x_lo, x_hi = min(0, math.floor(dx.min())), max(0, math.floor(dx.max()) + 1)
        y_lo, y_hi = min(0, math.floor(dy.min())), max(0, math.floor(dy.max()) + 1)
    else:
        s_v = np.array([k * C.DZ * math.tan(math.radians(vza)) / C.PX for k in active])
        ang_v = math.radians(vaa + 180.0)
        dx_v, dy_v = s_v * math.cos(ang_v), s_v * math.sin(ang_v)
        x_lo = min(0, math.floor(dx.min()), math.floor(dx_v.min()))
        x_hi = max(0, math.floor(dx.max()) + 1, math.floor(dx_v.max()) + 1)
        y_lo = min(0, math.floor(dy.min()), math.floor(dy_v.min()))
        y_hi = max(0, math.floor(dy.max()) + 1, math.floor(dy_v.max()) + 1)
    H, W = ny + y_hi - y_lo, nx + x_hi - x_lo
    sol = np.zeros((H, W))
    up = np.zeros((H, W))
    for j, (k, ddx, ddy) in enumerate(zip(active, dx, dy)):
        L = vol[k]
        ix, iy = math.floor(ddx), math.floor(ddy)
        fx, fy = ddx - ix, ddy - iy
        r, c = iy - y_lo, ix - x_lo
        sol[r:r + ny, c:c + nx] += (1 - fx) * (1 - fy) * L
        sol[r:r + ny, c + 1:c + 1 + nx] += fx * (1 - fy) * L
        sol[r + 1:r + 1 + ny, c:c + nx] += (1 - fx) * fy * L
        sol[r + 1:r + 1 + ny, c + 1:c + 1 + nx] += fx * fy * L
        if vza == 0.0:
            up[-y_lo:-y_lo + ny, -x_lo:-x_lo + nx] += L
        else:
            ix, iy = math.floor(dx_v[j]), math.floor(dy_v[j])
            fx, fy = dx_v[j] - ix, dy_v[j] - iy
            r, c = iy - y_lo, ix - x_lo
            up[r:r + ny, c:c + nx] += (1 - fx) * (1 - fy) * L
            up[r:r + ny, c + 1:c + 1 + nx] += fx * (1 - fy) * L
            up[r + 1:r + 1 + ny, c:c + nx] += (1 - fx) * fy * L
            up[r + 1:r + 1 + ny, c + 1:c + 1 + nx] += fx * fy * L
    a_s = 1.0 / math.cos(math.radians(sza))
    return (a_s * sol + a_v * up) / (a_s + a_v), y_lo, x_lo


def threshold(colmap: np.ndarray) -> float:
    """The highest threshold that fits the mass budget, without exceeding ZERO_THRESHOLD*maximum.

    The positive values are sorted and accumulated from the smallest: those that fit in
    MASS_BUDGET of the positive mass can go to zero."""
    v = np.sort(colmap[colmap > 0])
    cs = np.cumsum(v)
    i = np.searchsorted(cs, C.MASS_BUDGET * cs[-1], side="right")
    by_budget = v[i] if i < len(v) else v[-1]
    return float(min(C.ZERO_THRESHOLD * v[-1], by_budget))


def crop(colmap: np.ndarray, row0: int, col0: int):
    """Sets to zero whatever falls below the threshold (v < u, so also the negatives) and crops to
    the box with values. Returns (array, row, col, qa) with (row, col) of array[0, 0] in the vol
    frame."""
    peak = float(colmap.max())
    thr = threshold(colmap)
    before = float(colmap[colmap > 0].sum())
    m = np.where(colmap >= thr, colmap, 0.0)
    rows, cols = np.flatnonzero(m.any(axis=1)), np.flatnonzero(m.any(axis=0))
    r0, r1, c0, c1 = rows[0], rows[-1], cols[0], cols[-1]
    arr = m[r0:r1 + 1, c0:c1 + 1]
    qa = {"qa:threshold_ppb": thr, "qa:threshold_rel": thr / peak, "array:peak_ppb": peak,
          "qa:ime_before": before, "qa:ime_after": float(arr.sum())}
    return arr, row0 + r0, col0 + c0, qa
