"""Tests for ``core/geometry.py`` (plume injection geometry conventions).

Runs with pytest or directly:

    python tests/test_geometry.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy.ndimage import map_coordinates

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.geometry import (  # noqa: E402
    EMIT_VZA_DEG,
    math_to_meteo,
    meteo_to_math,
    parallax_displacement_px,
    raa_banco_emit,
    raa_banco_from_catalog,
    vaa_math,
    wind_dir_from_uv,
)

SAA_METEO = (15.0, 45.0, 100.0, 200.0, 300.0, 359.0)
WIND_DIR_MATH = (0.0, 30.0, 90.0, 217.0)


def _gaussian(shape, row0, col0, sigma):
    H, W = shape
    rr, cc = np.mgrid[0:H, 0:W].astype(np.float64)
    return np.exp(-0.5 * (((rr - row0) ** 2 + (cc - col0) ** 2) / sigma ** 2))


def _centroid(arr):
    tot = arr.sum()
    H, W = arr.shape
    rr, cc = np.mgrid[0:H, 0:W].astype(np.float64)
    return float((rr * arr).sum() / tot), float((cc * arr).sum() / tot)


def _shift(arr, drow, dcol, order=1):
    H, W = arr.shape
    rr, cc = np.mgrid[0:H, 0:W].astype(np.float64)
    return map_coordinates(arr, [rr - drow, cc - dcol],
                           order=order, mode="constant", cval=0.0)


# --------------------------------------------------------------------------
# T1a -- sign identity between the EMIT route and the catalog route
# --------------------------------------------------------------------------
def test_t1a_emit_equals_catalog():
    """EMIT and catalog routes agree on the same wind-relative azimuth.

    Since dataset v1.2.0 the catalog stores ``target:raa = (wind_dir_to - SAA) % 360``
    (the bank convention), so ``raa_banco_from_catalog`` is the identity.
    """
    n = 0
    for saa in SAA_METEO:
        for wd in WIND_DIR_MATH:
            wd_to = math_to_meteo(wd)
            target = (wd_to - saa) % 360.0
            r_emit = raa_banco_emit(saa, wd)
            r_cat = raa_banco_from_catalog(target)
            assert abs((r_emit - r_cat + 180.0) % 360.0 - 180.0) < 1e-9, (
                f"emit/catalog mismatch at SAA={saa}, wd_math={wd}: "
                f"{r_emit} vs {r_cat}")
            n += 1
    print(f"\nT1a: {n} emit/catalog pairs agree (target:raa in bank convention)")


# --------------------------------------------------------------------------
# T1b -- one concrete case, both routes
# --------------------------------------------------------------------------
def test_t1b_concrete_case():
    saa, wd = 90.0, 45.0
    wd_to = math_to_meteo(wd)
    target = (wd_to - saa) % 360.0
    r_emit = raa_banco_emit(saa, wd)
    r_cat = raa_banco_from_catalog(target)
    print(f"\nT1b  SAA_meteo={saa}  wind_dir_math={wd}  "
          f"wind_dir_to_meteo={wd_to}  target:raa={target}  "
          f"raa_emit={r_emit}  raa_catalog={r_cat}")
    assert r_emit == r_cat == 315.0


def test_raa_catalog_identity_regression():
    """Guard: since v1.2.0 ``target:raa`` is already in bank convention (no flip)."""
    assert raa_banco_from_catalog(315.0) == 315.0
    assert raa_banco_from_catalog(45.0) == 45.0


# --------------------------------------------------------------------------
# T9 -- parallax displacement recovers the synthetic centroid shift
# --------------------------------------------------------------------------
def test_t9_parallax_centroid():
    H = W = 320
    sigma = 14.0
    z_eff = 200.0
    vaa_meteo = 200.0
    vm = vaa_math(vaa_meteo)
    g = _gaussian((H, W), H / 2 + 0.37, W / 2 - 0.63, sigma)
    r0, c0 = _centroid(g)
    print(f"\nT9  gaussian  z_eff={z_eff} m  vza={EMIT_VZA_DEG}  "
          f"vaa_meteo={vaa_meteo} (math {vm})")
    for wd in WIND_DIR_MATH:
        drow, dcol = parallax_displacement_px(
            z_eff, vza=EMIT_VZA_DEG, vaa_math=vm, wind_dir_math=wd)
        out = _shift(g, drow, dcol, order=1)
        r1, c1 = _centroid(out)
        err = float(np.hypot((r1 - r0) - drow, (c1 - c0) - dcol))
        print(f"     wind_dir={wd:6.1f}  exp=(drow {drow:+.4f}, dcol {dcol:+.4f})  "
              f"meas=({r1 - r0:+.4f}, {c1 - c0:+.4f})  err={err:.4f} px")
        assert err < 0.2, f"parallax centroid error {err:.4f} px at wind_dir={wd}"


def test_wind_dir_from_uv():
    assert abs(wind_dir_from_uv(1.0, 0.0) - 0.0) < 1e-9      # blowing east
    assert abs(wind_dir_from_uv(0.0, 1.0) - 90.0) < 1e-9     # blowing north
    assert abs(wind_dir_from_uv(-1.0, 0.0) - 180.0) < 1e-9   # blowing west
    assert abs(wind_dir_from_uv(0.0, -1.0) - 270.0) < 1e-9   # blowing south


if __name__ == "__main__":
    test_wind_dir_from_uv()
    test_t1a_emit_equals_catalog()
    test_t1b_concrete_case()
    test_t9_parallax_centroid()
    print("\nALL TESTS PASSED")
