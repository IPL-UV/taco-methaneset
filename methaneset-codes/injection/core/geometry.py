"""Geometry conventions for the MethaneSET plume injection.

Two azimuth conventions meet in this pipeline and they are a classic source of
sign errors, so every conversion is explicit here:

* Meteorological (meteo): 0 deg = North, positive clockwise.  This is the
  convention of the TACO columns ``target:saa``, ``target:vaa`` and
  ``meteo:wind_dir``.
* Mathematical (math): 0 deg = East (+x), positive counterclockwise.  This is
  the frame of the plume bank (``sun:raa``, the wind-relative azimuth) and of
  ``atan2(v, u)``.

Conversion: ``math = (90 - meteo) % 360``.

Solar / wind relative azimuth (bank frame)
------------------------------------------
The bank stores ``sun:raa``: the sun azimuth measured counterclockwise from the
wind direction (``+x`` is the wind direction).  The solar-path image of the
plume sits at ``raa + 180``.

For EMIT the scene wind is a vector ``(u, v)``; its direction *towards which it
blows* in the mathematical frame is ``wind_dir = atan2(v, u)``.  The verified
relation is::

    raa_banco = (90 - saa_meteo - wind_dir_math) % 360

For the s2 / l89 catalogs the wind is stored as a meteo direction *to*, and since
dataset version 1.2.0 the catalog stores the bank convention directly::

    target:raa = (wind_dir_to_meteo - SAA_meteo) % 360   # == sun:raa

so ``raa_banco_from_catalog`` is the identity (before 1.2.0 the sign was opposite).

Both routes are the same number once ``wind_dir_to_meteo`` and
``wind_dir_math`` are related by ``wind_dir_to_meteo = (90 - wind_dir_math) % 360``.

View parallax
-------------
The bank is rendered at nadir (VZA = VAA = 0) on a 20 m grid aligned with the
wind.  An off-nadir view displaces gas at effective height ``z_eff`` on the
ground *away from the satellite* by ``z_eff * tan(VZA)`` along the bearing
``vaa_math + 180``, where ``vaa_math`` is the azimuth from the pixel **to** the
satellite (converted from ``target:vaa``).  The bank affine maps a +1-col
step to ``(cos w, sin w)`` and a +1-row step to ``(sin w, -cos w)`` in
(East, North), so the shift in bank pixels is::

    dcol = (ex * cos w + ny * sin w) / px
    drow = (ex * sin w - ny * cos w) / px

with ``(ex, ny)`` the East / North components of the shift and ``w`` the
mathematical wind direction.  This mirrors the validated prototype
``01-injection-notes/code/inyeccion/vaa_prototype.py``.
"""

from __future__ import annotations

import numpy as np

EMIT_VZA_DEG = 8.4        # median VZA of the 721 EMIT scenes
DEFAULT_PIXEL_RES = 20.0  # bank pixel size [m]


def meteo_to_math(az_meteo: float) -> float:
    """Meteorological azimuth (0=N, clockwise) -> mathematical (0=E, CCW)."""
    return (90.0 - az_meteo) % 360.0


def math_to_meteo(az_math: float) -> float:
    """Mathematical azimuth (0=E, CCW) -> meteorological (0=N, clockwise)."""
    return (90.0 - az_math) % 360.0


def raa_banco_emit(saa_mean_meteo: float, wind_dir_math: float) -> float:
    """Bank sun-relative azimuth ``sun:raa`` for an EMIT scene.

    Parameters
    ----------
    saa_mean_meteo:
        ``target:saa`` in meteorological convention (0=N, clockwise).
    wind_dir_math:
        Wind direction *towards which it blows* in the mathematical frame
        (0=E, CCW), i.e. ``degrees(atan2(v, u)) % 360``.

    Returns
    -------
    float
        ``(90 - saa_mean - wind_dir) % 360``, counterclockwise from the wind.
    """
    return (90.0 - saa_mean_meteo - wind_dir_math) % 360.0


def raa_banco_from_catalog(target_raa: float) -> float:
    """Bank ``sun:raa`` from a catalog's ``target:raa``.

    Since dataset version 1.2.0 the catalogs store ``target:raa`` directly in the
    bank convention (sun azimuth counterclockwise from the wind direction, equal
    to ``sun:raa``), so this is the **identity**. Before 1.2.0 the stored value
    used the opposite sign and this function applied ``(360 - x) % 360``.
    """
    return target_raa % 360.0


def wind_dir_from_uv(u: float, v: float) -> float:
    """Wind direction *towards which it blows*, math frame (0=E, CCW).

    From the eastward (``u``) and northward (``v``) wind components of
    ``wind.tif`` or of the catalog columns ``meteo:wind_u`` / ``meteo:wind_v``.
    """
    return float(np.degrees(np.arctan2(v, u)) % 360.0)


def vaa_math(vaa_meteo: float) -> float:
    """View azimuth (pixel -> satellite) from meteo to mathematical frame.

    ``target:vaa`` is meteorological (azimuth from the pixel to the
    satellite, 0=N, clockwise); this returns ``(90 - vaa_meteo) % 360``.
    """
    return meteo_to_math(vaa_meteo)


def parallax_displacement_px(z_eff: float, vza: float = EMIT_VZA_DEG,
                             vaa_math: float = 0.0, wind_dir_math: float = 0.0,
                             pixel_res: float = DEFAULT_PIXEL_RES):
    """First-order view-parallax shift in bank pixels.

    The gas at effective height ``z_eff`` [m] is displaced on the ground away
    from the satellite by ``z_eff * tan(VZA)`` along the bearing
    ``vaa_math + 180``.  The vector is then rotated by ``-wind_dir`` into the
    wind-aligned bank frame.

    Parameters
    ----------
    z_eff:
        Effective emission height [m].
    vza:
        View zenith angle [deg].  Defaults to the EMIT median.
    vaa_math:
        View azimuth (pixel -> satellite) in the mathematical frame; use
        :func:`vaa_math` to convert ``target:vaa``.
    wind_dir_math:
        Wind direction towards which it blows, math frame (0=E, CCW).
    pixel_res:
        Bank pixel size [m].  Must be positive.

    Returns
    -------
    (drow, dcol) : tuple of float
        Row and column shift in bank pixels.  A feature moves by ``(drow,
        dcol)``; to resample use ``out[r, c] = in[r - drow, c - dcol]``.
    """
    if pixel_res <= 0:
        raise ValueError("pixel_res must be positive")
    delta_d = z_eff * np.tan(np.deg2rad(vza))
    theta = np.deg2rad(vaa_math + 180.0)
    ex = delta_d * np.cos(theta)
    ny = delta_d * np.sin(theta)
    w = np.deg2rad(wind_dir_math)
    dcol = (ex * np.cos(w) + ny * np.sin(w)) / pixel_res
    drow = (ex * np.sin(w) - ny * np.cos(w)) / pixel_res
    return float(drow), float(dcol)
