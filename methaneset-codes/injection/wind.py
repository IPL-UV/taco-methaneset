"""
wind.py

Resolve wind conditions (speed + direction) for each plume source
and query the plume bank for the closest matching simulations.

Two modes:
- **Wind field** (``use_wind_field=True``):  read per-pixel U/V from the
  scene's ``wind.tif`` and extract the value at each source position.
- **Fixed wind** (``use_wind_field=False``):  use a single speed/direction
  for all sources (user-specified or drawn at random from the bank grid).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import rasterio

if TYPE_CHECKING:
    from bank import MethaneBank


def resolve_wind_and_query_bank(
    bank: MethaneBank,
    scene_dir: str,
    reading_window: rasterio.windows.Window,
    emit_positions: list[tuple[int, int]],
    sza_mean: float,
    saa_mean: float,
    source_type: str | None = None,
    use_wind_field: bool = False,
    wind_speed: float | None = None,
    wind_dir: float | None = None,
) -> tuple[list[dict], list[float], list[float], list[float]]:
    """Resolve per-plume wind and fetch matching bank entries.

    Parameters
    ----------
    bank:
        Initialised ``MethaneBank`` instance.
    scene_dir:
        GDAL VSI base path of the EMIT scene.
    reading_window:
        Rasterio window for the selected patch.
    emit_positions:
        Source pixel positions ``[(row, col), ...]``.
    sza_mean, saa_mean:
        Scene mean Solar Zenith Angle and Solar Azimuth Angle [deg].
    source_type:
        ``"area"`` or ``"multi"`` (passed to ``bank.query``).
    use_wind_field:
        If True, read ``wind.tif`` for per-source U/V.
    wind_speed:
        Fixed wind speed [m/s].  Ignored when *use_wind_field* is True.
        ``None`` -> random from bank grid.
    wind_dir:
        Fixed wind direction [deg, math convention].  Ignored when
        *use_wind_field* is True.  ``None`` -> random 0-359.

    Returns
    -------
    plume_metas : list[dict]
        Bank metadata dicts (one per source).
    wind_dirs : list[float]
    wind_speeds : list[float]
    raas : list[float]
        Relative Azimuth Angles used for each query.
    """
    n_plumes = len(emit_positions)

    if use_wind_field:
        return _from_wind_field(
            bank, scene_dir, reading_window, emit_positions,
            sza_mean, saa_mean, source_type,
        )

    return _from_fixed_wind(
        bank, emit_positions,
        sza_mean, saa_mean, source_type,
        wind_speed, wind_dir,
    )


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _from_wind_field(
    bank, scene_dir, reading_window, emit_positions,
    sza_mean, saa_mean, source_type,
):
    with rasterio.open(f"{scene_dir}/wind.tif") as src:
        wind_u = src.read(1, window=reading_window)
        wind_v = src.read(2, window=reading_window)

    wind_dirs, wind_speeds, raas = [], [], []
    plume_metas, used_ids = [], []

    for emit_r, emit_c in emit_positions:
        u = float(wind_u[emit_r, emit_c])
        v = float(wind_v[emit_r, emit_c])
        ws = float(np.hypot(u, v))
        wd = float(np.degrees(np.arctan2(v, u))) % 360
        raa = (90 - saa_mean - wd) % 360

        wind_speeds.append(ws)
        wind_dirs.append(wd)
        raas.append(raa)

        meta = bank.query(
            sza=sza_mean, raa=raa, wind_speed=ws,
            source_type=source_type, meta=True,
            exclude_ids=used_ids,
        )
        plume_metas.append(meta)
        used_ids.append(meta["id"])

    return plume_metas, wind_dirs, wind_speeds, raas


def _from_fixed_wind(
    bank, emit_positions,
    sza_mean, saa_mean, source_type,
    wind_speed, wind_dir,
):
    n_plumes = len(emit_positions)

    if wind_speed is None:
        wind_speed = float(np.random.choice(bank.wind_speeds))
    if wind_dir is None:
        wind_dir = float(np.random.randint(0, 360))

    raa = (90 - saa_mean - wind_dir) % 360

    if n_plumes == 1:
        plume_metas = [bank.query(
            sza=sza_mean, raa=raa, wind_speed=wind_speed,
            source_type=source_type, meta=True,
        )]
    else:
        plume_metas = bank.query(
            sza=sza_mean, raa=raa, wind_speed=wind_speed,
            source_type=source_type, meta=True, n=n_plumes,
        )

    wind_dirs = [wind_dir] * n_plumes
    wind_speeds = [wind_speed] * n_plumes
    raas = [raa] * n_plumes

    return plume_metas, wind_dirs, wind_speeds, raas
