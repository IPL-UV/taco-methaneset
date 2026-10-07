"""
project.py

Project bank enhancement maps onto the EMIT pixel grid and accumulate
the total enhancement for one or more plumes.

Reads each bank plume TIF, calls ``map_to_emit`` for the footprint
integration, scales by emission rate, and returns the combined result.
"""

from __future__ import annotations

import numpy as np
import rasterio

from config import ENHANCEMENT_PIXEL_RES, Q_MIN_KGH, Q_MAX_KGH, Q_REF_KGH
from trans_enhmap import map_to_emit


def project_plumes(
    plume_metas: list[dict],
    emit_positions: list[tuple[int, int]],
    wind_dirs: list[float],
    emit_lat: np.ndarray,
    emit_lon: np.ndarray,
    q_targets: np.ndarray | None = None,
    q_min: float = Q_MIN_KGH,
    q_max: float = Q_MAX_KGH,
    q_ref: float = Q_REF_KGH,
    q_target_fixed: float | None = None,
    pixel_res: float = ENHANCEMENT_PIXEL_RES,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Project every plume onto EMIT coordinates and accumulate.

    Parameters
    ----------
    plume_metas:
        Bank metadata dicts (one per plume), as returned by
        ``resolve_wind_and_query_bank``.
    emit_positions:
        Source pixel positions ``[(row, col), ...]``.
    wind_dirs:
        Per-plume wind direction [deg].
    emit_lat, emit_lon:
        (H, W) WGS-84 coordinate grids for the EMIT patch.
    q_targets:
        Pre-generated emission rates [kg/h], one per plume.
        If None, rates are drawn from ``[q_min, q_max]`` or set to
        *q_target_fixed*.
    q_min, q_max:
        Range for random emission rates [kg/h].
    q_ref:
        Bank reference emission rate [kg/h] (used for scaling).
    q_target_fixed:
        If set, all plumes use this emission rate [kg/h].
    pixel_res:
        Bank enhancement pixel resolution [m].

    Returns
    -------
    enhancement_total : (H, W)
        Accumulated scaled enhancement [ppb].
    enhancements_ref : list of (H, W)
        Per-plume enhancement at reference rate (for mask computation).
    plume_raws : list of (H_bank, W_bank)
        Raw bank enhancement maps (for plotting).
    q_targets : (n_plumes,)
        Emission rates used [kg/h].
    """
    n_plumes = len(plume_metas)
    H, W = emit_lat.shape

    # -- Resolve emission rates --
    if q_targets is None:
        q_targets = np.empty(n_plumes)
        for i in range(n_plumes):
            if q_target_fixed is not None:
                q_targets[i] = min(q_target_fixed, q_max)
            else:
                q_targets[i] = np.random.uniform(q_min, q_max)

    enhancement_total = np.zeros((H, W), dtype=np.float32)
    enhancements_ref = []
    plume_raws = []

    for i, (meta, (emit_r, emit_c)) in enumerate(
        zip(plume_metas, emit_positions)
    ):
        # Read bank enhancement map
        with rasterio.open(meta["gdal_vsi"]) as src:
            plume_ppb = src.read(1).astype(np.float32)
        plume_raws.append(plume_ppb)

        # Project at reference rate
        enhancement_i_ref = map_to_emit(
            enhancement=plume_ppb,
            source_row=meta["source_row"],
            source_col=meta["source_col"],
            emit_lat=emit_lat,
            emit_lon=emit_lon,
            emit_source_row=emit_r,
            emit_source_col=emit_c,
            pixel_res=pixel_res,
            wind_dir=wind_dirs[i],
        )
        enhancements_ref.append(enhancement_i_ref)

        # Scale by emission rate
        q_scale = q_targets[i] / q_ref
        enhancement_total += enhancement_i_ref * q_scale

    return enhancement_total, enhancements_ref, plume_raws, q_targets
