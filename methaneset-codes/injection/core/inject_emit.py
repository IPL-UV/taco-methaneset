"""
inject.py

Core spectral injection of a methane plume into EMIT radiance.

Physics summary
---------------
The plume changes the atmospheric methane column from the background
mixing ratio (mr_bckg) to (mr_bckg + Delta), where Delta is the enhancement
map projected onto the EMIT grid.

Using pre-computed LUT transmittances T(mr), the radiance at each
pixel is scaled by the ratio:

    L_modified = L_original x [ T(mr_bckg + Delta) / T(mr_bckg) ]

This ratio is computed at the LUT's high spectral resolution and then
convolved to the EMIT band resolution using a Gaussian Spectral
Response Function (SRF) derived from each band's FWHM.
"""

from __future__ import annotations

import numpy as np
from scipy import interpolate

from .config import CH4_WINDOWS, EMIT_NODATA
from .srf import build_srf_matrix  # re-exported for callers that used inject.build_srf_matrix

__all__ = ["get_methane_band_indices", "inject_plume", "build_srf_matrix"]


# ---------------------------------------------------------------------------
# 1.  Band selection
# ---------------------------------------------------------------------------


def get_methane_band_indices(emit_wvl: np.ndarray, ch4_windows=None) -> np.ndarray:
    """Return EMIT band indices that fall within the CH4 absorption windows.
        
        Parameters
        ----------
        emit_wvl : (n_bands,)
            EMIT central wavelengths [nm].
        ch4_windows : list of tuples, optional
            List of wavelength windows [nm] in which radiance wants to be modified.
            If None, defaults to CH4_WINDOWS from config.py.
    
        Returns
        -------
        np.ndarray of int, shape (n_selected,)
        """
    if ch4_windows is None:
        ch4_windows = CH4_WINDOWS
    mask = np.zeros(len(emit_wvl), dtype=bool)
    for lo, hi in ch4_windows:
        mask |= (emit_wvl >= lo) & (emit_wvl <= hi)
    return np.where(mask)[0]




# ---------------------------------------------------------------------------
# 2.  Plume injection
# ---------------------------------------------------------------------------
# The SRF convolution matrix lives in core/srf.py (build_srf_matrix).

def inject_plume(
    radiance: np.ndarray,
    enhancement: np.ndarray,
    mr_background: float,
    lut_transmittance: np.ndarray,
    lut_mixing_ratios: np.ndarray,
    srf_matrix: np.ndarray,
    band_indices: np.ndarray,
    coverage: np.ndarray,
    nodata: float = EMIT_NODATA,
) -> np.ndarray:
    """Inject a methane plume into an EMIT radiance cube (in-place).

    Steps
    -----
    1. Build a 1-D interpolator T(mr) from the LUT transmittance curves.
    2. For every pixel where enhancement > 0:
       a. Compute T(mr_bckg + Delta) and T(mr_bckg) at LUT resolution.
       b. Form the ratio T_plume / T_bckg  ->  shape ``(N_active, n_lut)``.
       c. Convolve to EMIT bands:  ``ratio @ srf_matrix``  ->  ``(N_active, n_sel)``.
       d. Multiply the original radiance by the convolved factor,
          skipping nodata (``-9999``) pixels.

    Parameters
    ----------
    radiance : (n_bands, H, W)
        EMIT radiance cube.  **Modified in-place**.
    enhancement : (H, W)
        Projected enhancement map [ppb].
    mr_background : float
        Background CH4 mixing ratio [ppb].
    lut_transmittance : (n_mr, n_lut)
        Transmittance spectra from ``read_luts``.
    lut_mixing_ratios : (n_mr,)
        Mixing ratios [ppb] corresponding to rows of *lut_transmittance*.
    srf_matrix : (n_lut, n_sel)
        Convolution matrix from ``build_srf_matrix``.
    band_indices : (n_sel,)
        EMIT band indices that correspond to columns of *srf_matrix*.
    coverage : (n_sel,)
        Fraction of each band's Gaussian that falls within the LUT range.
    nodata : float
        Radiance nodata EMIT.

    Returns
    -------
    radiance : same array, modified in-place for convenience.
    """
    # --- 1. Build mixing-ratio -> transmittance interpolator ----------------
    # Input axes:
    #   X: lut_mixing_ratios  (n_mr,)
    #   Y: lut_transmittance  (n_mr, n_lut)
    # The interpolation collapses axis 0 (mixing-ratio).
    f_transmittance = interpolate.interp1d(
        lut_mixing_ratios, lut_transmittance, axis=0
    )

    # Background transmittance at LUT resolution -- shape (n_lut,).
    t_background = f_transmittance(mr_background)

    # --- 2. Identify active (plume-containing) pixels ----------------------
    active_mask = enhancement > 0
    active_rows, active_cols = np.where(active_mask)

    if len(active_rows) == 0:
        return radiance  # Nothing to inject.

    # Total mixing ratio per active pixel -- shape (N_active,).
    mr_total = mr_background + enhancement[active_mask]
    mr_total = np.clip(mr_total, lut_mixing_ratios.min(), lut_mixing_ratios.max())

    # --- 3. Transmittance ratio at LUT resolution --------------------------
    # t_plume: (N_active, n_lut)  --  transmittance with the plume.
    t_plume = f_transmittance(mr_total)

    # ratio:  (N_active, n_lut)  --  fractional change in transmittance.
    ratio = t_plume / t_background  # broadcasting (N_active, n_lut) / (n_lut,)

    # --- 4. Convolve ratio to EMIT band resolution -------------------------
    # (N_active, n_lut) @ (n_lut, n_sel)  ->  (N_active, n_sel)
    transmission_factor = ratio @ srf_matrix
    transmission_factor = transmission_factor * coverage + (1.0 - coverage)

    # --- 5. Apply to radiance, respecting nodata ---------------------------
    for col_idx, band in enumerate(band_indices):
        pixel_rad = radiance[band, active_rows, active_cols]
        valid = pixel_rad != nodata
        radiance[band, active_rows[valid], active_cols[valid]] *= \
            transmission_factor[valid, col_idx]

    return radiance