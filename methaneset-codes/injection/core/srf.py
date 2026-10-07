"""
srf.py

Spectral Response Function (SRF) convolution matrix for EMIT bands.

Extracted from CARLOS/inject.py (section 2) so the SRF construction lives
on its own, independently of the plume injection.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erf


def build_srf_matrix(
    emit_wvl: np.ndarray,
    emit_fwhm: np.ndarray,
    lut_wvl: np.ndarray,
    band_indices: np.ndarray,
    n_sigma: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the Gaussian SRF convolution matrix.

    Returns
    -------
    W : np.ndarray, shape (n_lut, n_sel)
        Column-normalised weight matrix.
    coverage : np.ndarray, shape (n_sel,)
        Fraction of each band's Gaussian that falls within the LUT range.
        1.0 = fully covered, <1.0 = truncated at LUT boundary.
    """
    n_lut = len(lut_wvl)
    n_sel = len(band_indices)

    dw = np.empty(n_lut)
    dw[:-1] = np.diff(lut_wvl)
    dw[-1] = dw[-2]

    W = np.zeros((n_lut, n_sel), dtype=np.float64)
    coverage = np.ones(n_sel, dtype=np.float64)

    for col, k in enumerate(band_indices):
        centre = emit_wvl[k]
        sigma  = emit_fwhm[k] / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        radius = n_sigma * sigma

        lo_bound = centre - radius
        hi_bound = centre + radius
        lo_clipped = max(lo_bound, lut_wvl[0])
        hi_clipped = min(hi_bound, lut_wvl[-1])

        s2 = sigma * np.sqrt(2.0)
        full = erf((hi_bound - centre) / s2) - erf((lo_bound - centre) / s2)
        captured = erf((hi_clipped - centre) / s2) - erf((lo_clipped - centre) / s2)
        coverage[col] = captured / full if full > 0 else 1.0

        i0 = np.searchsorted(lut_wvl, lo_bound)
        i1 = min(np.searchsorted(lut_wvl, hi_bound), n_lut - 1)
        sl = np.arange(i0, i1)

        weights = np.exp(-0.5 * ((lut_wvl[sl] - centre) / sigma) ** 2) * dw[sl]
        total = weights.sum()
        if total > 0:
            W[sl, col] = weights / total

    return W, coverage
