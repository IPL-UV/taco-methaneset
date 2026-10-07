"""
pipeline.py

Orchestrate the spectral injection of a methane enhancement map
into an EMIT radiance cube.

Chains: band selection -> LUT interpolation -> enhancement clamping ->
SRF convolution matrix -> inject_plume.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from config import CH4_WINDOWS, EMIT_NODATA, MR_BACKGROUND_PPB
from inject import build_srf_matrix, get_methane_band_indices, inject_plume
from lut import air_mass_factor, read_luts


def run_injection(
    radiance: np.ndarray,
    enhancement: np.ndarray,
    emit_wvl: np.ndarray,
    emit_fwhm: np.ndarray,
    sza: float,
    vza: float,
    mr_background: float = MR_BACKGROUND_PPB,
    ch4_windows: list[tuple[float, float]] | None = None,
    verbose: bool = True,
) -> np.ndarray:
    """Inject a methane plume into EMIT radiance (in-place).

    Parameters
    ----------
    radiance : (n_bands, H, W)
        EMIT radiance cube.  **Modified in-place**.
    enhancement : (H, W)
        Projected enhancement map [ppb].
    emit_wvl : (n_bands,)
        EMIT central wavelengths [nm].
    emit_fwhm : (n_bands,)
        EMIT band FWHM [nm].
    sza, vza:
        Solar and Viewing Zenith Angles [deg].
    mr_background:
        Background CH4 mixing ratio [ppb].
    ch4_windows:
        Wavelength windows for CH4 absorption.
        Defaults to ``CH4_WINDOWS`` from config.
    verbose:
        Print progress messages.

    Returns
    -------
    radiance : same array, modified in-place.
    """
    # -- Band selection --
    band_indices = get_methane_band_indices(emit_wvl, ch4_windows)
    if verbose:
        print(
            f"CH4 bands: {len(band_indices)} "
            f"({emit_wvl[band_indices[0]]:.0f}-"
            f"{emit_wvl[band_indices[-1]]:.0f} nm)"
        )

    # -- LUT interpolation --
    amf = air_mass_factor(sza, vza)
    lut_wvl, lut_t, lut_mr = read_luts(amf)
    if verbose:
        print(f"AMF = {amf:.3f}")

    # -- Clamp enhancement to LUT range --
    max_enhancement = float(lut_mr.max()) - mr_background
    clipped = int((enhancement > max_enhancement).sum())
    if clipped > 0:
        if verbose:
            print(f"Clamping {clipped} pixels to {max_enhancement:.0f} ppb")
        enhancement = np.minimum(enhancement, max_enhancement)

    # -- SRF convolution matrix --
    srf_matrix, coverage = build_srf_matrix(
        emit_wvl, emit_fwhm, lut_wvl, band_indices,
    )

    # -- Inject --
    inject_plume(
        radiance=radiance,
        enhancement=enhancement,
        mr_background=mr_background,
        lut_transmittance=lut_t,
        lut_mixing_ratios=lut_mr,
        srf_matrix=srf_matrix,
        band_indices=band_indices,
        coverage=coverage,
    )

    if verbose:
        print("Injection complete.")

    return radiance
