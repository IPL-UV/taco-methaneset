"""
lut.py

From https://github.com/UNEP-IMEO-MARS/marshsi/blob/main/marshsi

Unified module for reading LUT (Look-Up Table) files for methane retrieval.

This module centralizes all LUT file reading operations. It handles the loading
of pre-calculated Radiative Transfer Model (RTM) outputs, specifically:
1.  **Methane Transmittance (T_CH4)**: The atmospheric transmission due to methane absorption.
3.  **Mixing Ratios**: The concentration of Methane in the atmospheric column.

Units assumed throughout:
- Wavelengths: Nanometers [nm]
- Mixing Ratios: Parts per billion [ppb]
- Transmittance: Unitless [0-1]
- Air Mass Factor (AMF): Unitless geometric factor
"""

import os
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import interpolate

import xarray as xr

# Default LUT file path
FILE_LUT_GAS = os.path.join(os.path.dirname(__file__), "output_Tch4_LUT_AMF_VZA_0_v2.nc")


def read_luts(amf: float, file_lut: str = FILE_LUT_GAS) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Interpolates Methane LUTs to a specific observation geometry (AMF).

    This function adjusts the generic LUTs to the specific geometric path length
    of the current observation. It performs linear interpolation along the AMF axis.

    The effective optical path is defined by the Air Mass Factor (AMF):
    AMF = 1/cos(theta_SZA) + 1/cos(theta_VZA)

    Args:
        amf (float): The specific Air Mass Factor for the observation.
        file_lut (str): Path to LUT file. Defaults to FILE_LUT_GAS.

    Returns:
        Tuple[NDArray, NDArray, NDArray]:

            1. **wvl_mod**: Model wavelength array.
               Shape: (n_model_wavelengths,) | Unit: [nm]

            2. **t_arr**: Transmittance spectrum interpolated to the specific AMF.
               Shape: (n_methane_ratios, n_model_wavelengths) | Unit: [unitless]

            3. **mr_arr**: Mixing ratios corresponding to the transmittance curves, interpolated to AMF.
               Shape: (n_methane_ratios,) | Unit: [ppb]
    """
    ds =xr.open_dataset(file_lut, cache=False)

    # Shape: (n_model_wavelengths,)
    wvl_mod = np.array(ds["wvl_mod"].values)

    # Shape: (n_air_mass_factors, n_methane_ratios, n_model_wavelengths)
    # Note: Transposed. Axis 0 represents the AMF dimension we will collapse via interpolation.
    t_arr_full = np.array(ds["t_ch4_arr"].values).T

    # Shape: (n_air_mass_factors, n_methane_ratios)
    # Note: Transposed. Axis 0 represents the AMF dimension.
    mr_arr_all = np.array(ds["mr_ch4_arr"].values).T

    # Shape: (n_air_mass_factors,)
    amf_arr = np.array(ds["amf_arr"].values)

    ds.close()

    # --- Interpolation Setup ---

    # We create interpolators along Axis 0 (AMF) to map generic geometry -> specific observation geometry.
    # Method: Linear Interpolation
    # Equation: y = y0 + (x - x0) * (y1 - y0) / (x1 - x0)

    # Input X: amf_arr (n_air_mass_factors,)
    # Input Y: t_arr_full (n_air_mass_factors, n_methane_ratios, n_model_wavelengths)
    f_t = interpolate.interp1d(amf_arr, t_arr_full, axis=0)

    # Input X: amf_arr (n_air_mass_factors,)
    # Input Y: mr_arr_all (n_air_mass_factors, n_methane_ratios)
    f_mr = interpolate.interp1d(amf_arr, mr_arr_all, axis=0)

    # --- Interpolation Execution ---

    # Operation: Evaluate interpolator at scalar `amf`.
    # Dimensionality Change: (n_air_mass_factors, ...) -> (...)

    # Final Shape: (n_methane_ratios, n_model_wavelengths)
    # Physical: The family of transmittance curves valid for this specific geometric path.
    t_arr = f_t(amf)

    # Final Shape: (n_methane_ratios,)
    # Physical: The mixing ratios associated with the rows of t_arr for this geometry.
    mr_arr = f_mr(amf)

    return wvl_mod, t_arr, mr_arr


def air_mass_factor(sza: float, vza: float) -> float:
    """
    Air Mass Factor (AMF) for a given Solar Zenith Angle (SZA) and Viewing Zenith Angle (VZA).
    The AMF is given by the formula:
    AMF = 1 / cos(VZA) + 1 / cos(SZA)

    Args:
        sza (float): Solar Zenith Angle in degrees
        vza (float): Viewing Zenith Angle in degrees

    Returns:
        float: Air Mass Factor
    """
    return 1.0 / np.cos(np.radians(vza)) + 1.0 / np.cos(np.radians(sza))
