"""
lut_emit.py · methane transmittance LUT reader and injection model.

The look-up table ``output_Tch4_LUT_AMF_VZA_0_v2.nc`` gives the methane
transmittance ``T(wvl)`` for a small set of methane columns (scaling factors
0.8, 1.0, 1.15, ... of a background column) and a set of air-mass factors
(AMF). A plume injection scales the measured radiance by the ratio
``T(mr_bg + Delta) / T(mr_bg)``.

Two modelling choices, both physically motivated:

1. ``read_lut`` interpolates the AMF axis in ``log(T)`` instead of ``T``.
   Beer-Lambert says ``T = exp(-tau)``, and the optical depth ``tau`` is the
   quantity that varies smoothly (near-linearly) with the absorption path,
   so interpolating ``log(T)`` and exponentiating is more faithful than
   interpolating ``T`` directly. Same convention as the vault module
   ``inyeccion/lut_ch4.py``.
2. ``TransmittanceModel`` uses the LUT node with scale factor 1.0 as the
   background (``mr_bg``), not a hard-coded 1900 ppb. That makes the
   ``Delta = 0`` ratio exactly 1.0, and the mixeratio interpolation along the
   column axis is also done in ``log(T)`` (exponential in methane).

Units: wavelength [nm], mixing ratio [ppb], transmittance [0-1], AMF unitless.
"""

from __future__ import annotations

import warnings

import numpy as np
import xarray as xr

# Absolute path to the LUT shipped with the working tree. Not bundled in the
# package (too large); callers should pass an explicit path in production.
DEFAULT_LUT_PATH = (
    "/data/users/julio/Notes/01-Projects/methanset/01-injection-notes/code/"
    "injection-carlos-2026-10-05/github/output_Tch4_LUT_AMF_VZA_0_v2.nc"
)


def _open_lut(path: str):
    """Load the raw LUT arrays. Returns ``(wvl, t, mr, amf, ch4_sc)``.

    Shapes follow the NetCDF file:
        wvl    : (num_wvl,)                     [nm]
        t      : (num_wvl, n_ch4, num_amf)      [0-1]
        mr     : (n_ch4, num_amf)               [ppb]
        amf    : (num_amf,)
        ch4_sc : (n_ch4,)                       scaling factors (0.8, 1.0, ...)
    """
    ds = xr.open_dataset(path, cache=False)
    try:
        wvl = np.asarray(ds["wvl_mod"].values, dtype=np.float64)
        t = np.asarray(ds["t_ch4_arr"].values, dtype=np.float64)
        mr = np.asarray(ds["mr_ch4_arr"].values, dtype=np.float64)
        amf = np.asarray(ds["amf_arr"].values, dtype=np.float64)
        ch4_sc = np.asarray(ds["ch4_sc_arr"].values, dtype=np.float64)
    finally:
        ds.close()
    return wvl, t, mr, amf, ch4_sc


def _amf_weights(amf_arr: np.ndarray, amf: float) -> tuple[int, float, float]:
    """Bracketing index and weight for linear interpolation along ``amf_arr``.

    The value is clipped into the LUT range. Returns ``(j, w, amf_clipped)``
    such that ``y = (1 - w) * y[j-1] + w * y[j]``.
    """
    lo, hi = float(amf_arr[0]), float(amf_arr[-1])
    amf_c = float(np.clip(amf, lo, hi))
    if not (lo <= float(amf) <= hi):
        warnings.warn(
            f"AMF {float(amf):.4f} outside LUT range [{lo:.3f}, {hi:.3f}]; "
            f"clipped to {amf_c:.4f}.",
            stacklevel=3,
        )
    j = int(np.searchsorted(amf_arr, amf_c))
    j = int(np.clip(j, 1, len(amf_arr) - 1))
    w = (amf_c - amf_arr[j - 1]) / (amf_arr[j] - amf_arr[j - 1])
    return j, float(w), amf_c


def read_lut(path: str, amf: float):
    """Read the methane LUT and collapse the AMF axis at ``amf``.

    Interpolation along AMF is done in ``log(T)`` (see module docstring), so
    the returned transmittance is ``exp`` of the linearly interpolated optical
    depth. This mirrors ``CARLOS/lut.py`` but in log space; the arrays keep the
    same orientation.

    Parameters
    ----------
    path : str
        Path to ``output_Tch4_LUT_AMF_VZA_0_v2.nc``.
    amf : float
        Air-mass factor of the observation (see :func:`air_mass_factor`).
        Values outside the LUT grid are clipped with a warning.

    Returns
    -------
    wvl : (num_wvl,) [nm]
    t : (n_ch4, num_wvl) [0-1]
        Transmittance curves at the requested AMF.
    mr : (n_ch4,) [ppb]
        Methane mixing ratios of the rows of ``t``.
    """
    wvl, t_full, mr_full, amf_arr, _ = _open_lut(path)
    j, w, _ = _amf_weights(amf_arr, amf)

    # (num_wvl, n_ch4, num_amf) -> interpolate in log-T -> transpose to
    # (n_ch4, num_wvl), matching CARLOS/lut.py.
    log_t = (1.0 - w) * np.log(t_full[:, :, j - 1]) + w * np.log(t_full[:, :, j])
    t = np.exp(log_t).T

    mr = (1.0 - w) * mr_full[:, j - 1] + w * mr_full[:, j]
    return wvl, t, mr


def interp_log_transmittance(
    mr_nodes: np.ndarray,
    logt_nodes: np.ndarray,
    mr: np.ndarray,
    clip: bool = True,
) -> np.ndarray:
    """Linearly interpolate ``log(T)`` along the mixing-ratio axis.

    ``T`` is exponential in the methane column, so the node-to-node variation
    is close to linear in ``log(T)`` (Beer-Lambert).

    Parameters
    ----------
    mr_nodes : (n_nodes,) [ppb], strictly increasing.
    logt_nodes : (n_nodes, num_wvl) [log transmittance].
    mr : scalar or (n,) [ppb].
    clip : bool, default True
        Clip ``mr`` into ``[mr_nodes[0], mr_nodes[-1]]`` so that no
        extrapolation happens beyond the tabulated columns. Set to False to
        allow log-linear extrapolation (used to validate the model against
        held-out LUT nodes).

    Returns
    -------
    (n, num_wvl) array (or (num_wvl,) for scalar input).
    """
    scalar = np.ndim(mr) == 0
    mr_arr = np.atleast_1d(np.asarray(mr, dtype=np.float64))
    mr_c = np.clip(mr_arr, mr_nodes[0], mr_nodes[-1]) if clip else mr_arr

    k = np.searchsorted(mr_nodes, mr_c)
    k = np.clip(k, 1, len(mr_nodes) - 1)
    w = (mr_c - mr_nodes[k - 1]) / (mr_nodes[k] - mr_nodes[k - 1])

    out = (1.0 - w)[:, None] * logt_nodes[k - 1] + w[:, None] * logt_nodes[k]
    return out[0] if scalar else out


class TransmittanceModel:
    """Spectral transmittance ratio ``T(mr_bg + Delta) / T(mr_bg)``.

    Parameters
    ----------
    path : str
        Path to the methane LUT.
    amf : float
        Observation air-mass factor.
    mr_bg : float, optional
        Background total methane mixing ratio [ppb]. Defaults to the LUT node
        with scale factor 1.0 at this AMF. See notes below.

    Notes
    -----
    The default ``mr_bg`` is *not* a fixed 1900 ppb. The LUT rows satisfy
    ``mr_ch4_arr = ch4_sc_arr[:, None] * column(amf)``, i.e. the ratio
    ``mr_ch4_arr[:, i_amf] / ch4_sc_arr`` is the AMF background column. The
    scale-1.0 node is therefore the row whose scaling factor is 1.0 (fallback:
    the row where ``mr_ch4_arr[:, i_amf] / ch4_sc_arr`` coincides with
    ``mr_ch4_arr[:, i_amf]``). Using that node as background makes the
    ``Delta = 0`` ratio exactly 1.0 by construction.

    Clamping: the plume column ``mr_bg + Delta`` is clipped to the tabulated
    range ``[min(mr), max(mr)]``. Deltas larger than what the LUT covers are
    saturated at the top node (flat response), never extrapolated.
    """

    def __init__(self, path: str, amf: float, mr_bg: float | None = None):
        wvl, t_full, mr_full, amf_arr, ch4_sc = _open_lut(path)
        j, w, self.amf = _amf_weights(amf_arr, amf)

        log_t = (1.0 - w) * np.log(t_full[:, :, j - 1]) + w * np.log(t_full[:, :, j])
        self.wvl = wvl
        self.t = np.exp(log_t).T  # (n_ch4, num_wvl)
        self.logt = log_t.T  # (n_ch4, num_wvl)
        self.mr = (1.0 - w) * mr_full[:, j - 1] + w * mr_full[:, j]  # (n_ch4,)

        if not np.all(np.diff(self.mr) > 0):
            raise ValueError("LUT mixing-ratio nodes are not strictly increasing")

        # Locate the scale-1.0 node.
        self.i_bg = int(np.argmin(np.abs(ch4_sc - 1.0)))
        self.mr_bg = float(self.mr[self.i_bg]) if mr_bg is None else float(mr_bg)

    def log_ratio(self, delta_ppb):
        """``log[T(mr_bg + Delta) / T(mr_bg)]`` at LUT spectral resolution.

        Scalar ``delta_ppb`` -> ``(num_wvl,)``; array -> ``(n, num_wvl)``.
        """
        scalar = np.ndim(delta_ppb) == 0
        d = np.atleast_1d(np.asarray(delta_ppb, dtype=np.float64))
        logt_plume = interp_log_transmittance(self.mr, self.logt, self.mr_bg + d)
        out = logt_plume - self.logt[self.i_bg]
        return out[0] if scalar else out

    def ratio(self, delta_ppb):
        """``T(mr_bg + Delta) / T(mr_bg)`` at LUT spectral resolution.

        ``delta_ppb = 0`` returns 1.0 exactly (same node, both log-T values
        cancel). Scalar -> ``(num_wvl,)``; array -> ``(n, num_wvl)``.
        Deltas are clamped to the tabulated column range (no extrapolation).
        """
        return np.exp(self.log_ratio(delta_ppb))


def air_mass_factor(sza: float, vza: float) -> float:
    """Geometric air-mass factor ``1 / cos(vza) + 1 / cos(sza)``.

    Parameters
    ----------
    sza, vza : float
        Solar and viewing zenith angles [degrees].
    """
    return 1.0 / np.cos(np.radians(vza)) + 1.0 / np.cos(np.radians(sza))
