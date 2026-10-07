"""Tests for the methane transmittance LUT reader and injection model.

Run from the ``injection/`` directory:

    python -m pytest tests/test_lut.py -q

Covers:
    T1  read_lut / air_mass_factor sanity.
    T2  Delta = 0 gives ratio exactly 1; ratio decreases monotonically with
        Delta inside the CH4 absorption band.
    T3  a held-out internal LUT node (scale 1.30) is reproduced from the
        scale 1.0 and 1.15 nodes by log-T (Beer-Lambert) extrapolation, and
        the log-T error beats linear-T.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.lut_emit import (
    DEFAULT_LUT_PATH,
    TransmittanceModel,
    air_mass_factor,
    interp_log_transmittance,
    read_lut,
)

LUT = DEFAULT_LUT_PATH
AMF = 2.0
# Strong CH4 absorption window; the reference log-T vs linear errors were
# measured here.
CH4_WINDOW = (2250.0, 2400.0)


def test_t1_read_lut_and_amf():
    assert air_mass_factor(0.0, 0.0) == pytest.approx(2.0, abs=1e-12)
    assert air_mass_factor(45.0, 45.0) == pytest.approx(2.0 * np.sqrt(2.0), rel=1e-12)

    wvl, t, mr = read_lut(LUT, AMF)
    assert wvl.shape == (11936,)
    assert t.shape == (9, wvl.size)
    assert mr.shape == (9,)
    assert np.all(np.diff(mr) > 0)
    assert np.all(t > 0.0) and np.all(t <= 1.0)


def test_t2_zero_delta_is_one_and_ratio_decreases():
    wvl, _, _ = read_lut(LUT, AMF)
    model = TransmittanceModel(LUT, AMF)

    r0 = model.ratio(0.0)
    assert np.max(np.abs(r0 - 1.0)) < 1e-12

    deltas = np.array([0.0, 100.0, 300.0, 600.0, 1200.0])
    ratio = model.ratio(deltas)
    assert ratio.shape == (deltas.size, wvl.size)

    band = (wvl >= CH4_WINDOW[0]) & (wvl <= CH4_WINDOW[1])
    assert band.sum() > 0
    # More methane -> lower transmittance -> lower ratio, strictly.
    assert np.all(np.diff(ratio[:, band], axis=0) < 0.0)


def test_t3_log_t_beats_linear_t_on_held_out_node():
    wvl, t, mr = read_lut(LUT, AMF)

    # Held-out node: scale 1.30 (index 3). Predicted from the scale 1.0 and
    # 1.15 nodes (index 1 and 2) with log-T extrapolation, vs linear-T.
    i1, i2, it = 1, 2, 3
    logt = np.log(t)
    pred_log = np.exp(
        interp_log_transmittance(
            mr[[i1, i2]], logt[[i1, i2]], np.array([mr[it]]), clip=False
        )
    )[0]

    w = (mr[it] - mr[i1]) / (mr[i2] - mr[i1])
    pred_lin = t[i1] + (t[i2] - t[i1]) * w

    band = (wvl >= CH4_WINDOW[0]) & (wvl <= CH4_WINDOW[1])
    err_log = float(np.mean(np.abs(pred_log[band] - t[it][band]) / t[it][band] * 100.0))
    err_lin = float(np.mean(np.abs(pred_lin[band] - t[it][band]) / t[it][band] * 100.0))

    print(f"\nT3 log-T mean error {err_log:.4f}% vs linear-T {err_lin:.4f}%")
    assert err_log < 0.1
    assert err_log < err_lin
