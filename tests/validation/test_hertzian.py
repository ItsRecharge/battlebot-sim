"""Physics validation: the Hertzian peak-pressure code vs. the closed-form solution.

The failure verdict rests entirely on the Hertzian peak pressure, so this pins the
implementation against the textbook sphere-on-flat result, derived here
independently of the production code:

    contact radius   a   = (3 F R / (4 E*))^(1/3)
    peak pressure    p0  = 3 F / (2 pi a^2)

The model uses the algebraically-equivalent closed form
``p0 = (6 F E*^2 / (pi^3 R^2))^(1/3)``; this test proves the two agree.
"""
from __future__ import annotations

import numpy as np
import pytest

from battlebot_sim.damage.model import _hertzian_peak_pressure

pytestmark = pytest.mark.validation


def _p0_reference(force: float, eff_modulus: float, radius: float) -> float:
    """Textbook sphere-on-flat peak contact pressure, computed from scratch."""
    a = (3.0 * force * radius / (4.0 * eff_modulus)) ** (1.0 / 3.0)
    return 3.0 * force / (2.0 * np.pi * a**2)


@pytest.mark.parametrize("force", [50.0, 500.0, 5000.0, 50000.0])
@pytest.mark.parametrize("radius", [2e-3, 1e-2, 5e-2])
def test_hertzian_matches_closed_form(force, radius):
    eff_modulus = 70e9
    measured = _hertzian_peak_pressure(force, eff_modulus, radius)
    reference = _p0_reference(force, eff_modulus, radius)
    rel_err = abs(measured - reference) / reference
    assert rel_err < 1e-9, f"p0={measured:.4e} vs closed-form {reference:.4e}"


def test_hertzian_scales_with_cube_root_of_force():
    """p0 ~ F^(1/3): doubling 8x the force should ~double the peak pressure."""
    e, r = 70e9, 1e-2
    low = _hertzian_peak_pressure(1000.0, e, r)
    high = _hertzian_peak_pressure(8000.0, e, r)
    assert np.isclose(high / low, 2.0, rtol=1e-9)
