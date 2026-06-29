"""Closed-form validation of the analytic stress functions (damage/structural.py).

Each test pins a function against an independent textbook formula or a known
value, so a regression in the absolute-margin model fails loudly here rather than
silently shifting the failure verdict.
"""

import numpy as np
import pytest

from battlebot_sim.damage import structural as st

# --- contact: subsurface von Mises onset ------------------------------------

def test_vm_factor_matches_known_value_for_nu_0_3():
    # Johnson: max von Mises ~ 0.62 * p0 for nu = 0.3 (first yield at p0 ~ 1.60Y).
    assert st.vm_factor_from_poisson(0.3) == pytest.approx(0.62, abs=0.01)


def test_vm_factor_decreases_with_poisson():
    # Higher Poisson ratio -> less subsurface shear -> smaller von-Mises coefficient.
    assert st.vm_factor_from_poisson(0.2) > st.vm_factor_from_poisson(0.45)


def test_contact_yield_onset_is_subsurface():
    """A part yields in contact when p0 >= yield / vm_factor (~1.60 * yield),
    NOT when p0 >= yield. Verify the corrected criterion brackets that point."""
    yield_pa = 250e6
    vm = st.vm_factor_from_poisson(0.3)
    p0_onset = yield_pa / vm                      # ~ 1.60 * yield
    just_below = st.contact_von_mises_peak(0.99 * p0_onset, vm)
    just_above = st.contact_von_mises_peak(1.01 * p0_onset, vm)
    assert just_below < yield_pa <= just_above
    # The old surface criterion (p0 vs yield) would have failed ~1.6x too early.
    assert p0_onset > 1.5 * yield_pa


# --- combined-curvature contact radius --------------------------------------

def test_effective_radius_sharp_on_flat_is_indenter():
    assert st.effective_contact_radius(4e-3, np.inf) == pytest.approx(4e-3)


def test_effective_radius_equal_radii_halves():
    assert st.effective_contact_radius(0.01, 0.01) == pytest.approx(0.005)


def test_effective_radius_smaller_than_either_input():
    r = st.effective_contact_radius(4e-3, 0.05)
    assert r < 4e-3 and r < 0.05


# --- oriented section (PCA) --------------------------------------------------

def _box_corners(extents):
    """The 8 corners of an axis-aligned box centred at the origin."""
    hx, hy, hz = (np.asarray(extents) / 2.0)
    return np.array([[sx * hx, sy * hy, sz * hz]
                     for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])


def test_section_recovers_thickness_of_axis_aligned_plate():
    sec = st.section_from_vertices(_box_corners([0.20, 0.10, 0.01]))
    assert sec.t == pytest.approx(0.01, rel=1e-6)
    assert sec.w == pytest.approx(0.10, rel=1e-6)
    assert sec.L == pytest.approx(0.20, rel=1e-6)


def test_section_recovers_thickness_of_rotated_plate():
    """PCA must recover the true 1 cm thickness even when the plate is tilted in
    world axes (an AABB would wrongly inflate it)."""
    corners = _box_corners([0.20, 0.10, 0.01])
    # Rotate 35 deg about an arbitrary axis.
    axis = np.array([1.0, 2.0, 3.0])
    axis = axis / np.linalg.norm(axis)
    a = np.radians(35.0)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)
    sec = st.section_from_vertices(corners @ R.T)
    assert sec.t == pytest.approx(0.01, rel=1e-6)
    assert sec.w == pytest.approx(0.10, rel=1e-6)
    assert sec.L == pytest.approx(0.20, rel=1e-6)


def test_section_clamps_degenerate_input():
    sec = st.section_from_vertices(np.zeros((1, 3)))
    assert sec.t > 0 and sec.w > 0 and sec.L > 0


# --- bending vs M c / I ------------------------------------------------------

def test_bending_stress_matches_Mc_over_I():
    F, span, t, w = 1000.0, 0.20, 0.01, 0.10
    bc = 0.25
    sigma = st.bending_stress(F, span, t, w, bc)
    # Independent closed form: M = bc*F*span, second moment I = w*t^3/12, c = t/2.
    moment = bc * F * span
    second_moment = w * t**3 / 12.0
    c = t / 2.0
    assert sigma == pytest.approx(moment * c / second_moment, rel=1e-12)


def test_bending_stress_scales_inverse_square_with_thickness():
    s1 = st.bending_stress(1000.0, 0.2, 0.01, 0.1, 0.25)
    s2 = st.bending_stress(1000.0, 0.2, 0.02, 0.1, 0.25)
    assert s1 / s2 == pytest.approx(4.0, rel=1e-9)   # ~ 1/t^2


def test_membrane_stress_is_force_over_area():
    assert st.membrane_stress(500.0, 0.001) == pytest.approx(5.0e5)


# --- governing mode ----------------------------------------------------------

def test_governing_picks_contact_when_it_dominates():
    gov, mode = st.governing_stress(300e6, 50e6, 10e6)
    assert mode == "contact" and gov == pytest.approx(300e6)


def test_governing_sums_bending_and_membrane_when_structural_dominates():
    gov, mode = st.governing_stress(100e6, 200e6, 50e6)
    assert mode == "bending" and gov == pytest.approx(250e6)


def test_governing_reports_membrane_when_it_is_the_larger_structural_term():
    gov, mode = st.governing_stress(50e6, 20e6, 120e6)
    assert mode == "membrane" and gov == pytest.approx(140e6)
