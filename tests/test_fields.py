"""Tests for the damage field helpers (normalize, vertex_scalars)."""
from __future__ import annotations

import numpy as np

from battlebot_sim.damage.fields import normalize, vertex_scalars


def test_normalize_linear_minmax():
    out = normalize(np.array([0.0, 5.0, 10.0]))
    assert out.min() == 0.0
    assert out.max() == 1.0
    assert np.isclose(out[1], 0.5)


def test_normalize_constant_is_zeros():
    assert np.allclose(normalize(np.array([3.0, 3.0, 3.0])), 0.0)


def test_normalize_empty_is_empty():
    out = normalize(np.array([]))
    assert out.size == 0


def test_normalize_log_mode_is_monotonic():
    out = normalize(np.array([0.0, 9.0, 99.0]), mode="log")
    assert out[0] == 0.0
    assert out[2] == 1.0
    assert out[0] < out[1] < out[2]


def test_vertex_scalars_average():
    # Two triangles sharing vertex 1; vertex 1 averages both faces' values.
    faces = np.array([[0, 1, 2], [1, 3, 4]])
    out = vertex_scalars(faces, 5, np.array([2.0, 4.0]))
    assert out[0] == 2.0          # only in face 0
    assert out[1] == 3.0          # mean of faces 0 and 1
    assert out[3] == 4.0          # only in face 1


def test_vertex_scalars_untouched_vertices_are_zero():
    faces = np.array([[0, 1, 2]])
    out = vertex_scalars(faces, 5, np.array([6.0]))
    assert out[4] == 0.0          # vertex 4 is in no face
    assert np.allclose(out[:3], 6.0)
