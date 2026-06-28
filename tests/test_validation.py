"""Tests for boundary input validation."""
from __future__ import annotations

import numpy as np
import pytest
import trimesh

from battlebot_sim.errors import ValidationError
from battlebot_sim.validation import (
    validate_mass,
    validate_mesh,
    validate_run_params,
    validate_scale,
)


def test_validate_scale_rejects_nonpositive_and_nonfinite():
    assert validate_scale(0.001) == 0.001
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValidationError):
            validate_scale(bad)


def test_validate_mesh_accepts_a_normal_box():
    validate_mesh(trimesh.creation.box(extents=(0.1, 0.1, 0.1)))  # no raise


def test_validate_mesh_rejects_nan_vertices():
    good = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    verts = good.vertices.copy()
    verts[0] = [np.nan, 0.0, 0.0]
    nan_mesh = trimesh.Trimesh(vertices=verts, faces=good.faces.copy(), process=False)
    with pytest.raises(ValidationError):
        validate_mesh(nan_mesh)


def test_validate_mesh_rejects_insane_bbox():
    huge = trimesh.creation.box(extents=(1e5, 1e5, 1e5))  # ~100 km — a units mistake
    with pytest.raises(ValidationError):
        validate_mesh(huge)


def test_validate_run_params_bounds():
    assert validate_run_params(3, 60) == (3, 60)
    for bad_trials in (0, -1, 1001, 1.5):
        with pytest.raises(ValidationError):
            validate_run_params(bad_trials, 60)
    for bad_fps in (0, -5, 5000):
        with pytest.raises(ValidationError):
            validate_run_params(1, bad_fps)


def test_validate_mass_rejects_negative_and_nonfinite():
    assert validate_mass(2.5) == 2.5
    for bad in (-0.1, float("nan"), float("inf")):
        with pytest.raises(ValidationError):
            validate_mass(bad)


def test_load_bot_rejects_bad_scale():
    from battlebot_sim.mesh.segment import load_bot, sample_bot_path

    with pytest.raises(ValidationError):
        load_bot(sample_bot_path(), scale_to_m=0.0)


def test_weight_check_rejects_nan_mass():
    from battlebot_sim.materials.assign import NHRL_CLASSES, validate_weight_class

    with pytest.raises(ValidationError):
        validate_weight_class(float("nan"), NHRL_CLASSES["3lb"])
