"""Boundary input validation: reject malformed input early with clear errors.

These run at the program's edges (mesh load, run setup, weight check) so a bad
scale, a NaN-laden mesh, or an out-of-range trial count fails loudly with a
helpful message instead of silently producing garbage or crashing deep in the
physics. They raise :class:`battlebot_sim.errors.ValidationError`, which the UI
catches and surfaces via its existing failure channel.

The finite-number guards mirror the ``np.isfinite`` idiom already used in
``mesh/segment.py`` for mass properties.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from battlebot_sim.errors import ValidationError

if TYPE_CHECKING:
    import trimesh

# Sanity bounds (metres) for a loaded mesh's bounding box *after* scaling. A combat
# bot spans centimetres to metres; anything outside this is almost certainly a
# units mistake (e.g. a millimetre STL loaded at scale 1.0) or a degenerate shape.
_MAX_EXTENT_M = 1.0e3
_MIN_EXTENT_M = 1.0e-6
_MAX_TRIALS = 1000
_MAX_FPS = 1000


def validate_scale(scale_to_m: float) -> float:
    """A mesh scale factor must be finite and strictly positive. Returns it."""
    s = float(scale_to_m)
    if not np.isfinite(s) or s <= 0.0:
        raise ValidationError(
            f"scale_to_m must be a finite positive number, got {scale_to_m!r}"
        )
    return s


def validate_mesh(mesh: trimesh.Trimesh) -> None:
    """Reject a mesh with no faces, non-finite vertices, or an insane bbox."""
    if len(getattr(mesh, "faces", [])) == 0:
        raise ValidationError("mesh has no faces")
    verts = np.asarray(mesh.vertices, dtype=float)
    if verts.size == 0:
        raise ValidationError("mesh has no vertices")
    if not np.all(np.isfinite(verts)):
        raise ValidationError("mesh has non-finite (NaN/inf) vertices")
    extent = float(np.max(verts.max(axis=0) - verts.min(axis=0)))
    if not np.isfinite(extent) or extent <= 0.0:
        raise ValidationError("mesh is degenerate (zero-size bounding box)")
    if not (_MIN_EXTENT_M <= extent <= _MAX_EXTENT_M):
        raise ValidationError(
            f"mesh bounding box {extent:.3g} m is outside the sane range "
            f"[{_MIN_EXTENT_M:g}, {_MAX_EXTENT_M:g}] m — check the scale / units"
        )


def validate_run_params(n_trials: int, fps: int) -> tuple[int, int]:
    """Battery trial count and capture fps must be sane positive integers."""
    for name, val, hi in (("n_trials", n_trials, _MAX_TRIALS), ("fps", fps, _MAX_FPS)):
        try:
            ival = int(val)
        except (TypeError, ValueError):
            raise ValidationError(f"{name} must be an integer, got {val!r}") from None
        if ival != val or not (1 <= ival <= hi):
            raise ValidationError(
                f"{name} must be an integer in [1, {hi}], got {val!r}"
            )
    return int(n_trials), int(fps)


def validate_mass(mass_kg: float) -> float:
    """A bot mass must be finite and non-negative. Returns it."""
    m = float(mass_kg)
    if not np.isfinite(m) or m < 0.0:
        raise ValidationError(f"mass must be finite and non-negative, got {mass_kg!r}")
    return m
