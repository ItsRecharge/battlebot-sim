"""Helpers to turn raw per-face damage values into render-ready scalars."""

from __future__ import annotations

import numpy as np


def normalize(values: np.ndarray, mode: str = "linear") -> np.ndarray:
    """Scale values to 0..1 for colour mapping.

    mode="linear": min-max. mode="log": log1p then min-max (good for energy,
    which spans orders of magnitude).
    """
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return v                       # nothing to scale; avoid min() on empty
    if mode == "log":
        v = np.log1p(np.clip(v, 0.0, None))
    vmin, vmax = float(v.min()), float(v.max())
    if vmax - vmin < 1e-15:
        return np.zeros_like(v)
    return (v - vmin) / (vmax - vmin)


def vertex_scalars(faces: np.ndarray, n_vertices: int, face_values: np.ndarray) -> np.ndarray:
    """Average per-face values onto vertices for smooth shading.

    faces: (n_faces, 3) vertex indices. Returns (n_vertices,) array; vertices
    touched by no face are 0.
    """
    faces = np.asarray(faces)
    fv = np.asarray(face_values, dtype=float)
    acc = np.zeros(n_vertices)
    count = np.zeros(n_vertices)
    for col in range(faces.shape[1]):
        np.add.at(acc, faces[:, col], fv)
        np.add.at(count, faces[:, col], 1.0)
    return np.divide(acc, count, out=np.zeros_like(acc), where=count > 0)
