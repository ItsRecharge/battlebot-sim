"""Brace load-sharing: a heuristic that lets brace-tagged parts shed stress.

When a part sits next to a brace, the brace carries part of the impact load, so
the part's local peak stress drops by a factor 1/(1 + k), where k is the brace's
normalised axial stiffness k = E * A / L  (A = average cross-section, L = span).
A fraction of the shed stress is transferred onto the brace itself.

Stiffer / thicker braces (larger A) give larger k and therefore more relief —
a deliberately simple beam-style heuristic, NOT structural FEA.
"""

from __future__ import annotations

import copy

import numpy as np

from battlebot_sim.config import DEFAULT_CONFIG, BraceConfig
from battlebot_sim.damage.model import DamageResult
from battlebot_sim.mesh.segment import BotModel

# Brace load-sharing constants live on ``BraceConfig`` (battlebot_sim/config.py)
# and are threaded in as ``cfg`` so a study can vary them per run.


def _extents(part) -> np.ndarray:
    b = part.bounds
    return np.asarray(b[1] - b[0], dtype=float)


def _brace_k(part, k_ref: float = DEFAULT_CONFIG.brace.k_ref) -> float:
    ext = _extents(part)
    span = float(max(ext.max(), 1e-6))
    cross_section = part.volume_m3 / span      # average cross-sectional area
    E = part.material.youngs_pa if part.material else 1.0
    k_axial = E * cross_section / span
    return k_axial / k_ref


def _aabb_adjacent(a_part, b_part, tol: float) -> bool:
    a, b = a_part.bounds, b_part.bounds
    for ax in range(3):
        lo = max(a[0][ax], b[0][ax])
        hi = min(a[1][ax], b[1][ax])
        if (lo - hi) > tol:        # separated by more than tol on this axis
            return False
    return True


def apply_brace_sharing(result: DamageResult, bot: BotModel,
                        cfg: BraceConfig = DEFAULT_CONFIG.brace) -> DamageResult:
    """Return a new DamageResult with brace stress relief applied."""
    braces = [p for p in bot.parts if p.is_brace]
    if not braces:
        return result

    out = copy.deepcopy(result)
    others = [p for p in bot.parts if not p.is_brace]

    for brace in braces:
        k = _brace_k(brace, cfg.k_ref)
        reduction = 1.0 / (1.0 + k)
        brace_faces = brace.face_ids
        for part in others:
            if not _aabb_adjacent(part, brace, cfg.adjacency_tol):
                continue
            faces = part.face_ids
            shed = out.peak_stress_per_face[faces] * (1.0 - reduction)
            out.peak_stress_per_face[faces] *= reduction
            out.failure_margin_per_face[faces] *= reduction
            # Transfer a share of the shed stress onto the brace.
            if len(brace_faces):
                out.peak_stress_per_face[brace_faces] += cfg.transfer * float(shed.max())

        # The transferred load raises the brace's own stress; recompute its
        # failure margin from the brace's yield so the absorbed load reaches the
        # verdict/heatmap (mirrors compute_damage: no/non-finite yield -> 0).
        if len(brace_faces):
            y = brace.material.yield_pa if brace.material else np.inf
            if np.isfinite(y) and y > 0:
                out.failure_margin_per_face[brace_faces] = (
                    out.peak_stress_per_face[brace_faces] / y
                )

    # Recompute per-part summaries from the modified fields.
    out.part_max_margin = {
        p.index: float(out.failure_margin_per_face[p.face_ids].max())
        if len(p.face_ids) else 0.0
        for p in bot.parts
    }
    return out
