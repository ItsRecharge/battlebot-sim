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
    """Return a new DamageResult with brace stress relief applied.

    A brace relieves the *structural* response of the parts it bridges — their
    bending and membrane stress drop by ``1/(1+k)`` and a share of the shed load
    is carried by the brace itself — but it does **not** relieve the local
    *contact* von Mises (a strut behind a plate can't stop a surface dent). The
    absolute verdict (``part_stress`` / ``part_max_margin``) is recomputed from
    the relieved structural stress; the Gaussian heatmap field is scaled the same
    way it always was, for the rendered picture.
    """
    braces = [p for p in bot.parts if p.is_brace]
    if not braces:
        return result

    out = copy.deepcopy(result)
    others = [p for p in bot.parts if not p.is_brace]
    ps_by_idx = {ps.part_index: ps for ps in out.part_stress}

    for brace in braces:
        k = _brace_k(brace, cfg.k_ref)
        reduction = 1.0 / (1.0 + k)
        brace_faces = brace.face_ids
        for part in others:
            if not _aabb_adjacent(part, brace, cfg.adjacency_tol):
                continue
            faces = part.face_ids
            # --- rendered heatmap field (unchanged behaviour) ---
            shed = out.peak_stress_per_face[faces] * (1.0 - reduction)
            out.peak_stress_per_face[faces] *= reduction
            out.failure_margin_per_face[faces] *= reduction
            if len(brace_faces):
                out.peak_stress_per_face[brace_faces] += cfg.transfer * float(shed.max())
            # --- absolute verdict: relieve the braced part's bending+membrane ---
            # The brace's own structural margin comes from the contacts it takes,
            # not from a synthetic transfer: piling a neighbour's shed stress onto
            # the brace as fake bending produced unphysical margins, so a brace
            # only *relieves* here.
            ps = ps_by_idx.get(part.index)
            if ps is not None:
                ps.bending_stress *= reduction
                ps.membrane_stress *= reduction

        # The transferred load raises the brace's own heatmap stress; recompute
        # its per-face margin so the absorbed load reaches the rendered map
        # (mirrors compute_damage: no/non-finite yield -> 0).
        if len(brace_faces):
            y = brace.material.yield_pa if brace.material else np.inf
            if np.isfinite(y) and y > 0:
                out.failure_margin_per_face[brace_faces] = (
                    out.peak_stress_per_face[brace_faces] / y
                )

    if out.part_stress:
        # Recompute every part's governing stress / margin from the relieved
        # structural stress (contact is left untouched — braces don't stop dents).
        yld = {p.index: (p.material.yield_pa if p.material else np.inf) for p in bot.parts}
        ult = {p.index: (p.material.ultimate_pa if p.material else np.inf) for p in bot.parts}
        for ps in out.part_stress:
            struct = ps.bending_stress + ps.membrane_stress
            if ps.contact_stress >= struct:
                ps.governing_stress, ps.governing_mode = ps.contact_stress, "contact"
            else:
                ps.governing_stress = struct
                ps.governing_mode = ("bending" if ps.bending_stress >= ps.membrane_stress
                                     else "membrane")
            y, u = yld[ps.part_index], ult[ps.part_index]
            ps.margin = float(ps.governing_stress / y) if np.isfinite(y) and y > 0 else 0.0
            ps.yields = ps.margin >= 1.0
            ps.fractures = bool(np.isfinite(u) and u > 0 and ps.governing_stress >= u)
            out.part_max_margin[ps.part_index] = ps.margin
    else:
        # Legacy/hand-built result with no per-part structural data: fall back to
        # summarising the rendered per-face field (preserves older behaviour).
        out.part_max_margin = {
            p.index: float(out.failure_margin_per_face[p.face_ids].max())
            if len(p.face_ids) else 0.0
            for p in bot.parts
        }
    return out
