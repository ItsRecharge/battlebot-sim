"""Auto-detect structural braces at import time.

A brace is a stiff, elongated member that *bridges* two or more other parts and
carries load between them. A part is flagged only when all three hold — elongated
geometry, a stiff/strong material, and a verified load path (it spans a gap and
has non-trivial axial stiffness) — so a plate, a blob, or a lone strut is never
mistaken for one. Every auto-flag stays user-overridable via the per-part
checkbox; this never clears a manual choice.

The geometry/stiffness maths can take a moment on a many-part model, so an
optional ``progress`` callback drives the import progress bar.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from battlebot_sim.config import DEFAULT_CONFIG, BraceConfig, BraceDetectConfig
from battlebot_sim.damage import structural as st
from battlebot_sim.damage.braces import _aabb_adjacent, _brace_k
from battlebot_sim.logging_setup import get_logger
from battlebot_sim.mesh.segment import BotModel, Part

logger = get_logger(__name__)


def _bridges_two_sides(part: Part, neighbours: list[Part], axis: np.ndarray) -> bool:
    """True when neighbours sit on *both* sides of the part along its long axis —
    i.e. it spans a gap rather than just abutting one cluster."""
    c0 = part.centroid
    positive = negative = False
    for nb in neighbours:
        s = float(np.dot(nb.centroid - c0, axis))
        if s > 1e-9:
            positive = True
        elif s < -1e-9:
            negative = True
        if positive and negative:
            return True
    return False


def auto_detect_braces(
    bot: BotModel,
    cfg: BraceConfig = DEFAULT_CONFIG.brace,
    detect_cfg: BraceDetectConfig = DEFAULT_CONFIG.brace_detect,
    progress: Callable[[int, int], None] | None = None,
) -> list[int]:
    """Flag elongated, stiff, load-bridging parts as braces.

    Returns the list of part indices newly flagged. Requires materials to be
    assigned (the stiffness gate reads them); parts with no material are skipped.
    """
    parts = bot.parts
    n = len(parts)
    flagged: list[int] = []
    for i, part in enumerate(parts):
        if progress is not None:
            progress(i, n)
        if part.is_brace:                       # respect an existing manual flag
            continue
        mat = part.material
        if mat is None:
            continue
        verts = np.asarray(part.mesh.vertices, dtype=float)
        sec = st.section_from_vertices(verts)
        elongated = (sec.L / max(sec.t, 1e-9)) >= detect_cfg.min_aspect
        stiff = (mat.youngs_pa >= detect_cfg.min_modulus_pa
                 and mat.yield_pa >= detect_cfg.min_yield_pa)
        if not (elongated and stiff):
            continue
        neighbours = [q for q in parts
                      if q.index != part.index
                      and _aabb_adjacent(part, q, cfg.adjacency_tol)]
        if len(neighbours) < 2:
            continue
        if not _bridges_two_sides(part, neighbours, st.long_axis(verts)):
            continue
        if _brace_k(part, cfg.k_ref) < detect_cfg.min_k:
            continue
        bot.set_brace(part.index, True)
        flagged.append(part.index)
    if progress is not None:
        progress(n, n)
    if flagged:
        logger.info("auto-detected %d brace(s): %s", len(flagged), flagged)
    return flagged
