"""Physics validation: brace load transfer is a *live* effect, not a no-op.

A code review once found the transferred brace load silently dropped before it
reached the failure verdict. This pins the behaviour: with the transfer fraction
set to zero a stressed-neighbours/unstressed-brace setup leaves the brace at zero
margin; with the default fraction the brace must carry load. Also serves as a
second proof that BraceConfig is genuinely threaded through.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import trimesh

from battlebot_sim.config import DEFAULT_CONFIG
from battlebot_sim.damage.braces import apply_brace_sharing
from battlebot_sim.damage.model import DamageResult
from battlebot_sim.materials.library import load_default_library
from battlebot_sim.mesh.segment import BotModel, segment_mesh

pytestmark = pytest.mark.validation


def _braced_bot():
    library = load_default_library()
    a = trimesh.creation.box(extents=(0.08, 0.08, 0.05))
    a.apply_translation((-0.10, 0, 0))
    b = trimesh.creation.box(extents=(0.08, 0.08, 0.05))
    b.apply_translation((0.10, 0, 0))
    bar = trimesh.creation.box(extents=(0.12, 0.02, 0.02))   # bridges the gap
    combined = trimesh.util.concatenate([a, b, bar])
    bot = BotModel(combined, segment_mesh(combined))
    bot.assign_material_to_all(library.get("Aluminum 6061-T6"))
    bot.set_brace(len(bot.parts) - 1, True)                  # smallest part = brace
    return bot


def _stress_cubes_only(bot) -> DamageResult:
    n_faces = len(bot.original.faces)
    face_part = np.zeros(n_faces, dtype=np.int64)
    for p in bot.parts:
        face_part[p.face_ids] = p.index
    peak = np.zeros(n_faces)
    for p in bot.parts:
        if not p.is_brace:
            peak[p.face_ids] = 200e6
    yield_pa = bot.parts[0].material.yield_pa
    margin = peak / yield_pa
    return DamageResult(
        energy_per_face=np.zeros(n_faces),
        peak_stress_per_face=peak.copy(),
        failure_margin_per_face=margin.copy(),
        face_part=face_part,
        part_max_margin={p.index: float(margin[p.face_ids].max()) for p in bot.parts},
        part_total_energy={p.index: 0.0 for p in bot.parts},
    )


def test_brace_transfer_is_a_live_effect():
    bot = _braced_bot()
    brace_idx = len(bot.parts) - 1
    base = _stress_cubes_only(bot)
    assert base.part_max_margin[brace_idx] == 0.0          # brace starts unstressed

    off = apply_brace_sharing(base, bot, cfg=replace(DEFAULT_CONFIG.brace, transfer=0.0))
    on = apply_brace_sharing(base, bot, cfg=DEFAULT_CONFIG.brace)   # transfer=0.5

    assert off.part_max_margin[brace_idx] == 0.0          # no transfer -> no load
    assert on.part_max_margin[brace_idx] > 0.0            # default -> brace carries load
