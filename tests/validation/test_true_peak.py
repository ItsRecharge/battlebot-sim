"""Physics validation: the failure verdict is independent of the heatmap-spread
constants.

The old model read the verdict off the Gaussian-painted per-face field, so on a
coarse mesh the "appearance-only" sigma constants dominated the margin (the
caveat documented in docs/uncertainty.md). The absolute verdict now reads the
un-attenuated governing stress at the true contact point, so changing
``stress_sigma_frac`` / ``sigma_patch_factor`` must move the *rendered heatmap*
but never the *verdict*.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import trimesh

from battlebot_sim.arena.nhrl import build_arena
from battlebot_sim.config import DEFAULT_CONFIG
from battlebot_sim.damage.model import compute_damage
from battlebot_sim.materials.assign import NHRL_CLASSES
from battlebot_sim.materials.library import load_default_library
from battlebot_sim.mesh.segment import BotModel, segment_mesh
from battlebot_sim.sim.battery import StressBattery, run_battery
from battlebot_sim.sim.engine import SimEngine

pytestmark = [pytest.mark.validation, pytest.mark.native_isolated]


def _trace_and_bot():
    library = load_default_library()
    cls = NHRL_CLASSES["3lb"]
    # A deliberately coarse cube: 12 faces, so no centroid lands on a contact.
    cube = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    bot = BotModel(cube, segment_mesh(cube))
    bot.assign_material_to_all(library.get("Aluminum 6061-T6"))
    arena = build_arena(cls)
    engine = SimEngine(arena, bot, timestep=5e-4)
    trace = run_battery(engine, StressBattery(arena, cls, n_trials=1, seed=0), fps=30)
    return trace, bot, arena, library


def test_verdict_is_sigma_independent_but_heatmap_is_not():
    trace, bot, arena, library = _trace_and_bot()
    tight = replace(DEFAULT_CONFIG.damage, stress_sigma_frac=0.2, sigma_patch_factor=4.0)
    loose = replace(DEFAULT_CONFIG.damage, stress_sigma_frac=0.8, sigma_patch_factor=16.0)

    r_tight = compute_damage(trace, bot, arena, library, cfg=tight)
    r_loose = compute_damage(trace, bot, arena, library, cfg=loose)

    # Verdict: identical across a 4x sigma swing.
    assert max(r_tight.part_max_margin.values()) > 0
    assert np.isclose(max(r_tight.part_max_margin.values()),
                      max(r_loose.part_max_margin.values()), rtol=1e-9)

    # Heatmap appearance: the painted field genuinely responds to the sigma.
    assert not np.allclose(r_tight.failure_margin_per_face,
                           r_loose.failure_margin_per_face)
