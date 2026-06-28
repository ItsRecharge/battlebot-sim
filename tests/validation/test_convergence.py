"""Physics validation: the failure verdict is converged w.r.t. the sim timestep.

The worst failure margin is driven by the hardest contact, which the MuJoCo solver
resolves essentially identically across a 2x timestep change — so the default
timestep (5e-4 s) sits firmly in the converged regime. If this test ever fails, the
verdict has become timestep-sensitive and the default needs tightening (a
legitimate, documented finding, not just a red build).
"""
from __future__ import annotations

import numpy as np
import pytest
import trimesh

from battlebot_sim.arena.nhrl import build_arena
from battlebot_sim.damage.model import compute_damage
from battlebot_sim.materials.assign import NHRL_CLASSES
from battlebot_sim.materials.library import load_default_library
from battlebot_sim.mesh.segment import BotModel, segment_mesh
from battlebot_sim.sim.battery import StressBattery, run_battery
from battlebot_sim.sim.engine import SimEngine

pytestmark = [pytest.mark.validation, pytest.mark.native_isolated]


def _peak_margin(timestep: float) -> float:
    library = load_default_library()
    cls = NHRL_CLASSES["3lb"]
    cube = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    bot = BotModel(cube, segment_mesh(cube))
    bot.assign_material_to_all(library.get("Aluminum 6061-T6"))
    arena = build_arena(cls)
    engine = SimEngine(arena, bot, timestep=timestep)
    trace = run_battery(engine, StressBattery(arena, cls, n_trials=1, seed=0), fps=30)
    result = compute_damage(trace, bot, arena, library)
    return max(result.part_max_margin.values())


def test_failure_verdict_is_timestep_converged():
    coarse = _peak_margin(1e-3)
    fine = _peak_margin(5e-4)          # the default
    assert np.isfinite(coarse) and coarse > 0
    assert np.isfinite(fine) and fine > 0
    assert np.isclose(coarse, fine, rtol=1e-3), (
        f"verdict drifted with timestep: {coarse:.6e} (1e-3) vs {fine:.6e} (5e-4); "
        "the default timestep may no longer be in the converged regime"
    )
