"""Unit tests for the MuJoCo engine wrapper."""
from __future__ import annotations

import numpy as np
import pytest
import trimesh

from battlebot_sim.arena.nhrl import build_arena
from battlebot_sim.config import DEFAULT_CONFIG
from battlebot_sim.materials.assign import NHRL_CLASSES
from battlebot_sim.mesh.segment import BotModel, segment_mesh
from battlebot_sim.sim.engine import SimEngine

pytestmark = pytest.mark.native_isolated


def _cube_bot(library):
    cube = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    bot = BotModel(cube, segment_mesh(cube))
    bot.assign_material_to_all(library.get("Aluminum 6061-T6"))
    return bot


def test_engine_uses_config_timestep(library):
    eng = SimEngine(build_arena(NHRL_CLASSES["3lb"]), _cube_bot(library))
    assert eng.timestep == DEFAULT_CONFIG.sim.timestep


def test_engine_custom_timestep(library):
    eng = SimEngine(build_arena(NHRL_CLASSES["3lb"]), _cube_bot(library), timestep=1e-3)
    assert eng.timestep == 1e-3


def test_engine_step_keeps_state_finite(library):
    eng = SimEngine(build_arena(NHRL_CLASSES["3lb"]), _cube_bot(library))
    eng.set_pose((0.0, 0.0, 0.3))
    eng.set_velocity((1.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    for _ in range(100):
        eng.step()
    pos, quat = eng.get_pose()
    assert np.all(np.isfinite(pos))
    assert np.all(np.isfinite(quat))
    assert abs(float(np.linalg.norm(quat)) - 1.0) < 1e-6  # quaternion stays unit


def test_engine_reads_floor_contacts_on_drop(library):
    bot = _cube_bot(library)
    eng = SimEngine(build_arena(NHRL_CLASSES["3lb"]), bot)
    eng.set_pose((0.0, 0.0, 0.15))          # drop the cube onto the floor
    part_indices = {p.index for p in bot.parts}
    saw_contact = False
    for _ in range(500):
        eng.step()
        for c in eng.read_contacts():
            assert c["normal_force"] > 0.0
            assert c["part_index"] in part_indices
            saw_contact = True
        if saw_contact:
            break
    assert saw_contact, "a dropped cube should register a floor contact"
