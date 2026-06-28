"""Physics validation: impact energy is conserved when spread onto the mesh.

Each contact's work (normal_force * closing_speed * dt) is painted onto nearby
faces with a Gaussian whose weights sum to 1, so the total energy on the mesh must
equal the work delivered — energy can spread, not appear or vanish. This is the
bug class the earlier code review flagged in the brace path.
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
from battlebot_sim.sim.recorder import ContactEvent, SimTrace

pytestmark = pytest.mark.validation


def _cube_setup():
    library = load_default_library()
    cube = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    bot = BotModel(cube, segment_mesh(cube))
    bot.assign_material_to_all(library.get("Aluminum 6061-T6"))
    return bot, build_arena(NHRL_CLASSES["3lb"]), library


def _contact(local_pos, force, speed):
    return ContactEvent(
        time=0.0, event="strike",
        pos=np.asarray(local_pos, float), local_pos=np.asarray(local_pos, float),
        normal=np.array([1.0, 0.0, 0.0]),
        normal_force=force, tangential_force=0.0, rel_speed=speed,
        part_index=0, other="opponent_weapon",
    )


def test_single_contact_energy_is_conserved():
    bot, arena, library = _cube_setup()
    force, speed, dt = 2000.0, 5.0, 5e-4
    trace = SimTrace(dt=dt, n_parts=1, contacts=[_contact([0.05, 0.0, 0.0], force, speed)])

    result = compute_damage(trace, bot, arena, library)
    work = force * speed * dt          # = 5.0 J
    assert np.isclose(result.energy_per_face.sum(), work, rtol=1e-9)
    assert np.isclose(sum(result.part_total_energy.values()), work, rtol=1e-9)


def test_energy_sums_across_multiple_contacts():
    bot, arena, library = _cube_setup()
    dt = 5e-4
    contacts = [
        _contact([0.05, 0.0, 0.0], 1000.0, 4.0),
        _contact([0.05, 0.02, 0.0], 1500.0, 3.0),
    ]
    trace = SimTrace(dt=dt, n_parts=1, contacts=contacts)

    result = compute_damage(trace, bot, arena, library)
    work = sum(c.normal_force * c.rel_speed * dt for c in contacts)
    assert np.isclose(result.energy_per_face.sum(), work, rtol=1e-9)
