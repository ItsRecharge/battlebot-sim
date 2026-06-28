"""Unit tests for the MJCF XML builder."""
from __future__ import annotations

import numpy as np
import pytest
import trimesh

from battlebot_sim.arena.nhrl import build_arena
from battlebot_sim.config import DEFAULT_CONFIG
from battlebot_sim.materials.assign import NHRL_CLASSES
from battlebot_sim.mesh.segment import BotModel, Part, segment_mesh
from battlebot_sim.sim.mjcf import _hull_vertices, build_mjcf

pytestmark = pytest.mark.native_isolated


def _cube_bot(library):
    cube = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    bot = BotModel(cube, segment_mesh(cube))
    bot.assign_material_to_all(library.get("Aluminum 6061-T6"))
    return bot


def test_build_mjcf_structure(library):
    bot = _cube_bot(library)
    xml, geom_map = build_mjcf(build_arena(NHRL_CLASSES["3lb"]), bot)
    assert xml.lstrip().startswith("<mujoco")
    assert "</mujoco>" in xml
    assert f'gravity="0 0 {DEFAULT_CONFIG.contact.gravity}"' in xml  # from config
    assert len(geom_map) == len(bot.parts)
    assert xml.count('<mesh name="hull_') == len(bot.parts)  # one hull per part


def test_build_mjcf_timestep_passthrough(library):
    bot = _cube_bot(library)
    xml, _ = build_mjcf(build_arena(NHRL_CLASSES["3lb"]), bot, timestep=1e-3)
    assert 'timestep="0.001"' in xml


def test_hull_vertices_pads_degenerate_part():
    # A single-triangle "part" has only 3 coplanar vertices; MuJoCo needs a 3D
    # hull, so the builder pads it into a tiny tetrahedron.
    deg = trimesh.Trimesh(
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]], process=False
    )
    part = Part(index=0, mesh=deg, face_ids=np.array([0]))
    verts = _hull_vertices(part)
    assert len(verts) >= 4
    assert np.all(np.isfinite(verts))
