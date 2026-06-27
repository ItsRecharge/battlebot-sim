"""Tests for the selection/coloring helpers that back click-to-pick material
assignment. These are pure (no OpenGL), so they run headless like the rest."""

import numpy as np
import trimesh

from battlebot_sim import viz
from battlebot_sim.mesh.segment import BotModel, segment_mesh


def _bot(two_box_bot):
    a, b, bar = two_box_bot
    combined = trimesh.util.concatenate([a, b, bar])
    return BotModel(combined, segment_mesh(combined))


def test_face_part_array_partitions_faces(two_box_bot):
    bot = _bot(two_box_bot)
    fp = viz.face_part_array(bot)
    assert fp.shape == (len(bot.original.faces),)
    for p in bot.parts:
        assert np.all(fp[p.face_ids] == p.index)
    assert set(np.unique(fp).tolist()) == {p.index for p in bot.parts}


def test_material_color_deterministic_and_in_range():
    c1 = viz.material_color("Aluminum 6061-T6")
    c2 = viz.material_color("Aluminum 6061-T6")
    assert c1 == c2
    assert all(0.0 <= ch <= 1.0 for ch in c1)
    assert viz.material_color("Steel 4140") != viz.material_color("Titanium Grade 5")


def test_face_material_colors_assigned_vs_unassigned(two_box_bot, aluminum):
    bot = _bot(two_box_bot)
    colors = viz.face_material_colors(bot)
    assert colors.shape == (len(bot.original.faces), 3)
    assert colors.dtype == np.uint8
    # All unassigned -> one uniform neutral colour.
    assert len(np.unique(colors, axis=0)) == 1

    bot.assign_material(0, aluminum)
    colors2 = viz.face_material_colors(bot)
    part0 = colors2[bot.parts[0].face_ids]
    rest = colors2[bot.parts[1].face_ids]
    assert len(np.unique(part0, axis=0)) == 1            # uniform within a part
    assert not np.array_equal(part0[0], rest[0])         # differs from unassigned


def test_part_at_cell_lookup(two_box_bot):
    bot = _bot(two_box_bot)
    fp = viz.face_part_array(bot)
    assert viz.part_at_cell(fp, int(bot.parts[1].face_ids[0])) == 1
    assert viz.part_at_cell(fp, -1) is None
    assert viz.part_at_cell(fp, 10**9) is None
