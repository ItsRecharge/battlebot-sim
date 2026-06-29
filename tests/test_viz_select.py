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


def test_attach_field_smooth_subdivides_to_point_scalars(two_box_bot):
    """A coarse bot mesh must be subdivided into a denser POINT-scalar field so
    the heatmap reads as a smooth gradient — guards against the subdivision
    silently failing back to the coarse per-vertex mesh."""
    from battlebot_sim.damage.model import DamageResult

    bot = _bot(two_box_bot)
    n = len(bot.original.faces)
    energy = np.zeros(n)
    energy[0] = 5.0                                   # one hot face
    result = DamageResult(
        energy_per_face=energy,
        peak_stress_per_face=np.zeros(n),
        failure_margin_per_face=np.zeros(n),
        face_part=viz.face_part_array(bot),
        part_max_margin={}, part_total_energy={})

    base = viz.bot_polydata(bot)
    heat, cmap, clim, title, log_scale = viz.attach_field_smooth(base, bot, result, "energy")
    assert heat.n_faces > base.n_faces                # coarse mesh got subdivided
    assert "energy" in heat.point_data                # smooth point scalars, not cells
    assert len(heat.point_data["energy"]) == heat.n_points
    assert log_scale is True                          # energy stays log-scaled


def _result_with(bot, *, energy=None, margin=None):
    from battlebot_sim.damage.model import DamageResult

    n = len(bot.original.faces)
    return DamageResult(
        energy_per_face=energy if energy is not None else np.zeros(n),
        peak_stress_per_face=np.zeros(n),
        failure_margin_per_face=margin if margin is not None else np.zeros(n),
        face_part=viz.face_part_array(bot),
        part_max_margin={}, part_total_energy={})


def test_attach_field_smooth_untested_at_key_bottom_not_grey(two_box_bot):
    """Untested/undamaged geometry must render at the bottom of the colour key
    (zero / black), not as a greyed-out NaN patch. Covers both the linear
    failure field and the log-scaled energy field."""
    bot = _bot(two_box_bot)
    n = len(bot.original.faces)
    base = viz.bot_polydata(bot)

    # failure (linear): undamaged faces land exactly on clim[0] == 0.
    margin = np.zeros(n)
    margin[0] = 2.0
    heat, _c, clim, _t, log_scale = viz.attach_field_smooth(
        base, bot, _result_with(bot, margin=margin), "failure")
    vals = heat.point_data["failure"]
    assert not np.isnan(vals).any()                   # nothing greyed out
    assert log_scale is False and clim[0] == 0.0
    assert float(vals.min()) == 0.0                   # untested == the key's zero

    # energy (log): no true zero, so undamaged is pinned to vmin == clim[0].
    energy = np.zeros(n)
    energy[0] = 5.0
    heat_e, _ce, clim_e, _te, log_e = viz.attach_field_smooth(
        base, bot, _result_with(bot, energy=energy), "energy")
    vals_e = heat_e.point_data["energy"]
    assert log_e is True
    assert not np.isnan(vals_e).any()                 # nothing greyed out
    assert float(vals_e.min()) == clim_e[0] > 0.0     # untested == bottom of the key
