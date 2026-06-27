"""Tests for segmentation and aggregate mass properties."""

import numpy as np
import pytest
import trimesh

from battlebot_sim.mesh.segment import BotModel, load_bot, segment_mesh


def _two_named_boxes() -> trimesh.Scene:
    a = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    a.apply_translation((-0.2, 0, 0))
    b = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    b.apply_translation((0.2, 0, 0))
    return trimesh.Scene({"chassis": a, "drum": b})


def test_load_glb_preserves_named_parts(tmp_path):
    """A multi-body GLB keeps each body's CAD name instead of part_0/part_1, and
    its face ids still partition the combined original mesh."""
    out = tmp_path / "bot.glb"
    _two_named_boxes().export(out)
    bot = load_bot(str(out), scale_to_m=1.0)
    assert len(bot.parts) == 2
    assert {p.name for p in bot.parts} == {"chassis", "drum"}
    all_faces = np.concatenate([p.face_ids for p in bot.parts])
    assert sorted(all_faces.tolist()) == list(range(len(bot.original.faces)))


def test_load_3mf_preserves_named_parts(tmp_path):
    pytest.importorskip("lxml")  # trimesh's 3MF reader needs lxml
    out = tmp_path / "bot.3mf"
    _two_named_boxes().export(out)
    bot = load_bot(str(out), scale_to_m=1.0)
    assert {p.name for p in bot.parts} == {"chassis", "drum"}


def test_scene_scale_to_m_applied(tmp_path):
    """scale_to_m converts a millimetre-authored multi-body export into metres."""
    a = trimesh.creation.box(extents=(100, 100, 100))
    a.apply_translation((-200, 0, 0))
    b = trimesh.creation.box(extents=(100, 100, 100))
    b.apply_translation((200, 0, 0))
    out = tmp_path / "bot_mm.glb"
    trimesh.Scene({"a": a, "b": b}).export(out)
    bot = load_bot(str(out), scale_to_m=1e-3)
    for p in bot.parts:                       # 100 mm -> 0.1 m edges
        extents = p.mesh.bounds[1] - p.mesh.bounds[0]
        assert np.allclose(extents, [0.1, 0.1, 0.1], atol=1e-6)


def test_single_body_scene_falls_back_to_components(tmp_path):
    """A scene with one geometry (no per-part names) splits by connected
    components, matching STL behaviour."""
    a = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    a.apply_translation((-0.2, 0, 0))
    b = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    b.apply_translation((0.2, 0, 0))
    single = trimesh.util.concatenate([a, b])
    out = tmp_path / "single.glb"
    trimesh.Scene({"body": single}).export(out)
    bot = load_bot(str(out), scale_to_m=1.0)
    assert len(bot.parts) == 2  # split by components, not by name


def test_segment_splits_disjoint_solids(two_box_bot):
    a, b, bar = two_box_bot
    combined = trimesh.util.concatenate([a, b, bar])
    parts = segment_mesh(combined)
    assert len(parts) == 3
    # Face ids partition the original faces exactly.
    all_faces = np.concatenate([p.face_ids for p in parts])
    assert sorted(all_faces.tolist()) == list(range(len(combined.faces)))


def test_single_solid_is_one_part(alu_cube_10cm):
    parts = segment_mesh(alu_cube_10cm)
    assert len(parts) == 1


def test_cube_mass_matches_density(alu_cube_10cm, aluminum):
    model = BotModel(alu_cube_10cm, segment_mesh(alu_cube_10cm))
    model.assign_material_to_all(aluminum)
    # 0.1 m cube = 1e-3 m^3; * 2700 kg/m^3 = 2.70 kg.
    assert np.isclose(model.total_mass(), 2.70, rtol=1e-3)


def test_cube_inertia_matches_analytic(alu_cube_10cm, aluminum):
    model = BotModel(alu_cube_10cm, segment_mesh(alu_cube_10cm))
    model.assign_material_to_all(aluminum)
    inertia = model.inertia_tensor()
    # Solid cube: I = (1/6) m a^2 about each principal axis.
    m, a = 2.70, 0.1
    expected = (1.0 / 6.0) * m * a**2  # = 0.0045
    diag = np.diag(inertia)
    assert np.allclose(diag, expected, rtol=1e-3)
    # Off-diagonal terms should be ~0 for an axis-aligned cube at origin.
    off = inertia - np.diag(diag)
    assert np.allclose(off, 0.0, atol=1e-6)


def test_center_of_mass_of_symmetric_pair(aluminum):
    # Two equal cubes symmetric about origin -> COM at origin.
    a = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    a.apply_translation((-0.2, 0, 0))
    b = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    b.apply_translation((0.2, 0, 0))
    combined = trimesh.util.concatenate([a, b])
    model = BotModel(combined, segment_mesh(combined))
    model.assign_material_to_all(aluminum)
    assert np.allclose(model.center_of_mass(), [0, 0, 0], atol=1e-6)


def test_degenerate_stray_triangle_does_not_break_mass(aluminum):
    """A stray single triangle (a common CAD-export tessellation artifact)
    segments into a non-watertight part whose convex hull is undefined (only 3
    points). Mass must degrade to ~0 for that part rather than raising
    QhullError, which otherwise silently aborts the whole load."""
    box = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    tri = trimesh.Trimesh(
        vertices=np.array([[1.0, 0, 0], [1.1, 0, 0], [1.0, 0.1, 0]]),
        faces=np.array([[0, 1, 2]]),
        process=False,
    )
    combined = trimesh.util.concatenate([box, tri])
    model = BotModel(combined, segment_mesh(combined))
    model.assign_material_to_all(aluminum)
    mass = model.total_mass()
    assert np.isfinite(mass)
    # Only the box has volume; the stray triangle contributes ~0.
    assert np.isclose(mass, 2.70, rtol=1e-3)


def test_zero_volume_part_keeps_aggregates_finite(aluminum):
    """A zero-volume (flat) part must not poison the bot COM/inertia with NaN
    (0 mass * NaN centroid = NaN, which would corrupt the whole bot)."""
    box = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    quad = trimesh.Trimesh(
        vertices=np.array([[1, 0, 0], [1.1, 0, 0], [1.1, 0.1, 0], [1, 0.1, 0]],
                          dtype=float),
        faces=np.array([[0, 1, 2], [0, 2, 3]]),
        process=False,
    )
    combined = trimesh.util.concatenate([box, quad])
    model = BotModel(combined, segment_mesh(combined))
    model.assign_material_to_all(aluminum)
    assert np.all(np.isfinite(model.center_of_mass()))
    assert np.all(np.isfinite(model.inertia_tensor()))


def test_sample_bot_path_exists_and_loads():
    """The bundled sample resolves and segments — the UI's 'Load sample bot'
    relies on this for an instant known-good run."""
    import os

    from battlebot_sim.mesh.segment import load_bot, sample_bot_path

    assert os.path.exists(sample_bot_path())
    bot = load_bot(sample_bot_path(), scale_to_m=1.0)
    assert len(bot.parts) >= 1


def test_merge_reduces_part_count(two_box_bot, aluminum):
    a, b, bar = two_box_bot
    combined = trimesh.util.concatenate([a, b, bar])
    model = BotModel(combined, segment_mesh(combined))
    model.assign_material_to_all(aluminum)
    before = len(model.parts)
    mass_before = model.total_mass()
    model.merge([0, 1])
    assert len(model.parts) == before - 1
    # Mass is conserved across a merge.
    assert np.isclose(model.total_mass(), mass_before, rtol=1e-6)
