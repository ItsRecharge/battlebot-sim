"""Tests for automatic brace detection at import (mesh/brace_detect.py)."""

import trimesh

from battlebot_sim.materials.library import load_default_library
from battlebot_sim.mesh.brace_detect import auto_detect_braces
from battlebot_sim.mesh.segment import BotModel, segment_mesh


def _bot(parts_meshes, material="Aluminum 6061-T6"):
    combined = trimesh.util.concatenate(parts_meshes)
    bot = BotModel(combined, segment_mesh(combined))
    bot.assign_material_to_all(load_default_library().get(material))
    return bot


def _two_cubes_and_bar():
    a = trimesh.creation.box(extents=(0.08, 0.08, 0.05))
    a.apply_translation((-0.10, 0, 0))
    b = trimesh.creation.box(extents=(0.08, 0.08, 0.05))
    b.apply_translation((0.10, 0, 0))
    bar = trimesh.creation.box(extents=(0.12, 0.02, 0.02))   # bridges the cubes
    return _bot([a, b, bar])


def test_auto_detects_bridging_bar():
    bot = _two_cubes_and_bar()
    bar_idx = len(bot.parts) - 1                 # smallest component = the bar
    flagged = auto_detect_braces(bot)
    assert bar_idx in flagged
    assert bot.parts[bar_idx].is_brace
    # The two chunky cubes are not elongated, so they stay non-braces.
    assert not any(bot.parts[i].is_brace for i in range(bar_idx))


def test_progress_callback_is_called():
    bot = _two_cubes_and_bar()
    calls = []
    auto_detect_braces(bot, progress=lambda i, n: calls.append((i, n)))
    assert calls, "progress callback was never invoked"
    assert all(n == len(bot.parts) for _, n in calls)
    assert calls[-1] == (len(bot.parts), len(bot.parts))


def test_manual_flag_is_never_cleared():
    bot = _two_cubes_and_bar()
    bot.set_brace(0, True)                        # user marked a cube as a brace
    auto_detect_braces(bot)
    assert bot.parts[0].is_brace, "auto-detect must not clear a manual flag"


def test_non_bridging_bar_not_flagged():
    # A long stiff bar that touches only one cube does not bridge anything.
    cube = trimesh.creation.box(extents=(0.08, 0.08, 0.05))
    cube.apply_translation((-0.10, 0, 0))
    bar = trimesh.creation.box(extents=(0.12, 0.02, 0.02))
    bar.apply_translation((-0.10, 0, 0))         # sits on the lone cube
    bot = _bot([cube, bar])
    flagged = auto_detect_braces(bot)
    assert flagged == []
    assert not any(p.is_brace for p in bot.parts)


def test_plastic_bar_not_flagged():
    # Same bridging geometry but a floppy plastic fails the stiffness gate.
    a = trimesh.creation.box(extents=(0.08, 0.08, 0.05))
    a.apply_translation((-0.10, 0, 0))
    b = trimesh.creation.box(extents=(0.08, 0.08, 0.05))
    b.apply_translation((0.10, 0, 0))
    bar = trimesh.creation.box(extents=(0.12, 0.02, 0.02))
    bot = _bot([a, b, bar], material="UHMW-PE")
    assert auto_detect_braces(bot) == []
