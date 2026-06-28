"""Tests for the damage model, brace load-sharing, and field helpers."""

import numpy as np
import trimesh

from battlebot_sim.arena.nhrl import build_arena
from battlebot_sim.damage.braces import apply_brace_sharing
from battlebot_sim.damage.fields import normalize, vertex_scalars
from battlebot_sim.damage.model import (
    _effective_modulus,
    _hertzian_peak_pressure,
    compute_damage,
)
from battlebot_sim.materials.assign import NHRL_CLASSES
from battlebot_sim.materials.library import load_default_library
from battlebot_sim.mesh.segment import BotModel, segment_mesh
from battlebot_sim.sim.battery import StressBattery, run_battery
from battlebot_sim.sim.engine import SimEngine


def _braced_bot(library, brace_thickness=0.02):
    """Two cubes bridged by a brace bar of the given thickness."""
    a = trimesh.creation.box(extents=(0.08, 0.08, 0.05))
    a.apply_translation((-0.10, 0, 0))
    b = trimesh.creation.box(extents=(0.08, 0.08, 0.05))
    b.apply_translation((0.10, 0, 0))
    bar = trimesh.creation.box(extents=(0.12, brace_thickness, brace_thickness))
    combined = trimesh.util.concatenate([a, b, bar])
    model = BotModel(combined, segment_mesh(combined))
    alu = library.get("Aluminum 6061-T6")
    model.assign_material_to_all(alu)
    # The smallest connected component (the bar) is the brace -> last part.
    model.set_brace(len(model.parts) - 1, True)
    return model


# ---- analytic pieces -------------------------------------------------------

def test_hertzian_pressure_increases_with_force():
    E = _effective_modulus(68.9e9, 200e9)
    p_low = _hertzian_peak_pressure(500, E, 0.02)
    p_high = _hertzian_peak_pressure(5000, E, 0.02)
    assert p_high > p_low > 0


def test_effective_modulus_between_components():
    E = _effective_modulus(70e9, 200e9)
    assert 0 < E < 70e9  # series compliance -> softer than the stiffer member


def test_normalize_ranges():
    out = normalize(np.array([0.0, 5.0, 10.0]))
    assert np.isclose(out.min(), 0.0) and np.isclose(out.max(), 1.0)
    flat = normalize(np.array([3.0, 3.0, 3.0]))
    assert np.allclose(flat, 0.0)


def test_vertex_scalars_average():
    faces = np.array([[0, 1, 2]])
    out = vertex_scalars(faces, 3, np.array([6.0]))
    assert np.allclose(out, [6.0, 6.0, 6.0])


# ---- end-to-end damage -----------------------------------------------------

def test_compute_damage_populates_fields():
    library = load_default_library()
    cls = NHRL_CLASSES["3lb"]
    arena = build_arena(cls)
    bot = _braced_bot(library)
    engine = SimEngine(arena, bot)
    trace = run_battery(engine, StressBattery(arena, cls), fps=30)

    result = compute_damage(trace, bot, arena, library)
    n_faces = len(bot.original.faces)
    assert result.energy_per_face.shape == (n_faces,)
    assert result.failure_margin_per_face.shape == (n_faces,)
    assert np.all(np.isfinite(result.energy_per_face))
    assert np.all(np.isfinite(result.failure_margin_per_face))
    assert result.energy_per_face.sum() > 0          # something absorbed energy
    assert result.failure_margin_per_face.max() > 0  # something was stressed


def test_brace_relieves_stress_monotonically():
    library = load_default_library()

    # Identical synthetic stress field on the braced parts; vary brace thickness.
    def braced_margin(thickness):
        bot = _braced_bot(library, brace_thickness=thickness)
        from battlebot_sim.damage.model import DamageResult
        n_faces = len(bot.original.faces)
        face_part = np.zeros(n_faces, dtype=np.int64)
        for p in bot.parts:
            face_part[p.face_ids] = p.index
        # Uniform 200 MPa peak stress everywhere; aluminium yield 276 MPa.
        peak = np.full(n_faces, 200e6)
        margin = peak / 276e6
        result = DamageResult(
            energy_per_face=np.zeros(n_faces),
            peak_stress_per_face=peak.copy(),
            failure_margin_per_face=margin.copy(),
            face_part=face_part,
            part_max_margin={p.index: float(margin[p.face_ids].max()) for p in bot.parts},
            part_total_energy={p.index: 0.0 for p in bot.parts},
        )
        shared = apply_brace_sharing(result, bot)
        # Part 0 is one of the cubes adjacent to the brace.
        return shared.part_max_margin[0]

    thin = braced_margin(0.01)
    thick = braced_margin(0.05)
    assert thick < thin, "thicker brace should relieve more stress"
    assert thin < 200e6 / 276e6  # any brace relieves some stress


def test_brace_absorbs_transferred_load():
    """Load shed by adjacent parts must surface in the brace's own margin."""
    library = load_default_library()
    bot = _braced_bot(library, brace_thickness=0.02)
    from battlebot_sim.damage.model import DamageResult

    n_faces = len(bot.original.faces)
    face_part = np.zeros(n_faces, dtype=np.int64)
    for p in bot.parts:
        face_part[p.face_ids] = p.index
    brace_idx = len(bot.parts) - 1  # the bar, as set by _braced_bot

    # Stress only the cubes; the brace starts completely unstressed.
    peak = np.zeros(n_faces)
    for p in bot.parts:
        if not p.is_brace:
            peak[p.face_ids] = 200e6
    yield_pa = library.get("Aluminum 6061-T6").yield_pa
    margin = peak / yield_pa
    result = DamageResult(
        energy_per_face=np.zeros(n_faces),
        peak_stress_per_face=peak.copy(),
        failure_margin_per_face=margin.copy(),
        face_part=face_part,
        part_max_margin={p.index: float(margin[p.face_ids].max()) for p in bot.parts},
        part_total_energy={p.index: 0.0 for p in bot.parts},
    )
    assert result.part_max_margin[brace_idx] == 0.0  # unstressed before sharing

    shared = apply_brace_sharing(result, bot)
    # The brace now carries transferred load, so its reported margin rises.
    assert shared.part_max_margin[brace_idx] > 0.0


def test_no_braces_is_noop():
    library = load_default_library()
    bot = _braced_bot(library)
    for p in bot.parts:
        p.is_brace = False
    from battlebot_sim.damage.model import DamageResult
    n_faces = len(bot.original.faces)
    peak = np.full(n_faces, 100e6)
    result = DamageResult(
        energy_per_face=np.zeros(n_faces),
        peak_stress_per_face=peak.copy(),
        failure_margin_per_face=(peak / 276e6).copy(),
        face_part=np.zeros(n_faces, dtype=np.int64),
        part_max_margin={}, part_total_energy={},
    )
    out = apply_brace_sharing(result, bot)
    assert np.allclose(out.peak_stress_per_face, peak)


def test_accumulator_batched_matches_oneshot():
    """Ingesting contacts in arbitrary consecutive batches must equal a single
    one-shot ingest — the guarantee that the live stream and the offline path
    produce identical damage fields."""
    from battlebot_sim.damage.model import DamageAccumulator

    library = load_default_library()
    cls = NHRL_CLASSES["3lb"]
    arena = build_arena(cls)
    bot = _braced_bot(library)
    engine = SimEngine(arena, bot)
    trace = run_battery(engine, StressBattery(arena, cls), fps=30)

    one = compute_damage(trace, bot, arena, library)

    acc = DamageAccumulator(bot, arena, library)
    contacts = trace.contacts
    for i in range(0, len(contacts), 7):           # odd batch size on purpose
        acc.ingest(contacts[i:i + 7], trace.dt)
    batched = acc.finalize()

    assert np.allclose(one.energy_per_face, batched.energy_per_face)
    assert np.allclose(one.peak_stress_per_face, batched.peak_stress_per_face)
    assert np.allclose(one.failure_margin_per_face, batched.failure_margin_per_face)
    assert one.part_max_margin == batched.part_max_margin
