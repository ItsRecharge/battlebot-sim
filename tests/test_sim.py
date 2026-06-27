"""Integration tests for the MuJoCo engine, MJCF build, and stress battery."""

import numpy as np
import trimesh

from battlebot_sim.arena.nhrl import build_arena
from battlebot_sim.materials.assign import NHRL_CLASSES
from battlebot_sim.mesh.segment import BotModel, segment_mesh
from battlebot_sim.sim.engine import SimEngine
from battlebot_sim.sim.battery import (
    StressBattery, run_battery, class_strike_energy,
)


def _make_bot(aluminum, offset=(0.0, 0.0, 0.0)):
    """A two-cube + brace bot, all aluminium, in metres.

    ``offset`` translates the whole mesh so the geometry centre no longer sits at
    the body origin — exercises the containment clamp's AABB anchoring.
    """
    a = trimesh.creation.box(extents=(0.08, 0.08, 0.05))
    a.apply_translation((-0.10, 0, 0))
    b = trimesh.creation.box(extents=(0.08, 0.08, 0.05))
    b.apply_translation((0.10, 0, 0))
    bar = trimesh.creation.box(extents=(0.12, 0.02, 0.02))
    combined = trimesh.util.concatenate([a, b, bar])
    if any(offset):
        combined.apply_translation(offset)
    model = BotModel(combined, segment_mesh(combined))
    model.assign_material_to_all(aluminum)
    return model


def test_engine_compiles_and_steps(aluminum):
    arena = build_arena(NHRL_CLASSES["3lb"])
    bot = _make_bot(aluminum)
    engine = SimEngine(arena, bot)
    assert engine.model.ngeom >= 6 + len(bot.parts)  # arena + bot parts
    engine.reset()
    engine.set_pose(arena.center_point())
    for _ in range(100):
        engine.step()
    pos, quat = engine.get_pose()
    assert np.all(np.isfinite(pos)) and np.all(np.isfinite(quat))


def test_drop_produces_floor_contacts(aluminum):
    arena = build_arena(NHRL_CLASSES["3lb"])
    bot = _make_bot(aluminum)
    engine = SimEngine(arena, bot)
    engine.reset()
    engine.set_pose(np.array([0.0, 0.0, 0.3]))  # above the floor
    saw_contact = False
    for _ in range(2000):
        engine.step()
        if any(c["other"] == "floor" for c in engine.read_contacts()):
            saw_contact = True
            break
    assert saw_contact, "bot never contacted the floor after a drop"


def test_full_battery_runs(aluminum):
    arena = build_arena(NHRL_CLASSES["12lb"])
    bot = _make_bot(aluminum)
    engine = SimEngine(arena, bot)
    battery = StressBattery(arena, NHRL_CLASSES["12lb"])
    trace = run_battery(engine, battery, fps=30)

    assert len(battery.events) >= 6
    assert len(trace.frames) > 0
    assert trace.total_contacts() > 0
    # Every contact maps to a real part and carries finite force.
    for c in trace.contacts:
        assert 0 <= c.part_index < len(bot.parts)
        assert np.isfinite(c.normal_force)
    # The opponent strike should be represented.
    assert any(c.other == "opponent_weapon" for c in trace.contacts)


def _assert_contained(trace, bot, arena, tol=0.05):
    """Every recorded bot pose keeps the bot's AABB inside the cage interior."""
    L, W, H = arena.interior
    lo = np.array([-L / 2.0, -W / 2.0, 0.0])
    hi = np.array([L / 2.0, W / 2.0, H])
    bmin, bmax = bot.original.bounds[0], bot.original.bounds[1]
    for f in trace.frames:
        assert np.all(f.pos + bmin >= lo - tol), \
            f"bot escaped low at event {f.event}: pos={f.pos}"
        assert np.all(f.pos + bmax <= hi + tol), \
            f"bot escaped high at event {f.event}: pos={f.pos}"


def test_battery_keeps_bot_in_chamber(aluminum):
    # Worst case: heaviest class (fastest launches), extra trials, fixed seed.
    wc = NHRL_CLASSES["30lb"]
    arena = build_arena(wc)
    bot = _make_bot(aluminum)
    engine = SimEngine(arena, bot)
    trace = run_battery(engine, StressBattery(arena, wc, n_trials=2, seed=0))
    _assert_contained(trace, bot, arena)


def test_containment_handles_offset_bot(aluminum):
    # Body origin != geometry centre: the clamp must anchor on pos + bounds.
    wc = NHRL_CLASSES["30lb"]
    arena = build_arena(wc)
    bot = _make_bot(aluminum, offset=(0.2, 0.0, 0.1))
    engine = SimEngine(arena, bot)
    trace = run_battery(engine, StressBattery(arena, wc, n_trials=2, seed=0))
    _assert_contained(trace, bot, arena)


def test_strike_damage_severity_unchanged_by_force_cap(aluminum):
    # Capping the physical launch must NOT change the reported impact severity
    # (normal_force / rel_speed) that the damage model reads.
    wc = NHRL_CLASSES["30lb"]
    arena = build_arena(wc)
    bot = _make_bot(aluminum)
    engine = SimEngine(arena, bot)
    trace = run_battery(engine, StressBattery(arena, wc, n_trials=1, seed=0))

    mass = max(bot.total_mass(), 1e-6)
    dv = np.sqrt(2.0 * class_strike_energy(wc) / mass)
    window = 0.01                          # strike t_end - t_start
    expected_force = mass * dv / window

    strikes = [c for c in trace.contacts if c.other == "opponent_weapon"]
    assert strikes, "no opponent-weapon contact recorded"
    for c in strikes:
        assert np.isclose(c.rel_speed, dv, rtol=1e-6)
        assert np.isclose(c.normal_force, expected_force, rtol=1e-6)


def test_overweight_bot_still_simulates(aluminum):
    # Steel-dense scaling is irrelevant here; just confirm no crash with a big bot.
    arena = build_arena(NHRL_CLASSES["30lb"])
    bot = _make_bot(aluminum)
    engine = SimEngine(arena, bot)
    engine.set_pose(arena.center_point())
    for _ in range(200):
        engine.step()
    assert np.all(np.isfinite(engine.get_pose()[0]))


# ---- streaming generator ---------------------------------------------------

def _drain(gen):
    """Run a generator to exhaustion; return its StopIteration value."""
    try:
        while True:
            next(gen)
    except StopIteration as stop:
        return stop.value


def test_iter_battery_emits_every_contact_exactly_once(aluminum):
    """Each contact in the final trace was streamed in exactly one chunk (same
    object), so a live consumer that ingests chunks never drops or double-counts."""
    from battlebot_sim.sim.battery import iter_battery

    wc = NHRL_CLASSES["3lb"]
    arena = build_arena(wc)
    bot = _make_bot(aluminum)
    engine = SimEngine(arena, bot)
    gen = iter_battery(engine, StressBattery(arena, wc, n_trials=1, seed=0), fps=30)

    streamed, n_chunks = [], 0
    try:
        while True:
            chunk = next(gen)
            n_chunks += 1
            assert 0 <= chunk.event_index < chunk.n_events
            streamed.extend(chunk.new_contacts)
    except StopIteration as stop:
        trace = stop.value

    assert n_chunks > 0
    assert len(trace.frames) > 0
    assert len(streamed) == len(trace.contacts)
    # Identity, not equality: the chunk holds the very objects in the trace.
    assert all(a is b for a, b in zip(streamed, trace.contacts))


def test_iter_battery_trace_matches_run_battery(aluminum):
    """Draining the generator yields a trace identical (deterministically) to the
    run_battery drainer on a fresh engine."""
    from battlebot_sim.sim.battery import iter_battery

    wc = NHRL_CLASSES["3lb"]
    arena = build_arena(wc)

    e1 = SimEngine(arena, _make_bot(aluminum))
    t_run = run_battery(e1, StressBattery(arena, wc, n_trials=1, seed=0), fps=30)

    e2 = SimEngine(arena, _make_bot(aluminum))
    t_iter = _drain(iter_battery(e2, StressBattery(arena, wc, n_trials=1, seed=0), fps=30))

    assert t_iter is not None
    assert len(t_run.frames) == len(t_iter.frames)
    assert len(t_run.contacts) == len(t_iter.contacts)


def test_iter_battery_cancel_returns_partial_trace(aluminum):
    """Closing the generator mid-run finalizes and returns the partial trace."""
    from battlebot_sim.sim.battery import iter_battery

    wc = NHRL_CLASSES["3lb"]
    arena = build_arena(wc)
    bot = _make_bot(aluminum)
    engine = SimEngine(arena, bot)
    gen = iter_battery(engine, StressBattery(arena, wc, n_trials=1, seed=0), fps=30)

    for _ in range(5):
        next(gen)
    try:
        partial = gen.throw(GeneratorExit)
    except StopIteration as stop:
        partial = stop.value
    except GeneratorExit:
        partial = None

    assert partial is not None
    assert len(partial.frames) >= 5
    # A full run has many more frames than the 5 we consumed before cancelling.
    full = _drain(iter_battery(SimEngine(arena, _make_bot(aluminum)),
                               StressBattery(arena, wc, n_trials=1, seed=0), fps=30))
    assert len(partial.frames) < len(full.frames)
