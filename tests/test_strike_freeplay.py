"""Unit tests for the shared strike helper, the freeplay pipeline, the verdict
formatter, and the pure pixel->plane projection. None of these need a GL context."""

import os

import numpy as np

SAMPLE = os.path.join(
    os.path.dirname(__file__), "..", "data", "sample_bots", "wedge_bot.stl"
)


def test_ray_plane_intersect_hits_and_parallel():
    from gauntlet.ui.freeplay_view import ray_plane_intersect

    # A ray along -z from z=2 to z=0 crosses the plane z=1 at z=1.
    hit = ray_plane_intersect([0, 0, 2], [0, 0, 0], [5, 5, 1], [0, 0, 1])
    assert np.allclose(hit, [0, 0, 1])

    # A ray travelling within a z-parallel direction never crosses a z=const plane.
    assert ray_plane_intersect([0, 0, 1], [1, 0, 1], [0, 0, 5], [0, 0, 1]) is None


def test_strike_contact_converts_energy_to_force():
    from gauntlet.sim.strike import strike_contact

    mass, energy, window = 1.0, 8.0, 5e-4
    ce, dv = strike_contact(
        world_point=[1, 2, 3], local_point=[0, 0, 0], direction=[0, 0, -1],
        energy_j=energy, mass=mass, window=window, part_index=2, time=0.0, event="t")
    assert np.isclose(dv, np.sqrt(2 * energy / mass))   # 4.0 m/s
    assert np.isclose(ce.normal_force, mass * dv / window)
    assert np.allclose(ce.normal, [0, 0, 1])            # normal = -direction (into bot)
    assert ce.part_index == 2
    assert ce.other == "opponent_weapon"


def test_freeplay_strike_local_point_and_pipeline():
    from gauntlet.arena.nhrl import build_arena
    from gauntlet.config import DEFAULT_CONFIG
    from gauntlet.damage.braces import apply_brace_sharing
    from gauntlet.damage.model import DamageAccumulator
    from gauntlet.materials.assign import NHRL_CLASSES
    from gauntlet.materials.library import load_default_library
    from gauntlet.mesh.segment import load_bot
    from gauntlet.sim.strike import freeplay_strike, nearest_part

    lib = load_default_library()
    wc = NHRL_CLASSES["3lb"]
    bot = load_bot(os.path.abspath(SAMPLE), scale_to_m=1.0)
    bot.assign_material_to_all(lib.get("Aluminum 6061-T6"))
    arena = build_arena(wc)
    dt = DEFAULT_CONFIG.sim.timestep

    # Strike the top of the bot straight down, with a non-zero rest lift so the
    # world->local mapping is exercised.
    rest_t = np.array([0.0, 0.0, 0.25])
    top = np.array([bot.original.centroid[0], bot.original.centroid[1],
                    bot.original.bounds[1][2]]) + rest_t
    direction = np.array([0.0, 0.0, -1.0])
    pidx = nearest_part(bot, top - rest_t, np.zeros(3), [1, 0, 0, 0])

    ce = freeplay_strike(bot, rest_t, top, pidx, direction, 500.0, dt)
    # local_point = world_point - rest_translation (identity rest rotation).
    assert np.allclose(ce.local_pos, top - rest_t)
    assert np.allclose(ce.normal, [0, 0, 1])

    accum = DamageAccumulator(bot, arena, lib)
    accum.ingest([ce], dt)
    result = apply_brace_sharing(accum.finalize(), bot)
    # A 500 J strike on aluminium leaves a measurable margin somewhere.
    assert max(result.part_max_margin.values()) > 0.0


def test_verdict_label_and_summary_are_plain_text():
    from gauntlet.report.verdict import summarize_result, verdict_label

    assert verdict_label(None, 0.5) == "ok"
    assert verdict_label(None, 1.5) == "FAIL"

    class _PS:
        fractures = True

    assert verdict_label(_PS(), 0.1) == "FRACTURE"

    from gauntlet.materials.library import load_default_library
    from gauntlet.mesh.segment import load_bot

    lib = load_default_library()
    bot = load_bot(os.path.abspath(SAMPLE), scale_to_m=1.0)
    bot.assign_material_to_all(lib.get("Aluminum 6061-T6"))

    from gauntlet.arena.nhrl import build_arena
    from gauntlet.config import DEFAULT_CONFIG
    from gauntlet.damage.braces import apply_brace_sharing
    from gauntlet.damage.model import DamageAccumulator
    from gauntlet.materials.assign import NHRL_CLASSES
    from gauntlet.sim.strike import freeplay_strike, nearest_part

    wc = NHRL_CLASSES["3lb"]
    arena = build_arena(wc)
    dt = DEFAULT_CONFIG.sim.timestep
    top = np.array([bot.original.centroid[0], bot.original.centroid[1],
                    bot.original.bounds[1][2]])
    pidx = nearest_part(bot, top, np.zeros(3), [1, 0, 0, 0])
    ce = freeplay_strike(bot, np.zeros(3), top, pidx, np.array([0.0, 0.0, -1.0]),
                         200.0, dt)
    accum = DamageAccumulator(bot, arena, lib)
    accum.ingest([ce], dt)
    result = apply_brace_sharing(accum.finalize(), bot)

    lines = summarize_result(bot, result)
    assert lines and all(line.isascii() for line in lines)   # no emoji


def test_battery_and_freeplay_strike_agree_on_force():
    """A battery strike and a freeplay strike of the same energy/point/mass must
    report the same normal force (they share strike_contact)."""
    from gauntlet.sim.strike import strike_contact

    mass, energy, window = 1.3, 200.0, 5e-4
    a, _ = strike_contact(world_point=[0, 0, 0], local_point=[0, 0, 0],
                          direction=[0, 0, -1], energy_j=energy, mass=mass,
                          window=window, part_index=0, time=0.0, event="opponent")
    b, _ = strike_contact(world_point=[0, 0, 0], local_point=[0, 0, 0],
                          direction=[0, 0, -1], energy_j=energy, mass=mass,
                          window=window, part_index=0, time=0.0, event="freeplay")
    assert np.isclose(a.normal_force, b.normal_force)
