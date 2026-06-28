"""Shared test helpers for the headless damage pipeline.

The ``--selftest`` path (see ``battlebot_sim.__main__``) exercises the whole
offline pipeline: load the bundled sample bot, run one seeded stress battery, and
compute the damage fields. Several test modules (the golden-baseline regression
oracle, and the fidelity-validation suite) need that exact result, so the runner
lives here once. No VTK / Pillow is imported, keeping this off the native
DLL-load-crash path so it can run in the shared test process.
"""
from __future__ import annotations

import numpy as np

from battlebot_sim.arena.nhrl import build_arena
from battlebot_sim.damage.model import DamageResult, compute_damage
from battlebot_sim.materials.assign import NHRL_CLASSES
from battlebot_sim.materials.library import load_default_library
from battlebot_sim.mesh.segment import load_bot, sample_bot_path
from battlebot_sim.sim.battery import StressBattery, run_battery
from battlebot_sim.sim.recorder import SimTrace


def run_selftest_pipeline(seed: int = 0, n_trials: int = 1, fps: int = 30):
    """Run the seeded offline pipeline and return ``(trace, result)``.

    Mirrors ``battlebot_sim.__main__._run_selftest`` (minus the VTK render) so the
    numbers a test pins here are the numbers the shipped self-test produces.
    """
    # Import SimEngine lazily so merely importing this module doesn't pull in
    # MuJoCo (lets tests use summarize_result without the native engine).
    from battlebot_sim.sim.engine import SimEngine

    library = load_default_library()
    bot = load_bot(str(sample_bot_path()), scale_to_m=1.0)
    bot.assign_material_to_all(library.get("Aluminum 6061-T6"))
    cls = NHRL_CLASSES["3lb"]
    arena = build_arena(cls)
    engine = SimEngine(arena, bot)
    battery = StressBattery(arena, cls, n_trials=n_trials, seed=seed)
    trace = run_battery(engine, battery, fps=fps)
    result = compute_damage(trace, bot, arena, library)
    return trace, result


def summarize_result(trace: SimTrace, result: DamageResult) -> dict:
    """Reduce a damage result to JSON-friendly scalar summaries for snapshotting."""
    return {
        "n_parts": len(result.part_max_margin),
        "total_contacts": int(trace.total_contacts()),
        "energy_sum": float(np.nansum(result.energy_per_face)),
        "stress_sum": float(np.nansum(result.peak_stress_per_face)),
        "margin_sum": float(np.nansum(result.failure_margin_per_face)),
        "part_max_margin": {
            int(k): float(v) for k, v in sorted(result.part_max_margin.items())
        },
        "part_total_energy": {
            int(k): float(v) for k, v in sorted(result.part_total_energy.items())
        },
    }
