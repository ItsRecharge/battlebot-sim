"""End-to-end test: load the sample STL, run the pipeline, export a report."""

import os

from battlebot_sim.arena.nhrl import build_arena
from battlebot_sim.damage.braces import apply_brace_sharing
from battlebot_sim.damage.model import compute_damage
from battlebot_sim.materials.assign import NHRL_CLASSES
from battlebot_sim.materials.library import load_default_library
from battlebot_sim.mesh.segment import load_bot
from battlebot_sim.report.export import export_report
from battlebot_sim.sim.battery import StressBattery, run_battery
from battlebot_sim.sim.engine import SimEngine

SAMPLE = os.path.join(os.path.dirname(__file__), "..", "data", "sample_bots", "wedge_bot.stl")


def test_full_pipeline_and_report(tmp_path):
    library = load_default_library()
    cls = NHRL_CLASSES["12lb"]

    bot = load_bot(os.path.abspath(SAMPLE), scale_to_m=1.0)
    assert len(bot.parts) == 5
    bot.assign_material_to_all(library.get("Aluminum 6061-T6"))
    # Tag the cross-bar (smallest-ish) as a brace if present.
    bot.set_brace(len(bot.parts) - 1, True)

    arena = build_arena(cls)
    engine = SimEngine(arena, bot)
    trace = run_battery(engine, StressBattery(arena, cls), fps=30)
    result = apply_brace_sharing(compute_damage(trace, bot, arena, library), bot)

    paths = export_report(bot, result, cls, str(tmp_path), trace=trace)
    for key in ("report", "energy_png", "failure_png"):
        assert os.path.exists(paths[key]), f"missing {key}"
        assert os.path.getsize(paths[key]) > 0

    # The heatmaps are real rendered images, not empty stubs.
    assert os.path.getsize(paths["energy_png"]) > 1000
    assert os.path.getsize(paths["failure_png"]) > 1000

    report_text = open(paths["report"], encoding="utf-8").read()
    assert "BattleBot Damage Report" in report_text
    assert "Per-part damage" in report_text
    assert "Worst impacts" in report_text
    # The report must reach an explicit pass/fail verdict.
    assert "## Verdict" in report_text
    assert ("predicted to yield" in report_text) or ("No part exceeded" in report_text)
