"""Plain-text damage verdict formatting, shared by the live results panel, the
Freeplay panel, and the markdown report so the wording stays in one place."""

from __future__ import annotations

from gauntlet.damage.model import DamageResult
from gauntlet.damage.structural import PartStress
from gauntlet.mesh.segment import BotModel


def verdict_label(part_stress: PartStress | None, margin: float) -> str:
    """One part's verdict: 'FRACTURE', 'FAIL', or 'ok'. Fracture takes precedence,
    then yield (margin >= 1)."""
    if part_stress is not None and part_stress.fractures:
        return "FRACTURE"
    if margin >= 1.0:
        return "FAIL"
    return "ok"


def summarize_result(bot: BotModel, result: DamageResult) -> list[str]:
    """The per-part damage summary lines: a headline (fracture/yield/clean) followed
    by one line per part with its margin, governing mode, and section thickness."""
    ps_by_idx = {ps.part_index: ps for ps in result.part_stress}
    failing = result.parts_that_fail()
    fracturing = [ps.part_index for ps in result.part_stress if ps.fractures]

    lines: list[str] = []
    if fracturing:
        names = ", ".join(bot.parts[i].name for i in fracturing)
        lines.append(f"{len(fracturing)} part(s) predicted to FRACTURE: {names}")
    if failing:
        names = ", ".join(bot.parts[i].name for i in failing)
        lines.append(f"{len(failing)} part(s) predicted to yield: {names}")
    if not failing and not fracturing:
        lines.append("No part exceeded its material yield.")
    lines.append("")

    for p in bot.parts:
        ps = ps_by_idx.get(p.index)
        m = result.part_max_margin.get(p.index, 0.0)
        if ps is not None:
            label = verdict_label(ps, m)
            tag = "" if label == "ok" else f"  {label}"
            lines.append(
                f"{p.name}: margin {m:.2f} "
                f"[{ps.governing_mode}, t={ps.thickness_used * 1e3:.1f} mm]{tag}")
        else:
            lines.append(f"{p.name}: max margin {m:.2f}"
                         + ("  FAIL" if m >= 1.0 else ""))
    return lines
