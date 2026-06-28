"""Write a self-contained damage report: two heatmap PNGs + a markdown summary."""

from __future__ import annotations

import os
from datetime import datetime

from battlebot_sim import viz
from battlebot_sim.damage.model import DamageResult
from battlebot_sim.materials.assign import WeightClass, validate_weight_class
from battlebot_sim.mesh.segment import BotModel
from battlebot_sim.sim.recorder import SimTrace


def _worst_contacts(trace: SimTrace, n: int = 8):
    return sorted(trace.contacts, key=lambda c: abs(c.normal_force), reverse=True)[:n]


def export_report(
    bot: BotModel,
    result: DamageResult,
    weight_class: WeightClass,
    out_dir: str,
    trace: SimTrace | None = None,
) -> dict[str, str]:
    """Render heatmaps and write report.md into out_dir. Returns written paths."""
    os.makedirs(out_dir, exist_ok=True)
    energy_png = viz.render_heatmap_png(bot, result, "energy", os.path.join(out_dir, "heatmap_energy.png"))
    failure_png = viz.render_heatmap_png(bot, result, "failure", os.path.join(out_dir, "heatmap_failure.png"))

    total_mass = bot.total_mass()
    com = bot.center_of_mass()
    check = validate_weight_class(total_mass, weight_class)

    lines: list[str] = []
    lines.append(f"# BattleBot Damage Report — {weight_class.name}")
    lines.append("")
    lines.append(f"_Generated {datetime.now():%Y-%m-%d %H:%M}_")
    lines.append("")
    lines.append("## Bot summary")
    lines.append("")
    lines.append(f"- **Total mass:** {total_mass:.3f} kg ({total_mass / 0.45359237:.2f} lb)")
    lines.append(f"- **Centre of mass:** ({com[0]:.3f}, {com[1]:.3f}, {com[2]:.3f}) m")
    status = "✅ within limit" if check.ok else "❌ OVER WEIGHT"
    lines.append(f"- **Weight class:** {status} — {check.message}")
    lines.append(f"- **Parts:** {len(bot.parts)}")
    lines.append("")

    lines.append("## Per-part damage")
    lines.append("")
    lines.append("| # | Part | Material | Brace | Mass (kg) | Max failure margin | Verdict | Impact energy (J) |")
    lines.append("|---|------|----------|-------|-----------|--------------------|---------|-------------------|")
    for p in bot.parts:
        margin = result.part_max_margin.get(p.index, 0.0)
        energy = result.part_total_energy.get(p.index, 0.0)
        verdict = "FAIL ⚠️" if margin >= 1.0 else "ok"
        mat = p.material.name if p.material else "—"
        lines.append(
            f"| {p.index} | {p.name} | {mat} | {'yes' if p.is_brace else ''} | "
            f"{p.mass_kg:.3f} | {margin:.2f} | {verdict} | {energy:.2f} |"
        )
    lines.append("")

    failing = result.parts_that_fail()
    lines.append("## Verdict")
    lines.append("")
    if failing:
        names = ", ".join(bot.parts[i].name for i in failing)
        lines.append(f"⚠️ **{len(failing)} part(s) predicted to yield:** {names}.")
        lines.append("Consider a stronger material, thicker section, or added bracing there.")
    else:
        lines.append("✅ No part exceeded its material yield in this battery.")
    lines.append("")

    if trace is not None:
        lines.append("## Worst impacts")
        lines.append("")
        lines.append("| Event | Part | Normal force (N) | Impact angle (deg) | Hit |")
        lines.append("|-------|------|------------------|--------------------|-----|")
        for c in _worst_contacts(trace):
            lines.append(
                f"| {c.event} | {bot.parts[c.part_index].name} | "
                f"{abs(c.normal_force):.0f} | {c.impact_angle_deg:.0f} | {c.other} |"
            )
        lines.append("")

    lines.append("## Heatmaps")
    lines.append("")
    lines.append("![Failure margin](heatmap_failure.png)")
    lines.append("")
    lines.append("![Impact energy](heatmap_energy.png)")
    lines.append("")
    lines.append("---")
    lines.append("_Hybrid model: simplified Hertzian contact stress + brace load-sharing. "
                 "Results are comparative, not certification-grade._")

    md_path = os.path.join(out_dir, "report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return {"report": md_path, "energy_png": energy_png, "failure_png": failure_png}
