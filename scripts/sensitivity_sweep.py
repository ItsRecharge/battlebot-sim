"""Sensitivity sweep: rank how much each tuning constant moves the failure verdict.

Runs the seeded stress battery once, then re-derives the damage map while varying
each ``DamageConfig`` / ``BraceConfig`` constant by +/- a percentage. Reports the
relative swing in the worst failure margin, ranked — telling you which uncalibrated
heuristic the verdict is most sensitive to (i.e. which to calibrate first).

This is an analysis tool, not a gating test. Run:

    python scripts/sensitivity_sweep.py [--pct 0.25] [--out docs]

Because the damage/brace constants don't affect the physics, the battery runs only
once; each perturbation just re-runs the cheap damage computation.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import trimesh

from battlebot_sim.arena.nhrl import build_arena
from battlebot_sim.config import DEFAULT_CONFIG
from battlebot_sim.damage.braces import apply_brace_sharing
from battlebot_sim.damage.model import compute_damage
from battlebot_sim.materials.assign import NHRL_CLASSES
from battlebot_sim.materials.library import load_default_library
from battlebot_sim.mesh.segment import BotModel, segment_mesh
from battlebot_sim.sim.battery import StressBattery, run_battery
from battlebot_sim.sim.engine import SimEngine

DAMAGE_FIELDS = [
    "sigma_patch_factor", "sigma_min_frac", "sigma_part_frac",
    "stress_sigma_frac", "kernel_radius_sigmas", "opponent_modulus_pa", "poisson",
]
BRACE_FIELDS = ["k_ref", "transfer", "adjacency_tol"]


def _braced_bot(library):
    """Two cubes bridged by a bar; the bar is tagged as a brace."""
    a = trimesh.creation.box(extents=(0.08, 0.08, 0.05))
    a.apply_translation((-0.10, 0, 0))
    b = trimesh.creation.box(extents=(0.08, 0.08, 0.05))
    b.apply_translation((0.10, 0, 0))
    bar = trimesh.creation.box(extents=(0.12, 0.02, 0.02))
    bot = BotModel(trimesh.util.concatenate([a, b, bar]),
                   segment_mesh(trimesh.util.concatenate([a, b, bar])))
    bot.assign_material_to_all(library.get("Aluminum 6061-T6"))
    bot.set_brace(len(bot.parts) - 1, True)
    return bot


def _peak_margin(trace, bot, arena, library, dmg_cfg, brace_cfg):
    result = apply_brace_sharing(
        compute_damage(trace, bot, arena, library, cfg=dmg_cfg), bot, cfg=brace_cfg
    )
    return max(result.part_max_margin.values())


def run_sweep(pct: float):
    library = load_default_library()
    cls = NHRL_CLASSES["3lb"]
    bot = _braced_bot(library)
    arena = build_arena(cls)
    engine = SimEngine(arena, bot, timestep=DEFAULT_CONFIG.sim.timestep)
    trace = run_battery(engine, StressBattery(arena, cls, n_trials=1, seed=0), fps=30)

    d0, b0 = DEFAULT_CONFIG.damage, DEFAULT_CONFIG.brace
    base = _peak_margin(trace, bot, arena, library, d0, b0)

    rows = []
    for field in DAMAGE_FIELDS:
        v = getattr(d0, field)
        lo = _peak_margin(trace, bot, arena, library, replace(d0, **{field: v * (1 - pct)}), b0)
        hi = _peak_margin(trace, bot, arena, library, replace(d0, **{field: v * (1 + pct)}), b0)
        rows.append((f"damage.{field}", lo, hi, abs(hi - lo) / max(base, 1e-30)))
    for field in BRACE_FIELDS:
        v = getattr(b0, field)
        lo = _peak_margin(trace, bot, arena, library, d0, replace(b0, **{field: v * (1 - pct)}))
        hi = _peak_margin(trace, bot, arena, library, d0, replace(b0, **{field: v * (1 + pct)}))
        rows.append((f"brace.{field}", lo, hi, abs(hi - lo) / max(base, 1e-30)))

    rows.sort(key=lambda r: r[3], reverse=True)
    return base, rows


def render_markdown(base: float, rows, pct: float) -> str:
    lines = [
        f"# Verdict sensitivity (+/- {pct * 100:.0f}% per constant)",
        "",
        f"Base worst failure margin: `{base:.4e}`. Constants ranked by how much they",
        "move the verdict (relative swing); the top ones are what to calibrate first.",
        "",
        "| Constant | margin (low) | margin (high) | relative swing |",
        "|----------|--------------|---------------|----------------|",
    ]
    for name, lo, hi, swing in rows:
        lines.append(f"| `{name}` | {lo:.4e} | {hi:.4e} | {swing:.3f} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pct", type=float, default=0.25, help="perturbation fraction")
    ap.add_argument("--out", type=str, default=None, help="dir to write sensitivity.md")
    args = ap.parse_args()

    base, rows = run_sweep(args.pct)
    md = render_markdown(base, rows, args.pct)
    print(md)
    if args.out:
        path = Path(args.out) / "sensitivity.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(md, encoding="utf-8")
        print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
