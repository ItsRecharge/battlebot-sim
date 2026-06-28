"""Damage modelling: turn a SimTrace into per-face energy and failure fields,
including brace load-sharing."""

from battlebot_sim.damage.braces import apply_brace_sharing
from battlebot_sim.damage.fields import normalize, vertex_scalars
from battlebot_sim.damage.model import DamageResult, compute_damage

__all__ = [
    "DamageResult",
    "compute_damage",
    "apply_brace_sharing",
    "normalize",
    "vertex_scalars",
]
