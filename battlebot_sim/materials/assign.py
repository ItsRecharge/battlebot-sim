"""NHRL weight classes and weight-class validation.

NHRL (National Havoc Robot League) runs three combat classes. The mass limits
are authoritative; cage dimensions are first-order approximations (the NHRL cage
is a polycarbonate box with a steel floor) and are fully parametric so they can
be tuned against the official spec later.
"""

from __future__ import annotations

from dataclasses import dataclass

from battlebot_sim.validation import validate_mass

LB_TO_KG = 0.45359237


@dataclass(frozen=True)
class WeightClass:
    """An NHRL combat class and the test cage used for it."""

    key: str                 # short id, e.g. "3lb"
    name: str                # display name
    max_mass_kg: float       # mass limit
    cage_length_m: float     # interior X
    cage_width_m: float      # interior Y
    cage_height_m: float     # interior Z (floor to ceiling)
    wall_material: str = "Polycarbonate (Lexan)"
    floor_material: str = "Mild Steel (A36)"

    @property
    def max_mass_lb(self) -> float:
        return self.max_mass_kg / LB_TO_KG


# Approximate NHRL cage sizes; mass limits are exact.
NHRL_CLASSES: dict[str, WeightClass] = {
    "3lb": WeightClass(
        key="3lb", name="Beetleweight (3 lb)", max_mass_kg=3 * LB_TO_KG,
        cage_length_m=1.22, cage_width_m=1.22, cage_height_m=0.61,
    ),
    "12lb": WeightClass(
        key="12lb", name="Hobbyweight (12 lb)", max_mass_kg=12 * LB_TO_KG,
        cage_length_m=1.83, cage_width_m=1.83, cage_height_m=0.91,
    ),
    "30lb": WeightClass(
        key="30lb", name="Featherweight (30 lb)", max_mass_kg=30 * LB_TO_KG,
        cage_length_m=2.44, cage_width_m=2.44, cage_height_m=1.22,
    ),
}


@dataclass(frozen=True)
class WeightCheck:
    """Result of comparing a bot's mass against a class limit."""

    ok: bool
    total_mass_kg: float
    max_mass_kg: float
    message: str

    @property
    def over_by_kg(self) -> float:
        return max(0.0, self.total_mass_kg - self.max_mass_kg)


def validate_weight_class(total_mass_kg: float, weight_class: WeightClass) -> WeightCheck:
    """Check a bot's total mass against its class limit.

    Returns a WeightCheck; `ok` is False when the bot exceeds the class limit.
    """
    total_mass_kg = validate_mass(total_mass_kg)
    limit = weight_class.max_mass_kg
    if total_mass_kg <= limit:
        margin = limit - total_mass_kg
        msg = (
            f"OK: {total_mass_kg:.3f} kg is within the {weight_class.name} "
            f"limit of {limit:.3f} kg ({margin:.3f} kg to spare)."
        )
        return WeightCheck(True, total_mass_kg, limit, msg)

    over = total_mass_kg - limit
    msg = (
        f"OVER WEIGHT: {total_mass_kg:.3f} kg exceeds the {weight_class.name} "
        f"limit of {limit:.3f} kg by {over:.3f} kg."
    )
    return WeightCheck(False, total_mass_kg, limit, msg)
