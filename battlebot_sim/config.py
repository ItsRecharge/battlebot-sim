"""Central, typed configuration for the physics + damage tuning constants.

Every tuning constant that used to be a magic number scattered across the sim
and damage modules lives here, grouped into frozen dataclasses with documented
defaults. The defaults reproduce the historical behaviour exactly (the golden
self-test baseline pins this), so the app runs with zero config files.

Design:

* **Source of truth in code.** Defaults live on the dataclasses; there is always
  a usable :data:`DEFAULT_CONFIG` even with no TOML present.
* **Threaded as defaulted arguments.** The sweep-relevant groups
  (:class:`DamageConfig`, :class:`BraceConfig`, sim timestep) are passed into the
  functions that use them as ``cfg=DEFAULT_CONFIG.<group>`` defaults, rather than
  read from a module global inside the body. That lets a study vary one constant
  per run (see the Phase 6 sensitivity sweep) without monkeypatching.
* **Optional TOML overlay.** :func:`load_config` overlays a user TOML onto the
  defaults for the few constants worth exposing, leaving the rest untouched.

Frozen dataclasses are immutable, so it is safe to share a single
``DEFAULT_CONFIG`` instance everywhere; use :func:`dataclasses.replace` (or
``load_config``) to derive a tweaked copy.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path


@dataclass(frozen=True)
class SimConfig:
    """MuJoCo integration settings."""

    timestep: float = 5.0e-4          # physics step (s)


@dataclass(frozen=True)
class BatteryConfig:
    """Stress-battery scaling and containment safety-net constants."""

    # An opponent strike delivers full energy to the damage model, but the
    # physical launch is capped to this multiple of class speed so the bot
    # bounces in the cage instead of being fired through a wall.
    strike_dv_cap_factor: float = 1.5
    # Restitution when the containment net reflects the bot off an interior wall.
    contain_restitution: float = 0.3
    # Representative opponent-weapon energy: this many joules per kg of class limit.
    strike_energy_per_kg: float = 200.0
    # Representative collision speed (m/s) per class, with a fallback.
    default_class_speed: float = 7.0
    class_speed: dict = field(
        default_factory=lambda: {"3lb": 6.0, "12lb": 8.0, "30lb": 10.0}
    )


@dataclass(frozen=True)
class DamageConfig:
    """Hertzian-contact and heatmap-spread constants for the damage model."""

    poisson: float = 0.3
    opponent_modulus_pa: float = 200e9     # treat the opponent weapon as steel
    # Spatial spread of each contact onto the mesh (see damage/model.py).
    sigma_patch_factor: float = 8.0        # energy sigma ~ this * Hertzian patch radius
    sigma_min_frac: float = 0.025          # floor: sigma >= this * bot diagonal …
    sigma_abs_min: float = 3.0e-3          # … but never below this many metres
    sigma_part_frac: float = 0.5           # cap: sigma <= this * struck part max extent
    stress_sigma_frac: float = 0.4         # stress sigma = this * energy sigma (tighter)
    kernel_radius_sigmas: float = 3.0      # gather faces within this many sigma of a hit


@dataclass(frozen=True)
class BraceConfig:
    """Brace load-sharing heuristic constants."""

    k_ref: float = 5.0e6          # reference axial stiffness (N/m) that gives k ~ 1
    adjacency_tol: float = 5.0e-3  # parts within this many metres count as connected
    transfer: float = 0.5          # fraction of shed stress pushed into the brace


@dataclass(frozen=True)
class ContactConfig:
    """MuJoCo contact tuning per material category, plus gravity."""

    gravity: float = -9.81         # m/s^2 along -z
    # (sliding friction, restitution-ish bounce 0..1) per material category.
    material_friction_bounce: dict = field(
        default_factory=lambda: {
            "metal": (0.6, 0.2),
            "plastic": (0.4, 0.1),
            "composite": (0.5, 0.15),
        }
    )


@dataclass(frozen=True)
class AppConfig:
    """The whole tunable surface, one immutable bundle."""

    sim: SimConfig = field(default_factory=SimConfig)
    battery: BatteryConfig = field(default_factory=BatteryConfig)
    damage: DamageConfig = field(default_factory=DamageConfig)
    brace: BraceConfig = field(default_factory=BraceConfig)
    contact: ContactConfig = field(default_factory=ContactConfig)


#: The default configuration. Immutable; derive copies with ``dataclasses.replace``.
DEFAULT_CONFIG = AppConfig()

_SECTIONS = ("sim", "battery", "damage", "brace", "contact")


def load_config(path: str | Path, base: AppConfig = DEFAULT_CONFIG) -> AppConfig:
    """Overlay a TOML file onto ``base`` and return a new :class:`AppConfig`.

    Only the keys present in the file are overridden; everything else keeps the
    default. Unknown sections/keys raise so typos surface instead of silently
    doing nothing. Example TOML::

        [damage]
        sigma_patch_factor = 6.0

        [sim]
        timestep = 2.5e-4
    """
    try:
        import tomllib as _toml  # Python 3.11+
    except ModuleNotFoundError:          # pragma: no cover - 3.10 fallback
        import tomli as _toml  # type: ignore[no-redef]

    with open(path, "rb") as fh:
        data = _toml.load(fh)

    cfg = base
    for section, values in data.items():
        if section not in _SECTIONS:
            raise ValueError(f"unknown config section [{section}]")
        sub = getattr(cfg, section)
        known = {f for f in sub.__dataclass_fields__}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"unknown keys in [{section}]: {sorted(unknown)}")
        cfg = replace(cfg, **{section: replace(sub, **values)})
    return cfg
