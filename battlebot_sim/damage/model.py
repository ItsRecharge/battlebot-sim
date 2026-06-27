"""Map recorded contacts onto the bot mesh and build two damage fields.

Both fields are per-face arrays over the *original* (full) mesh:

1. Accumulated impact energy (J): for each contact, work = normal_force *
   closing_speed * dt is spread over nearby faces with a Gaussian falloff and
   summed across the whole battery. Answers "what took the most punishment".

2. Failure margin (peak stress / yield): contact stress is estimated with a
   Hertzian peak-pressure model
       p0 = (6 * F * E*^2 / (pi^3 * R^2)) ** (1/3)
   where E* is the effective modulus of the bot material and the surface it hit
   (1/E* = (1-v^2)/E_bot + (1-v^2)/E_other, v = 0.3), and R is the part's
   equivalent-sphere radius. The peak ratio p0 / yield is tracked per face.
   Answers "where will it actually break" (>= 1 means yielding).

This is a simplified, comparative model (Hybrid accuracy) — not FEA.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from battlebot_sim.arena.nhrl import Arena
from battlebot_sim.materials.library import Material, MaterialLibrary
from battlebot_sim.mesh.segment import BotModel
from battlebot_sim.sim.recorder import SimTrace

POISSON = 0.3
_OPPONENT_MODULUS_PA = 200e9   # treat the opponent weapon as steel

# Spatial spread of each contact onto the mesh. Kept tight so the heatmap shows
# crisp, localized hotspots (smaller sections of the bot) rather than a smear:
# energy spreads over ~1.2% of the bot diagonal, stress over 40% of that.
ENERGY_FALLOFF_FRAC = 0.012
STRESS_RADIUS_FRAC = 0.4


@dataclass
class DamageResult:
    """Per-face damage fields plus per-part summaries (original-mesh indexing)."""

    energy_per_face: np.ndarray        # (n_faces,) joules
    peak_stress_per_face: np.ndarray   # (n_faces,) pascals
    failure_margin_per_face: np.ndarray  # (n_faces,) stress / yield
    face_part: np.ndarray              # (n_faces,) owning part index
    part_max_margin: dict[int, float]
    part_total_energy: dict[int, float]

    def parts_that_fail(self, threshold: float = 1.0) -> list[int]:
        return [i for i, m in self.part_max_margin.items() if m >= threshold]


def _effective_modulus(bot_E: float, other_E: float) -> float:
    inv = (1 - POISSON**2) / max(bot_E, 1.0) + (1 - POISSON**2) / max(other_E, 1.0)
    return 1.0 / inv


def _part_radius(volume_m3: float) -> float:
    """Equivalent-sphere radius of a part, clamped to a sane minimum."""
    r = (3.0 * max(volume_m3, 1e-12) / (4.0 * np.pi)) ** (1.0 / 3.0)
    return max(r, 2e-3)


def _hertzian_peak_pressure(force: float, eff_modulus: float, radius: float) -> float:
    return (6.0 * force * eff_modulus**2 / (np.pi**3 * radius**2)) ** (1.0 / 3.0)


def _other_modulus(other: str, arena: Arena, library: MaterialLibrary) -> float:
    if other == "opponent_weapon":
        return _OPPONENT_MODULUS_PA
    try:
        return library.get(arena.material_of(other)).youngs_pa
    except KeyError:
        return _OPPONENT_MODULUS_PA


def compute_damage(
    trace: SimTrace,
    bot: BotModel,
    arena: Arena,
    library: MaterialLibrary,
    energy_falloff: float | None = None,
) -> DamageResult:
    """Build energy + failure-margin fields from a battery trace."""
    mesh = bot.original
    n_faces = len(mesh.faces)
    centroids = mesh.triangles_center           # (n_faces, 3) body frame
    tree = cKDTree(centroids)

    # Per-face owning part + that part's material/radius.
    face_part = np.zeros(n_faces, dtype=np.int64)
    part_radius = {}
    part_yield = {}
    part_modulus = {}
    for p in bot.parts:
        face_part[p.face_ids] = p.index
        part_radius[p.index] = _part_radius(p.volume_m3)
        mat = p.material
        part_yield[p.index] = mat.yield_pa if mat else np.inf
        part_modulus[p.index] = mat.youngs_pa if mat else 1.0

    # Falloff radius for energy spreading: a small fraction of the bot's size.
    if energy_falloff is None:
        diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
        energy_falloff = max(diag * ENERGY_FALLOFF_FRAC, 5e-3)
    stress_radius = energy_falloff * STRESS_RADIUS_FRAC

    energy = np.zeros(n_faces)
    peak_stress = np.zeros(n_faces)

    for c in trace.contacts:
        pidx = int(c.part_index)
        # --- energy: spread work over nearby faces (Gaussian weights) ---
        work = abs(c.normal_force) * max(c.rel_speed, 0.0) * trace.dt
        if work > 0:
            idx = tree.query_ball_point(c.local_pos, energy_falloff)
            if not idx:
                idx = [int(tree.query(c.local_pos)[1])]
            d = np.linalg.norm(centroids[idx] - c.local_pos, axis=1)
            w = np.exp(-(d**2) / (2.0 * (energy_falloff / 2.0) ** 2))
            w = w / w.sum()
            energy[idx] += work * w

        # --- failure margin: Hertzian peak pressure at the impact site ---
        eff_E = _effective_modulus(
            part_modulus[pidx], _other_modulus(c.other, arena, library)
        )
        p0 = _hertzian_peak_pressure(abs(c.normal_force), eff_E, part_radius[pidx])
        sidx = tree.query_ball_point(c.local_pos, stress_radius)
        if not sidx:
            sidx = [int(tree.query(c.local_pos)[1])]
        peak_stress[sidx] = np.maximum(peak_stress[sidx], p0)

    # Failure margin = peak stress / owning part's yield.
    yield_per_face = np.array([part_yield[int(face_part[f])] for f in range(n_faces)])
    failure_margin = np.divide(
        peak_stress, yield_per_face,
        out=np.zeros_like(peak_stress), where=np.isfinite(yield_per_face),
    )

    part_max_margin = {
        p.index: float(failure_margin[p.face_ids].max()) if len(p.face_ids) else 0.0
        for p in bot.parts
    }
    part_total_energy = {
        p.index: float(energy[p.face_ids].sum()) for p in bot.parts
    }

    return DamageResult(
        energy_per_face=energy,
        peak_stress_per_face=peak_stress,
        failure_margin_per_face=failure_margin,
        face_part=face_part,
        part_max_margin=part_max_margin,
        part_total_energy=part_total_energy,
    )
