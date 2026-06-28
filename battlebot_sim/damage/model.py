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
from battlebot_sim.config import DEFAULT_CONFIG, DamageConfig
from battlebot_sim.materials.library import MaterialLibrary
from battlebot_sim.mesh.segment import BotModel
from battlebot_sim.sim.recorder import SimTrace

# The Hertzian-contact and heatmap-spread tuning constants live on ``DamageConfig``
# (battlebot_sim/config.py). Each impact paints a smooth exponential hotspot whose
# sigma is tied to the Hertzian contact-patch radius, floored for visibility (a
# fraction of the bot diagonal, plus an absolute minimum) and capped per struck
# part. These drive the *appearance*; the peak values — and so the failure verdict
# — come from the unchanged Hertzian model below. ``cfg`` is threaded in as a
# defaulted argument so a study can vary one constant per run.

# Legacy fixed-spread fractions, used only when an explicit ``energy_falloff`` is
# passed (overrides the per-contact sigma).
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


def _effective_modulus(bot_E: float, other_E: float,
                       poisson: float = DEFAULT_CONFIG.damage.poisson) -> float:
    inv = (1 - poisson**2) / max(bot_E, 1.0) + (1 - poisson**2) / max(other_E, 1.0)
    return 1.0 / inv


def _part_radius(volume_m3: float) -> float:
    """Equivalent-sphere radius of a part, clamped to a sane minimum."""
    r = (3.0 * max(volume_m3, 1e-12) / (4.0 * np.pi)) ** (1.0 / 3.0)
    return max(r, 2e-3)


def _hertzian_peak_pressure(force: float, eff_modulus: float, radius: float) -> float:
    return (6.0 * force * eff_modulus**2 / (np.pi**3 * radius**2)) ** (1.0 / 3.0)


def _hertzian_patch_radius(force: float, eff_modulus: float, radius: float) -> float:
    """Radius of the Hertzian contact patch for a sphere-on-flat contact (m):
    ``a = (3 F R / (4 E*))^(1/3)``. Sets the physical width of an impact's
    footprint, which the heatmap spreads damage over."""
    return (3.0 * max(force, 0.0) * radius / (4.0 * max(eff_modulus, 1.0))) ** (1.0 / 3.0)


def _other_modulus(
    other: str, arena: Arena, library: MaterialLibrary,
    opponent_modulus_pa: float = DEFAULT_CONFIG.damage.opponent_modulus_pa,
) -> float:
    if other == "opponent_weapon":
        return opponent_modulus_pa
    try:
        return library.get(arena.material_of(other)).youngs_pa
    except KeyError:
        return opponent_modulus_pa


class DamageAccumulator:
    """Builds the per-face damage fields incrementally from contacts.

    This is the single implementation behind both the offline ``compute_damage``
    (one ``ingest`` of the whole trace) and the live run (the streaming worker
    calls ``ingest`` per frame-chunk as contacts arrive). Ingesting the same
    contacts in the same order — whether all at once or in arbitrary
    consecutive batches — produces identical fields (energy is summed, peak
    stress is an order-independent max), so the live and offline paths agree.
    """

    def __init__(
        self,
        bot: BotModel,
        arena: Arena,
        library: MaterialLibrary,
        energy_falloff: float | None = None,
        cfg: DamageConfig = DEFAULT_CONFIG.damage,
    ):
        self.bot = bot
        self.arena = arena
        self.library = library
        self.cfg = cfg

        mesh = bot.original
        self.mesh = mesh
        self.n_faces = len(mesh.faces)
        self.centroids = mesh.triangles_center      # (n_faces, 3) body frame
        self.tree = cKDTree(self.centroids)

        # Per-face owning part + that part's material/radius.
        self.face_part = np.zeros(self.n_faces, dtype=np.int64)
        self.part_radius: dict[int, float] = {}
        self.part_yield: dict[int, float] = {}
        self.part_modulus: dict[int, float] = {}
        self.part_char_len: dict[int, float] = {}
        for p in bot.parts:
            self.face_part[p.face_ids] = p.index
            self.part_radius[p.index] = _part_radius(p.volume_m3)
            ext = np.asarray(p.bounds[1] - p.bounds[0], dtype=float)
            self.part_char_len[p.index] = float(max(ext.max(), 1e-3))
            mat = p.material
            self.part_yield[p.index] = mat.yield_pa if mat else np.inf
            self.part_modulus[p.index] = mat.youngs_pa if mat else 1.0

        # Spreading sigma config. With an explicit ``energy_falloff`` we keep a
        # fixed spread (legacy/override); otherwise each contact picks a sigma
        # from its Hertzian patch radius, floored for visibility (see ingest).
        self.diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
        self._fixed_sigma = energy_falloff
        self.sigma_floor = max(cfg.sigma_min_frac * self.diag, cfg.sigma_abs_min)

        self.energy = np.zeros(self.n_faces)
        self.peak_stress = np.zeros(self.n_faces)
        # Per-face yield (fixed once parts/materials are set), for cheap margins.
        self.yield_per_face = np.array(
            [self.part_yield[int(self.face_part[f])] for f in range(self.n_faces)]
        )

    def _contact_sigma(self, force: float, eff_E: float, pidx: int) -> float:
        """The energy-spread Gaussian sigma for one contact (metres).

        Tied to the Hertzian contact-patch radius, then floored for visibility
        and capped at half the struck part's size."""
        if self._fixed_sigma is not None:
            return float(self._fixed_sigma)
        cfg = self.cfg
        a = _hertzian_patch_radius(force, eff_E, self.part_radius[pidx])
        sigma = cfg.sigma_patch_factor * a
        sigma = max(sigma, self.sigma_floor)
        sigma = min(sigma, cfg.sigma_part_frac * self.part_char_len[pidx])
        return max(sigma, cfg.sigma_abs_min)

    def ingest(self, contacts, dt: float) -> None:
        """Fold a batch of contacts into the running energy/stress fields.

        Each contact paints a smooth exponential blob: energy is spread with a
        conserved Gaussian (weights sum to 1, so the absorbed joules are exact),
        and contact stress is painted as a Gaussian-weighted peak combined by MAX
        — a gradient inside the footprint whose centre keeps the full Hertzian p0,
        so repeated hits never inflate the failure verdict.
        """
        for c in contacts:
            pidx = int(c.part_index)
            force = abs(c.normal_force)
            eff_E = _effective_modulus(
                self.part_modulus[pidx],
                _other_modulus(c.other, self.arena, self.library,
                               self.cfg.opponent_modulus_pa),
                self.cfg.poisson,
            )
            p0 = _hertzian_peak_pressure(force, eff_E, self.part_radius[pidx])
            sigma_e = self._contact_sigma(force, eff_E, pidx)
            sigma_s = self.cfg.stress_sigma_frac * sigma_e

            # --- energy: conserved Gaussian spread over nearby faces ---
            work = force * max(c.rel_speed, 0.0) * dt
            if work > 0:
                idx = self.tree.query_ball_point(
                    c.local_pos, self.cfg.kernel_radius_sigmas * sigma_e)
                if not idx:
                    idx = [int(self.tree.query(c.local_pos)[1])]
                d = np.linalg.norm(self.centroids[idx] - c.local_pos, axis=1)
                w = np.exp(-(d**2) / (2.0 * sigma_e**2))
                total = float(w.sum())
                if total > 0:
                    self.energy[idx] += work * (w / total)

            # --- failure margin: Gaussian-weighted Hertzian peak, MAX-combined ---
            sidx = self.tree.query_ball_point(
                c.local_pos, self.cfg.kernel_radius_sigmas * sigma_s)
            if not sidx:
                sidx = [int(self.tree.query(c.local_pos)[1])]
            ds = np.linalg.norm(self.centroids[sidx] - c.local_pos, axis=1)
            ws = np.exp(-(ds**2) / (2.0 * sigma_s**2))
            self.peak_stress[sidx] = np.maximum(self.peak_stress[sidx], p0 * ws)

    def _failure_margin(self) -> np.ndarray:
        """Per-face peak stress / owning-part yield (0 where yield is non-finite)."""
        return np.divide(
            self.peak_stress, self.yield_per_face,
            out=np.zeros_like(self.peak_stress), where=np.isfinite(self.yield_per_face),
        )

    def snapshot_face_fields(self) -> tuple[np.ndarray, np.ndarray]:
        """Cheap (energy, failure_margin) copies of the running fields — for a
        live view to paint mid-run without touching the accumulator's arrays."""
        return self.energy.copy(), self._failure_margin()

    def current_max_margin(self) -> float:
        """Running worst failure margin across all faces (for live metrics)."""
        fm = self._failure_margin()
        return float(fm.max()) if fm.size else 0.0

    def finalize(self) -> DamageResult:
        """Snapshot the accumulated fields into a DamageResult."""
        failure_margin = self._failure_margin()
        part_max_margin = {
            p.index: float(failure_margin[p.face_ids].max()) if len(p.face_ids) else 0.0
            for p in self.bot.parts
        }
        part_total_energy = {
            p.index: float(self.energy[p.face_ids].sum()) for p in self.bot.parts
        }
        return DamageResult(
            energy_per_face=self.energy,
            peak_stress_per_face=self.peak_stress,
            failure_margin_per_face=failure_margin,
            face_part=self.face_part,
            part_max_margin=part_max_margin,
            part_total_energy=part_total_energy,
        )


def compute_damage(
    trace: SimTrace,
    bot: BotModel,
    arena: Arena,
    library: MaterialLibrary,
    energy_falloff: float | None = None,
    cfg: DamageConfig = DEFAULT_CONFIG.damage,
) -> DamageResult:
    """Build energy + failure-margin fields from a complete battery trace.

    A thin wrapper over DamageAccumulator so the offline report/selftest path and
    the live streaming path share one implementation.
    """
    acc = DamageAccumulator(bot, arena, library, energy_falloff, cfg=cfg)
    acc.ingest(trace.contacts, trace.dt)
    return acc.finalize()
