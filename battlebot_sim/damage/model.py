"""Map recorded contacts onto the bot mesh and build two damage fields.

Both fields are per-face arrays over the *original* (full) mesh:

1. Accumulated impact energy (J): for each contact, work = normal_force *
   closing_speed * dt is spread over nearby faces with a Gaussian falloff and
   summed across the whole battery. Answers "what took the most punishment".

2. Failure margin (governing stress / yield): an *absolute* per-part verdict
   built from classical mechanics of materials (see ``damage/structural.py``):
     * contact: the Hertzian peak pressure ``p0`` is taken to its subsurface
       von-Mises peak (``sigma_vm,max = vm_factor * p0``, first yield at
       ``p0 ~= 1.60 * yield``) using a combined-curvature radius that models the
       opponent weapon as a sharp striker;
     * structural: the impact also bends the part (``sigma_b = 6 b F L / w t^2``)
       and crushes it (``sigma_m = F / A``);
   the governing (larger) stress over yield is the margin (>= 1 means yielding).
   This verdict is read at the true contact point, so it is independent of the
   Gaussian heatmap-spread constants below.

The heatmap fields are still a smoothed, comparative *picture*; the verdict is an
analytic, hand-calc-grade absolute estimate — not FEA.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from battlebot_sim.arena.nhrl import Arena
from battlebot_sim.config import DEFAULT_CONFIG, DamageConfig
from battlebot_sim.damage import structural as st
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
    peak_stress_per_face: np.ndarray   # (n_faces,) pascals (Gaussian-painted, render)
    failure_margin_per_face: np.ndarray  # (n_faces,) stress / yield (render heatmap)
    face_part: np.ndarray              # (n_faces,) owning part index
    part_max_margin: dict[int, float]  # absolute governing margin per part
    part_total_energy: dict[int, float]
    # Absolute per-part stress verdict (see damage/structural.py). ``part_stress``
    # is parallel to ``bot.parts``; ``true_peak_stress_per_face`` is the
    # un-attenuated governing stress at each contact's nearest face (verdict-grade,
    # independent of the Gaussian heatmap spread above).
    part_stress: list = field(default_factory=list)
    true_peak_stress_per_face: np.ndarray | None = None

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


def _indenter_radius(other: str, cfg: DamageConfig = DEFAULT_CONFIG.damage) -> float:
    """Effective tip radius (m) of whatever struck the part. The opponent weapon
    is a sharp striker (small radius -> high contact pressure); the arena
    floor/wall is treated as near-flat (large radius)."""
    if other == "opponent_weapon":
        return cfg.weapon_tip_radius_m
    return cfg.flat_surface_radius_m


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
        # Absolute-verdict precomputes (oriented section, surface curvature, ult).
        self.part_section: dict[int, st.Section] = {}
        self.part_surface_radius: dict[int, float] = {}
        self.part_ultimate: dict[int, float] = {}
        self.part_bends: dict[int, bool] = {}   # slender enough for beam bending?
        for p in bot.parts:
            self.face_part[p.face_ids] = p.index
            self.part_radius[p.index] = _part_radius(p.volume_m3)
            ext = np.asarray(p.bounds[1] - p.bounds[0], dtype=float)
            self.part_char_len[p.index] = float(max(ext.max(), 1e-3))
            mat = p.material
            self.part_yield[p.index] = mat.yield_pa if mat else np.inf
            self.part_modulus[p.index] = mat.youngs_pa if mat else 1.0
            sec = st.section_from_vertices(np.asarray(p.mesh.vertices, dtype=float))
            self.part_section[p.index] = sec
            # A plate-like part is locally flat (R_surface = inf) so a sharp
            # striker reads its true high contact stress; a chunky part keeps its
            # equivalent-sphere curvature.
            plate_like = (sec.t / max(sec.L, 1e-9)) < cfg.plate_flat_threshold
            self.part_surface_radius[p.index] = (
                np.inf if plate_like else self.part_radius[p.index])
            self.part_ultimate[p.index] = mat.ultimate_pa if mat else np.inf
            # Beam bending is only physical for a slender part that is also a real
            # (not sub-mm shell/sliver) member; everything else is governed by
            # contact/membrane stress instead.
            self.part_bends[p.index] = (
                sec.t >= cfg.bending_min_thickness_m
                and sec.L / max(sec.t, 1e-9) >= cfg.bending_min_aspect)
        # Subsurface von-Mises coefficient: configured, or derived from Poisson.
        self.vm_factor = (cfg.contact_vm_factor if cfg.contact_vm_factor > 0
                          else st.vm_factor_from_poisson(cfg.poisson))
        # Per-part running maxima for the absolute verdict (all order-independent).
        self.part_contact_vm = {p.index: 0.0 for p in bot.parts}
        self.part_bending = {p.index: 0.0 for p in bot.parts}
        self.part_membrane = {p.index: 0.0 for p in bot.parts}
        self.part_struct = {p.index: 0.0 for p in bot.parts}

        # Spreading sigma config. With an explicit ``energy_falloff`` we keep a
        # fixed spread (legacy/override); otherwise each contact picks a sigma
        # from its Hertzian patch radius, floored for visibility (see ingest).
        self.diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
        self._fixed_sigma = energy_falloff
        self.sigma_floor = max(cfg.sigma_min_frac * self.diag, cfg.sigma_abs_min)

        self.energy = np.zeros(self.n_faces)
        self.peak_stress = np.zeros(self.n_faces)
        # Un-attenuated governing stress at each contact's nearest face (verdict).
        self.true_peak_stress = np.zeros(self.n_faces)
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

            # --- absolute verdict: un-attenuated structural pass (no painting) ---
            # Evaluated at the true contact point with combined-curvature radius,
            # so the verdict is independent of the heatmap sigma constants.
            sec = self.part_section[pidx]
            r_eff = st.effective_contact_radius(
                _indenter_radius(c.other, self.cfg), self.part_surface_radius[pidx])
            p0_true = _hertzian_peak_pressure(force, eff_E, r_eff)
            contact_vm = st.contact_von_mises_peak(p0_true, self.vm_factor)
            sigma_b = (st.bending_stress(
                force, sec.L, sec.t, sec.w, self.cfg.bending_bc_factor)
                if self.part_bends[pidx] else 0.0)
            sigma_m = st.membrane_stress(force, sec.area)
            struct = sigma_b + sigma_m
            self.part_contact_vm[pidx] = max(self.part_contact_vm[pidx], contact_vm)
            self.part_bending[pidx] = max(self.part_bending[pidx], sigma_b)
            self.part_membrane[pidx] = max(self.part_membrane[pidx], sigma_m)
            self.part_struct[pidx] = max(self.part_struct[pidx], struct)
            nf = int(self.tree.query(c.local_pos)[1])
            self.true_peak_stress[nf] = max(
                self.true_peak_stress[nf], max(contact_vm, struct))

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
        """Running worst *governing* margin across parts (for live metrics).

        Uses the absolute per-part verdict (the larger of the subsurface contact
        von Mises and the structural bending+membrane stress, over the part's
        yield), not the Gaussian-painted heatmap field."""
        best = 0.0
        for p in self.bot.parts:
            y = self.part_yield[p.index]
            if not (np.isfinite(y) and y > 0):
                continue
            gov = max(self.part_contact_vm[p.index], self.part_struct[p.index])
            best = max(best, gov / y)
        return float(best)

    def finalize(self) -> DamageResult:
        """Snapshot the accumulated fields into a DamageResult.

        ``part_max_margin`` is the *absolute* governing margin per part (the
        larger of the subsurface contact von Mises and the structural
        bending+membrane stress, over yield) — independent of the Gaussian
        heatmap spread. ``failure_margin_per_face`` is retained for rendering.
        """
        failure_margin = self._failure_margin()
        part_stress: list[st.PartStress] = []
        part_max_margin: dict[int, float] = {}
        for p in self.bot.parts:
            i = p.index
            cvm = self.part_contact_vm[i]
            sb, sm, struct = (self.part_bending[i], self.part_membrane[i],
                              self.part_struct[i])
            if cvm >= struct:
                gov, mode = cvm, "contact"
            else:
                gov, mode = struct, ("bending" if sb >= sm else "membrane")
            y, ult = self.part_yield[i], self.part_ultimate[i]
            margin = float(gov / y) if np.isfinite(y) and y > 0 else 0.0
            sec = self.part_section[i]
            part_stress.append(st.PartStress(
                part_index=i, contact_stress=cvm, bending_stress=sb,
                membrane_stress=sm, governing_stress=gov, governing_mode=mode,
                span_used=sec.L, thickness_used=sec.t, margin=margin,
                yields=margin >= 1.0,
                fractures=bool(np.isfinite(ult) and ult > 0 and gov >= ult)))
            part_max_margin[i] = margin
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
            part_stress=part_stress,
            true_peak_stress_per_face=self.true_peak_stress.copy(),
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
