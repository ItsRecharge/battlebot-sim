# Damage model: assumptions, scope & limitations

BattleBot Sim predicts an **absolute** per-part failure margin — read it as a
yield/fracture prediction, not merely a ranking — to help you reason about material
and geometry choices. It is a *hybrid* tool: rigid-body physics for the motion,
classical mechanics of materials for the stress. It is hand-calc grade, **not** a
finite-element or certification-grade analysis. Read this before trusting a number.

## What the model computes

For every contact recorded during the stress battery, two per-face fields are
accumulated over the whole run:

1. **Impact energy (J)** — `work = normal_force × closing_speed × dt`, spread over
   nearby faces with a conserved Gaussian (the weights sum to 1, so total energy
   is preserved; see `tests/validation/test_energy_conservation.py`).
2. **Failure margin** — `governing_stress / yield_strength`, the larger of two
   physically distinct drivers (`damage/structural.py`); a margin ≥ 1 means the
   material is predicted to yield:
   - **Contact:** the Hertzian peak pressure `p0 = (6 F E*² / (π³ R²))^(1/3)`
     (validated in `tests/validation/test_hertzian.py`) taken to its **subsurface
     von Mises peak** `σ_vm,max = 0.62·p0`. First yield in contact is subsurface, so
     it occurs at `p0 ≈ 1.60·yield`, not at `p0 = yield`.
   - **Structural:** the impact also **bends** the part — `σ_b = 6·b·F·L/(w·t²)` for
     a plate of thickness `t`, width `w`, span `L` — and **crushes** it — `σ_m = F/A`.

   The margin is read at the **true contact point** with the un-attenuated stress,
   so it is independent of the heatmap-spread constants (see `uncertainty.md`). A
   separate **fracture** flag trips when the governing stress exceeds the ultimate
   strength.

## Key assumptions

- **Absolute, but first-order.** Margins are calibrated to first yield (and the
  fracture flag to ultimate strength), so they read as predictions — but they come
  from closed-form hand calculations, not a full-field solve. Treat them as a
  conservative engineering screen, not a certification.
- **Combined-curvature contact.** The contact radius combines the struck part's
  local curvature with the *indenter*: the opponent weapon is a sharp striker
  (`weapon_tip_radius_m`, default 4 mm), the arena floor/wall near-flat, via
  `1/R_eff = 1/R_indenter + 1/R_surface`. A flat plate struck by a sharp weapon now
  reads its true high contact stress instead of looking artificially safe.
- **Beam/plate structural model.** Each part is reduced to an oriented rectangular
  section (thickness = its thinnest principal extent, found by PCA) and loaded as a
  simply-supported beam (`M = bc·F·L`, `bc = 0.25`, configurable). Bending is applied
  **only** to genuinely beam-like parts — slender (`L/t ≥ 8`) and not a sub-millimetre
  shell/sliver; stubby, blocky, or shell parts are governed by contact and membrane
  stress instead, so a thick block never reports an impossible bending stress. A brace
  *relieves* the bending of the parts it bridges (its own margin still comes from the
  contacts it takes). Real parts are not ideal beams; treat this as a conservative
  first-order screen.
- **Effective modulus.** `1/E* = (1−ν²)/E_bot + (1−ν²)/E_other` with Poisson's
  ratio ν = 0.3 for both bodies. The opponent weapon and unmaterialed surfaces are
  treated as steel (`opponent_modulus_pa = 200 GPa`).
- **Convex-hull collision.** MuJoCo collides each part as its convex hull, so
  concavities don't catch and interlock; degenerate slivers are padded to a tiny
  tetrahedron so the solver accepts them.
- **Heatmap spread is appearance only.** The Gaussian blob sizes (`sigma_*`
  constants) control how the rendered hotspot *looks*; the failure verdict is read
  from the un-attenuated stress at the true contact point, so tuning the sigmas
  never changes whether a part is reported as failing (proven in
  `tests/validation/test_true_peak.py`).
- **Deterministic & seeded.** The stress battery is fully seeded, so a given bot +
  class + seed always produces the same numbers (pinned by the golden baseline).

## Valid input ranges

Enforced at the boundary (`battlebot_sim/validation.py`):

- **Mesh:** finite vertices, ≥ 1 face, bounding box within `[1 µm, 1 km]` after
  scaling. Outside this is treated as a units mistake.
- **Scale:** finite and strictly positive.
- **Mass:** finite and non-negative.
- **Run params:** `n_trials` and `fps` integers in `[1, 1000]`.

Geometry is SI metres throughout; an STL exported in millimetres needs
`scale_to_m = 0.001`.

## Not captured (do not rely on the model for these)

- Fatigue / repeated-load failure, crack growth, or fracture mechanics.
- Plastic deformation and energy absorbed by permanent bending (only the *onset*
  of yield is flagged).
- Fastener / joint / weld failure and load paths through them.
- Strain-rate, thermal, or material-nonlinearity effects.
- Dynamic stress-wave propagation through the structure.
- Anisotropic materials (composites are treated with isotropic properties).

## Tuning constants

Every constant above lives on typed dataclasses in
[`battlebot_sim/config.py`](../battlebot_sim/config.py) with documented defaults,
overridable via a TOML file (`load_config`). Their influence on the verdict is
quantified in [uncertainty.md](uncertainty.md).
