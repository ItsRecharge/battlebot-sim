# Damage model: assumptions, scope & limitations

BattleBot Sim predicts **comparative** impact damage to help you reason about
material and geometry choices. It is a *hybrid* tool — rigid-body physics for the
motion, an analytic contact-mechanics model for the stress — **not** a
finite-element or certification-grade analysis. Read this before trusting a number.

## What the model computes

For every contact recorded during the stress battery, two per-face fields are
accumulated over the whole run:

1. **Impact energy (J)** — `work = normal_force × closing_speed × dt`, spread over
   nearby faces with a conserved Gaussian (the weights sum to 1, so total energy
   is preserved; see `tests/validation/test_energy_conservation.py`).
2. **Failure margin** — `peak_stress / yield_strength`, where the peak stress is
   the Hertzian sphere-on-flat contact pressure
   `p0 = (6 F E*² / (π³ R²))^(1/3)` (validated against the closed form in
   `tests/validation/test_hertzian.py`). A margin ≥ 1 means the material is
   predicted to yield.

## Key assumptions

- **Comparative, not absolute.** Treat margins as a ranking ("part A is closer to
  failing than part B"), not a guaranteed real-world yield prediction.
- **Hertzian contact.** Each part is approximated as an equivalent sphere of
  radius `R = (3V/4π)^(1/3)` from its volume (clamped to ≥ 2 mm). Real parts are
  not spheres; sharp edges concentrate stress more than this predicts.
- **Effective modulus.** `1/E* = (1−ν²)/E_bot + (1−ν²)/E_other` with Poisson's
  ratio ν = 0.3 for both bodies. The opponent weapon and unmaterialed surfaces are
  treated as steel (`opponent_modulus_pa = 200 GPa`).
- **Convex-hull collision.** MuJoCo collides each part as its convex hull, so
  concavities don't catch and interlock; degenerate slivers are padded to a tiny
  tetrahedron so the solver accepts them.
- **Heatmap spread is appearance, not physics.** The Gaussian blob sizes
  (`sigma_*` constants) only control how the hotspot *looks*; the peak value at the
  centre — and therefore the failure verdict — comes from the unchanged Hertzian
  model. Tuning the sigmas never changes whether a part is reported as failing.
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
