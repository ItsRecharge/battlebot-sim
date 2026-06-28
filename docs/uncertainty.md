# Uncertainty & convergence

How much to trust the failure verdict. Two questions: is it converged (independent
of the numerical timestep), and how sensitive is it to the uncalibrated tuning
constants? Both are answered by reproducible artifacts in the repo.

## Timestep convergence — converged

`tests/validation/test_convergence.py` runs the seeded battery at two timesteps and
compares the worst failure margin:

| timestep | worst failure margin |
|----------|----------------------|
| 1.0e-3 s | 1.060123e-05 |
| 5.0e-4 s (default) | 1.060123e-05 |
| 2.5e-4 s | 1.060123e-05 |

The verdict is **identical to six significant figures** across a 4× timestep range:
it is driven by the hardest single contact, which the solver resolves the same way
regardless of step size. **The default 5e-4 s is firmly in the converged regime.**
(Contact *count* scales with 1/timestep, but the peak — and so the verdict — does
not.)

## Sensitivity to tuning constants

`scripts/sensitivity_sweep.py` perturbs each constant ±25% and measures the relative
swing in the worst failure margin. Representative ranking (braced two-cube bot):

| constant | relative swing | reading |
|----------|----------------|---------|
| `damage.stress_sigma_frac` | ~19 | **dominant** |
| `damage.sigma_patch_factor` | ~19 | **dominant** |
| `damage.opponent_modulus_pa` | ~0.69 | moderate (`p0 ∝ E*^(2/3)`) |
| `brace.transfer` | ~0.50 | moderate |
| `damage.poisson` | ~0.46 | moderate |
| `brace.k_ref`, `damage.sigma_min_frac` | ~0.01 | minor |
| `damage.sigma_part_frac`, `kernel_radius_sigmas`, `brace.adjacency_tol` | ~0 | negligible |

### The important caveat (a real finding, not a bug)

The sigma spread constants (`sigma_patch_factor`, `stress_sigma_frac`) **dominate the
verdict on coarse meshes**, even though `docs/model_assumptions.md` describes them as
"appearance only". The reason is mesh resolution: the peak stress is sampled at the
nearest *face centroid*, and on a coarse mesh no face sits exactly at the contact
point, so a Gaussian-weighted value `p0 · exp(−d²/2σ²) < p0` is recorded. A tighter
sigma drops that weight steeply; a looser one approaches the true `p0`. On a
**finely meshed** part (a face near every contact) this coupling vanishes and the
constants behave as appearance-only.

**Implication:** absolute margins on coarse meshes are sigma-dependent and should be
read *comparatively*, not as absolute yield predictions. To harden this, refine the
mesh near contacts (so a centroid lands at the peak) before relying on absolute
numbers — the single highest-value calibration follow-up.

## Reproducing

```powershell
.\.venv\Scripts\python.exe scripts\sensitivity_sweep.py --pct 0.25 --out docs
.\.venv\Scripts\python.exe -m pytest tests\validation\test_convergence.py
```

Both are seeded and deterministic.
