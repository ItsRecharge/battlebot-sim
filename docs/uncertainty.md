# Uncertainty & convergence

How much to trust the failure verdict. Two questions: is it converged (independent
of the numerical timestep), and how sensitive is it to the uncalibrated tuning
constants? Both are answered by reproducible artifacts in the repo.

## Timestep convergence — converged

`tests/validation/test_convergence.py` runs the seeded battery at two timesteps and
compares the worst failure margin:

| timestep | worst failure margin |
|----------|----------------------|
| 1.0e-3 s | 1.236084e+01 |
| 5.0e-4 s (default) | 1.236084e+01 |
| 2.5e-4 s | 1.236084e+01 |

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
| `damage.contact_vm_factor` | ~0.50 | **dominant** (scales the contact von Mises directly) |
| `damage.weapon_tip_radius_m` | ~0.32 | **dominant** (`p0 ∝ R_eff^(-2/3)`) |
| `damage.opponent_modulus_pa` | ~0.09 | moderate (`p0 ∝ E*^(2/3)`) |
| `damage.poisson` | ~0.07 | moderate |
| `damage.sigma_*` (all heatmap-spread), `brace.*` | **0.000** | negligible — the verdict is now decoupled from them |

The verdict now responds **only** to physically meaningful contact-mechanics
constants; the previously-dominant "appearance-only" sigma constants move it by
exactly zero (see the `0.000` rows in `docs/sensitivity.md`). That is the headline
result of the absolute-model rework. The single highest-value calibration target is
now `weapon_tip_radius_m` (the assumed opponent striker sharpness).

### Resolved: the verdict no longer depends on the spread constants

This used to be the model's biggest fidelity risk. The verdict was read off the
Gaussian-painted per-face field, so on a coarse mesh — where no face centroid sits
exactly at the contact point — a tighter sigma recorded `p0 · exp(−d²/2σ²) < p0` and
the "appearance-only" constants (`sigma_patch_factor`, `stress_sigma_frac`) dominated
the margin.

The absolute model removed the coupling entirely: the verdict is now computed from
the **un-attenuated governing stress at the true contact point**, independent of any
sigma. `tests/validation/test_true_peak.py` pins this — a 4× sigma swing moves the
rendered heatmap but leaves `part_max_margin` bit-identical. The sigmas now affect
**only** the picture, exactly as `docs/model_assumptions.md` claims — the sensitivity
table above confirms a `0.000` swing for every one of them.

## Reproducing

```powershell
.\.venv\Scripts\python.exe scripts\sensitivity_sweep.py --pct 0.25 --out docs
.\.venv\Scripts\python.exe -m pytest tests\validation\test_convergence.py
```

Both are seeded and deterministic.
