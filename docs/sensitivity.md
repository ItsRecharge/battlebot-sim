# Verdict sensitivity (+/- 25% per constant)

Base worst failure margin: `1.1830e+01`. Constants ranked by how much they
move the verdict (relative swing); the top ones are what to calibrate first.

| Constant | margin (low) | margin (high) | relative swing |
|----------|--------------|---------------|----------------|
| `damage.contact_vm_factor` | 8.8722e+00 | 1.4787e+01 | 0.500 |
| `damage.weapon_tip_radius_m` | 1.4124e+01 | 1.0340e+01 | 0.320 |
| `damage.opponent_modulus_pa` | 1.1201e+01 | 1.2252e+01 | 0.089 |
| `damage.poisson` | 1.1500e+01 | 1.2290e+01 | 0.067 |
| `damage.bending_bc_factor` | 1.1830e+01 | 1.1830e+01 | 0.000 |
| `damage.plate_flat_threshold` | 1.1830e+01 | 1.1830e+01 | 0.000 |
| `damage.sigma_patch_factor` | 1.1830e+01 | 1.1830e+01 | 0.000 |
| `damage.sigma_min_frac` | 1.1830e+01 | 1.1830e+01 | 0.000 |
| `damage.sigma_part_frac` | 1.1830e+01 | 1.1830e+01 | 0.000 |
| `damage.stress_sigma_frac` | 1.1830e+01 | 1.1830e+01 | 0.000 |
| `damage.kernel_radius_sigmas` | 1.1830e+01 | 1.1830e+01 | 0.000 |
| `brace.k_ref` | 1.1830e+01 | 1.1830e+01 | 0.000 |
| `brace.transfer` | 1.1830e+01 | 1.1830e+01 | 0.000 |
| `brace.adjacency_tol` | 1.1830e+01 | 1.1830e+01 | 0.000 |
