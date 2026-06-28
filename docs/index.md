# BattleBot Sim documentation

Impact-damage simulator for combat robots: assign materials to a CAD model, run
an NHRL stress battery in a MuJoCo cage, and read damage heatmaps.

## Contents

- [Model assumptions & limitations](model_assumptions.md) — what the damage model
  does and does not capture, valid input ranges, and the comparative (not FEA)
  accuracy scope.
- [Uncertainty & convergence](uncertainty.md) — how trustworthy the failure
  verdict is: timestep convergence and sensitivity to the tuning constants.

## Configuration

All physics/damage tuning constants live on typed dataclasses in
[`battlebot_sim/config.py`](../battlebot_sim/config.py). The defaults reproduce
the historical behaviour; override a subset with a TOML file via
`battlebot_sim.config.load_config(path)`:

```toml
[damage]
sigma_patch_factor = 6.0

[sim]
timestep = 2.5e-4
```

## Developing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for dev setup, the native-library test
quirk, and how to verify the 3D viewport.
