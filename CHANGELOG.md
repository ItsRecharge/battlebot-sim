# Changelog

All notable changes to BattleBot Sim are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Central, typed configuration (`battlebot_sim/config.py`): every physics/damage
  tuning constant grouped into frozen dataclasses with documented defaults and an
  optional TOML overlay (`load_config`).
- Structured logging (`battlebot_sim/logging_setup.py`) with `--log-level` /
  `--log-file` CLI flags; previously-silent `except` guards now log with context.
- Boundary input validation (`battlebot_sim/validation.py`, `errors.py`):
  mesh, scale, run-parameter and mass checks that raise `ValidationError`.
- Golden-baseline regression test pinning the seeded self-test pipeline so
  behavior-preserving refactors can prove they changed no numbers.
- Tooling: ruff (lint + format), mypy, coverage, pre-commit, `.editorconfig`,
  GitHub Actions CI (Windows primary), `py.typed` marker.
- Project metadata: license, authors, URLs, classifiers; `LICENSE`,
  `CONTRIBUTING.md`, `docs/`.

### Changed
- Damage and brace models, the MuJoCo engine/MJCF builder, and the stress battery
  now source their tuning constants from `config.py` (defaults unchanged).

## [0.1.1] - 2026-06-26

### Fixed
- Brace load transfer now surfaces in the failure margin (was a silent no-op).

## [0.1.0] - 2026-06-26

### Added
- Initial release: STL/3MF/glTF import, material assignment, NHRL stress battery,
  MuJoCo physics, Hertzian damage model, and damage heatmaps.
