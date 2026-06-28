# Contributing to BattleBot Sim

Thanks for helping improve BattleBot Sim. This guide covers the dev setup, the
test quirks specific to this project's native-library stack, and the quality
gates CI enforces.

## Dev setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\pre-commit install        # optional but recommended
```

## Quality gates

CI runs these on every push; run them locally first:

```powershell
.\.venv\Scripts\ruff check .
.\.venv\Scripts\mypy battlebot_sim
```

`ruff check . --fix` applies most lint fixes automatically. Formatting is being
migrated to `ruff format` incrementally — the pre-commit hook formats the files
you touch, so run `ruff format <changed files>` before committing rather than
reformatting the whole tree at once.

## Running tests

Set the offscreen Qt platform so the headless tests don't need a display:

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
.\.venv\Scripts\python.exe -m pytest
```

### The native-library crash (important)

Loading NumPy, SciPy, VTK, MuJoCo and Pillow into **one** process can
intermittently crash the interpreter (`0xC0000005` / `0xC0000409`), usually on
shutdown *after* the tests have already passed. This is a DLL load/teardown race,
not a logic bug. To run the whole suite reliably, run each heavy test file in its
own process:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_sim.py
.\.venv\Scripts\python.exe -m pytest tests\test_ui_smoke.py
# … or use the helper that loops over every test file:
.\scripts\run_isolated_tests.ps1
```

CI uses the same per-file isolation. Tests that load the heavy native stack are
marked `@pytest.mark.native_isolated`; pure-logic tests run together in one fast
process via `pytest -m "not native_isolated"`.

## Verifying the 3D viewport

`battlebot_sim/ui/viewport.py` needs a **real** OpenGL context — offscreen VTK
does not fully exercise the embedded `QtInteractor`, so it is excluded from the
coverage gate. Verify viewport/material/heatmap changes by driving
`MainWindow` / `BotViewport` from a short script on a real display, not headless.

## Building the standalone .exe

```powershell
.\.venv\Scripts\pyinstaller build\battlebot_sim.spec --noconfirm
```

The build is occasionally flaky (the same native-library race). Build from a
fresh shell and retry a couple of times if a compile segfaults.

## Commit style

Conventional Commits (`feat:`, `fix:`, `refactor:`, `docs:`, `test:`). Add a
`CHANGELOG.md` entry under **Unreleased** for user-visible changes.
