# BattleBot Damage Simulator

A Windows desktop app that predicts **where a combat robot will take damage** before
you cut any metal. Load your bot's model (STL, 3MF, or glTF), assign real materials to
its parts, pick an NHRL weight class, and the app runs your bot through an automated
"flinging" stress battery inside a physics-simulated test cage — then shows you a replay
and two damage heatmaps.

> **Hybrid-accuracy tool.** A real rigid-body physics engine (MuJoCo) handles the
> flinging and collisions; a simplified analytical model (Hertzian contact stress +
> brace load-sharing) estimates damage. Results are **comparative** — great for
> "which design survives better" — not certification-grade FEA.

## What it does

1. **Load a model** → a named multi-body file (3MF / glTF) keeps each body as its own
   part with its CAD name; an STL is auto-segmented into connected solid parts named
   `part_0`, `part_1`… (chassis, armor, wedge, brace…). Or hit **Load sample bot** to try
   the whole pipeline instantly.
2. **Assign materials** per part from a library (aluminums, steels, titanium, polycarbonate,
   UHMW, HDPE, TPU, carbon fiber — all editable). **Click parts in the 3D view to select
   them** (Ctrl/Shift to multi-select), then assign a material to the whole selection at
   once; each part is tinted by its material so the table doubles as a legend. Mass, centre
   of mass, and inertia are computed and **validated against the NHRL weight class**
   (3 / 12 / 30 lb).
3. **Tag braces** so the model accounts for structural load-sharing.
4. **Run the stress battery** — drops, wall slams at several speeds/angles, a tumble, and
   opponent-weapon strikes, all scaled to the weight class, simulated in MuJoCo inside the
   class-sized cage.
5. **See the results**:
   - a **replay** of the bot being flung around the cage,
   - an **Impact-Energy** heatmap (what took the most punishment),
   - a **Failure-Margin** heatmap (peak stress ÷ yield; red ≥ 1 means it would yield),
   - an exportable **report** (two PNG heatmaps + a markdown summary with a per-part table
     and worst impacts).

## Input formats

| Format | Part segmentation | Recommended |
|--------|-------------------|-------------|
| **3MF** | Each named body → one part, keeps the CAD name | ✅ best for multi-part bots |
| **glTF / GLB** | Each named node → one part, keeps the name | ✅ |
| **STL** | Auto-split by connected components, generic `part_N` names | ok (no names) |
| **OBJ** | Merged on import → connected-component split | ok (no names) |

Export **3MF** (or glTF) from your CAD — Fusion 360, SolidWorks, Onshape, FreeCAD and
Blender all do — so every body arrives as a distinct, named part you can pick and assign
individually. STEP/IGES/Parasolid/JT/ACIS need a CAD kernel this app doesn't bundle;
convert them to 3MF/glTF first. Units aren't carried by STL (and vary by exporter), so set
the **Model units** dropdown to match your file.

## Run from source

Requires Python 3.10–3.12 on Windows.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m battlebot_sim
```

A sample bot is included at `data/sample_bots/wedge_bot.stl` — or just click **Load
sample bot** in the app.

## Build the standalone .exe

```powershell
.\.venv\Scripts\python.exe -m pip install pyinstaller
.\.venv\Scripts\pyinstaller build\battlebot_sim.spec --noconfirm
```

This produces a portable one-folder app at `dist\BattleBotSim\` — run
`dist\BattleBotSim\BattleBotSim.exe`. Zip the `BattleBotSim` folder to share it.
(One-folder rather than one-file because VTK is large; one-file would unpack hundreds
of MB to a temp dir on every launch.)

## Tests

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
.\.venv\Scripts\python.exe -m pytest -q
```

All 45 tests pass. See the stability note below if a run aborts.

## Known issue: intermittent native-DLL crash on this machine

When NumPy, SciPy, VTK, MuJoCo and Pillow are all loaded into a *single* process,
this environment intermittently throws a native access violation (Windows
`0xC0000005`/`0xC0000409`) during library initialization — a DLL load-order race,
not a bug in this code. It shows up two ways:

- **Running the whole test suite at once** can abort partway. Workaround: run each
  test file in its own process (each passes reliably in isolation):
  ```powershell
  Get-ChildItem tests\test_*.py | ForEach-Object { .\.venv\Scripts\python.exe -m pytest $_.FullName -q }
  ```
- **`pyinstaller` builds** crash during analysis ~half the time. Workaround: just
  retry — the build succeeds within a few attempts (the included build loop does
  this automatically). The Qt-module excludes in the spec already cut the rate.

The packaged app itself ran the full sim + render pipeline cleanly. If you hit a
rare crash launching the app, relaunch it.

## How the damage model works

| Quantity | Method |
|----------|--------|
| Mass / COM / inertia | Per-part volume × material density, aggregated (parallel-axis). |
| Flinging & collisions | MuJoCo rigid-body sim; bot = union of per-part convex hulls. |
| Impact energy field | Σ (normal force × closing speed × dt), spread over nearby faces (Gaussian). |
| Failure margin field | Hertzian peak contact pressure `p0 = (6·F·E*²/(π³·R²))^⅓` ÷ material yield. |
| Effective modulus E* | Series compliance of the bot material and the surface it hit (ν = 0.3). |
| Brace load-sharing | Adjacent-to-brace parts shed stress by `1/(1+k)`, `k = E·A/L` (normalized). |

### Known limitations
- Lumped per-part stress with an equivalent-sphere contact radius — order-of-magnitude.
- Brace sharing is a beam-style heuristic, not FEA.
- Collision shapes are per-part convex hulls (concave detail is approximated).
- NHRL cage dimensions are first-order approximations (mass limits are exact); the cage
  is parametric and easy to retune.

## Architecture

```
battlebot_sim/
  mesh/segment.py      STL/3MF/glTF load, named-body + connected-component segmentation, mass properties
  materials/           material library + NHRL classes + weight validation
  arena/nhrl.py        class-scaled cage geometry
  sim/                 MJCF builder, MuJoCo engine wrapper, stress battery, recorder
  damage/              contact→face mapping, energy & failure fields, brace sharing
  viz.py               PyVista rendering (shared by viewport + report)
  ui/                  PySide6 window, panels, embedded 3D viewport
  report/export.py     PNG heatmaps + markdown summary
```
