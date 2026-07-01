# Combat Robot Stress-Test Gauntlet

> **Not FEA, not certified engineering analysis.** This is a quick sanity-check tool:
> a general guide to help you find the likely weak points on a combat robot before you
> build it. Treat the numbers as a rough hand-calc, not a guarantee that a part will
> survive a real fight. See [docs/model_assumptions.md](docs/model_assumptions.md)
> before you trust any single number.

A Windows desktop app that estimates **where a combat robot will take damage** before
you cut any metal. Load your bot's model (STL, 3MF, or glTF), assign materials to its
parts, pick an NHRL weight class, and the app throws the bot through an automated
stress battery inside a physics-simulated test cage. You get a live fly-around, four
metric-vs-time graphs, and two damage heatmaps.

![Failure-margin heatmap of the sample bot, with hotspots on the wheel hubs and corners and a labelled YIELD key](docs/img/heatmap_failure.png)

> **How the model works, briefly.** A rigid-body physics engine (MuJoCo) handles the
> throwing and collisions. An analytic mechanics-of-materials model then reads the
> contact forces into a per-part failure verdict: subsurface von-Mises contact yield
> (first yield at p0 ≈ 1.6·yield), a combined-curvature contact that treats the
> opponent weapon as a sharp striker, and per-part beam/plate bending and membrane
> stress with brace load-sharing. It is hand-calc grade, not FEA.

## The app has three tabs

- **Setup** defines the bot: load a model, set its units and weight class, and assign
  materials and brace flags to each part.
- **Simulate** runs the automated stress battery and shows the live fly-around, the
  metric graphs, and the damage heatmaps.
- **Freeplay** lets you draw your own impacts: click a point on the bot, drag to aim a
  strike, add a few, then run them to see the resulting damage.

## What it does

1. **Load a model.** A named multi-body file (3MF / glTF) keeps each body as its own
   part with its CAD name. An STL is auto-segmented into connected solid parts named
   `part_0`, `part_1`, and so on (chassis, armor, wedge, brace). Or click **Load sample
   bot** to try the whole pipeline right away.
2. **Assign materials** per part from a library (aluminums, steels, titanium,
   polycarbonate, UHMW, HDPE, TPU, carbon fiber, all editable). Click parts in the 3D
   view to select them (Ctrl/Shift to multi-select), then assign a material to the whole
   selection at once. Each part is tinted by its material, so the table doubles as a
   legend. Mass, centre of mass, and inertia are computed and validated against the
   NHRL weight class (3 / 12 / 30 lb).
3. **Tag braces** so the model accounts for structural load-sharing.
4. **Run the stress battery.** Drops, wall slams at several speeds and angles, a tumble,
   and opponent-weapon strikes, all scaled to the weight class and simulated in MuJoCo
   inside the class-sized cage. It runs live: watch the bot fly around the cage in real
   time (with a playback-speed control and a Stop button) while four graphs fill in as
   the impacts happen: peak contact force, cumulative impact energy, worst failure
   margin, and bot speed / hit-rate.

   ![Control-room dashboard: the cage fly-around and live graphs on the left, a bot-only turntable in the middle, controls on the right](docs/img/dashboard.png)
   <!-- live-UI shot: capture the running app (Win+Shift+S) and overwrite docs/img/dashboard.png -->

5. **See the results:**
   - the live fly-around in the cage, with a slider to re-scrub it afterwards,
   - a separate bot-only view that auto-rotates the finished bot on a turntable,
   - an Impact-Energy heatmap and a Failure-Margin heatmap (peak stress / yield; 1 or
     above means it would yield), drawn as a smooth gradient of hotspots with a labelled
     key. The damage is spread from the physics-engine contacts, not painted onto single
     triangles.
   - an exportable report (two PNG heatmaps plus a markdown summary with a per-part table
     and the worst impacts).

| Impact energy (log scale) | Failure margin (stress / yield) |
|---|---|
| ![Impact-energy heatmap](docs/img/heatmap_energy.png) | ![Failure-margin heatmap](docs/img/heatmap_failure.png) |

<!-- live-UI shots to capture and drop in: -->
<!-- ![Live metric graphs](docs/img/live_graphs.png) -->
<!-- ![Bot-only turntable](docs/img/turntable.png) -->

## Under the hood

Most of the work happens out of sight. Here is what each piece does and how it is
checked. Each piece is independently tested.

**The failure verdict is absolute mechanics-of-materials, not a relative score.** For
every part, at the contact point, the model computes:
- **Subsurface von-Mises contact yield.** Hertzian contact puts the peak shear below the
  surface, so first yield is at `σ_vm,max = 0.62·p0`, i.e. `p0 ≈ 1.6·σ_yield`, not at
  the surface pressure.
- **Combined-curvature contact radius.** The opponent weapon is modelled as a sharp
  striker, so a thin spinner tooth concentrates load far more than a flat slam.
- **PCA-oriented beam/plate section** with bending (`σ = 6·b·F·L / w·t²`) and membrane
  (`F/A`) stress. The model takes the governing stress and flags a fracture when it
  reaches ultimate. Bending only applies to genuinely slender members, so a stubby block
  cannot report impossible beam stress.

This margin is decoupled from the heatmap-spread constants: change the cosmetic smoothing
and the verdict does not move (see [docs/sensitivity.md](docs/sensitivity.md)).

**Braces share load.** A part flagged as a brace (auto-detected on import for elongated,
stiff, load-bridging members, and user-overridable) relieves the bending and membrane
stress of the parts it bridges, and the governing margin is recomputed, rather than
dumping a synthetic extra load onto the verdict.

**The heatmaps are physics-grounded.** Each MuJoCo contact is spread as a Gaussian sized
from its Hertzian contact patch and accumulated into smooth hotspots, so colour reflects
where energy actually concentrated rather than which triangle happened to be hit.

**The live view matches the analysis.** The battery is a generator: the live fly-around
drains the same stream that produces the offline trace, so what you watch is the data
that gets analysed. The worker runs on a background thread, paced to wall-clock with a
~30 Hz render throttle and a 0.25-4x live-speed control.

**Reproducible and checked.**
- **Seeded trials.** The speed/angle envelope is seeded, so a run is repeatable.
- **Golden-baseline regression** pins the whole pipeline's numbers against drift across
  refactors.
- A **validation suite** checks the physics against closed form: Hertzian peak pressure,
  energy conservation, live brace transfer, and timestep convergence (the verdict is
  converged at the default `5e-4 s` step). Findings are written up in
  [docs/uncertainty.md](docs/uncertainty.md) and [docs/sensitivity.md](docs/sensitivity.md).

**Handles messy real-world CAD.** On import the mesh is vertex-welded to stitch
triangle-soup seams, the part count is capped (64) by fusing the smallest fragments, and
coplanar hulls are inflated so MuJoCo accepts flat plates and armour instead of rejecting
them.

**Tunable.** Every physics/damage constant lives on a frozen `config.py` dataclass with a
documented default and an optional TOML overlay, so there are no magic numbers buried in
the code.

## Input formats

| Format | Part segmentation | Recommended |
|--------|-------------------|-------------|
| **3MF** | Each named body becomes one part, keeps the CAD name | Best for multi-part bots |
| **glTF / GLB** | Each named node becomes one part, keeps the name | Yes |
| **STL** | Auto-split by connected components, generic `part_N` names | OK (no names) |
| **OBJ** | Merged on import, then connected-component split | OK (no names) |

Export 3MF (or glTF) from your CAD. Fusion 360, SolidWorks, Onshape, FreeCAD, and Blender
all do, so every body arrives as a distinct, named part you can pick and assign
individually. STEP/IGES/Parasolid/JT/ACIS need a CAD kernel this app does not bundle;
convert them to 3MF/glTF first. Units are not carried by STL (and vary by exporter), so
set the **Model units** dropdown to match your file.

## Run from source

Requires Python 3.10-3.12 on Windows.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m gauntlet
```

A sample bot is included at `data/sample_bots/bot_test_1.stl`, or just click **Load sample
bot** in the app.

## Build the standalone .exe

```powershell
.\.venv\Scripts\python.exe -m pip install pyinstaller
.\.venv\Scripts\pyinstaller build\gauntlet.spec --noconfirm
```

This produces a portable one-folder app at `dist\Gauntlet\`; run
`dist\Gauntlet\Gauntlet.exe`. Zip the `Gauntlet` folder to share it. (One-folder rather
than one-file because VTK is large; one-file would unpack hundreds of MB to a temp dir on
every launch.)

## Tests

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
.\.venv\Scripts\python.exe -m pytest -q
```

The full suite (including a `tests/validation/` set that checks the physics against closed
form) passes. See the stability note below if a run aborts.

## Known issue: intermittent native-DLL crash on this machine

When NumPy, SciPy, VTK, MuJoCo, and Pillow are all loaded into a single process, this
environment intermittently throws a native access violation (Windows
`0xC0000005`/`0xC0000409`) during library initialization. It is a DLL load-order race
rather than a bug in this code. It shows up two ways:

- **Running the whole test suite at once** can abort partway. Workaround: run each test
  file in its own process (each passes reliably in isolation). The included
  `scripts/run_isolated_tests.ps1` does exactly this and judges pass/fail from JUnit XML:
  ```powershell
  .\scripts\run_isolated_tests.ps1
  ```
- **`pyinstaller` builds** crash during analysis about half the time. Workaround: retry;
  the build succeeds within a few attempts. The Qt-module excludes in the spec already cut
  the rate.

The packaged app itself ran the full sim and render pipeline cleanly. If you hit a rare
crash launching the app, relaunch it.

## How the damage model works

| Quantity | Method |
|----------|--------|
| Mass / COM / inertia | Per-part volume × material density, aggregated (parallel-axis). |
| Throwing & collisions | MuJoCo rigid-body sim (Newton + elliptic cone + multiccd); bot = union of per-part convex hulls, per-material restitution. |
| Contact pressure | Hertzian peak `p0 = (6·F·E*²/(π³·R²))^⅓`; effective modulus `E*` is the series compliance of the bot material and the surface it hit (ν = 0.3). |
| Failure verdict | **Absolute**: subsurface von-Mises yield (`p0 ≈ 1.6·σ_yield`) vs. governing bending/membrane stress on the PCA-oriented section; margin = governing stress ÷ yield (≥ 1 yields), fracture flag at ultimate. |
| Impact-energy field | Σ (normal force × closing speed × dt), spread over nearby faces by a Hertzian-sized Gaussian. |
| Brace load-sharing | A brace relieves the bending/membrane of the members it bridges; the governing margin is recomputed. |

### Known limitations
- Lumped per-part stress with an equivalent contact radius: order-of-magnitude, not FEA.
- Brace sharing is a beam-style heuristic.
- Collision shapes are per-part convex hulls (concave detail is approximated).
- NHRL cage dimensions are first-order approximations (mass limits are exact); the cage is
  parametric and easy to retune.

## Architecture

```
gauntlet/
  mesh/segment.py      STL/3MF/glTF load, named-body + connected-component segmentation, mass properties
  materials/           material library + NHRL classes + weight validation
  arena/nhrl.py        class-scaled cage geometry
  sim/                 MJCF builder, MuJoCo engine wrapper, stress battery (streaming generator), single-strike helper, recorder
  damage/              contact->face mapping, structural verdict, energy & failure fields, brace sharing
  viz.py               PyVista rendering (shared by viewport + report)
  ui/                  PySide6 tabbed window (Setup / Simulate / Freeplay), panels, embedded 3D viewports, live charts
  report/              PNG heatmaps + markdown summary + shared verdict text
  config.py            frozen tuning dataclasses (+ optional TOML overlay)
```
