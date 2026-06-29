"""Entry point: launch the Combat Robot Stress-Test Gauntlet desktop app.

    python -m gauntlet                         # normal GUI
    python -m gauntlet --selftest              # headless smoke test, exit
    python -m gauntlet --log-level DEBUG        # verbose engineering log
    python -m gauntlet --log-file run.log       # also tee logs to a file

--selftest is used to smoke-test the packaged .exe: it verifies imports, the Qt
window, the VTK viewport, and bundled-data resolution all work, then quits.

Made by Neel Bansal.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _sample_bot_path() -> Path:
    """Locate the bundled sample STL in both dev and frozen (.exe) runs."""
    from gauntlet.mesh.segment import sample_bot_path
    return Path(sample_bot_path())


def app_icon_path() -> str:
    """Absolute path to the platform app icon, resolved in dev and frozen runs.

    Mirrors :func:`gauntlet.mesh.segment.sample_bot_path`: under PyInstaller the
    bundled ``assets/`` lives in ``sys._MEIPASS``; in a source checkout it sits
    at the project root. ``.icns`` on macOS, ``.ico`` everywhere else.
    """
    name = "gauntlet.icns" if sys.platform == "darwin" else "gauntlet.ico"
    if getattr(sys, "frozen", False):
        return str(Path(sys._MEIPASS) / "assets" / name)   # PyInstaller unpack dir
    return str(Path(__file__).resolve().parents[1] / "build" / name)  # source checkout


def _run_selftest() -> int:
    """Headless validation of the frozen bundle's hard parts.

    Verifies, without opening a GUI window (which can't be automated cleanly):
    bundled-data resolution, trimesh STL load + segmentation, the MuJoCo native
    engine, and VTK *off-screen* rendering. Writes the outcome to a log file
    next to the temp dir (a windowed .exe has no console stderr) and returns a
    clear exit code.
    """
    import os
    import tempfile
    import traceback

    log_path = os.path.join(tempfile.gettempdir(), "gauntlet_selftest.log")

    def log(msg: str) -> None:
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(msg + "\n")

    try:
        open(log_path, "w").close()
        sample = _sample_bot_path()
        log(f"sample: {sample} exists={sample.exists()}")
        if not sample.exists():
            return 2

        from gauntlet import viz
        from gauntlet.arena.nhrl import build_arena
        from gauntlet.damage.model import compute_damage
        from gauntlet.materials.assign import NHRL_CLASSES
        from gauntlet.materials.library import load_default_library
        from gauntlet.mesh.segment import SAMPLE_SCALE_TO_M, load_bot
        from gauntlet.sim.battery import StressBattery, run_battery
        from gauntlet.sim.engine import SimEngine

        library = load_default_library()
        bot = load_bot(str(sample), scale_to_m=SAMPLE_SCALE_TO_M)
        bot.assign_material_to_all(library.get("Aluminum 6061-T6"))
        log(f"loaded bot: {len(bot.parts)} parts, mass={bot.total_mass():.3f} kg")
        if len(bot.parts) == 0:
            return 3

        cls = NHRL_CLASSES["3lb"]
        arena = build_arena(cls)
        engine = SimEngine(arena, bot)            # exercises MuJoCo native lib
        trace = run_battery(engine, StressBattery(arena, cls), fps=30)
        result = compute_damage(trace, bot, arena, library)
        log(f"sim OK: {trace.total_contacts()} contacts")

        # VTK off-screen render needs a GL context. Headless CI runners
        # (e.g. GitHub windows-latest) have none, so the render access-violates
        # there even though the freeze is sound. Skip just this step when asked;
        # everything above (imports, bundled data, trimesh, the MuJoCo native
        # engine) has already been exercised. Verify the render locally / on a
        # real desktop, where it runs by default.
        if os.environ.get("GAUNTLET_SELFTEST_SKIP_RENDER"):
            log("render SKIPPED (GAUNTLET_SELFTEST_SKIP_RENDER set — headless CI)")
        else:
            png = os.path.join(tempfile.gettempdir(), "gauntlet_selftest.png")
            viz.render_heatmap_png(bot, result, "failure", png)   # VTK off-screen
            log(f"render OK: {png} ({os.path.getsize(png)} bytes)")

        log("SELFTEST OK")
        return 0
    except Exception:
        log("SELFTEST FAILED:\n" + traceback.format_exc())
        return 1


def _parse_log_args(argv: list[str]) -> tuple[str, str | None]:
    """Pull --log-level / --log-file out of argv without disturbing Qt's args."""
    def _opt(flag: str, default):
        if flag in argv and argv.index(flag) + 1 < len(argv):
            return argv[argv.index(flag) + 1]
        return default

    return _opt("--log-level", "INFO"), _opt("--log-file", None)


def main() -> int:
    from gauntlet._bootstrap import preload_native_libraries
    from gauntlet.logging_setup import configure_logging

    level, logfile = _parse_log_args(sys.argv)
    configure_logging(level=level, logfile=logfile)

    if "--selftest" in sys.argv:
        preload_native_libraries()  # headless path: pin load order, no GUI/splash
        return _run_selftest()

    # GUI path: bring up Qt and a splash *before* the slow native preload, so the
    # loading bar covers it (Qt isn't in the pinned native set, so showing it
    # first doesn't disturb _bootstrap's ordering).
    from PySide6 import QtGui, QtWidgets

    from gauntlet.ui.splash import StartupSplash

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    if sys.platform == "darwin":
        # Force Fusion on macOS: the native Aqua push-button bezel clips longer
        # labels ("Run stress battery"). Windows keeps its native style.
        app.setStyle("Fusion")
    icon = QtGui.QIcon(app_icon_path())
    app.setWindowIcon(icon)

    splash = StartupSplash(icon)
    splash.show()
    app.processEvents()

    splash.step(15, "Loading physics engine…")
    preload_native_libraries()      # pin native-lib load order before anything else

    splash.step(70, "Building interface…")
    from gauntlet.ui.main_window import MainWindow

    window = MainWindow()
    window.setWindowIcon(icon)
    splash.step(100, "Ready")
    window.show()
    splash.finish(window)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
