"""Main application window: wires the panels and viewport to the pipeline."""

from __future__ import annotations

import os

from PySide6 import QtCore, QtWidgets

from battlebot_sim.arena.nhrl import build_arena
from battlebot_sim.damage.braces import apply_brace_sharing
from battlebot_sim.damage.model import compute_damage
from battlebot_sim.materials.assign import NHRL_CLASSES, validate_weight_class
from battlebot_sim.materials.library import load_default_library
from battlebot_sim.mesh.segment import load_bot
from battlebot_sim.report.export import export_report
from battlebot_sim.sim.battery import StressBattery, run_battery
from battlebot_sim.sim.engine import SimEngine
from battlebot_sim.ui.panels import PartsPanel, ResultsPanel, SetupPanel
from battlebot_sim.ui.viewport import BotViewport


class SimWorker(QtCore.QObject):
    """Runs the battery + damage model off the UI thread."""

    finished = QtCore.Signal(object, object)   # (trace, result)
    failed = QtCore.Signal(str)

    def __init__(self, bot, arena, weight_class, library, n_trials=1):
        super().__init__()
        self.bot, self.arena = bot, arena
        self.weight_class, self.library = weight_class, library
        self.n_trials = n_trials

    @QtCore.Slot()
    def run(self) -> None:
        try:
            engine = SimEngine(self.arena, self.bot)
            battery = StressBattery(self.arena, self.weight_class,
                                    n_trials=self.n_trials)
            # 60 fps capture gives a smooth, realistic flinging replay.
            trace = run_battery(engine, battery, fps=60)
            result = compute_damage(trace, self.bot, self.arena, self.library)
            result = apply_brace_sharing(result, self.bot)
            self.finished.emit(trace, result)
        except Exception as exc:        # surface failures to the UI, don't swallow
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BattleBot Damage Simulator")
        self.resize(1280, 820)

        self.library = load_default_library()
        self.bot = None
        self.arena = None
        self.trace = None
        self.result = None
        self._thread = None
        self._worker = None

        # --- widgets ---
        self.viewport = BotViewport(self)
        self.setup_panel = SetupPanel()
        self.parts_panel = PartsPanel(self.library)
        self.results_panel = ResultsPanel()

        side = QtWidgets.QWidget()
        side_layout = QtWidgets.QVBoxLayout(side)
        side_layout.addWidget(self.setup_panel)
        side_layout.addWidget(self.parts_panel)
        side_layout.addWidget(self.results_panel)
        side.setMaximumWidth(460)

        splitter = QtWidgets.QSplitter()
        splitter.addWidget(self.viewport)
        splitter.addWidget(side)
        splitter.setStretchFactor(0, 1)
        self.setCentralWidget(splitter)
        self.statusBar().showMessage("Load an STL to begin.")

        # --- replay timer (~60 fps to match the 60 fps capture) ---
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._advance_frame)

        # --- signals ---
        self.setup_panel.load_requested.connect(self.load_stl)
        self.setup_panel.run_requested.connect(self.run_simulation)
        self.setup_panel.cage_check.toggled.connect(self._on_cage_toggled)
        self.parts_panel.changed.connect(self._update_weight_check)
        self.parts_panel.changed.connect(self.viewport.refresh_materials)
        self.parts_panel.selection_changed.connect(self.viewport.set_selection)
        self.viewport.part_clicked.connect(self._on_part_picked)
        self.results_panel.mode_changed.connect(self._on_mode)
        self.results_panel.frame_changed.connect(self.viewport.show_frame)
        self.results_panel.play_toggled.connect(self._on_play)
        self.results_panel.export_requested.connect(self.export)

    # ---- load / setup ----------------------------------------------------
    @QtCore.Slot(str, float)
    def load_stl(self, path: str, scale_to_m: float) -> None:
        """Load a model file and install it. The *entire* load + setup runs under
        one guard: a failure anywhere (parse, segmentation, viewport, panels)
        surfaces a dialog and resets to a clean state, instead of silently
        leaving the Run button disabled against a half-loaded bot."""
        try:
            bot = load_bot(path, scale_to_m=scale_to_m)
            self._install_bot(bot, path)
        except Exception as exc:
            self._fail_load(path, exc)

    def _install_bot(self, bot, path: str) -> None:
        """Place a freshly loaded bot into the viewport and panels, then enable
        Run. Any exception here propagates to load_stl's guard."""
        self.bot = bot
        self.result = self.trace = None
        self.results_panel.set_enabled(False)
        self.results_panel.solid_btn.setChecked(True)

        wc = NHRL_CLASSES[self.setup_panel.current_class_key()]
        self.arena = build_arena(wc)
        self.viewport.clear()
        # Show the bot, then populate the table: the table's one-shot `changed`
        # repaints the bot in its material colours. The arena cage stays hidden
        # during setup (unless "show cage" is ticked) so parts are easy to click.
        self.viewport.set_bot(self.bot)
        self.parts_panel.set_bot(self.bot)
        if self.setup_panel.cage_check.isChecked():
            self.viewport.show_arena(self.arena)
        else:
            self.viewport.hide_arena()
        self._update_weight_check()
        self.setup_panel.run_btn.setEnabled(True)
        self.statusBar().showMessage(
            f"Loaded {os.path.basename(path)} — {len(self.bot.parts)} parts.")

    def _fail_load(self, path: str, exc: Exception) -> None:
        """Report a load/setup failure and reset to a safe state so Run is never
        left enabled against a broken bot — nor silently disabled with no reason.
        The full traceback goes in the dialog's details to help diagnose the
        offending mesh (degenerate parts, non-finite geometry, etc.)."""
        import traceback
        self.bot = self.arena = self.result = self.trace = None
        self.setup_panel.run_btn.setEnabled(False)
        self.results_panel.set_enabled(False)
        self.statusBar().showMessage("Load failed.")
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Icon.Critical)
        box.setWindowTitle("Load failed")
        box.setText(f"Could not load {os.path.basename(path)}.\n\n"
                    f"{type(exc).__name__}: {exc}")
        box.setDetailedText(traceback.format_exc())
        box.exec()

    def _update_weight_check(self) -> None:
        if self.bot is None:
            return
        wc = NHRL_CLASSES[self.setup_panel.current_class_key()]
        check = validate_weight_class(self.bot.total_mass(), wc)
        self.parts_panel.update_weight_check(check.message, check.ok)

    # ---- part selection (3D <-> table) ----------------------------------
    @QtCore.Slot(int, bool)
    def _on_part_picked(self, idx: int, additive: bool) -> None:
        """A 3D click selects a part: replace the selection, or toggle it when
        Ctrl/Shift is held. Mirror the result to both the table and the view."""
        current = set(self.parts_panel.selected_indices())
        if additive:
            current ^= {idx}
        else:
            current = {idx}
        sel = sorted(current)
        self.parts_panel.select_parts(sel)
        self.viewport.set_selection(sel)
        # A plain single-part click opens that part's material dropdown so it can
        # be changed straight from the 3D view.
        if not additive and len(sel) == 1:
            self.parts_panel.focus_material_editor(sel[0])

    @QtCore.Slot(bool)
    def _on_cage_toggled(self, on: bool) -> None:
        if self.bot is None or self.arena is None:
            return
        if on:
            self.viewport.show_arena(self.arena)
        else:
            self.viewport.hide_arena()

    # ---- run simulation --------------------------------------------------
    @QtCore.Slot(str)
    def run_simulation(self, class_key: str) -> None:
        if self.bot is None:
            return
        wc = NHRL_CLASSES[class_key]
        n_trials = self.setup_panel.current_trials()
        self.arena = build_arena(wc)
        self.viewport.show_arena(self.arena)
        self.setup_panel.run_btn.setEnabled(False)
        self.setup_panel.progress.show()
        self.statusBar().showMessage(
            f"Running stress battery ({n_trials} trial(s))…")

        self._thread = QtCore.QThread(self)
        self._worker = SimWorker(self.bot, self.arena, wc, self.library, n_trials)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_sim_done)
        self._worker.failed.connect(self._on_sim_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.start()

    @QtCore.Slot(object, object)
    def _on_sim_done(self, trace, result) -> None:
        self.trace, self.result = trace, result
        self.viewport.set_trace(trace)
        self.viewport.set_result(result)
        self.results_panel.configure_slider(self.viewport.n_frames)
        self.results_panel.set_enabled(True)
        self.setup_panel.progress.hide()
        self.setup_panel.run_btn.setEnabled(True)
        self._write_summary()
        self.statusBar().showMessage(
            f"Done: {trace.total_contacts()} contacts, {self.viewport.n_frames} frames.")
        # Immediately play the flinging replay once so the user sees the bot get
        # thrown around the cage without having to hit Play. Checking the button
        # drives the existing play path (solid view + replay timer).
        if self.viewport.n_frames:
            self.results_panel.solid_btn.setChecked(True)
            self.results_panel.slider.setValue(0)
            self.results_panel.play_btn.setChecked(True)

    @QtCore.Slot(str)
    def _on_sim_failed(self, message: str) -> None:
        self.setup_panel.progress.hide()
        self.setup_panel.run_btn.setEnabled(True)
        QtWidgets.QMessageBox.critical(self, "Simulation failed", message)
        self.statusBar().showMessage("Simulation failed.")

    def _write_summary(self) -> None:
        if self.result is None:
            return
        failing = self.result.parts_that_fail()
        lines = []
        if failing:
            names = ", ".join(self.bot.parts[i].name for i in failing)
            lines.append(f"⚠️ {len(failing)} part(s) predicted to yield: {names}")
        else:
            lines.append("✅ No part exceeded its material yield.")
        lines.append("")
        for p in self.bot.parts:
            m = self.result.part_max_margin.get(p.index, 0.0)
            lines.append(f"{p.name}: max margin {m:.2f}"
                         + ("  FAIL" if m >= 1.0 else ""))
        self.results_panel.summary.setPlainText("\n".join(lines))

    # ---- results interactions -------------------------------------------
    @QtCore.Slot(str)
    def _on_mode(self, mode: str) -> None:
        self.viewport.show_heatmap(mode)

    @QtCore.Slot(bool)
    def _on_play(self, playing: bool) -> None:
        if playing and self.viewport.n_frames:
            # Replay shows motion on the plain surface.
            self.results_panel.solid_btn.setChecked(True)
            # If we're parked at the end of a previous play-through, rewind first.
            if self.results_panel.slider.value() >= self.viewport.n_frames - 1:
                self.results_panel.slider.setValue(0)
            self._timer.start()
        else:
            self._timer.stop()

    def _advance_frame(self) -> None:
        cur = self.results_panel.slider.value()
        if cur >= self.viewport.n_frames - 1:
            # Reached the end: play through once, then stop (no looping).
            self._timer.stop()
            self.results_panel.play_btn.setChecked(False)
            return
        self.results_panel.slider.setValue(cur + 1)

    # ---- export ----------------------------------------------------------
    @QtCore.Slot()
    def export(self) -> None:
        if self.result is None:
            return
        out_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose output folder")
        if not out_dir:
            return
        wc = NHRL_CLASSES[self.setup_panel.current_class_key()]
        try:
            paths = export_report(self.bot, self.result, wc, out_dir, trace=self.trace)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Export failed", str(exc))
            return
        QtWidgets.QMessageBox.information(
            self, "Report exported", f"Wrote:\n{paths['report']}")
