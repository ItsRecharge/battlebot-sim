"""Main application window: wires the panels and viewport to the pipeline."""

from __future__ import annotations

import os
import threading
import time

from PySide6 import QtCore, QtWidgets

from battlebot_sim.arena.nhrl import build_arena
from battlebot_sim.damage.braces import apply_brace_sharing
from battlebot_sim.damage.model import DamageAccumulator
from battlebot_sim.errors import ValidationError
from battlebot_sim.logging_setup import get_logger
from battlebot_sim.materials.assign import NHRL_CLASSES, validate_weight_class
from battlebot_sim.materials.library import load_default_library
from battlebot_sim.mesh.segment import load_bot
from battlebot_sim.report.export import export_report
from battlebot_sim.sim.battery import StressBattery, iter_battery
from battlebot_sim.sim.engine import SimEngine
from battlebot_sim.sim.recorder import SimTrace
from battlebot_sim.ui.charts import LiveCharts, MetricStreamer
from battlebot_sim.ui.pacing import pace_schedule
from battlebot_sim.ui.panels import PartsPanel, ResultsPanel, SetupPanel
from battlebot_sim.ui.viewport import BotOnlyView, BotViewport
from battlebot_sim.validation import validate_run_params

logger = get_logger(__name__)


class StreamWorker(QtCore.QObject):
    """Runs the battery off the UI thread and *streams* it live.

    The battery is driven one captured frame at a time via ``iter_battery``;
    contacts fold into a ``DamageAccumulator`` as they arrive, and the bot pose +
    progress are emitted as Qt signals (auto-queued to the UI thread). Emission is
    throttled to ~30 Hz and the loop is paced to wall-clock so the fly-around is
    watchable. ``cancel`` stops it within ~50 ms and still finalises the partial
    result. This object NEVER touches Qt widgets or VTK — only signals cross the
    thread boundary.
    """

    finished = QtCore.Signal(object, object)   # (trace, result) — same as before
    failed = QtCore.Signal(str)
    chunk = QtCore.Signal(object)              # StreamChunk (live pose + contacts)
    metrics = QtCore.Signal(object)            # MetricSample (live graphs)
    progressed = QtCore.Signal(int, int)       # (event_index, n_events)

    def __init__(self, bot, arena, weight_class, library,
                 n_trials=1, fps=60, speed=1.0):
        super().__init__()
        self.bot, self.arena = bot, arena
        self.weight_class, self.library = weight_class, library
        self.n_trials = n_trials
        self.fps = fps
        self._speed = max(0.05, float(speed))
        # threading.Event: written from the UI thread (cancel), read in the worker
        # loop. Its set/is_set are atomic, so no torn reads across the boundary.
        self._cancel = threading.Event()

    @QtCore.Slot(float)
    def set_speed(self, value: float) -> None:
        # Delivered via Qt's queued connection (set_speed is a Slot), so the
        # assignment runs on the worker thread — already serialised, no lock needed.
        self._speed = max(0.05, float(value))

    @QtCore.Slot()
    def cancel(self) -> None:
        self._cancel.set()

    @QtCore.Slot()
    def run(self) -> None:
        try:
            engine = SimEngine(self.arena, self.bot)
            battery = StressBattery(self.arena, self.weight_class,
                                    n_trials=self.n_trials)
            accum = DamageAccumulator(self.bot, self.arena, self.library)
            mstream = MetricStreamer(self.bot)
            dt = engine.timestep
            frame_period = 1.0 / self.fps
            emit_every = max(1, self.fps // 30)        # render ~30 Hz, not 60
            trace = SimTrace(dt=dt, n_parts=len(self.bot.parts))

            gen = iter_battery(engine, battery, fps=self.fps)
            next_wall = time.perf_counter()
            i = 0
            while True:
                if self._cancel.is_set():
                    try:                               # finalise the partial trace
                        stopped = gen.throw(GeneratorExit)
                    except StopIteration as stop:
                        stopped = stop.value
                    except GeneratorExit:
                        stopped = None
                    if stopped is not None:
                        trace = stopped
                    break
                try:
                    ch = next(gen)
                except StopIteration as stop:
                    if stop.value is not None:
                        trace = stop.value
                    break

                accum.ingest(ch.new_contacts, dt)
                sample = mstream.update(ch, accum)
                if i % emit_every == 0 or ch.sim_done:
                    self.chunk.emit(ch)
                    self.metrics.emit(sample)
                    self.progressed.emit(ch.event_index, ch.n_events)
                    mstream.flush()
                i += 1

                sleep, next_wall = pace_schedule(
                    next_wall, frame_period, self._speed, time.perf_counter())
                while sleep > 0.0 and not self._cancel.is_set():  # cancel lands fast
                    step = min(sleep, 0.05)
                    QtCore.QThread.msleep(int(step * 1000))
                    sleep -= step

            result = apply_brace_sharing(accum.finalize(), self.bot)
            self.finished.emit(trace, result)
        except Exception as exc:        # surface failures to the UI, don't swallow
            logger.exception("stream worker failed")   # full traceback to the log
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
        self._render_busy = False   # coalesce live pose updates (keep newest)
        self._pending = None

        # --- widgets ---
        self.viewport = BotViewport(self)
        self.setup_panel = SetupPanel()
        self.parts_panel = PartsPanel(self.library)
        self.results_panel = ResultsPanel()
        self.graphs = LiveCharts()            # live metric-vs-time strip charts

        # Second 3D view: just the bot, for the final damage turntable.
        self.bot_view = BotOnlyView(self)

        side = QtWidgets.QWidget()
        side_layout = QtWidgets.QVBoxLayout(side)
        side_layout.addWidget(self.setup_panel)
        side_layout.addWidget(self.parts_panel)
        side_layout.addWidget(self.results_panel)
        side.setMaximumWidth(460)

        # Control-room dashboard: the cage view + live graphs on the left, the
        # bot-only turntable in the centre, the controls on the right — all at
        # once so a battery run reads like a test bench.
        left = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        left.addWidget(self.viewport)
        left.addWidget(self.graphs)
        left.setStretchFactor(0, 3)
        left.setStretchFactor(1, 1)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(self.bot_view)
        splitter.addWidget(side)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setStretchFactor(2, 0)
        self.setCentralWidget(splitter)
        self.statusBar().showMessage("Load a model to begin.")

        # --- replay timer (~60 fps to match the 60 fps capture) ---
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._advance_frame)

        # --- signals ---
        self.setup_panel.load_requested.connect(self.load_stl)
        self.setup_panel.run_requested.connect(self.run_simulation)
        # Stop/speed are wired once here (not per-run) to a forwarder, so repeated
        # runs never stack connections onto soon-to-be-deleted workers.
        self.setup_panel.stop_requested.connect(self._on_stop_requested)
        self.setup_panel.speed_changed.connect(self._on_speed_changed)
        self.setup_panel.cage_check.toggled.connect(self._on_cage_toggled)
        self.parts_panel.changed.connect(self._update_weight_check)
        self.parts_panel.changed.connect(self.viewport.refresh_materials)
        self.parts_panel.changed.connect(self.bot_view.refresh_materials)
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
        self.bot_view.set_bot(self.bot)         # mirror into the bot-only view
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
        if self._thread is not None and self._thread.isRunning():
            return                              # already running: ignore re-clicks
        wc = NHRL_CLASSES[class_key]
        n_trials = self.setup_panel.current_trials()
        speed = self.setup_panel.current_speed()
        try:                                    # reject bad run settings up front
            n_trials, fps = validate_run_params(n_trials, 60)
        except ValidationError as exc:
            logger.warning("invalid run settings: %s", exc)
            QtWidgets.QMessageBox.critical(self, "Invalid run settings", str(exc))
            self.statusBar().showMessage("Invalid run settings.")
            return
        self.arena = build_arena(wc)
        self.viewport.show_arena(self.arena)
        self.viewport.begin_live()              # solid bot at rest, ready to fly
        self.bot_view.begin_live()              # bot-only view idles until results
        self.graphs.reset()                     # clear the live charts for a fresh run
        self.results_panel.set_enabled(False)
        self.results_panel.solid_btn.setChecked(True)
        self.setup_panel.show_running(True)
        self.setup_panel.progress.setRange(0, 1)
        self.setup_panel.progress.setValue(0)
        self._render_busy = False
        self._pending = None
        self.statusBar().showMessage(
            f"Running stress battery ({n_trials} trial(s)) — live…")

        self._thread = QtCore.QThread(self)
        self._worker = StreamWorker(self.bot, self.arena, wc, self.library,
                                    n_trials, fps=fps, speed=speed)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.chunk.connect(self._on_chunk)
        self._worker.metrics.connect(self._on_metrics)
        self._worker.progressed.connect(self._on_progress)
        self._worker.finished.connect(self._on_sim_done)
        self._worker.failed.connect(self._on_sim_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._on_thread_finished)
        self._thread.start()

    @QtCore.Slot()
    def _on_stop_requested(self) -> None:
        if self._worker is not None:
            self._worker.cancel()

    @QtCore.Slot(float)
    def _on_speed_changed(self, value: float) -> None:
        if self._worker is not None:
            self._worker.set_speed(value)

    @QtCore.Slot(object)
    def _on_chunk(self, chunk) -> None:
        """Render the latest live pose, coalescing if a render is in flight so a
        burst of frames never backs up (keep only the newest pose)."""
        if self._render_busy:
            self._pending = chunk
            return
        self._render_busy = True
        try:
            f = chunk.frame
            self.viewport.show_live_pose(f.pos, f.quat, f.event)
        finally:
            self._render_busy = False
        if self._pending is not None:
            # Defer the queued re-render past any event processing the VTK render
            # may have triggered, so it can't interleave with _on_sim_done.
            nxt, self._pending = self._pending, None
            QtCore.QTimer.singleShot(0, lambda c=nxt: self._on_chunk(c))

    @QtCore.Slot(object)
    def _on_metrics(self, sample) -> None:
        self.graphs.append(sample)

    @QtCore.Slot(int, int)
    def _on_progress(self, event_index: int, n_events: int) -> None:
        self.setup_panel.progress.setRange(0, max(1, n_events))
        self.setup_panel.progress.setValue(min(event_index + 1, n_events))

    @QtCore.Slot()
    def _on_thread_finished(self) -> None:
        self._thread = None
        self._worker = None

    @QtCore.Slot(object, object)
    def _on_sim_done(self, trace, result) -> None:
        # Cancelled before any data was collected: don't present an empty,
        # misleading "result" — just return to idle.
        if trace.total_contacts() == 0 and not trace.frames:
            self.setup_panel.show_running(False)
            self.statusBar().showMessage("Cancelled — no data collected.")
            return
        self.trace, self.result = trace, result
        self.viewport.set_trace(trace)
        self.viewport.set_result(result)
        self.results_panel.configure_slider(self.viewport.n_frames)
        self.results_panel.set_enabled(True)
        self.setup_panel.show_running(False)
        self._write_summary()
        self.statusBar().showMessage(
            f"Done: {trace.total_contacts()} contacts, {self.viewport.n_frames} frames.")
        # The fly-around already played live; show the failure heatmap as the
        # headline result (checking the radio emits mode_changed -> show_heatmap).
        # The slider + Play replay remain to re-scrub the completed trace.
        self.bot_view.show_final(result, "failure")   # final-heatmap turntable
        if self.results_panel.failure_btn.isChecked():
            self._on_mode("failure")        # already checked: trigger explicitly
        else:
            self.results_panel.failure_btn.setChecked(True)

    @QtCore.Slot(str)
    def _on_sim_failed(self, message: str) -> None:
        self.setup_panel.show_running(False)
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
        self.bot_view.set_mode(mode)            # keep the bot-only view in sync

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

    # ---- teardown --------------------------------------------------------
    def closeEvent(self, event) -> None:
        """Stop the animation timers and cleanly drain a running battery so no
        signal or timer fires into a half-destroyed window / VTK render window."""
        self._timer.stop()
        self.bot_view._spin.stop()
        worker, thread = self._worker, self._thread
        if worker is not None:
            # Detach the UI slots (keep finished->thread.quit) then cancel + wait,
            # so a late finish can't repaint a closing window.
            for sig, slot in (
                (worker.chunk, self._on_chunk),
                (worker.metrics, self._on_metrics),
                (worker.progressed, self._on_progress),
                (worker.finished, self._on_sim_done),
                (worker.failed, self._on_sim_failed),
            ):
                try:
                    sig.disconnect(slot)
                except (RuntimeError, TypeError):
                    pass
            worker.cancel()
        if thread is not None and thread.isRunning():
            thread.quit()
            thread.wait(3000)
        super().closeEvent(event)
