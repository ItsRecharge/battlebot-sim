"""Main application window: wires the panels and viewport to the pipeline."""

from __future__ import annotations

import os
import threading
import time

import numpy as np
from PySide6 import QtCore, QtWidgets

from gauntlet.arena.nhrl import build_arena
from gauntlet.damage.braces import apply_brace_sharing
from gauntlet.damage.model import DamageAccumulator
from gauntlet.errors import ValidationError
from gauntlet.logging_setup import get_logger
from gauntlet.materials.assign import NHRL_CLASSES, validate_weight_class
from gauntlet.materials.library import load_default_library
from gauntlet.mesh.brace_detect import auto_detect_braces
from gauntlet.mesh.segment import load_bot
from gauntlet.report.export import export_report
from gauntlet.report.verdict import summarize_result
from gauntlet.sim.battery import (
    BatteryEvent,
    StressBattery,
    Strike,
    class_strike_energy,
    iter_battery,
    run_battery,
)
from gauntlet.sim.engine import SimEngine
from gauntlet.sim.recorder import SimTrace
from gauntlet.sim.strike import FREEPLAY_STRIKE_WINDOW
from gauntlet.ui.charts import LiveCharts, MetricStreamer
from gauntlet.ui.freeplay_view import FreeplayViewport
from gauntlet.ui.pacing import pace_schedule
from gauntlet.ui.panels import (
    FreeplayPanel,
    ModelSetupPanel,
    PartsPanel,
    ResultsPanel,
    RunControlsPanel,
)
from gauntlet.ui.viewport import BotOnlyView, BotViewport
from gauntlet.validation import validate_run_params

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
                 n_trials=1, fps=60, speed=1.0,
                 velocity_range=None, drop_angle_range=None, seed=0):
        super().__init__()
        self.bot, self.arena = bot, arena
        self.weight_class, self.library = weight_class, library
        self.n_trials = n_trials
        self.fps = fps
        self._speed = max(0.05, float(speed))
        self.velocity_range = velocity_range
        self.drop_angle_range = drop_angle_range
        self.seed = int(seed)
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
            battery_kw = {"n_trials": self.n_trials, "seed": self.seed}
            if self.velocity_range is not None:
                battery_kw["velocity_range"] = self.velocity_range
            if self.drop_angle_range is not None:
                battery_kw["drop_tilt_range_deg"] = self.drop_angle_range
            battery = StressBattery(self.arena, self.weight_class, **battery_kw)
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
            logger.exception("stream worker failed")
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Combat Robot Stress-Test Gauntlet")
        self.resize(1280, 820)

        self.library = load_default_library()
        self.bot = None
        self._loaded_path = None     # current model's path, for live unit re-scale
        self.arena = None
        self.trace = None
        self.result = None
        self._thread = None
        self._worker = None
        self._render_busy = False   # coalesce live pose updates (keep newest)
        self._pending = None

        # --- widgets ---
        self.viewport = BotViewport(self)                 # in-cage fly-around (Simulate)
        self.model_panel = ModelSetupPanel()              # load / class / units (Setup)
        self.run_panel = RunControlsPanel()               # trial sweep + Run (Simulate)
        self.parts_panel = PartsPanel(self.library)       # materials + braces (Setup)
        self.results_panel = ResultsPanel()               # heatmap + replay (Simulate)
        self.graphs = LiveCharts()                        # live metric-vs-time charts
        self.bot_view = BotOnlyView(self)                 # final damage turntable (Simulate)
        self.setup_preview = BotOnlyView(self)            # solid bot preview (Setup)
        self.freeplay_view = FreeplayViewport(self)       # draw-a-strike sandbox (Freeplay)
        self.freeplay_panel = FreeplayPanel()             # strike list + verdict (Freeplay)

        # Every viewport/panel that renders a loaded bot, fanned out by _install_bot.
        self._bot_consumers = [self.viewport, self.bot_view, self.setup_preview,
                               self.parts_panel, self.freeplay_view]

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._build_setup_tab(), "Setup")
        self.tabs.addTab(self._build_simulate_tab(), "Simulate")
        self.tabs.addTab(self._build_freeplay_tab(), "Freeplay")
        self.setCentralWidget(self.tabs)
        self.statusBar().showMessage("Load a model to begin.")

        # --- replay timer (~60 fps to match the 60 fps capture) ---
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._advance_frame)

        # --- freeplay bounce-replay timer ---
        self._fp_timer = QtCore.QTimer(self)
        self._fp_timer.setInterval(16)
        self._fp_timer.timeout.connect(self._fp_advance)
        self._fp_frame = 0

        # --- signals ---
        self.model_panel.load_requested.connect(self.load_stl)
        self.model_panel.units_changed.connect(self._on_units_changed)
        self.model_panel.class_changed.connect(self.run_panel.apply_class_speed_defaults)
        self.model_panel.cage_check.toggled.connect(self._on_cage_toggled)
        self.run_panel.run_requested.connect(self.run_simulation)
        # Stop/speed are wired once here (not per-run) to a forwarder, so repeated
        # runs never stack connections onto soon-to-be-deleted workers.
        self.run_panel.stop_requested.connect(self._on_stop_requested)
        self.run_panel.speed_changed.connect(self._on_speed_changed)
        self.run_panel.apply_class_speed_defaults(self.model_panel.current_class_key())
        self.parts_panel.changed.connect(self._update_weight_check)
        self.parts_panel.changed.connect(self.viewport.refresh_materials)
        self.parts_panel.changed.connect(self.bot_view.refresh_materials)
        self.parts_panel.changed.connect(self.setup_preview.refresh_materials)
        self.parts_panel.changed.connect(self.freeplay_view.refresh_materials)
        self.parts_panel.selection_changed.connect(self.viewport.set_selection)
        self.viewport.part_clicked.connect(self._on_part_picked)
        self.results_panel.mode_changed.connect(self._on_mode)
        self.results_panel.frame_changed.connect(self.viewport.show_frame)
        self.results_panel.play_toggled.connect(self._on_play)
        self.results_panel.export_requested.connect(self.export)

        # Freeplay: the panel arms the draw tool and lists strikes; the viewport
        # reports each placed strike back to the list.
        self.freeplay_panel.arm_toggled.connect(self.freeplay_view.arm_draw)
        self.freeplay_panel.energy_changed.connect(self.freeplay_view.set_energy)
        self.freeplay_panel.strike_edited.connect(self.freeplay_view.update_strike)
        self.freeplay_panel.strike_removed.connect(self.freeplay_view.remove_strike)
        self.freeplay_panel.clear_requested.connect(self.freeplay_view.clear_arrows)
        self.freeplay_panel.run_requested.connect(self.run_freeplay)
        self.freeplay_panel.mode_changed.connect(self.freeplay_view.show_heatmap)
        self.freeplay_view.strike_added.connect(self.freeplay_panel.add_strike_row)
        self.freeplay_view.set_energy(self.freeplay_panel.current_energy())

    # ---- tab layouts -----------------------------------------------------
    def _build_setup_tab(self) -> QtWidgets.QWidget:
        """Define the bot: a solid preview on the left, load/class/units and the
        parts-and-materials table on the right."""
        side = QtWidgets.QWidget()
        sl = QtWidgets.QVBoxLayout(side)
        sl.addWidget(self.model_panel)
        sl.addWidget(self.parts_panel, 1)
        side.setMaximumWidth(460)

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        split.addWidget(self.setup_preview)
        split.addWidget(side)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 0)
        return split

    def _build_simulate_tab(self) -> QtWidgets.QWidget:
        """The control-room dashboard: the cage view + live graphs on the left, the
        bot-only turntable in the centre, run controls + results on the right."""
        left = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        left.addWidget(self.viewport)
        left.addWidget(self.graphs)
        left.setStretchFactor(0, 3)
        left.setStretchFactor(1, 1)

        side = QtWidgets.QWidget()
        sl = QtWidgets.QVBoxLayout(side)
        sl.addWidget(self.run_panel)
        sl.addWidget(self.results_panel, 1)
        side.setMaximumWidth(460)

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        split.addWidget(left)
        split.addWidget(self.bot_view)
        split.addWidget(side)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setStretchFactor(2, 0)
        return split

    def _build_freeplay_tab(self) -> QtWidgets.QWidget:
        """Draw-a-strike sandbox: the bot on the left, the strike list + verdict on
        the right."""
        side = QtWidgets.QWidget()
        sl = QtWidgets.QVBoxLayout(side)
        sl.addWidget(self.freeplay_panel, 1)
        side.setMaximumWidth(460)

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        split.addWidget(self.freeplay_view)
        split.addWidget(side)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 0)
        return split

    # ---- load / setup ----------------------------------------------------
    @QtCore.Slot(str, float)
    def load_stl(self, path: str, scale_to_m: float) -> None:
        """Load a model file and install it. The *entire* load + setup runs under
        one guard: a failure anywhere (parse, segmentation, viewport, panels)
        surfaces a dialog and resets to a clean state, instead of silently
        leaving the Run button disabled against a half-loaded bot."""
        # Segmenting a large mesh takes a moment; show a wait cursor so the window
        # doesn't read as frozen while it works.
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        self._begin_load_progress()
        try:
            bot = load_bot(path, scale_to_m=scale_to_m,
                           max_parts=self.model_panel.current_max_parts())
            self._loaded_path = path
            self._install_bot(bot, path)
        except Exception as exc:
            self._fail_load(path, exc)
        finally:
            self._end_load_progress()
            QtWidgets.QApplication.restoreOverrideCursor()

    # ---- load progress + live unit re-scale ------------------------------
    def _begin_load_progress(self) -> None:
        """Show an indeterminate 'busy' bar so a slow import never reads as a hang."""
        p = self.model_panel.progress
        p.setRange(0, 0)            # 0,0 == Qt busy animation
        p.show()
        QtWidgets.QApplication.processEvents()

    def _on_load_progress(self, i: int, n: int) -> None:
        """Determinate progress for the per-part brace analysis."""
        p = self.model_panel.progress
        p.setRange(0, max(1, n))
        p.setValue(min(i, n))
        QtWidgets.QApplication.processEvents()

    def _end_load_progress(self) -> None:
        p = self.model_panel.progress
        p.hide()
        p.setRange(0, 1)
        p.setValue(0)

    @QtCore.Slot()
    def _on_units_changed(self) -> None:
        """Re-scale the loaded model when the Model-units dropdown changes.

        The bundled sample bot is authored in centimetres and ignores the dropdown,
        so it is excluded; with nothing loaded this is a no-op."""
        from gauntlet.mesh.segment import sample_bot_path
        if not self._loaded_path or self._loaded_path == sample_bot_path():
            return
        self.load_stl(self._loaded_path, self.model_panel.current_scale_to_m())

    def _install_bot(self, bot, path: str) -> None:
        """Place a freshly loaded bot into the viewport and panels, then enable
        Run. Any exception here propagates to load_stl's guard."""
        self.bot = bot
        self.result = self.trace = None
        self.results_panel.set_enabled(False)
        self.results_panel.solid_btn.setChecked(True)

        wc = NHRL_CLASSES[self.model_panel.current_class_key()]
        self.arena = build_arena(wc)
        self.viewport.clear()
        # Fan the bot out to every view/panel that renders it. The parts table's
        # one-shot `changed` then repaints each view in material colours. The arena
        # cage stays hidden during setup (unless "show cage" is ticked).
        for consumer in self._bot_consumers:
            consumer.set_bot(self.bot)
        # Auto-flag structural braces now that parts have (default) materials; the
        # user can still tick/untick any of them. Drives the import progress bar.
        auto_detect_braces(self.bot, progress=self._on_load_progress)
        self.parts_panel.refresh_braces()
        if self.model_panel.cage_check.isChecked():
            self.viewport.show_arena(self.arena)
        else:
            self.viewport.hide_arena()
        self._update_weight_check()
        self.run_panel.run_btn.setEnabled(True)
        # Reset freeplay for the new bot: show the cage the strikes bounce it against,
        # tell the panel the bot mass (force<->energy), and suggest a class strike.
        self.freeplay_view.show_arena(self.arena)
        self.freeplay_panel.clear_rows()
        self.freeplay_panel.set_strike_scale(self.bot.total_mass())
        self.freeplay_panel.set_suggested_energy(class_strike_energy(wc))
        n_parts = len(self.bot.parts)
        fragments = getattr(self.bot, "source_fragments", n_parts)
        if fragments > n_parts:
            msg = (f"Loaded {os.path.basename(path)} — simplified "
                   f"{fragments} → {n_parts} parts.")
        else:
            msg = f"Loaded {os.path.basename(path)} — {n_parts} parts."
        self.statusBar().showMessage(msg)

    def _fail_load(self, path: str, exc: Exception) -> None:
        """Report a load/setup failure and reset to a safe state so Run is never
        left enabled against a broken bot — nor silently disabled with no reason.
        The full traceback goes in the dialog's details to help diagnose the
        offending mesh (degenerate parts, non-finite geometry, etc.)."""
        import traceback
        self.bot = self.arena = self.result = self.trace = None
        self.run_panel.run_btn.setEnabled(False)
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
        wc = NHRL_CLASSES[self.model_panel.current_class_key()]
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
    @QtCore.Slot()
    def run_simulation(self) -> None:
        if self.bot is None:
            return
        if self._thread is not None and self._thread.isRunning():
            return                              # already running: ignore re-clicks
        wc = NHRL_CLASSES[self.model_panel.current_class_key()]
        n_trials = self.run_panel.current_trials()
        speed = self.run_panel.current_speed()
        velocity_range = self.run_panel.current_velocity_range()
        drop_angle_range = self.run_panel.current_drop_angle_range()
        seed = self.run_panel.current_seed()
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
        self._set_running(True)
        self.run_panel.progress.setRange(0, 1)
        self.run_panel.progress.setValue(0)
        self._render_busy = False
        self._pending = None
        self.statusBar().showMessage(
            f"Running stress battery ({n_trials} trial(s)) — live…")

        self._thread = QtCore.QThread(self)
        self._worker = StreamWorker(self.bot, self.arena, wc, self.library,
                                    n_trials, fps=fps, speed=speed,
                                    velocity_range=velocity_range,
                                    drop_angle_range=drop_angle_range, seed=seed)
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
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _set_running(self, running: bool) -> None:
        """Lock the run controls and the model panel together while a battery runs,
        so neither the trial settings nor the loaded bot can change underneath the
        worker."""
        self.run_panel.show_running(running)
        self.model_panel.set_locked(running)

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
        self.run_panel.progress.setRange(0, max(1, n_events))
        self.run_panel.progress.setValue(min(event_index + 1, n_events))

    @QtCore.Slot()
    def _on_thread_finished(self) -> None:
        self._thread = None
        self._worker = None

    @QtCore.Slot(object, object)
    def _on_sim_done(self, trace, result) -> None:
        # Cancelled before any data was collected: don't present an empty,
        # misleading "result" — just return to idle.
        if trace.total_contacts() == 0 and not trace.frames:
            self._set_running(False)
            self.statusBar().showMessage("Cancelled — no data collected.")
            return
        self.trace, self.result = trace, result
        self.viewport.set_trace(trace)
        self.viewport.set_result(result)
        self.results_panel.configure_slider(self.viewport.n_frames)
        self.results_panel.set_enabled(True)
        self._set_running(False)
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
        self._set_running(False)
        QtWidgets.QMessageBox.critical(self, "Simulation failed", message)
        self.statusBar().showMessage("Simulation failed.")

    def _write_summary(self) -> None:
        if self.result is None:
            return
        lines = summarize_result(self.bot, self.result)
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

    # ---- freeplay --------------------------------------------------------
    @QtCore.Slot()
    def run_freeplay(self) -> None:
        """Fire the drawn strikes at the bot as real impulses inside the cage, so it
        recoils and bounces off the walls and floor. Damage from the strikes and the
        cage impacts is accumulated, the bounce is replayed, then the failure heatmap
        and per-part verdict are shown."""
        if self.bot is None or not self.freeplay_view.strikes:
            return
        wc = NHRL_CLASSES[self.model_panel.current_class_key()]
        if self.arena is None:
            self.arena = build_arena(wc)
        rest_t = self.freeplay_view.rest_translation()
        # Start the bot exactly where it was drawn (resting on the floor), then fire.
        min_z = float(self.bot.original.bounds[0][2])
        init_pos = np.array([0.0, 0.0, -min_z + 1e-3])
        strikes = [
            Strike(t_start=0.10, t_end=0.10 + FREEPLAY_STRIKE_WINDOW,
                   direction=np.asarray(s.direction, dtype=float),
                   energy_j=s.energy_j,
                   local_point=np.asarray(s.world_point, dtype=float) - rest_t)
            for s in self.freeplay_view.strikes
        ]
        event = BatteryEvent(name="freeplay", duration=2.0, init_pos=init_pos,
                             init_quat=np.array([1.0, 0, 0, 0]), strikes=strikes)
        battery = StressBattery(self.arena, wc, n_trials=1)
        battery.events = [event]

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            engine = SimEngine(self.arena, self.bot)
            trace = run_battery(engine, battery, fps=60)
            accum = DamageAccumulator(self.bot, self.arena, self.library)
            accum.ingest(trace.contacts, engine.timestep)
            result = apply_brace_sharing(accum.finalize(), self.bot)
        except Exception as exc:
            logger.exception("freeplay run failed")
            QtWidgets.QMessageBox.critical(
                self, "Freeplay failed", f"{type(exc).__name__}: {exc}")
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        self.freeplay_view.set_trace(trace)
        self.freeplay_view.set_result(result)
        self.freeplay_panel.set_summary("\n".join(summarize_result(self.bot, result)))
        self.statusBar().showMessage(
            f"Freeplay: fired {len(strikes)} strike(s), {trace.total_contacts()} contacts.")
        self._fp_play_bounce()

    def _fp_play_bounce(self) -> None:
        """Replay the bounce once in the freeplay viewport, then settle on the
        failure heatmap."""
        self._fp_frame = 0
        if self.freeplay_view.n_frames <= 1:
            self._fp_show_result()
            return
        self.freeplay_view.begin_live()          # solid bot, ready to pose per frame
        self.freeplay_view.set_arrows_visible(False)   # bot moves away from them
        self._fp_timer.start()

    def _fp_advance(self) -> None:
        if self._fp_frame >= self.freeplay_view.n_frames - 1:
            self._fp_timer.stop()
            self._fp_show_result()
            return
        self.freeplay_view.show_frame(self._fp_frame)
        self._fp_frame += 1

    def _fp_show_result(self) -> None:
        # Checking the radio drives show_heatmap and returns the bot to rest pose;
        # trigger it explicitly when it is already checked from a previous run.
        if self.freeplay_panel.failure_btn.isChecked():
            self.freeplay_view.show_heatmap("failure")
        else:
            self.freeplay_panel.failure_btn.setChecked(True)
        self.freeplay_view.set_arrows_visible(True)   # back at rest pose, arrows align

    # ---- export ----------------------------------------------------------
    @QtCore.Slot()
    def export(self) -> None:
        if self.result is None:
            return
        out_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose output folder")
        if not out_dir:
            return
        wc = NHRL_CLASSES[self.model_panel.current_class_key()]
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
        self._fp_timer.stop()
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
