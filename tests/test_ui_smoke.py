"""Headless smoke test for the Qt control panels and their wiring.

Uses the offscreen Qt platform. It deliberately does NOT construct the VTK
viewport (QtInteractor), which requires a real OpenGL context and cannot be
created under the offscreen platform. The full window + viewport is verified
separately by an on-display launch.
"""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

SAMPLE = os.path.join(
    os.path.dirname(__file__), "..", "data", "sample_bots", "wedge_bot.stl"
)


@pytest.fixture(scope="module")
def app():
    try:
        from PySide6 import QtWidgets
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"PySide6 unavailable: {exc}")
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield application


def test_parts_panel_assigns_materials_and_masses(app):
    from battlebot_sim.materials.library import load_default_library
    from battlebot_sim.mesh.segment import load_bot
    from battlebot_sim.ui.panels import PartsPanel

    library = load_default_library()
    bot = load_bot(os.path.abspath(SAMPLE), scale_to_m=1.0)

    panel = PartsPanel(library)
    panel.set_bot(bot)
    # Defaults applied -> every part has a material and positive total mass.
    assert bot.assigned()
    assert bot.total_mass() > 0
    assert panel.table.rowCount() == len(bot.parts)

    # Toggling a brace flag propagates to the model.
    panel._on_brace(0, True)
    assert bot.parts[0].is_brace


def test_parts_panel_bulk_assign_to_selected(app):
    from battlebot_sim.materials.library import load_default_library
    from battlebot_sim.mesh.segment import load_bot
    from battlebot_sim.ui.panels import PartsPanel

    library = load_default_library()
    bot = load_bot(os.path.abspath(SAMPLE), scale_to_m=1.0)
    panel = PartsPanel(library)
    panel.set_bot(bot)

    panel.select_parts([0, 2])
    target = "Titanium Ti-6Al-4V"
    panel.bulk_combo.setCurrentText(target)
    panel._assign_to_selected()
    assert bot.parts[0].material.name == target
    assert bot.parts[2].material.name == target
    # An unselected part keeps the default (first) material.
    assert bot.parts[1].material.name == library.names()[0]


def test_parts_panel_recalc_button(app):
    from battlebot_sim.materials.library import load_default_library
    from battlebot_sim.mesh.segment import load_bot
    from battlebot_sim.ui.panels import PartsPanel

    library = load_default_library()
    bot = load_bot(os.path.abspath(SAMPLE), scale_to_m=1.0)
    panel = PartsPanel(library)
    panel.set_bot(bot)

    # The explicit recalc button reflects the model's current total mass.
    panel._recalculate()
    assert panel.total_label.text() == f"Total mass: {bot.total_mass():.3f} kg"


def test_parts_panel_searchable_combo_and_guard(app):
    from PySide6 import QtCore

    from battlebot_sim.materials.library import load_default_library
    from battlebot_sim.mesh.segment import load_bot
    from battlebot_sim.ui.panels import PartsPanel

    library = load_default_library()
    bot = load_bot(os.path.abspath(SAMPLE), scale_to_m=1.0)
    panel = PartsPanel(library)
    panel.set_bot(bot)

    # Per-row combos are editable with a contains-filter completer for search.
    combo = panel.table.cellWidget(0, 1)
    assert combo.isEditable()
    assert combo.completer().filterMode() == QtCore.Qt.MatchFlag.MatchContains

    # Partial / invalid typed text must not raise or change the assignment.
    before = bot.parts[0].material.name
    panel._on_material(0, "alumin")          # mid-typing, not a real material
    assert bot.parts[0].material.name == before

    # Focusing a part's editor (as a 3D click does) must not raise.
    panel.focus_material_editor(0)


def test_parts_panel_selection_signal_no_echo(app):
    from battlebot_sim.materials.library import load_default_library
    from battlebot_sim.mesh.segment import load_bot
    from battlebot_sim.ui.panels import PartsPanel

    library = load_default_library()
    bot = load_bot(os.path.abspath(SAMPLE), scale_to_m=1.0)
    panel = PartsPanel(library)
    panel.set_bot(bot)

    received = []
    panel.selection_changed.connect(received.append)
    panel.table.selectRow(1)                 # a user selection emits indices
    assert received and received[-1] == [1]
    received.clear()
    panel.select_parts([0, 2])               # a 3D-pick sync must NOT echo back
    assert received == []
    assert panel.selected_indices() == [0, 2]


def test_setup_and_results_panel_wiring(app):
    from battlebot_sim.ui.panels import ResultsPanel, SetupPanel

    setup = SetupPanel()
    assert setup.current_class_key() in {"3lb", "12lb", "30lb"}
    assert not setup.run_btn.isEnabled()  # disabled until a bot loads

    results = ResultsPanel()
    received = []
    results.mode_changed.connect(received.append)
    results.failure_btn.setChecked(True)
    assert received == ["failure"]

    results.set_enabled(True)
    results.configure_slider(50)
    assert results.slider.maximum() == 49


def test_load_sample_emits_known_good_path(app):
    from battlebot_sim.mesh.segment import sample_bot_path
    from battlebot_sim.ui.panels import SetupPanel

    setup = SetupPanel()
    captured = []
    setup.load_requested.connect(lambda p, s: captured.append((p, s)))
    # Even with a non-metre unit selected, the sample loads at scale 1.0
    # (it is authored in metres), so the demo can never be mis-scaled.
    setup.unit_combo.setCurrentText("millimetres")
    setup._load_sample()
    assert captured == [(sample_bot_path(), 1.0)]


def test_setup_panel_unit_scales(app):
    from battlebot_sim.ui.panels import SetupPanel

    setup = SetupPanel()
    units = [setup.unit_combo.itemText(i)
             for i in range(setup.unit_combo.count())]
    # Metric units plus the imperial additions, all wired to a scale factor.
    assert units == ["millimetres", "centimetres", "metres", "inches", "feet"]

    expected = {"millimetres": 1e-3, "centimetres": 1e-2, "metres": 1.0,
                "inches": 0.0254, "feet": 0.3048}
    for unit, factor in expected.items():
        setup.unit_combo.setCurrentText(unit)
        assert setup._scale_to_m() == factor


def test_setup_panel_running_controls(app):
    from battlebot_sim.ui.panels import SetupPanel

    setup = SetupPanel()
    assert setup.stop_btn.isHidden()             # Stop hidden until a run starts
    assert setup.current_speed() == 1.0

    speeds = []
    setup.speed_changed.connect(speeds.append)
    setup.speed_spin.setValue(2.5)
    assert speeds and abs(speeds[-1] - 2.5) < 1e-9
    assert setup.current_speed() == 2.5

    stops = []
    setup.stop_requested.connect(lambda: stops.append(True))
    setup.run_btn.setEnabled(True)
    setup.show_running(True)                      # running: Run off, Stop on, load blocked
    assert not setup.run_btn.isEnabled()
    assert not setup.stop_btn.isHidden()
    assert not setup.load_btn.isEnabled()
    setup.stop_btn.click()
    assert stops == [True]

    setup.show_running(False)                     # idle again
    assert setup.run_btn.isEnabled()
    assert setup.load_btn.isEnabled()
    assert setup.stop_btn.isHidden()


def test_stream_worker_streams_and_finishes(app):
    """Drive StreamWorker synchronously (no QThread) at huge speed so pacing
    sleeps collapse: it must stream chunks and emit a finished trace+result."""
    import numpy as np

    from battlebot_sim.arena.nhrl import build_arena
    from battlebot_sim.materials.assign import NHRL_CLASSES
    from battlebot_sim.materials.library import load_default_library
    from battlebot_sim.mesh.segment import load_bot
    from battlebot_sim.ui.main_window import StreamWorker

    lib = load_default_library()
    wc = NHRL_CLASSES["3lb"]
    arena = build_arena(wc)
    bot = load_bot(os.path.abspath(SAMPLE), 1.0)
    bot.assign_material_to_all(lib.get("Aluminum 6061-T6"))

    worker = StreamWorker(bot, arena, wc, lib, n_trials=1, fps=30, speed=1000.0)
    chunks, progress, done, failed = [], [], {}, []
    worker.chunk.connect(chunks.append)
    worker.progressed.connect(lambda i, n: progress.append((i, n)))
    worker.finished.connect(lambda t, r: done.update(trace=t, result=r))
    worker.failed.connect(failed.append)
    worker.run()

    assert not failed, failed
    assert len(chunks) > 0
    assert progress and progress[-1][0] <= progress[-1][1]
    assert done["trace"].total_contacts() > 0
    assert np.all(np.isfinite(done["result"].failure_margin_per_face))


def test_live_charts_rolling_window(app):
    """LiveCharts builds under offscreen Qt and bounds each series to the window."""
    from battlebot_sim.ui.charts import LiveCharts, MetricSample

    charts = LiveCharts()
    charts.reset()
    for i in range(700):  # overflow the rolling window to exercise eviction
        charts.append(MetricSample(
            t=i * 0.1, peak_force=float(i), cum_energy=float(i) * 2,
            max_margin=min(1.4, i / 500.0), speed=i * 0.01, hit_rate=float(i)))

    from battlebot_sim.ui.charts import WINDOW
    assert charts.force.series.count() <= WINDOW
    assert charts.motion.series2 is not None          # dual series (speed + hits)
    assert charts.motion.series2.count() <= WINDOW
    assert charts.margin.threshold == 1.0             # yield reference line present
    # Latest values made it in.
    assert charts.force.ys[-1] == 699.0


def test_stream_worker_emits_metrics(app):
    """The worker emits MetricSamples alongside chunks during a run."""
    from battlebot_sim.arena.nhrl import build_arena
    from battlebot_sim.materials.assign import NHRL_CLASSES
    from battlebot_sim.materials.library import load_default_library
    from battlebot_sim.mesh.segment import load_bot
    from battlebot_sim.ui.charts import MetricSample
    from battlebot_sim.ui.main_window import StreamWorker

    lib = load_default_library()
    wc = NHRL_CLASSES["3lb"]
    arena = build_arena(wc)
    bot = load_bot(os.path.abspath(SAMPLE), 1.0)
    bot.assign_material_to_all(lib.get("Aluminum 6061-T6"))

    worker = StreamWorker(bot, arena, wc, lib, n_trials=1, fps=30, speed=1000.0)
    samples = []
    worker.metrics.connect(samples.append)
    worker.run()

    assert samples and all(isinstance(s, MetricSample) for s in samples)
    # Cumulative energy never decreases; some force was registered.
    energies = [s.cum_energy for s in samples]
    assert energies == sorted(energies)
    assert max(s.peak_force for s in samples) > 0


def test_stream_worker_cancel_finalizes_partial(app):
    """Cancelling mid-run still finalises: a (partial) trace+result is emitted."""
    from battlebot_sim.arena.nhrl import build_arena
    from battlebot_sim.materials.assign import NHRL_CLASSES
    from battlebot_sim.materials.library import load_default_library
    from battlebot_sim.mesh.segment import load_bot
    from battlebot_sim.ui.main_window import StreamWorker

    lib = load_default_library()
    wc = NHRL_CLASSES["3lb"]
    arena = build_arena(wc)
    bot = load_bot(os.path.abspath(SAMPLE), 1.0)
    bot.assign_material_to_all(lib.get("Aluminum 6061-T6"))

    worker = StreamWorker(bot, arena, wc, lib, n_trials=1, fps=30, speed=1000.0)
    done = {}
    counter = {"n": 0}

    def on_chunk(_ch):
        counter["n"] += 1
        if counter["n"] == 4:
            worker.cancel()

    worker.chunk.connect(on_chunk)
    worker.finished.connect(lambda t, r: done.update(trace=t, result=r))
    worker.run()

    assert "result" in done                      # finalised despite the cancel
    assert 0 < len(done["trace"].frames) < 200    # stopped early, not a full run


def test_stream_worker_cancel_before_start(app):
    """Cancelling before the first frame must finish cleanly (not raise / fail)
    with an empty trace — the worker never emits a spurious failure."""
    from battlebot_sim.arena.nhrl import build_arena
    from battlebot_sim.materials.assign import NHRL_CLASSES
    from battlebot_sim.materials.library import load_default_library
    from battlebot_sim.mesh.segment import load_bot
    from battlebot_sim.ui.main_window import StreamWorker

    lib = load_default_library()
    wc = NHRL_CLASSES["3lb"]
    arena = build_arena(wc)
    bot = load_bot(os.path.abspath(SAMPLE), 1.0)
    bot.assign_material_to_all(lib.get("Aluminum 6061-T6"))

    worker = StreamWorker(bot, arena, wc, lib, n_trials=1, fps=30, speed=1000.0)
    worker.cancel()                               # cancel before run() even starts
    done, failed = {}, []
    worker.finished.connect(lambda t, r: done.update(trace=t))
    worker.failed.connect(failed.append)
    worker.run()

    assert not failed                             # no spurious error dialog
    assert "trace" in done
    assert done["trace"].total_contacts() == 0


def test_stream_worker_cancel_from_another_thread(app):
    """Cancelling from a *different* thread mid-run must finalise a partial trace
    without hanging or failing — the cross-thread cancel uses a threading.Event."""
    import threading
    import time as _time

    from PySide6 import QtCore

    from battlebot_sim.arena.nhrl import build_arena
    from battlebot_sim.materials.assign import NHRL_CLASSES
    from battlebot_sim.materials.library import load_default_library
    from battlebot_sim.mesh.segment import load_bot
    from battlebot_sim.ui.main_window import StreamWorker

    lib = load_default_library()
    wc = NHRL_CLASSES["3lb"]
    arena = build_arena(wc)
    bot = load_bot(os.path.abspath(SAMPLE), 1.0)
    bot.assign_material_to_all(lib.get("Aluminum 6061-T6"))

    worker = StreamWorker(bot, arena, wc, lib, n_trials=1, fps=30, speed=1.0)
    done, failed = {}, []
    # Direct connections so the signals fire synchronously in the worker thread
    # (there is no Qt event loop here to deliver queued cross-thread signals).
    direct = QtCore.Qt.ConnectionType.DirectConnection
    worker.finished.connect(lambda t, r: done.update(trace=t), direct)
    worker.failed.connect(failed.append, direct)

    runner = threading.Thread(target=worker.run)
    runner.start()
    _time.sleep(0.1)                # let the run loop get going
    worker.cancel()                 # cancel from the main thread, mid-run
    runner.join(timeout=30)

    assert not runner.is_alive()    # finalised and returned; did not hang
    assert not failed               # cancellation is not a failure
    assert "trace" in done          # a (partial) trace was still emitted
