"""Control panels for the main window: parts/materials, setup, and results."""

from __future__ import annotations

import random
from contextlib import contextmanager

from PySide6 import QtCore, QtGui, QtWidgets

from gauntlet import viz
from gauntlet.materials.assign import NHRL_CLASSES
from gauntlet.materials.library import MaterialLibrary
from gauntlet.mesh.segment import DEFAULT_MAX_PARTS, BotModel
from gauntlet.sim.battery import (
    DEFAULT_DROP_TILT_RANGE_DEG,
    DEFAULT_VELOCITY_CEILING,
    class_speed,
)
from gauntlet.sim.strike import energy_from_force, force_from_energy


@contextmanager
def _blocked(*widgets):
    """Temporarily block Qt signals on ``widgets`` (avoids slider<->spinbox loops)."""
    for w in widgets:
        w.blockSignals(True)
    try:
        yield
    finally:
        for w in widgets:
            w.blockSignals(False)


class PartsPanel(QtWidgets.QGroupBox):
    """A table of parts with per-part material + brace selection, plus 3D-linked
    multi-select and bulk material assignment."""

    changed = QtCore.Signal()                 # any material/brace change
    selection_changed = QtCore.Signal(list)   # selected part indices (from table)

    def __init__(self, library: MaterialLibrary, parent=None):
        super().__init__("Parts & Materials", parent)
        self.library = library
        self.bot: BotModel | None = None
        self._syncing = False                 # guard against selection echo loops

        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Part", "Material", "Brace", "Mass (kg)"])
        self.table.horizontalHeader().setSectionResizeMode(
            1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        # Multi-select rows: lets several parts be bulk-assigned at once and keeps
        # 3D picks and the table in sync.
        self.table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)

        self.hint = QtWidgets.QLabel(
            "Tip: click a part in the 3D view to open its material dropdown, or "
            "Ctrl/Shift-click to multi-select then “Assign to selected”. "
            "Type in a dropdown to search materials.")
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet("color:#57606a")

        # Bulk assignment row.
        self.bulk_combo = self._make_material_combo()
        self.assign_sel_btn = QtWidgets.QPushButton("Assign to selected")
        self.assign_all_btn = QtWidgets.QPushButton("Assign to all")
        self.assign_sel_btn.clicked.connect(self._assign_to_selected)
        self.assign_all_btn.clicked.connect(self._assign_to_all)
        bulk = QtWidgets.QHBoxLayout()
        bulk.addWidget(self.bulk_combo, 1)
        bulk.addWidget(self.assign_sel_btn)
        bulk.addWidget(self.assign_all_btn)

        # Explicit confirm/recalc control (masses already update live).
        self.recalc_btn = QtWidgets.QPushButton("Recalculate weight")
        self.recalc_btn.clicked.connect(self._recalculate)

        self.total_label = QtWidgets.QLabel("Total mass: —")
        self.class_label = QtWidgets.QLabel("")
        self.class_label.setWordWrap(True)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.hint)
        layout.addWidget(self.table)
        layout.addLayout(bulk)
        layout.addWidget(self.recalc_btn)
        layout.addWidget(self.total_label)
        layout.addWidget(self.class_label)

    def set_bot(self, bot: BotModel) -> None:
        self.bot = bot
        # Populate with our own signals muted so applying N default materials
        # fires `changed`/`selection_changed` once (below), not once per part.
        self.blockSignals(True)
        try:
            self.table.clearSelection()
            self.table.setRowCount(len(bot.parts))
            for p in bot.parts:
                name_item = QtWidgets.QTableWidgetItem(p.name)
                name_item.setToolTip(f"Part {p.index}: {p.name}")
                self.table.setItem(p.index, 0, name_item)

                combo = self._make_material_combo()
                combo.currentTextChanged.connect(
                    lambda name, idx=p.index: self._on_material(idx, name))
                self.table.setCellWidget(p.index, 1, combo)

                check = QtWidgets.QCheckBox()
                check.setChecked(bool(p.is_brace))
                check.setToolTip(
                    "Structural brace: bridges and stiffens other parts. "
                    "Auto-detected on import; tick/untick to override.")
                check.stateChanged.connect(
                    lambda state, idx=p.index: self._on_brace(idx, state))
                holder = QtWidgets.QWidget()
                hl = QtWidgets.QHBoxLayout(holder)
                hl.addWidget(check)
                hl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                hl.setContentsMargins(0, 0, 0, 0)
                self.table.setCellWidget(p.index, 2, holder)

                self.table.setItem(p.index, 3, QtWidgets.QTableWidgetItem("—"))
                # Apply the default (first) material immediately.
                self._on_material(p.index, combo.currentText())
        finally:
            self.blockSignals(False)
        self.changed.emit()

    def _make_material_combo(self) -> QtWidgets.QComboBox:
        """A material picker that shows full names and filters as you type."""
        combo = QtWidgets.QComboBox()
        combo.addItems(self.library.names())
        combo.setEditable(True)
        combo.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
        completer = combo.completer()
        completer.setCompletionMode(
            QtWidgets.QCompleter.CompletionMode.PopupCompletion)
        completer.setFilterMode(QtCore.Qt.MatchFlag.MatchContains)
        completer.setCaseSensitivity(QtCore.Qt.CaseSensitivity.CaseInsensitive)
        return combo

    def focus_material_editor(self, idx: int) -> None:
        """Bring a part's material combo into view and open it for editing —
        used when a single part is picked in the 3D view."""
        item = self.table.item(idx, 0)
        if item is not None:
            self.table.scrollToItem(item)
        combo = self.table.cellWidget(idx, 1)
        if combo is not None:
            combo.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)
            combo.showPopup()

    def _recalculate(self) -> None:
        """Force a fresh mass / weight-class computation on demand. Live updates
        already keep this current; the button gives an explicit confirm step."""
        self._refresh_masses()
        self.changed.emit()

    def _on_material(self, idx: int, name: str) -> None:
        # Editable combos emit on every keystroke; only commit real materials.
        if self.bot is None or name not in self.library:
            return
        self.bot.assign_material(idx, self.library.get(name))
        self._apply_swatch(idx)
        self._refresh_masses()
        self.changed.emit()

    def _on_brace(self, idx: int, state) -> None:
        if self.bot is None:
            return
        self.bot.set_brace(idx, bool(state))
        self.changed.emit()

    def refresh_braces(self) -> None:
        """Re-sync the brace checkboxes to the model (e.g. after auto-detection)
        without emitting change signals or clobbering a user's manual choice."""
        if self.bot is None:
            return
        for p in self.bot.parts:
            holder = self.table.cellWidget(p.index, 2)
            check = holder.findChild(QtWidgets.QCheckBox) if holder else None
            if check is None:
                continue
            check.blockSignals(True)
            check.setChecked(bool(p.is_brace))
            check.blockSignals(False)

    def _refresh_masses(self) -> None:
        if self.bot is None:
            return
        for p in self.bot.parts:
            item = self.table.item(p.index, 3)
            if item:
                item.setText(f"{p.mass_kg:.3f}")
        self.total_label.setText(f"Total mass: {self.bot.total_mass():.3f} kg")

    def update_weight_check(self, message: str, ok: bool) -> None:
        color = "#1a7f37" if ok else "#cf222e"
        self.class_label.setText(f"<span style='color:{color}'>{message}</span>")

    # ---- colour swatches -------------------------------------------------
    def _apply_swatch(self, idx: int) -> None:
        """Paint the part's name cell with its material colour so the table is a
        legend for the 3D view."""
        item = self.table.item(idx, 0)
        if item is None or self.bot is None:
            return
        mat = self.bot.parts[idx].material
        if mat is None:
            return
        r, g, b = viz.material_color(mat.name)
        item.setBackground(QtGui.QColor(int(r * 255), int(g * 255), int(b * 255)))

    # ---- bulk assignment -------------------------------------------------
    def _assign_to_selected(self) -> None:
        self._assign_material_to(self.selected_indices(), self.bulk_combo.currentText())

    def _assign_to_all(self) -> None:
        if self.bot is not None:
            self._assign_material_to([p.index for p in self.bot.parts],
                                     self.bulk_combo.currentText())

    def _assign_material_to(self, indices, name: str) -> None:
        """Set one material on many parts in a single action — model, per-row
        combos and swatches — then refresh once."""
        if self.bot is None or not indices or name not in self.library:
            return
        mat = self.library.get(name)
        for idx in indices:
            combo = self.table.cellWidget(idx, 1)
            if combo is not None:
                combo.blockSignals(True)
                combo.setCurrentText(name)
                combo.blockSignals(False)
            self.bot.assign_material(idx, mat)
            self._apply_swatch(idx)
        self._refresh_masses()
        self.changed.emit()

    # ---- selection sync with the 3D view ---------------------------------
    def selected_indices(self) -> list[int]:
        return sorted({ix.row() for ix in self.table.selectionModel().selectedRows()})

    def _on_selection_changed(self) -> None:
        if not self._syncing:
            self.selection_changed.emit(self.selected_indices())

    def select_parts(self, indices) -> None:
        """Set the table selection from a 3D pick, without echoing it back out."""
        sm = self.table.selectionModel()
        model = self.table.model()
        self._syncing = True
        try:
            sm.clearSelection()
            if indices:
                sel = QtCore.QItemSelection()
                last = self.table.columnCount() - 1
                for idx in indices:
                    sel.select(model.index(idx, 0), model.index(idx, last))
                sm.select(sel, QtCore.QItemSelectionModel.SelectionFlag.Select)
                self.table.scrollToItem(self.table.item(indices[0], 0))
        finally:
            self._syncing = False


class ModelSetupPanel(QtWidgets.QGroupBox):
    """Load a model, choose its weight class and units, and cap segmentation. The
    'define the bot' half of setup; running the battery lives in RunControlsPanel."""

    load_requested = QtCore.Signal(str, float)   # path, scale_to_m
    units_changed = QtCore.Signal()              # Model-units dropdown changed
    class_changed = QtCore.Signal(str)           # weight-class key changed

    # STL import units -> factor converting one source unit to metres. Listed in
    # the order shown in the dropdown (metric first, then imperial); millimetres
    # stays first as the most common STL export unit.
    UNIT_SCALES_M = {
        "millimetres": 1e-3,
        "centimetres": 1e-2,
        "metres": 1.0,
        "inches": 0.0254,
        "feet": 0.3048,
    }

    def __init__(self, parent=None):
        super().__init__("Model", parent)
        self.class_combo = QtWidgets.QComboBox()
        for key, wc in NHRL_CLASSES.items():
            self.class_combo.addItem(wc.name, key)
        self.class_combo.currentIndexChanged.connect(self._emit_class_changed)

        self.unit_combo = QtWidgets.QComboBox()
        self.unit_combo.addItems(list(self.UNIT_SCALES_M))
        # Changing units after a model is loaded re-scales it (see MainWindow).
        self.unit_combo.currentIndexChanged.connect(self.units_changed.emit)

        self.load_btn = QtWidgets.QPushButton("Load model…")
        self.sample_btn = QtWidgets.QPushButton("Load sample bot")
        self.sample_btn.setToolTip(
            "Load the bundled demo bot to try the full pipeline immediately.")

        # Max parts: cap on segmentation so a fragmented CAD/STL still loads & runs.
        self.maxparts_spin = QtWidgets.QSpinBox()
        self.maxparts_spin.setRange(8, 512)
        self.maxparts_spin.setValue(DEFAULT_MAX_PARTS)
        self.maxparts_spin.setToolTip(
            "Maximum number of parts a model is split into. A messy mesh that fragments "
            "past this is simplified (smallest fragments merged) so it still loads.")

        self.cage_check = QtWidgets.QCheckBox("Show arena cage during setup")
        self.cage_check.setToolTip(
            "Off by default so parts are easy to see and click; the cage always "
            "appears when you run the battery.")
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 1)            # determinate: filled per event
        self.progress.hide()

        form = QtWidgets.QFormLayout(self)
        form.addRow("Weight class:", self.class_combo)
        form.addRow("Model units:", self.unit_combo)
        form.addRow(self.load_btn)
        form.addRow(self.sample_btn)
        form.addRow("Max parts:", self.maxparts_spin)
        form.addRow(self.cage_check)
        form.addRow(self.progress)

        self.load_btn.clicked.connect(self._choose_file)
        self.sample_btn.clicked.connect(self._load_sample)

    def _emit_class_changed(self, *_) -> None:
        key = self.class_combo.currentData()
        if key is not None:
            self.class_changed.emit(key)

    def _scale_to_m(self) -> float:
        return self.UNIT_SCALES_M[self.unit_combo.currentText()]

    def current_scale_to_m(self) -> float:
        """The metres-per-source-unit factor for the selected Model units."""
        return self._scale_to_m()

    def current_class_key(self) -> str:
        return self.class_combo.currentData()

    def current_max_parts(self) -> int:
        return int(self.maxparts_spin.value())

    def _choose_file(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open model", "",
            "Models (*.stl *.3mf *.gltf *.glb *.obj);;"
            "STL (*.stl);;3MF (*.3mf);;glTF (*.gltf *.glb);;OBJ (*.obj);;"
            "All files (*)")
        if path:
            self.load_requested.emit(path, self._scale_to_m())

    def _load_sample(self) -> None:
        """Load the bundled demo bot. It is authored in centimetres, so it ignores
        the units dropdown (fixed scale) and gives an instant known-good run."""
        from gauntlet.mesh.segment import SAMPLE_SCALE_TO_M, sample_bot_path
        self.load_requested.emit(sample_bot_path(), SAMPLE_SCALE_TO_M)

    def set_locked(self, locked: bool) -> None:
        """Block model changes while a battery runs so the worker's bot can't change
        underneath it."""
        for w in (self.load_btn, self.sample_btn, self.class_combo,
                  self.unit_combo, self.maxparts_spin):
            w.setEnabled(not locked)


class RunControlsPanel(QtWidgets.QGroupBox):
    """Trial-sweep settings (speed/angle/seed), playback speed, and Run/Stop for the
    stress battery."""

    run_requested = QtCore.Signal()               # start the battery
    stop_requested = QtCore.Signal()              # cancel a running battery
    speed_changed = QtCore.Signal(float)          # live playback speed multiplier

    def __init__(self, parent=None):
        super().__init__("Stress battery", parent)
        self.trials_spin = QtWidgets.QSpinBox()
        self.trials_spin.setRange(1, 50)
        self.trials_spin.setValue(1)
        self.trials_spin.setToolTip(
            "More trials test more impact angles and orientations (a systematic "
            "sweep plus seeded random extras). Damage accumulates across every "
            "trial into one worst-case map. Higher = more thorough but slower.")

        # --- per-trial randomisation envelope (higher/seeded velocity + drop angle) ---
        # Impact speed range (m/s), drawn per trial from the seed. min == max gives a
        # fixed manual speed; the slider drives the upper bound for quick tuning.
        self.vel_min_spin = QtWidgets.QDoubleSpinBox()
        self.vel_min_spin.setRange(0.5, 40.0)
        self.vel_min_spin.setSingleStep(0.5)
        self.vel_min_spin.setSuffix(" m/s")
        self.vel_min_spin.setToolTip(
            "Lowest impact speed a trial may use. Set equal to the max for a fixed speed.")
        self.vel_max_spin = QtWidgets.QDoubleSpinBox()
        self.vel_max_spin.setRange(0.5, 40.0)
        self.vel_max_spin.setSingleStep(0.5)
        self.vel_max_spin.setSuffix(" m/s")
        self.vel_max_spin.setToolTip(
            "Highest impact speed a trial may use (drag the slider to tune).")
        self.vel_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.vel_slider.setRange(1, 40)
        self.vel_slider.setToolTip("Manual impact-speed control — sets the maximum speed.")
        vel_row = QtWidgets.QHBoxLayout()
        vel_row.addWidget(self.vel_min_spin)
        vel_row.addWidget(QtWidgets.QLabel("to"))
        vel_row.addWidget(self.vel_max_spin)
        self.vel_min_spin.valueChanged.connect(self._on_vel_min_changed)
        self.vel_max_spin.valueChanged.connect(self._on_vel_max_changed)
        self.vel_slider.valueChanged.connect(self._on_vel_slider_changed)

        # Drop-angle (tilt) range in degrees, drawn per trial from the seed:
        # 0° is a flat drop, higher tilts land the bot on an edge or corner.
        self.drop_min_spin = QtWidgets.QSpinBox()
        self.drop_min_spin.setRange(0, 90)
        self.drop_min_spin.setValue(int(DEFAULT_DROP_TILT_RANGE_DEG[0]))
        self.drop_min_spin.setSuffix(" °")
        self.drop_max_spin = QtWidgets.QSpinBox()
        self.drop_max_spin.setRange(0, 90)
        self.drop_max_spin.setValue(int(DEFAULT_DROP_TILT_RANGE_DEG[1]))
        self.drop_max_spin.setSuffix(" °")
        for s in (self.drop_min_spin, self.drop_max_spin):
            s.setToolTip("Range of drop tilt angles tested (0° = flat, higher = on an edge).")
        drop_row = QtWidgets.QHBoxLayout()
        drop_row.addWidget(self.drop_min_spin)
        drop_row.addWidget(QtWidgets.QLabel("to"))
        drop_row.addWidget(self.drop_max_spin)

        # Seed makes the random trials reproducible; "New seed" rolls a fresh one.
        self.seed_spin = QtWidgets.QSpinBox()
        self.seed_spin.setRange(0, 2_147_483_647)
        self.seed_spin.setToolTip(
            "Random seed for the trial sweep. Same seed → identical battery; change it "
            "(or click New seed) to explore a different set of impacts.")
        self.new_seed_btn = QtWidgets.QPushButton("New seed")
        self.new_seed_btn.setToolTip("Roll a fresh random seed.")
        self.new_seed_btn.clicked.connect(self._roll_seed)
        seed_row = QtWidgets.QHBoxLayout()
        seed_row.addWidget(self.seed_spin, 1)
        seed_row.addWidget(self.new_seed_btn)

        # Velocity defaults follow the weight class; MainWindow seeds them once the
        # model panel exists and re-applies them on ModelSetupPanel.class_changed.

        # Live playback speed: the battery runs at (scaled) real time so the bot
        # is watchable flying around the cage. MuJoCo is much faster than real
        # time, so this throttles — 1x = real time, up to 4x to skim through.
        self.speed_spin = QtWidgets.QDoubleSpinBox()
        self.speed_spin.setRange(0.25, 4.0)
        self.speed_spin.setSingleStep(0.25)
        self.speed_spin.setValue(1.0)
        self.speed_spin.setSuffix(" ×")
        self.speed_spin.setToolTip(
            "Live playback speed of the fly-around (1× = real time). Adjustable "
            "while the battery runs.")
        self.speed_spin.valueChanged.connect(self.speed_changed.emit)

        self.run_btn = QtWidgets.QPushButton("Run stress battery")
        self.run_btn.setEnabled(False)
        self.run_btn.setToolTip("Load a model first.")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setToolTip("Cancel the running battery; partial results are kept.")
        self.stop_btn.hide()
        self.stop_btn.clicked.connect(self.stop_requested.emit)
        run_row = QtWidgets.QHBoxLayout()
        run_row.addWidget(self.run_btn, 1)
        run_row.addWidget(self.stop_btn)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 1)            # determinate: filled per event
        self.progress.hide()

        form = QtWidgets.QFormLayout(self)
        form.addRow("Trials:", self.trials_spin)
        form.addRow("Impact speed:", vel_row)
        form.addRow("", self.vel_slider)
        form.addRow("Drop angle:", drop_row)
        form.addRow("Seed:", seed_row)
        form.addRow("Playback speed:", self.speed_spin)
        form.addRow(run_row)
        form.addRow(self.progress)

        self.run_btn.clicked.connect(self.run_requested.emit)

    # ---- trial-randomisation controls -----------------------------------
    def apply_class_speed_defaults(self, key: str) -> None:
        """Seed the velocity range from a weight class's representative speed
        (min = class speed, max = a higher multiple of it)."""
        if key is None or key not in NHRL_CLASSES:
            return
        base = class_speed(NHRL_CLASSES[key])
        lo = round(base, 1)
        hi = round(min(base * DEFAULT_VELOCITY_CEILING, self.vel_max_spin.maximum()), 1)
        with _blocked(self.vel_min_spin, self.vel_max_spin, self.vel_slider):
            self.vel_min_spin.setValue(lo)
            self.vel_max_spin.setValue(hi)
            self.vel_slider.setValue(int(round(hi)))

    def _on_vel_min_changed(self, value: float) -> None:
        if value > self.vel_max_spin.value():       # keep min <= max
            with _blocked(self.vel_max_spin, self.vel_slider):
                self.vel_max_spin.setValue(value)
                self.vel_slider.setValue(int(round(value)))

    def _on_vel_max_changed(self, value: float) -> None:
        with _blocked(self.vel_slider):
            self.vel_slider.setValue(int(round(value)))
        if value < self.vel_min_spin.value():
            with _blocked(self.vel_min_spin):
                self.vel_min_spin.setValue(value)

    def _on_vel_slider_changed(self, value: int) -> None:
        with _blocked(self.vel_max_spin):
            self.vel_max_spin.setValue(float(value))
        if float(value) < self.vel_min_spin.value():
            with _blocked(self.vel_min_spin):
                self.vel_min_spin.setValue(float(value))

    def _roll_seed(self) -> None:
        self.seed_spin.setValue(random.randrange(0, 2_147_483_647))

    def current_trials(self) -> int:
        return int(self.trials_spin.value())

    def current_speed(self) -> float:
        return float(self.speed_spin.value())

    def current_velocity_range(self) -> tuple[float, float]:
        lo, hi = float(self.vel_min_spin.value()), float(self.vel_max_spin.value())
        return (lo, hi) if lo <= hi else (hi, lo)

    def current_drop_angle_range(self) -> tuple[float, float]:
        lo, hi = float(self.drop_min_spin.value()), float(self.drop_max_spin.value())
        return (lo, hi) if lo <= hi else (hi, lo)

    def current_seed(self) -> int:
        return int(self.seed_spin.value())

    def show_running(self, running: bool) -> None:
        """Flip the panel between idle and running: while a battery runs, Run is
        disabled and a Stop button + determinate progress bar appear. The trial
        settings lock, but playback speed stays live so the fly-around can be sped
        up. (ModelSetupPanel.set_locked blocks model changes in parallel.)"""
        self.run_btn.setEnabled(not running)
        self.stop_btn.setVisible(running)
        self.trials_spin.setEnabled(not running)
        for w in (self.vel_min_spin, self.vel_max_spin, self.vel_slider,
                  self.drop_min_spin, self.drop_max_spin, self.seed_spin,
                  self.new_seed_btn):
            w.setEnabled(not running)
        self.progress.setVisible(running)


class ResultsPanel(QtWidgets.QGroupBox):
    """Heatmap toggle, replay scrubber, summary text, and export."""

    mode_changed = QtCore.Signal(str)       # "solid" | "energy" | "failure"
    frame_changed = QtCore.Signal(int)
    play_toggled = QtCore.Signal(bool)
    export_requested = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__("Results", parent)
        self.solid_btn = QtWidgets.QRadioButton("Solid")
        self.energy_btn = QtWidgets.QRadioButton("Impact energy")
        self.failure_btn = QtWidgets.QRadioButton("Failure margin")
        self.solid_btn.setChecked(True)
        for b, mode in ((self.solid_btn, "solid"),
                        (self.energy_btn, "energy"),
                        (self.failure_btn, "failure")):
            b.toggled.connect(lambda on, m=mode: on and self.mode_changed.emit(m))

        self.play_btn = QtWidgets.QPushButton("Play replay")
        self.play_btn.setCheckable(True)
        self.play_btn.toggled.connect(self.play_toggled.emit)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self.frame_changed.emit)

        self.summary = QtWidgets.QTextEdit()
        self.summary.setReadOnly(True)
        self.export_btn = QtWidgets.QPushButton("Export report…")
        self.export_btn.clicked.connect(self.export_requested.emit)

        self.set_enabled(False)

        layout = QtWidgets.QVBoxLayout(self)
        mode_row = QtWidgets.QHBoxLayout()
        for b in (self.solid_btn, self.energy_btn, self.failure_btn):
            mode_row.addWidget(b)
        layout.addLayout(mode_row)
        layout.addWidget(self.play_btn)
        layout.addWidget(self.slider)
        layout.addWidget(self.summary)
        layout.addWidget(self.export_btn)

    def set_enabled(self, on: bool) -> None:
        for w in (self.energy_btn, self.failure_btn, self.play_btn,
                  self.slider, self.export_btn):
            w.setEnabled(on)

    def configure_slider(self, n_frames: int) -> None:
        self.slider.setRange(0, max(0, n_frames - 1))
        self.slider.setValue(0)


class FreeplayPanel(QtWidgets.QGroupBox):
    """Place impact arrows on the bot, set each one's force and angle, run them, and
    read the resulting per-part verdict."""

    _ID = QtCore.Qt.ItemDataRole.UserRole
    _ENERGY = QtCore.Qt.ItemDataRole.UserRole + 1
    _ELEV = QtCore.Qt.ItemDataRole.UserRole + 2

    arm_toggled = QtCore.Signal(bool)             # draw tool on/off
    energy_changed = QtCore.Signal(float)         # energy (J) for the next new strike
    strike_edited = QtCore.Signal(int, float, float)   # id, energy_j, elevation_deg
    strike_removed = QtCore.Signal(int)           # remove the strike with this id
    clear_requested = QtCore.Signal()             # remove all strikes
    run_requested = QtCore.Signal()               # run the placed strikes
    mode_changed = QtCore.Signal(str)             # "solid" | "energy" | "failure"

    def __init__(self, parent=None):
        super().__init__("Freeplay impacts", parent)
        self._mass = 1.0           # bot mass (kg), for force <-> energy conversion
        self._loading = False      # guard the editor while loading a selected row

        self.hint = QtWidgets.QLabel(
            "Click a point on the bot and drag to aim an incoming impact, then add a "
            "few and Run impacts. Select a strike to change its force or angle.")
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet("color:#57606a")

        self.add_btn = QtWidgets.QPushButton("Add impact")
        self.add_btn.setCheckable(True)
        self.add_btn.setToolTip("Arm the draw tool, then click-drag on the bot.")
        self.add_btn.toggled.connect(self.arm_toggled.emit)

        self.force_spin = self._force_spin(
            "Peak contact force the next strike delivers.")
        self.force_spin.valueChanged.connect(self._on_default_force)

        self.list = QtWidgets.QListWidget()
        self.list.setToolTip("Placed impacts. Select one to edit it, Delete to remove.")
        self.list.currentItemChanged.connect(self._on_row_changed)

        self.delete_btn = QtWidgets.QPushButton("Delete selected")
        self.delete_btn.clicked.connect(self._delete_selected)
        self.clear_btn = QtWidgets.QPushButton("Clear all")
        self.clear_btn.clicked.connect(self._clear_all)
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addWidget(self.delete_btn)
        btn_row.addWidget(self.clear_btn)

        # Per-strike editor: change the selected strike's force and angle after drawing.
        self.edit_force = self._force_spin("Force of the selected strike.")
        self.edit_angle = QtWidgets.QSpinBox()
        self.edit_angle.setRange(-90, 90)
        self.edit_angle.setSuffix(" °")
        self.edit_angle.setToolTip(
            "Angle of the selected strike below horizontal: 90 straight down, 0 level.")
        self.edit_force.valueChanged.connect(self._on_edit)
        self.edit_angle.valueChanged.connect(self._on_edit)
        edit_form = QtWidgets.QFormLayout()
        edit_form.addRow("Selected force:", self.edit_force)
        edit_form.addRow("Selected angle:", self.edit_angle)
        self.edit_box = QtWidgets.QGroupBox("Edit selected impact")
        self.edit_box.setLayout(edit_form)

        self.run_btn = QtWidgets.QPushButton("Run impacts")
        self.run_btn.setEnabled(False)
        self.run_btn.setToolTip("Add at least one impact first.")
        self.run_btn.clicked.connect(self.run_requested.emit)

        self.solid_btn = QtWidgets.QRadioButton("Solid")
        self.energy_btn = QtWidgets.QRadioButton("Impact energy")
        self.failure_btn = QtWidgets.QRadioButton("Failure margin")
        self.solid_btn.setChecked(True)
        for b, mode in ((self.solid_btn, "solid"),
                        (self.energy_btn, "energy"),
                        (self.failure_btn, "failure")):
            b.toggled.connect(lambda on, m=mode: on and self.mode_changed.emit(m))
        mode_row = QtWidgets.QHBoxLayout()
        for b in (self.solid_btn, self.energy_btn, self.failure_btn):
            mode_row.addWidget(b)

        self.summary = QtWidgets.QTextEdit()
        self.summary.setReadOnly(True)

        # Keep the draw/run/results buttons from grabbing the focus highlight while
        # the user is drawing a vector in the viewport.
        for btn in (self.add_btn, self.delete_btn, self.clear_btn, self.run_btn):
            btn.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
            btn.setAutoDefault(False)
            btn.setDefault(False)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.hint)
        form = QtWidgets.QFormLayout()
        form.addRow(self.add_btn)
        form.addRow("New impact force:", self.force_spin)
        layout.addLayout(form)
        layout.addWidget(self.list, 1)
        layout.addLayout(btn_row)
        layout.addWidget(self.edit_box)
        layout.addWidget(self.run_btn)
        layout.addLayout(mode_row)
        layout.addWidget(self.summary, 1)
        self._set_results_enabled(False)
        self._on_row_changed(None, None)        # editor disabled until a row exists

    @staticmethod
    def _force_spin(tooltip: str) -> QtWidgets.QDoubleSpinBox:
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(10.0, 200000.0)
        spin.setSingleStep(100.0)
        spin.setValue(1000.0)
        spin.setSuffix(" N")
        spin.setToolTip(tooltip)
        return spin

    def set_strike_scale(self, mass_kg: float) -> None:
        """Tell the panel the bot mass so it can convert between force and energy."""
        self._mass = max(float(mass_kg), 1e-9)

    def set_suggested_energy(self, energy_j: float) -> None:
        """Seed the new-strike force from a class-appropriate energy."""
        force = force_from_energy(energy_j, self._mass)
        with _blocked(self.force_spin):
            self.force_spin.setValue(self._clamp_force(force))
        self.energy_changed.emit(self.current_energy())

    def current_energy(self) -> float:
        return energy_from_force(self.force_spin.value(), self._mass)

    def add_strike_row(self, strike) -> None:
        force = force_from_energy(strike.energy_j, self._mass)
        item = QtWidgets.QListWidgetItem(
            self._row_text(strike.id, force, strike.elevation_deg))
        item.setData(self._ID, int(strike.id))
        item.setData(self._ENERGY, float(strike.energy_j))
        item.setData(self._ELEV, float(strike.elevation_deg))
        self.list.addItem(item)
        self.run_btn.setEnabled(True)

    def clear_rows(self) -> None:
        self.list.clear()
        self.run_btn.setEnabled(False)
        self.summary.clear()
        self._set_results_enabled(False)
        self.solid_btn.setChecked(True)
        self._on_row_changed(None, None)

    def set_summary(self, text: str) -> None:
        self.summary.setPlainText(text)
        self._set_results_enabled(True)

    # ---- internals -------------------------------------------------------
    def _clamp_force(self, force: float) -> float:
        return max(self.force_spin.minimum(), min(force, self.force_spin.maximum()))

    def _row_text(self, sid: int, force: float, elev: float) -> str:
        ftxt = f"{force / 1000:.1f} kN" if force >= 1000 else f"{force:.0f} N"
        return f"#{sid + 1}: {ftxt}, {int(round(elev))}°"

    def _on_default_force(self, _value: float) -> None:
        self.energy_changed.emit(self.current_energy())

    def _on_row_changed(self, current, _previous) -> None:
        self.edit_box.setEnabled(current is not None)
        if current is None:
            return
        self._loading = True
        force = force_from_energy(current.data(self._ENERGY), self._mass)
        with _blocked(self.edit_force, self.edit_angle):
            self.edit_force.setValue(self._clamp_force(force))
            self.edit_angle.setValue(int(round(current.data(self._ELEV))))
        self._loading = False

    def _on_edit(self, _value) -> None:
        item = self.list.currentItem()
        if self._loading or item is None:
            return
        sid = int(item.data(self._ID))
        energy = energy_from_force(self.edit_force.value(), self._mass)
        elev = float(self.edit_angle.value())
        item.setData(self._ENERGY, energy)
        item.setData(self._ELEV, elev)
        item.setText(self._row_text(sid, self.edit_force.value(), elev))
        self.strike_edited.emit(sid, energy, elev)

    def _set_results_enabled(self, on: bool) -> None:
        for w in (self.energy_btn, self.failure_btn):
            w.setEnabled(on)

    def _delete_selected(self) -> None:
        for item in self.list.selectedItems():
            sid = item.data(self._ID)
            self.list.takeItem(self.list.row(item))
            self.strike_removed.emit(int(sid))
        if self.list.count() == 0:
            self.run_btn.setEnabled(False)

    def _clear_all(self) -> None:
        self.clear_rows()
        self.clear_requested.emit()
