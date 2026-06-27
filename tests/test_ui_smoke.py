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
