"""Tests for the material library and NHRL weight-class validation."""

import math

from battlebot_sim.materials.assign import (
    LB_TO_KG,
    NHRL_CLASSES,
    validate_weight_class,
)
from battlebot_sim.materials.library import Material, MaterialLibrary, load_default_library


def test_default_library_loads_presets():
    lib = load_default_library()
    assert len(lib) >= 8
    assert "Aluminum 6061-T6" in lib
    alu = lib.get("Aluminum 6061-T6")
    assert alu.density == 2700
    assert math.isclose(alu.yield_pa, 276e6)
    assert math.isclose(alu.youngs_pa, 68.9e9)


def test_library_add_and_roundtrip(tmp_path):
    lib = MaterialLibrary()
    custom = Material("Unobtainium", 1234, 999, 1100, 300, "exotic")
    lib.add(custom)
    path = tmp_path / "mats.json"
    lib.save(path)
    reloaded = MaterialLibrary.from_file(path)
    assert reloaded.get("Unobtainium") == custom


def test_nhrl_class_mass_limits():
    assert math.isclose(NHRL_CLASSES["3lb"].max_mass_kg, 3 * LB_TO_KG)
    assert math.isclose(NHRL_CLASSES["12lb"].max_mass_kg, 12 * LB_TO_KG)
    assert math.isclose(NHRL_CLASSES["30lb"].max_mass_kg, 30 * LB_TO_KG)
    # Cages grow with class.
    assert (
        NHRL_CLASSES["3lb"].cage_length_m
        < NHRL_CLASSES["12lb"].cage_length_m
        < NHRL_CLASSES["30lb"].cage_length_m
    )


def test_validate_weight_class_under_and_over():
    cls3 = NHRL_CLASSES["3lb"]
    under = validate_weight_class(1.0, cls3)
    assert under.ok and "OK" in under.message

    over = validate_weight_class(2.0, cls3)
    assert not over.ok
    assert over.over_by_kg > 0
    assert "OVER WEIGHT" in over.message
