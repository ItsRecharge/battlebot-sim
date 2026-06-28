"""Shared test fixtures: synthetic bots built in SI metres."""

import pytest
import trimesh

from battlebot_sim.materials.library import load_default_library


@pytest.fixture
def library():
    return load_default_library()


@pytest.fixture
def aluminum(library):
    return library.get("Aluminum 6061-T6")


@pytest.fixture
def alu_cube_10cm():
    """A solid 10 cm aluminium cube, centred at the origin (in metres)."""
    return trimesh.creation.box(extents=(0.1, 0.1, 0.1))


@pytest.fixture(scope="session")
def selftest_result():
    """The finalized ``(trace, result)`` from the seeded offline pipeline.

    Session-scoped so the heavy battery runs once; reused by the golden-baseline
    regression oracle and the fidelity-validation suite.
    """
    from helpers import run_selftest_pipeline

    return run_selftest_pipeline()


@pytest.fixture
def two_box_bot():
    """Two disjoint boxes joined by a thin connecting bar (a 'brace').

    Layout along X: box A at x=-0.15, box B at x=+0.15, slender bar between
    them. Used to exercise segmentation, mass aggregation, and braces.
    """
    a = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    a.apply_translation((-0.15, 0, 0))
    b = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    b.apply_translation((0.15, 0, 0))
    bar = trimesh.creation.box(extents=(0.2, 0.02, 0.02))  # spans the gap
    # Keep them disjoint so segmentation yields 3 connected components.
    return a, b, bar
