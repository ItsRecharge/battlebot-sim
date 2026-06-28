"""Tests for the central logging setup and that guarded fallbacks are logged."""
from __future__ import annotations

import logging

import trimesh

from battlebot_sim.logging_setup import ROOT_LOGGER_NAME, configure_logging
from battlebot_sim.mesh.segment import _hull_or_none


def test_configure_logging_is_idempotent():
    """Repeat calls update the level but never stack another console handler."""
    root = logging.getLogger(ROOT_LOGGER_NAME)
    configure_logging("INFO", force=True)
    n_handlers = len(root.handlers)
    assert n_handlers >= 1

    configure_logging("DEBUG")            # second call must not add handlers
    assert len(root.handlers) == n_handlers
    assert root.level == logging.DEBUG    # but the level is updated


def test_force_rebuilds_handlers_without_growth():
    root = logging.getLogger(ROOT_LOGGER_NAME)
    configure_logging("INFO", force=True)
    n_handlers = len(root.handlers)
    configure_logging("INFO", force=True)  # tear down + rebuild
    assert len(root.handlers) == n_handlers


def test_guarded_hull_fallback_is_logged(caplog):
    """A degenerate (collinear) part trips the convex-hull guard, which must now
    emit a debug record instead of failing silently."""
    degenerate = trimesh.Trimesh(
        vertices=[[0, 0, 0], [1, 0, 0], [2, 0, 0]], faces=[[0, 1, 2]], process=False
    )
    with caplog.at_level(logging.DEBUG, logger="battlebot_sim.mesh.segment"):
        result = _hull_or_none(degenerate)

    assert result is None
    assert any("convex hull failed" in r.getMessage() for r in caplog.records)
