"""Unit tests for the pure wall-clock pacing helper (no Qt, fake clock)."""

from battlebot_sim.ui.pacing import pace_schedule


def test_on_time_schedules_one_period_ahead():
    sleep, nxt = pace_schedule(100.0, 1 / 60, 1.0, now=100.0)
    assert sleep == 0.0
    assert abs(nxt - (100.0 + 1 / 60)) < 1e-9


def test_ahead_sleeps_until_due():
    # The frame is due 0.05 s in the future -> sleep that long.
    sleep, nxt = pace_schedule(100.05, 1 / 60, 1.0, now=100.0)
    assert abs(sleep - 0.05) < 1e-9
    assert abs(nxt - (100.05 + 1 / 60)) < 1e-9


def test_behind_resyncs_without_sleeping():
    # Half a second behind (>> max_lag): no sleep, and the next due time is
    # anchored to *now* so we never accumulate a catch-up debt (no death spiral).
    sleep, nxt = pace_schedule(100.0, 1 / 60, 1.0, now=100.5)
    assert sleep == 0.0
    assert nxt > 100.5


def test_speed_scales_the_period():
    _, nxt1 = pace_schedule(100.0, 1 / 60, 1.0, now=100.0)
    _, nxt2 = pace_schedule(100.0, 1 / 60, 2.0, now=100.0)
    assert abs((nxt1 - 100.0) - 1 / 60) < 1e-9
    assert abs((nxt2 - 100.0) - 1 / 120) < 1e-9   # twice as fast -> half the period


def test_within_max_lag_does_not_resync():
    # Behind by less than max_lag: still no sleep, but schedule advances by one
    # period from the original due time (gentle catch-up), not from now.
    sleep, nxt = pace_schedule(100.0, 1 / 60, 1.0, now=100.1, max_lag=0.25)
    assert sleep == 0.0
    assert abs(nxt - (100.0 + 1 / 60)) < 1e-9
