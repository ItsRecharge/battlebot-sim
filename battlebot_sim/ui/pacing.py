"""Wall-clock pacing for the live simulation playback.

MuJoCo steps far faster than real time, so the streaming worker must throttle how
fast it hands captured frames to the UI, or the bot would flash across the cage in
a blur. ``pace_schedule`` is a pure function (no Qt, no clock of its own) so it can
be unit-tested with a fake clock: given when the *next* frame was due, the frame
period, a speed multiplier and the current time, it returns how long to sleep and
when the following frame is due.

If a frame's own work makes the worker fall too far behind (``max_lag``), it
resyncs to *now* instead of trying to catch up — no death spiral, just a brief
drop below real time.
"""

from __future__ import annotations


def pace_schedule(
    next_wall: float,
    frame_period: float,
    speed: float,
    now: float,
    max_lag: float = 0.25,
) -> tuple[float, float]:
    """Return ``(sleep_seconds, new_next_wall)`` for the next frame.

    - ``next_wall``    : wall-clock time the current frame was due.
    - ``frame_period`` : seconds of *sim* time each captured frame represents.
    - ``speed``        : playback multiplier (2.0 = twice real time).
    - ``now``          : current wall-clock time.
    - ``max_lag``      : if behind by more than this, resync (don't catch up).
    """
    period = frame_period / max(float(speed), 1e-6)
    if now - next_wall > max_lag:
        # Fallen too far behind (e.g. a heavy frame): resync so we never try to
        # "make up" a long sleep debt by racing ahead.
        return 0.0, now + period
    sleep = next_wall - now
    if sleep < 0.0:
        sleep = 0.0
    return sleep, next_wall + period
