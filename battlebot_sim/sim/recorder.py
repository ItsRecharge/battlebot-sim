"""Trace data captured during simulation: per-frame poses and contact events.

A SimTrace is the sole hand-off from the physics layer to the damage layer and
the viewer. It must be picklable/JSON-able plain data (numpy arrays only), so it
holds no references to MuJoCo objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ContactEvent:
    """A single contact between a bot part and an arena/opponent surface."""

    time: float
    event: str                 # name of the battery event that produced it
    pos: np.ndarray            # (3,) world contact point, metres
    local_pos: np.ndarray      # (3,) same point in the bot's body frame
    normal: np.ndarray         # (3,) unit contact normal (points into the bot)
    normal_force: float        # N, along the contact normal
    tangential_force: float    # N, friction magnitude in the contact plane
    rel_speed: float           # m/s, closing speed of the bot along the normal
    part_index: int            # which bot part was hit
    other: str                 # arena/opponent geom name

    @property
    def impact_angle_deg(self) -> float:
        """Angle of the closing velocity relative to the surface normal.

        0 deg = head-on (pure normal), 90 deg = grazing (pure shear).
        Derived from the normal vs. tangential force split.
        """
        n, t = abs(self.normal_force), abs(self.tangential_force)
        if n <= 0 and t <= 0:
            return 0.0
        return float(np.degrees(np.arctan2(t, max(n, 1e-12))))


@dataclass
class FrameSample:
    """The bot's rigid pose at one recorded instant (for replay)."""

    time: float
    pos: np.ndarray            # (3,) body-frame origin in world
    quat: np.ndarray           # (4,) orientation, MuJoCo order (w, x, y, z)
    event: str = ""


@dataclass
class SimTrace:
    """All recorded data from a battery run."""

    dt: float
    n_parts: int
    frames: list[FrameSample] = field(default_factory=list)
    contacts: list[ContactEvent] = field(default_factory=list)

    def contacts_for_part(self, part_index: int) -> list[ContactEvent]:
        return [c for c in self.contacts if c.part_index == part_index]

    def peak_normal_force(self) -> float:
        return max((abs(c.normal_force) for c in self.contacts), default=0.0)

    def total_contacts(self) -> int:
        return len(self.contacts)


@dataclass
class StreamChunk:
    """One increment of a *live* battery run, handed from the sim layer to the
    UI as the run progresses.

    Carries one captured frame plus every contact produced since the previous
    captured frame. ``frame`` is the same object appended to ``SimTrace.frames``
    and ``new_contacts`` are already in ``SimTrace.contacts`` — so draining the
    whole stream rebuilds the identical trace. Plain data only (numpy + the two
    dataclasses, no engine references) so it is safe to marshal across a Qt
    queued signal to the UI thread.
    """

    frame: FrameSample
    new_contacts: list[ContactEvent]
    event_index: int           # 0-based index of the event in progress
    n_events: int              # total events in the battery (for a progress bar)
    t_global: float            # battery time at this frame (s)
    sim_done: bool = False     # True on the final tail-flush chunk
