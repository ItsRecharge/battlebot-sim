"""Tests for the trace data classes (ContactEvent, FrameSample, SimTrace, StreamChunk)."""
from __future__ import annotations

import numpy as np

from battlebot_sim.sim.recorder import ContactEvent, FrameSample, SimTrace, StreamChunk


def _contact(part_index=0, nf=100.0, tf=0.0):
    return ContactEvent(
        time=0.1, event="drop",
        pos=np.zeros(3), local_pos=np.zeros(3), normal=np.array([0.0, 0.0, 1.0]),
        normal_force=nf, tangential_force=tf, rel_speed=2.0,
        part_index=part_index, other="floor",
    )


def test_simtrace_aggregates_contacts():
    trace = SimTrace(
        dt=5e-4, n_parts=2,
        contacts=[_contact(0, 100.0), _contact(1, 250.0), _contact(0, 50.0)],
    )
    assert trace.total_contacts() == 3
    assert trace.peak_normal_force() == 250.0
    assert len(trace.contacts_for_part(0)) == 2
    assert trace.contacts_for_part(1)[0].normal_force == 250.0


def test_simtrace_empty_defaults():
    trace = SimTrace(dt=1e-3, n_parts=1)
    assert trace.total_contacts() == 0
    assert trace.peak_normal_force() == 0.0
    assert trace.frames == []


def test_contact_impact_angle():
    head_on = _contact(nf=100.0, tf=0.0)
    grazing = _contact(nf=0.0, tf=100.0)
    assert head_on.impact_angle_deg == 0.0
    assert grazing.impact_angle_deg > 80.0
    # No force at all -> well-defined zero, not NaN.
    assert _contact(nf=0.0, tf=0.0).impact_angle_deg == 0.0


def test_framesample_and_streamchunk_defaults():
    frame = FrameSample(time=0.0, pos=np.zeros(3), quat=np.array([1.0, 0.0, 0.0, 0.0]))
    assert frame.event == ""
    chunk = StreamChunk(
        frame=frame, new_contacts=[_contact()],
        event_index=0, n_events=5, t_global=0.0,
    )
    assert chunk.sim_done is False
    assert len(chunk.new_contacts) == 1
    assert chunk.n_events == 5
