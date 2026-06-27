"""The automated stress battery: a deterministic sequence of combat-like events,
scaled by NHRL weight class, that fling the bot around the cage.

Events:
- drops      : released from near the ceiling at several orientations
- wall slams : launched into a wall at class speed and several incidence angles
- tumble     : launched spinning across the cage
- opponent   : a weapon strike modelled as a short, strong impulse at a face,
               which both physically launches the bot and emits a synthetic
               contact so the strike registers in the damage map

Energies scale with the class: heavier classes carry more kinetic energy and
absorb stronger weapon hits.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from battlebot_sim.arena.nhrl import Arena
from battlebot_sim.materials.assign import WeightClass
from battlebot_sim.mesh.segment import BotModel
from battlebot_sim.sim.engine import SimEngine
from battlebot_sim.sim.recorder import (
    ContactEvent, FrameSample, SimTrace, StreamChunk,
)


@dataclass
class Strike:
    """A scheduled opponent-weapon impulse during an event."""

    t_start: float
    t_end: float
    direction: np.ndarray     # unit force direction (into the bot)
    energy_j: float           # kinetic energy delivered


@dataclass
class BatteryEvent:
    """One scripted scenario the bot is put through."""

    name: str
    duration: float
    init_pos: np.ndarray
    init_quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0, 0, 0]))
    init_linvel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    init_angvel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    strike: Strike | None = None


# An opponent strike delivers its full energy to the *damage* model, but the
# *physical* launch is capped to a multiple of the class speed so the bot bounces
# around the cage instead of being fired clean through a wall (see run_battery).
STRIKE_DV_CAP_FACTOR = 1.5
# Restitution applied when the containment safety-net reflects the bot off an
# interior wall: a gentle bounce reads more naturally than a dead stop.
CONTAIN_RESTITUTION = 0.3


def class_speed(wc: WeightClass) -> float:
    """Representative collision speed (m/s) for a class."""
    return {"3lb": 6.0, "12lb": 8.0, "30lb": 10.0}.get(wc.key, 7.0)


def class_strike_energy(wc: WeightClass) -> float:
    """Representative opponent-weapon energy (J): ~200 J per kg of class limit."""
    return 200.0 * wc.max_mass_kg


def _random_unit(rng: np.random.Generator) -> np.ndarray:
    """A seeded random unit vector."""
    v = rng.normal(size=3)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])


def _random_quats(rng: np.random.Generator, n: int) -> list[np.ndarray]:
    """``n`` seeded random orientations as MuJoCo (w, x, y, z) quaternions."""
    from scipy.spatial.transform import Rotation
    q = np.atleast_2d(Rotation.random(n, random_state=rng).as_quat())  # (n, xyzw)
    return [np.array([row[3], row[0], row[1], row[2]]) for row in q]


class StressBattery:
    """Builds and holds the list of events for a class + arena.

    ``n_trials`` scales how many impact angles/orientations are tested: a
    systematic sweep (evenly-spaced wall-slam incidences fired toward walls all
    around the bot) plus that many seeded *random* extra drops, tumbles and
    weapon strikes. The seed makes the whole battery reproducible. Damage from
    every event accumulates into one worst-case map (see compute_damage).
    """

    def __init__(self, arena: Arena, weight_class: WeightClass,
                 n_trials: int = 1, seed: int = 0):
        self.arena = arena
        self.weight_class = weight_class
        self.n_trials = max(1, int(n_trials))
        self.rng = np.random.default_rng(seed)
        self.events = self._build()

    def _build(self) -> list[BatteryEvent]:
        L, W, H = self.arena.interior
        v = class_speed(self.weight_class)
        e_strike = class_strike_energy(self.weight_class)
        drop_z = max(H * 0.85, 0.2)
        n = self.n_trials
        events: list[BatteryEvent] = []

        # --- drops: three named orientations + n seeded random ones ---
        orientations = {
            "flat": np.array([1.0, 0, 0, 0]),
            "tilted": np.array([0.92, 0.38, 0, 0]),       # ~45 deg about X
            "corner": np.array([0.88, 0.33, 0.33, 0.0]),
        }
        for name, quat in orientations.items():
            events.append(BatteryEvent(
                name=f"drop_{name}", duration=1.2,
                init_pos=np.array([0.0, 0.0, drop_z]),
                init_quat=quat / np.linalg.norm(quat),
            ))
        for k, quat in enumerate(_random_quats(self.rng, n)):
            events.append(BatteryEvent(
                name=f"drop_rand{k}", duration=1.0,
                init_pos=np.array([0.0, 0.0, drop_z]), init_quat=quat,
            ))

        # --- wall slams: incidence-angle sweep fired toward walls all around ---
        # Capped at 50 deg: a steeper angle puts most of the launch speed into the
        # vertical and fires the bot at the ceiling rather than the wall.
        n_wall = 3 + n
        incidences = np.linspace(0.0, 50.0, n_wall)
        for k, ang_deg in enumerate(incidences):
            a = np.radians(ang_deg)
            az = 2.0 * np.pi * k / n_wall            # which wall to strike
            dir_h = np.array([np.cos(az), np.sin(az), 0.0])
            vel = (np.cos(a) * dir_h + np.sin(a) * np.array([0, 0, 1.0])) * v
            start = np.array([-0.25 * L * np.cos(az), -0.25 * W * np.sin(az), H * 0.4])
            events.append(BatteryEvent(
                name=f"wall_slam_{int(round(ang_deg))}deg_az{int(round(np.degrees(az)))}",
                duration=0.9, init_pos=start, init_linvel=vel,
            ))

        # --- tumbles: one scripted + n seeded random spins ---
        events.append(BatteryEvent(
            name="tumble", duration=1.8,
            init_pos=np.array([-L * 0.25, -W * 0.2, H * 0.5]),
            init_linvel=np.array([v * 0.7, v * 0.4, 0.0]),
            init_angvel=np.array([8.0, 12.0, 5.0]),
        ))
        for k in range(n):
            lin = self.rng.uniform(-1.0, 1.0, 3) * v * 0.6
            lin[2] = abs(lin[2]) * 0.3              # mostly horizontal launch
            events.append(BatteryEvent(
                name=f"tumble_rand{k}", duration=1.5,
                init_pos=np.array([0.0, 0.0, H * 0.5]),
                init_linvel=lin, init_angvel=self.rng.uniform(-15.0, 15.0, 3),
            ))

        # --- opponent weapon strikes: two fixed + n seeded random directions ---
        for name, d in (("side", np.array([-1.0, 0, 0])), ("top", np.array([0, 0, -1.0]))):
            events.append(BatteryEvent(
                name=f"opponent_{name}", duration=1.0,
                init_pos=np.array([0.0, 0.0, H * 0.25]),
                strike=Strike(t_start=0.15, t_end=0.16, direction=d, energy_j=e_strike),
            ))
        for k in range(n):
            events.append(BatteryEvent(
                name=f"opponent_rand{k}", duration=1.0,
                init_pos=np.array([0.0, 0.0, H * 0.25]),
                strike=Strike(t_start=0.15, t_end=0.16,
                              direction=_random_unit(self.rng), energy_j=e_strike),
            ))
        return events


def _quat_matrix(quat) -> np.ndarray:
    """Body->world rotation matrix from a MuJoCo (w, x, y, z) quaternion."""
    from scipy.spatial.transform import Rotation
    return Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()


def _nearest_part(bot: BotModel, world_point: np.ndarray, pos, quat) -> int:
    """Index of the part whose (transformed) centroid is closest to a point."""
    R = _quat_matrix(quat)
    best, best_d = 0, np.inf
    for p in bot.parts:
        c = R @ p.centroid + pos
        d = np.linalg.norm(c - world_point)
        if d < best_d:
            best, best_d = p.index, d
    return best


def _contain(engine: SimEngine, lo, hi, bmin, bmax,
             restitution: float = CONTAIN_RESTITUTION) -> None:
    """Pull the bot back inside the cage if its AABB has crossed an interior wall.

    A guaranteed safety net against high-speed tunneling through the thin cage
    walls. The outward linear-velocity component is reflected (a gentle bounce)
    so the replay shows the bot ricocheting rather than escaping. Anchored on the
    bot's geometry AABB (``pos + bounds``), never the body origin: for imported
    CAD the body origin is not the geometry centre.
    """
    pos, quat = engine.get_pose()
    lin, ang = engine.get_velocity()
    changed = False
    for ax in range(3):
        if bmax[ax] - bmin[ax] >= hi[ax] - lo[ax]:
            # Bot is larger than the cage on this axis: centre it and stop drift.
            pos[ax] = 0.5 * (lo[ax] + hi[ax]) - 0.5 * (bmin[ax] + bmax[ax])
            lin[ax] = 0.0
            changed = True
        elif pos[ax] + bmin[ax] < lo[ax]:
            pos[ax] = lo[ax] - bmin[ax]
            if lin[ax] < 0.0:
                lin[ax] = -restitution * lin[ax]
            changed = True
        elif pos[ax] + bmax[ax] > hi[ax]:
            pos[ax] = hi[ax] - bmax[ax]
            if lin[ax] > 0.0:
                lin[ax] = -restitution * lin[ax]
            changed = True
    if changed:
        engine.set_velocity(lin, ang)
        engine.set_pose(pos, quat)   # mj_forward refreshes contacts for read


def iter_battery(engine: SimEngine, battery: StressBattery, fps: int = 60):
    """Streaming twin of ``run_battery``: run every event and, at each captured
    frame, ``yield`` a :class:`StreamChunk` carrying that frame plus the contacts
    produced since the previous captured frame. ``return``\\s the complete
    ``SimTrace`` (available as ``StopIteration.value``).

    Draining this generator to exhaustion reproduces ``run_battery``'s trace
    byte-for-byte: frames are appended only at capture points and contacts every
    step, exactly as before. Contacts on non-capture steps batch into ``pending``
    and ride out on the next captured chunk's ``new_contacts`` (plus a final
    tail-flush chunk), so a consumer that ingests every chunk's contacts never
    misses one — only *renders* may be coalesced.

    Cancellation: closing the generator (``gen.close()`` /
    ``gen.throw(GeneratorExit)``) finalizes and returns the partial trace built
    so far, so a cancelled run still yields usable (partial) results.
    """
    dt = engine.timestep
    bot = engine.bot
    trace = SimTrace(dt=dt, n_parts=len(bot.parts))
    record_every = max(1, int(round((1.0 / fps) / dt)))

    # Bot half-size along each axis, for placing strike points on a face.
    half = (bot.original.bounds[1] - bot.original.bounds[0]) / 2.0
    center_local = bot.original.centroid
    mass = max(bot.total_mass(), 1e-6)
    # Cap on the physical launch speed a weapon strike imparts (the damage map
    # still sees the full energy; only the body push is limited).
    dv_cap = STRIKE_DV_CAP_FACTOR * class_speed(battery.weight_class)

    # Interior bounds + the bot's own AABB, for the containment safety net.
    L, W, H = battery.arena.interior
    lo = np.array([-L / 2.0, -W / 2.0, 0.0])
    hi = np.array([L / 2.0, W / 2.0, H])
    bmin, bmax = bot.original.bounds[0], bot.original.bounds[1]

    n_events = len(battery.events)
    pending: list[ContactEvent] = []   # contacts since the last yielded frame
    t_global = 0.0
    try:
        for ev_index, ev in enumerate(battery.events):
            engine.reset()
            engine.clear_applied()
            engine.set_pose(ev.init_pos, ev.init_quat)
            engine.set_velocity(ev.init_linvel, ev.init_angvel)

            n_steps = int(round(ev.duration / dt))
            for s in range(n_steps):
                t_local = s * dt
                engine.clear_applied()

                pos, quat = engine.get_pose()
                R = _quat_matrix(quat)            # body -> world rotation

                # Scheduled opponent strike: physical impulse + synthetic contact.
                if ev.strike and ev.strike.t_start <= t_local < ev.strike.t_end:
                    strike = ev.strike
                    window = max(strike.t_end - strike.t_start, dt)
                    dv = np.sqrt(2.0 * strike.energy_j / mass)
                    force_mag = mass * dv / window          # reported impact severity
                    force_phys = mass * min(dv, dv_cap) / window   # capped body push
                    # Land on the face whose outward normal opposes the strike dir.
                    axis = int(np.argmax(np.abs(strike.direction)))
                    offset = np.zeros(3)
                    offset[axis] = -np.sign(strike.direction[axis]) * half[axis]
                    local_point = center_local + offset
                    world_point = R @ local_point + pos
                    engine.apply_impulse(strike.direction * force_phys, world_point)
                    ce = ContactEvent(
                        time=t_global + t_local, event=ev.name,
                        pos=world_point, local_pos=local_point, normal=-strike.direction,
                        normal_force=force_mag, tangential_force=0.0,
                        rel_speed=dv,
                        part_index=_nearest_part(bot, world_point, pos, quat),
                        other="opponent_weapon",
                    )
                    trace.contacts.append(ce)
                    pending.append(ce)

                engine.step()
                _contain(engine, lo, hi, bmin, bmax)

                for c in engine.read_contacts():
                    local_pos = R.T @ (c["pos"] - pos)
                    ce = ContactEvent(
                        time=t_global + t_local, event=ev.name,
                        pos=c["pos"], local_pos=local_pos, normal=c["normal"],
                        normal_force=c["normal_force"],
                        tangential_force=c["tangential_force"],
                        rel_speed=c["rel_speed"],
                        part_index=c["part_index"], other=c["other"],
                    )
                    trace.contacts.append(ce)
                    pending.append(ce)

                if s % record_every == 0:
                    frame = FrameSample(
                        time=t_global + t_local, pos=pos, quat=quat, event=ev.name)
                    trace.frames.append(frame)
                    chunk = StreamChunk(
                        frame=frame, new_contacts=pending,
                        event_index=ev_index, n_events=n_events,
                        t_global=t_global + t_local, sim_done=False)
                    pending = []
                    yield chunk

            t_global += ev.duration

        # Flush contacts produced after the final captured frame so a streaming
        # consumer's damage accumulator sees every contact (the trace already
        # holds them; this only re-emits the un-yielded tail).
        if pending and trace.frames:
            yield StreamChunk(
                frame=trace.frames[-1], new_contacts=pending,
                event_index=n_events - 1, n_events=n_events,
                t_global=t_global, sim_done=True)
            pending = []
    except GeneratorExit:
        # Cancelled mid-run: hand back the partial trace built so far.
        return trace
    return trace


def run_battery(engine: SimEngine, battery: StressBattery, fps: int = 30) -> SimTrace:
    """Run every event, recording bot poses (for replay) and contacts (for
    damage). A thin drainer over :func:`iter_battery` so the offline path and the
    live streaming path share one simulation loop."""
    gen = iter_battery(engine, battery, fps=fps)
    try:
        while True:
            next(gen)
    except StopIteration as stop:
        return stop.value
