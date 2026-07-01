"""Turn a single impact strike (a direction, an energy, and where it lands) into the
synthetic contact the damage model ingests. Shared by the automated stress battery
and the interactive Freeplay mode so both report identical numbers."""

from __future__ import annotations

import numpy as np

from gauntlet.mesh.segment import BotModel
from gauntlet.sim.recorder import ContactEvent

# How long a freeplay strike is delivered over (s). Sets the reported force for a
# given energy; shared by the run path and the on-arrow force read-out so they agree.
FREEPLAY_STRIKE_WINDOW = 0.01


def force_from_energy(energy_j: float, mass: float,
                      window: float = FREEPLAY_STRIKE_WINDOW) -> float:
    """Reported peak contact force (N) for a strike of ``energy_j`` on a bot of
    ``mass`` (kg). Inverse of :func:`energy_from_force`."""
    return float(np.sqrt(2.0 * energy_j * mass) / window)


def energy_from_force(force_n: float, mass: float,
                      window: float = FREEPLAY_STRIKE_WINDOW) -> float:
    """Strike energy (J) that yields a given peak contact force (N)."""
    return float((force_n * window) ** 2 / (2.0 * max(mass, 1e-9)))


def quat_matrix(quat) -> np.ndarray:
    """Body->world rotation matrix from a MuJoCo (w, x, y, z) quaternion."""
    from scipy.spatial.transform import Rotation
    return Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()


def nearest_part(bot: BotModel, world_point: np.ndarray, pos, quat) -> int:
    """Index of the part whose (transformed) centroid is closest to a point."""
    R = quat_matrix(quat)
    best, best_d = 0, np.inf
    for p in bot.parts:
        c = R @ p.centroid + pos
        d = np.linalg.norm(c - world_point)
        if d < best_d:
            best, best_d = p.index, d
    return best


def strike_contact(*, world_point, local_point, direction, energy_j, mass, window,
                   part_index, time, event, other="opponent_weapon"):
    """A strike of ``energy_j`` along unit ``direction`` (pointing INTO the bot)
    landing at ``world_point`` becomes a ContactEvent. ``window`` (s) sets how long
    the energy is delivered over, which fixes the reported normal force; ``mass`` is
    the bot mass (kg). Returns ``(contact, dv)``."""
    dv = np.sqrt(2.0 * energy_j / mass)
    force_mag = mass * dv / window
    direction = np.asarray(direction, dtype=float)
    contact = ContactEvent(
        time=time, event=event,
        pos=np.asarray(world_point, dtype=float),
        local_pos=np.asarray(local_point, dtype=float),
        normal=-direction, normal_force=force_mag, tangential_force=0.0,
        rel_speed=dv, part_index=part_index, other=other,
    )
    return contact, dv


def battery_strike_contact(bot, strike, R, pos, quat, half, center_local, mass, dt,
                           time, event):
    """The battery's path: place the strike on the face whose outward normal opposes
    the strike direction, then build its contact. Returns ``(contact, world_point,
    dv)``; the caller applies the physical impulse at ``world_point``.

    A freeplay strike carrying an explicit ``local_point`` (the user's picked point in
    the bot's body frame) lands there instead of the synthesized face point."""
    window = max(strike.t_end - strike.t_start, dt)
    explicit = getattr(strike, "local_point", None)
    if explicit is not None:
        local_point = np.asarray(explicit, dtype=float)
    else:
        axis = int(np.argmax(np.abs(strike.direction)))
        offset = np.zeros(3)
        offset[axis] = -np.sign(strike.direction[axis]) * half[axis]
        local_point = center_local + offset
    world_point = R @ local_point + pos
    contact, dv = strike_contact(
        world_point=world_point, local_point=local_point, direction=strike.direction,
        energy_j=strike.energy_j, mass=mass, window=window,
        part_index=nearest_part(bot, world_point, pos, quat), time=time, event=event,
    )
    return contact, world_point, dv


def freeplay_strike(bot, rest_translation, world_point, part_index, direction,
                    energy_j, dt):
    """Freeplay's path: a real picked surface point and part index, with the bot at
    its identity-rotation rest pose so ``local_point = world_point - rest_translation``.
    ``direction`` is the unit impact direction (INTO the bot). Returns the contact.

    The identity-rotation assumption holds because the freeplay viewport always shows
    the bot in its rest pose; if that ever changes, transform the point properly.
    """
    mass = max(bot.total_mass(), 1e-6)
    world_point = np.asarray(world_point, dtype=float)
    local_point = world_point - np.asarray(rest_translation, dtype=float)
    contact, _dv = strike_contact(
        world_point=world_point, local_point=local_point, direction=direction,
        energy_j=energy_j, mass=mass, window=dt, part_index=part_index,
        time=0.0, event="freeplay",
    )
    return contact
