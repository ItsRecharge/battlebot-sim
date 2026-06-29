"""MuJoCo engine wrapper: build a world from an Arena + BotModel, drive the bot
freejoint, step, and read contact forces.

This is the only module that imports MuJoCo, so the rest of the app stays
engine-agnostic. It exposes the bot pose and per-step contacts as plain numpy.
"""

from __future__ import annotations

import mujoco
import numpy as np

from battlebot_sim.arena.nhrl import Arena
from battlebot_sim.config import DEFAULT_CONFIG
from battlebot_sim.logging_setup import get_logger
from battlebot_sim.mesh.segment import BotModel
from battlebot_sim.sim.mjcf import build_mjcf

logger = get_logger(__name__)


class SimEngine:
    """A MuJoCo world containing the cage and one free-floating bot."""

    def __init__(self, arena: Arena, bot: BotModel,
                 timestep: float = DEFAULT_CONFIG.sim.timestep):
        self.arena = arena
        self.bot = bot
        self.timestep = timestep
        self.model, self.geom_map = self._compile(arena, bot, timestep)
        self.data = mujoco.MjData(self.model)
        self._bot_geom_ids = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name): idx
            for name, idx in self.geom_map.items()
        }
        self._qadr = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "bot_free")
        ]
        self._vadr = self.model.jnt_dofadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "bot_free")
        ]

    @staticmethod
    def _compile(arena, bot, timestep):
        """Compile MJCF, falling back to geom-derived inertia if needed."""
        xml, geom_map = build_mjcf(arena, bot, timestep, use_explicit_inertia=True)
        try:
            return mujoco.MjModel.from_xml_string(xml), geom_map
        except Exception:
            logger.info("explicit-inertia MJCF compile failed; retrying with "
                        "geom-derived inertia", exc_info=True)
            xml, geom_map = build_mjcf(arena, bot, timestep, use_explicit_inertia=False)
            try:
                return mujoco.MjModel.from_xml_string(xml), geom_map
            except Exception as second_exc:
                # Both compiles failed: surface a clean error (with the geometry
                # cause) instead of letting an opaque MuJoCo exception propagate.
                raise RuntimeError(
                    "MuJoCo could not compile this bot's geometry "
                    f"({len(bot.parts)} parts): {second_exc}"
                ) from second_exc

    # ---- state access ----------------------------------------------------
    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def set_pose(self, pos, quat=(1.0, 0.0, 0.0, 0.0)) -> None:
        """Set the bot freejoint pose (quat in w,x,y,z)."""
        a = self._qadr
        self.data.qpos[a:a + 3] = np.asarray(pos, dtype=float)
        q = np.asarray(quat, dtype=float)
        self.data.qpos[a + 3:a + 7] = q / (np.linalg.norm(q) or 1.0)
        mujoco.mj_forward(self.model, self.data)

    def set_velocity(self, linear=(0, 0, 0), angular=(0, 0, 0)) -> None:
        v = self._vadr
        self.data.qvel[v:v + 3] = np.asarray(linear, dtype=float)
        self.data.qvel[v + 3:v + 6] = np.asarray(angular, dtype=float)

    def get_velocity(self):
        """Return the bot freejoint (linear, angular) velocity as numpy arrays."""
        v = self._vadr
        return (np.array(self.data.qvel[v:v + 3]),
                np.array(self.data.qvel[v + 3:v + 6]))

    def apply_impulse(self, force, point) -> None:
        """Apply an external force (N) at a world point for the next step.

        Used by the opponent-weapon event; cleared automatically each step.
        """
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "bot")
        torque = np.zeros(3)
        mujoco.mj_applyFT(
            self.model, self.data,
            np.asarray(force, dtype=float), torque,
            np.asarray(point, dtype=float), bid, self.data.qfrc_applied,
        )

    def clear_applied(self) -> None:
        self.data.qfrc_applied[:] = 0.0

    def get_pose(self):
        a = self._qadr
        pos = np.array(self.data.qpos[a:a + 3])
        quat = np.array(self.data.qpos[a + 3:a + 7])
        return pos, quat

    def step(self) -> None:
        mujoco.mj_step(self.model, self.data)

    # ---- contacts --------------------------------------------------------
    def read_contacts(self):
        """Yield dicts for every current contact that involves the bot.

        Each: pos(3), normal(3, into the bot), normal_force, tangential_force,
        rel_speed (closing speed along the normal), part_index, other(name).
        """
        d, m = self.data, self.model
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            bot_geom = part = other_geom = None
            if g1 in self._bot_geom_ids:
                bot_geom, part, other_geom = g1, self._bot_geom_ids[g1], g2
            elif g2 in self._bot_geom_ids:
                bot_geom, part, other_geom = g2, self._bot_geom_ids[g2], g1
            else:
                continue

            force6 = np.zeros(6)
            mujoco.mj_contactForce(m, d, i, force6)
            normal_force = float(force6[0])
            tangential_force = float(np.linalg.norm(force6[1:3]))
            if normal_force <= 0.0:
                continue

            # Contact frame: row 0 of c.frame is the normal (from geom1 to geom2).
            normal = np.array(c.frame[0:3])
            if bot_geom == g1:
                normal = -normal  # make it point into the bot
            rel_speed = self._closing_speed(bot_geom, c.pos, normal)

            other_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, other_geom) or ""
            yield {
                "pos": np.array(c.pos),
                "normal": normal,
                "normal_force": normal_force,
                "tangential_force": tangential_force,
                "rel_speed": rel_speed,
                "part_index": part,
                "other": other_name,
            }

    def _closing_speed(self, geom_id, point, normal) -> float:
        """Speed of the bot at `point` projected onto `normal` (>=0 = closing)."""
        vel6 = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_GEOM, geom_id, vel6, 0
        )
        ang, lin = vel6[0:3], vel6[3:6]
        gpos = self.data.geom_xpos[geom_id]
        v_point = lin + np.cross(ang, np.asarray(point) - gpos)
        # Closing speed = component moving INTO the surface (opposite the inward normal).
        return float(-np.dot(v_point, normal))
