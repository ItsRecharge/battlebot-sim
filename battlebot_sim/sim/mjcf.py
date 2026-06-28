"""Generate MuJoCo MJCF XML from an Arena and a BotModel.

The bot becomes a single free-floating body whose collision shape is the union
of per-part convex hulls (MuJoCo treats every mesh as convex for collision).
The arena walls/floor/ceiling are static boxes in the worldbody.

We supply the bot's mass and inertia explicitly from the BotModel so the
simulated dynamics match the mass the user validated against the weight class.
If MuJoCo rejects the explicit inertia as non-physical, the engine retries with
``inertiafromgeom`` (see engine.py).
"""

from __future__ import annotations

import numpy as np

from battlebot_sim.arena.nhrl import Arena
from battlebot_sim.config import DEFAULT_CONFIG
from battlebot_sim.logging_setup import get_logger
from battlebot_sim.mesh.segment import BotModel

logger = get_logger(__name__)

# Per-material contact tuning (sliding friction, restitution-ish bounce 0..1) and
# gravity now live on ``ContactConfig`` (battlebot_sim/config.py). Restitution is
# emulated via solref; we keep it modest and material-flavoured.
_CONTACT = DEFAULT_CONFIG.contact.material_friction_bounce


def _fmt(values) -> str:
    return " ".join(f"{float(v):.6g}" for v in np.ravel(values))


def _hull_vertices(part) -> np.ndarray:
    """Convex-hull vertices for a part, with a degenerate-shape guard."""
    try:
        verts = np.asarray(part.mesh.convex_hull.vertices, dtype=float)
    except Exception:
        logger.debug("convex hull failed for part %r; using raw vertices",
                     getattr(part, "name", "?"), exc_info=True)
        verts = np.asarray(part.mesh.vertices, dtype=float)
    if len(verts) < 4:
        # Pad a near-degenerate part into a tiny tetra so MuJoCo accepts it.
        c = verts.mean(axis=0) if len(verts) else np.zeros(3)
        eps = 1e-3
        verts = np.vstack([c + [eps, 0, 0], c - [eps, 0, 0],
                           c + [0, eps, 0], c + [0, 0, eps]])
    return verts


def build_mjcf(
    arena: Arena,
    bot: BotModel,
    timestep: float = DEFAULT_CONFIG.sim.timestep,
    use_explicit_inertia: bool = True,
) -> tuple[str, dict[str, int]]:
    """Return (mjcf_xml, geom_name -> part_index map)."""
    mass = bot.total_mass()
    com = bot.center_of_mass()
    inertia = bot.inertia_tensor()

    # --- assets: one convex-hull mesh per part ---------------------------
    asset_lines, geom_lines = [], []
    geom_map: dict[str, int] = {}
    for p in bot.parts:
        verts = _hull_vertices(p)
        mesh_name = f"hull_{p.index}"
        asset_lines.append(f'    <mesh name="{mesh_name}" vertex="{_fmt(verts)}"/>')
        geom_name = f"bot_part_{p.index}"
        geom_map[geom_name] = p.index
        cat = p.material.category if p.material else "metal"
        fric, bounce = _CONTACT.get(cat, _CONTACT["metal"])
        density = p.material.density if p.material else 1000.0
        # Realistic restitution: a softer damping ratio bounces more. Map the
        # material's bounce (0..1) onto solref's damping ratio so metal, plastic
        # and composite return impact energy differently.
        dampratio = max(0.2, 1.0 - 0.7 * bounce)
        geom_lines.append(
            f'      <geom name="{geom_name}" type="mesh" mesh="{mesh_name}" '
            f'friction="{fric} 0.02 0.001" density="{density:.3f}" '
            f'rgba="0.6 0.65 0.7 1" condim="3" solref="0.02 {dampratio:.3f}"/>'
        )

    # --- inertial element -------------------------------------------------
    if use_explicit_inertia and mass > 0:
        sym = 0.5 * (inertia + inertia.T)
        fullinertia = [sym[0, 0], sym[1, 1], sym[2, 2], sym[0, 1], sym[0, 2], sym[1, 2]]
        inertial = (
            f'      <inertial pos="{_fmt(com)}" mass="{mass:.6g}" '
            f'fullinertia="{_fmt(fullinertia)}"/>'
        )
        compiler = '  <compiler inertiafromgeom="false" angle="radian"/>'
    else:
        inertial = ""  # MuJoCo computes mass/inertia from geom densities
        compiler = '  <compiler inertiafromgeom="true" angle="radian"/>'

    # --- arena static geoms ----------------------------------------------
    arena_lines = []
    for g in arena.geoms:
        col = "0.4 0.45 0.5 0.25" if g.role != "floor" else "0.3 0.3 0.32 1"
        arena_lines.append(
            f'    <geom name="{g.name}" type="box" pos="{_fmt(g.center)}" '
            f'size="{_fmt(g.half_extents)}" rgba="{col}" condim="3" '
            f'friction="0.5 0.02 0.001"/>'
        )

    xml = f"""<mujoco model="battlebot">
{compiler}
  <option timestep="{timestep}" gravity="0 0 {DEFAULT_CONFIG.contact.gravity}" integrator="implicitfast"
          solver="Newton" iterations="50" tolerance="1e-10"
          cone="elliptic" impratio="2">
    <flag multiccd="enable"/>
  </option>
  <asset>
{chr(10).join(asset_lines)}
  </asset>
  <worldbody>
{chr(10).join(arena_lines)}
    <body name="bot" pos="0 0 0">
      <freejoint name="bot_free"/>
{inertial}
{chr(10).join(geom_lines)}
    </body>
  </worldbody>
</mujoco>
"""
    return xml, geom_map
