"""Embedded PyVista 3D viewport: shows the cage, the bot, the flinging replay,
and the damage heatmaps."""

from __future__ import annotations

import numpy as np
from PySide6 import QtCore
from pyvistaqt import QtInteractor
from vtkmodules.vtkRenderingCore import vtkCellPicker

from battlebot_sim import viz
from battlebot_sim.arena.nhrl import Arena
from battlebot_sim.damage.model import DamageResult
from battlebot_sim.mesh.segment import BotModel
from battlebot_sim.sim.recorder import SimTrace


def _matrix(pos, quat) -> np.ndarray:
    """4x4 body->world transform from pos + MuJoCo (w,x,y,z) quaternion."""
    from scipy.spatial.transform import Rotation
    R = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = np.asarray(pos, dtype=float)
    return M


def rest_matrix(min_z: float, clearance: float = 1e-3) -> np.ndarray:
    """Translation that lifts a mesh so its lowest point sits on the floor (z=0).

    The bot's body-frame geometry generally dips below z=0, so in the static
    rest pose it would poke through the cage floor when viewed from below. This
    shifts it up by ``-min_z`` (plus a hair of clearance). Replay uses the real
    world poses from physics and ignores this.
    """
    M = np.eye(4)
    M[2, 3] = -float(min_z) + clearance
    return M


class BotViewport(QtInteractor):
    """A QtInteractor that renders the bot, arena, replay and heatmaps."""

    # Emitted when the user clicks a part: (part_index, additive). `additive` is
    # True when Ctrl/Shift is held (extend the selection) rather than replace it.
    part_clicked = QtCore.Signal(int, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.bot: BotModel | None = None
        self.poly = None
        self.bot_actor = None
        self.trace: SimTrace | None = None
        self.result: DamageResult | None = None
        self._mode = "solid"   # "solid" | "energy" | "failure"
        self._face_part: np.ndarray | None = None   # cell id -> part index
        self._selection: set[int] = set()
        self._highlight_actor = None
        self._arena_shown = False
        # Lift applied to the bot in its static rest pose so it sits ON the floor
        # (z=0) instead of poking through it. Recomputed per bot in set_bot().
        self._rest_matrix = np.eye(4)
        self._last_event: str | None = None     # last replay event captioned
        self._init_picking()
        self._setup_studio()                     # realistic lighting + AA + backdrop

    def _setup_studio(self) -> None:
        """A clean, realistic 'studio' look: a soft gradient backdrop, screen-
        space anti-aliasing, SSAO, and neutral three-point lighting.

        Everything (bot and cage) lives on the single base renderer, so real
        depth ordering applies: the opaque floor occludes the (opaque) bot when
        the camera is below the cage, while the translucent walls composite
        over/under it in the standard transparency pass so it still reads clearly
        through them from above and the side. This single shared depth buffer is
        what fixes the old overlay's "bot always on top" artifact; depth peeling
        is enabled as a best-effort refinement for translucent-on-translucent
        ordering (it no-ops when the SSAA/SSAO passes own the render pipeline).
        """
        import pyvista as pv
        # Soft neutral gradient (lighter at top) — reads as a photographic
        # backdrop rather than a flat white void.
        self.set_background("#c7d0d9", top="#eef2f6")
        try:
            self.enable_anti_aliasing("ssaa")
        except Exception:
            pass
        # Best-effort order-independent transparency for overlapping cage walls.
        try:
            self.enable_depth_peeling(number_of_peels=8, occlusion_ratio=0.0)
        except Exception:
            pass
        try:
            self.enable_ssao(radius=0.05)
        except Exception:
            pass
        # Neutral three-point rig: warm key, cool fill, cool back/rim.
        rig = (
            dict(position=(6.0, -6.0, 8.0), color=(1.0, 0.97, 0.92), intensity=0.95),
            dict(position=(-7.0, -3.0, 4.0), color=(0.90, 0.94, 1.0), intensity=0.45),
            dict(position=(0.0, 7.0, 5.0), color=(0.96, 0.98, 1.0), intensity=0.55),
        )
        self.renderer.RemoveAllLights()
        for spec in rig:
            light = pv.Light(position=spec["position"], focal_point=(0, 0, 0),
                             color=spec["color"], intensity=spec["intensity"])
            light.positional = False              # distant/directional: even lighting
            self.renderer.AddLight(light)

    def _init_picking(self) -> None:
        """Left-click selects the part under the cursor. We pick against the base
        renderer (where the bot actor lives) and ignore drags so camera rotation
        still works. A press+release within a few pixels is a click."""
        self._picker = vtkCellPicker()
        self._picker.SetTolerance(0.001)
        self._press_xy = None
        iren = self.iren.interactor
        iren.AddObserver("LeftButtonPressEvent", self._on_press)
        iren.AddObserver("LeftButtonReleaseEvent", self._on_release)

    def _on_press(self, obj, event) -> None:
        self._press_xy = obj.GetEventPosition()

    def _on_release(self, obj, event) -> None:
        press, self._press_xy = self._press_xy, None
        if press is None or self.bot is None or self.bot_actor is None \
                or self._face_part is None:
            return
        if self._mode != "solid":
            return                       # picking only meaningful on the solid bot
        x, y = obj.GetEventPosition()
        if abs(x - press[0]) > 3 or abs(y - press[1]) > 3:
            return                       # a drag (rotate/pan), not a click
        self._picker.Pick(x, y, 0, self.renderer)
        if self._picker.GetActor() is not self.bot_actor:
            return                       # clicked empty space
        idx = viz.part_at_cell(self._face_part, self._picker.GetCellId())
        if idx is None:
            return
        additive = bool(obj.GetControlKey() or obj.GetShiftKey())
        self.part_clicked.emit(idx, additive)

    def _set_bot_actor(self, mesh=None, **kwargs) -> None:
        """(Re)create the bot actor on the base renderer, dropping the previous
        one. Depth peeling handles the cage's translucent walls, so the bot no
        longer needs a separate always-on-top renderer. ``mesh`` defaults to the
        solid pick mesh; the heatmap passes a smoothed/subdivided copy."""
        if self.bot_actor is not None:
            self.remove_actor(self.bot_actor, render=False)
        self.bot_actor = self.add_mesh(mesh if mesh is not None else self.poly, **kwargs)

    def _clear_scalar_bars(self) -> None:
        """Drop any heatmap colour-bars so toggling fields doesn't stack them."""
        try:
            for title in list(self.scalar_bars.keys()):
                self.remove_scalar_bar(title, render=False)
        except Exception:
            pass

    # ---- scene setup -----------------------------------------------------
    def set_bot(self, bot: BotModel) -> None:
        self.bot = bot
        self.poly = viz.bot_polydata(bot)
        self._face_part = viz.face_part_array(bot)
        # Lift the bot so it rests on the floor (z=0) in the static pose.
        self._rest_matrix = rest_matrix(float(bot.original.bounds[0][2]))
        self._last_event = None
        self.set_event_label("")               # clear any stale replay caption
        self._clear_highlight()
        self._selection = set()
        self._mode = "solid"
        self._show_solid()                 # colour by material from the start
        self.add_axes(color="black")
        self.reset_camera()
        self.render()

    def _show_solid(self) -> None:
        """Paint the bot by each part's assigned material (neutral grey where no
        material is set yet) using per-face RGB cell colours. A modest specular
        highlight gives the surface a realistic machined-metal sheen."""
        self._clear_scalar_bars()              # drop any heatmap key when going solid
        self.poly.cell_data["material_rgb"] = viz.face_material_colors(self.bot)
        self._set_bot_actor(scalars="material_rgb", rgb=True, preference="cell",
                            show_scalar_bar=False, show_edges=False,
                            ambient=0.18, diffuse=0.85, specular=0.35,
                            specular_power=18)
        self.bot_actor.user_matrix = self._rest_matrix

    def refresh_materials(self) -> None:
        """Re-colour after a material assignment (solid view only)."""
        if self.bot is None or self.poly is None or self._mode != "solid":
            return
        self._show_solid()                    # also re-applies the rest pose
        self.set_selection(self._selection)   # rebuild highlight on top + render

    # ---- selection -------------------------------------------------------
    def set_selection(self, indices) -> None:
        """Highlight the given part indices (driven by a 3D click or the table).
        Rebuilds a single bright overlay of just those parts' faces."""
        self._selection = {int(i) for i in indices}
        self._clear_highlight()
        if (self.bot is None or self.poly is None or self._face_part is None
                or not self._selection):
            self.render()
            return
        cell_ids = np.nonzero(np.isin(self._face_part, list(self._selection)))[0]
        if len(cell_ids):
            sel = self.poly.extract_cells(cell_ids)
            actor = self.add_mesh(
                sel, color="#ffcc00", show_edges=True, edge_color="#7a5b00",
                line_width=2, show_scalar_bar=False, reset_camera=False)
            if self.bot_actor is not None:
                actor.user_matrix = self.bot_actor.user_matrix
            self._highlight_actor = actor
        self.render()

    def _clear_highlight(self) -> None:
        if self._highlight_actor is not None:
            self.remove_actor(self._highlight_actor, render=False)
            self._highlight_actor = None

    # ---- arena cage ------------------------------------------------------
    def show_arena(self, arena: Arena) -> None:
        for g in arena.geoms:
            box = self._box(g.center, g.half_extents)
            opacity = 1.0 if g.role == "floor" else 0.12
            color = "#5b6168" if g.role == "floor" else "#88a0c0"
            self.add_mesh(box, color=color, opacity=opacity, name=f"arena_{g.name}",
                          ambient=0.2, diffuse=0.8, specular=0.1)
        self._add_floor_grid(arena)
        self._arena_shown = True
        self.reset_camera()
        self.render()

    def _add_floor_grid(self, arena: Arena) -> None:
        """A faint grid on the floor plane: reinforces scale and makes the floor
        read as a solid surface (named ``arena_*`` so hide_arena clears it too)."""
        import pyvista as pv
        L, W, _H = arena.interior
        grid = pv.Plane(center=(0.0, 0.0, 0.002), direction=(0.0, 0.0, 1.0),
                        i_size=L, j_size=W, i_resolution=12, j_resolution=12)
        self.add_mesh(grid, style="wireframe", color="#8b949d", opacity=0.35,
                      line_width=1, name="arena_grid", show_scalar_bar=False,
                      reset_camera=False, lighting=False)

    def hide_arena(self) -> None:
        """Remove the arena cage so parts are easy to see and click during setup.
        The bot, axes and any selection stay."""
        for name in [n for n in self.renderer.actors if n.startswith("arena_")]:
            self.remove_actor(name, render=False)
        self._arena_shown = False
        self.render()

    @staticmethod
    def _box(center, half):
        import pyvista as pv
        c, h = np.asarray(center), np.asarray(half)
        return pv.Box(bounds=(c[0] - h[0], c[0] + h[0],
                              c[1] - h[1], c[1] + h[1],
                              c[2] - h[2], c[2] + h[2]))

    # ---- replay ----------------------------------------------------------
    def set_trace(self, trace: SimTrace) -> None:
        self.trace = trace

    @property
    def n_frames(self) -> int:
        return len(self.trace.frames) if self.trace else 0

    def set_event_label(self, text: str) -> None:
        """Caption the current replay event in the corner (empty text clears it)."""
        if not text:
            try:
                self.remove_actor("event_label", render=False)
            except Exception:
                pass
            return
        self.add_text(text, position="upper_left", font_size=12,
                      color="#1b2733", name="event_label")

    def begin_live(self) -> None:
        """Prepare the viewport for a live run: show the solid bot at rest and
        clear any stale heatmap/caption so the fly-around starts clean."""
        if self.bot is None or self.poly is None:
            return
        self._mode = "solid"
        self._last_event = None
        self.set_event_label("")
        self._show_solid()                         # solid material view, rest pose
        self.render()

    def show_live_pose(self, pos, quat, event: str = "") -> None:
        """Pose the bot at a live (pos, quat) as the battery streams — the
        real-time fly-around. Same mechanism as replay's show_frame, but driven
        by the worker's frames instead of the slider."""
        if self.bot_actor is None:
            return
        m = _matrix(pos, quat)
        self.bot_actor.user_matrix = m
        if self._highlight_actor is not None:
            self._highlight_actor.user_matrix = m
        if event and event != self._last_event:
            self._last_event = event
            self.set_event_label(f"▶  {event}")
        self.render()

    def show_frame(self, i: int) -> None:
        """Pose the bot at recorded frame i (used during replay)."""
        if not self.trace or self.bot_actor is None or self.n_frames == 0:
            return
        i = max(0, min(i, self.n_frames - 1))
        f = self.trace.frames[i]
        m = _matrix(f.pos, f.quat)
        self.bot_actor.user_matrix = m
        if self._highlight_actor is not None:
            self._highlight_actor.user_matrix = m
        if f.event != self._last_event:           # caption updates on event change
            self._last_event = f.event
            self.set_event_label(f"▶  {f.event}")
        self.render()

    # ---- heatmaps --------------------------------------------------------
    def set_result(self, result: DamageResult) -> None:
        self.result = result

    def show_heatmap(self, mode: str) -> None:
        """Switch the bot surface to a damage field ('energy' or 'failure')
        or back to a plain surface ('solid'). Resets the bot to rest pose."""
        if self.bot is None or self.poly is None:
            return
        self._mode = mode
        if mode == "solid" or self.result is None:
            self._show_solid()                    # also re-applies the rest pose
            self.set_selection(self._selection)   # keep selection visible + render
            return
        self._clear_highlight()                   # selection irrelevant on a heatmap
        self._clear_scalar_bars()                 # avoid stacking energy+failure bars
        self._last_event = None
        self.set_event_label("")                  # drop the stale live/replay caption
        # Vertex-smoothed, subdivided field -> a continuous exponential gradient
        # of hotspots with a labelled key, instead of flat per-triangle cells.
        heat, cmap, clim, title, log_scale = viz.attach_field_smooth(
            self.poly, self.bot, self.result, mode)
        self._set_bot_actor(
            mesh=heat, **viz.heat_mesh_kwargs(mode, cmap, clim, title, log_scale))
        self.bot_actor.user_matrix = self._rest_matrix
        self.render()


class BotOnlyView(QtInteractor):
    """A second, lightweight 3D view of *just the bot* — no cage, no floor, no
    picking, no SSAO/depth-peeling (one opaque actor).

    During setup and the live run it shows the solid bot at rest (the cage view
    owns the live fly-around). When a run finishes it paints the final damage
    heatmap with its key and slowly auto-rotates it on a turntable, so the user
    gets a clean, framed, animated read of where the bot took damage — separate
    from the busy in-cage scene.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.bot: BotModel | None = None
        self.poly = None
        self.result: DamageResult | None = None
        self._mode = "failure"          # which field the turntable shows
        self._actor = None
        self._rest = np.eye(4)
        self.set_background("#c7d0d9", top="#eef2f6")
        try:
            self.enable_anti_aliasing("ssaa")
        except Exception:
            pass
        self._spin = QtCore.QTimer(self)
        self._spin.setInterval(33)      # ~30 Hz turntable
        self._spin.timeout.connect(self._turntable)

    # ---- scene ----------------------------------------------------------
    def set_bot(self, bot: BotModel) -> None:
        self._spin.stop()
        self.bot = bot
        self.poly = viz.bot_polydata(bot)
        self.result = None
        self._rest = rest_matrix(float(bot.original.bounds[0][2]))
        self._show_solid()
        self.reset_camera()             # frame just the bot, so it fills the view
        self.render()

    def _clear_scalar_bars(self) -> None:
        try:
            for title in list(self.scalar_bars.keys()):
                self.remove_scalar_bar(title, render=False)
        except Exception:
            pass

    def _show_solid(self) -> None:
        if self.bot is None or self.poly is None:
            return
        if self._actor is not None:
            self.remove_actor(self._actor, render=False)
        self._clear_scalar_bars()
        self.poly.cell_data["material_rgb"] = viz.face_material_colors(self.bot)
        self._actor = self.add_mesh(
            self.poly, scalars="material_rgb", rgb=True, preference="cell",
            show_scalar_bar=False, show_edges=False,
            ambient=0.2, diffuse=0.85, specular=0.3, specular_power=18)
        self._actor.user_matrix = self._rest

    def refresh_materials(self) -> None:
        """Re-colour the solid bot after a material change (ignored once a result
        heatmap is showing)."""
        if self.bot is None or self.result is not None:
            return
        self._show_solid()
        self.render()

    def begin_live(self) -> None:
        """Idle (solid bot, no spin) while the cage view runs the live battery."""
        self._spin.stop()
        self.result = None
        if self.bot is not None:
            self._show_solid()
            self.render()

    # ---- final heatmap turntable ---------------------------------------
    def show_final(self, result: DamageResult, mode: str = "failure") -> None:
        """Paint the final damage field and start the auto-rotating turntable."""
        self.result = result
        self._mode = mode if mode in ("energy", "failure") else "failure"
        self._paint_heatmap()
        self._spin.start()

    def set_mode(self, mode: str) -> None:
        """Follow the results panel's field toggle (energy / failure / solid)."""
        if mode == "solid" or self.result is None:
            self._spin.stop()
            if self.bot is not None:
                self._show_solid()
                self.render()
            return
        self._mode = mode
        self._paint_heatmap()
        if not self._spin.isActive():
            self._spin.start()

    def _paint_heatmap(self) -> None:
        if self.bot is None or self.poly is None or self.result is None:
            return
        if self._actor is not None:
            self.remove_actor(self._actor, render=False)
        self._clear_scalar_bars()
        heat, cmap, clim, title, log_scale = viz.attach_field_smooth(
            self.poly, self.bot, self.result, self._mode)
        self._actor = self.add_mesh(
            heat, **viz.heat_mesh_kwargs(self._mode, cmap, clim, title, log_scale))
        self._actor.user_matrix = self._rest
        self.render()

    def _turntable(self) -> None:
        try:
            self.camera.Azimuth(0.6)        # vtkCamera method (degrees)
            self.render()
        except Exception:
            pass
