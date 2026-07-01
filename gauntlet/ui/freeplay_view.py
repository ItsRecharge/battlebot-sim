"""Freeplay 3D view: draw impact vectors on the bot and read the resulting damage.

The user clicks a point on the bot surface and drags to aim an incoming strike (an
arrow whose head sits on the surface). Each arrow is labelled with its force and
angle, can be edited after drawing, and several can be fired together. Geometry only
lives here; turning strikes into damage is done in MainWindow via the stress battery.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PySide6 import QtCore

from gauntlet import viz
from gauntlet.logging_setup import get_logger
from gauntlet.mesh.segment import BotModel
from gauntlet.sim.strike import force_from_energy
from gauntlet.ui.viewport import BotViewport

logger = get_logger(__name__)


def ray_plane_intersect(near, far, plane_pt, plane_n):
    """World point where the segment ``near`` -> ``far`` crosses the plane through
    ``plane_pt`` with normal ``plane_n``. Returns ``None`` when the ray is parallel
    to the plane. Pure geometry, so it is unit-testable without a GL context."""
    near = np.asarray(near, dtype=float)
    far = np.asarray(far, dtype=float)
    plane_pt = np.asarray(plane_pt, dtype=float)
    plane_n = np.asarray(plane_n, dtype=float)
    d = far - near
    denom = float(np.dot(d, plane_n))
    if abs(denom) < 1e-12:
        return None
    t = float(np.dot(plane_pt - near, plane_n) / denom)
    return near + t * d


def elevation_deg(direction) -> float:
    """Angle (deg) of a unit direction below the horizontal: +90 straight down, 0
    horizontal, -90 straight up."""
    d = np.asarray(direction, dtype=float)
    return float(np.degrees(np.arcsin(np.clip(-d[2], -1.0, 1.0))))


def direction_from(elevation_deg_value: float, azimuth_rad: float) -> np.ndarray:
    """Unit direction (into the bot) from an elevation below horizontal and a
    compass azimuth, the inverse of :func:`elevation_deg`."""
    e = np.radians(elevation_deg_value)
    return np.array([np.cos(e) * np.cos(azimuth_rad),
                     np.cos(e) * np.sin(azimuth_rad),
                     -np.sin(e)])


@dataclass
class FreeplayStrike:
    """One user-drawn impact: where it lands, which part it hits, the unit direction
    INTO the bot, the energy delivered, and the arrow's visual length."""

    id: int
    world_point: np.ndarray      # impact point in rest-pose world coords (m)
    part_index: int
    direction: np.ndarray        # unit vector pointing into the bot
    energy_j: float
    arrow_len: float = 0.0

    @property
    def elevation_deg(self) -> float:
        return elevation_deg(self.direction)

    @property
    def azimuth_rad(self) -> float:
        return float(np.arctan2(self.direction[1], self.direction[0]))


@dataclass
class _Drag:
    origin: np.ndarray
    part: int
    plane_n: np.ndarray


class FreeplayViewport(BotViewport):
    """A solid bot view with a draw tool: click a surface point and drag to place an
    incoming-strike arrow. Inherits the bot/heatmap rendering and the cell picker from
    BotViewport; adds high-priority mouse observers that take over from the camera
    only while the draw tool is armed."""

    strike_added = QtCore.Signal(object)     # a FreeplayStrike

    ARROW_COLOR = "#d83a2f"
    PREVIEW_COLOR = "#f08c00"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._draw_armed = False
        self._energy_j = 0.0
        self._drag: _Drag | None = None
        self._preview_actor = None
        self._actors: dict[int, list] = {}   # strike id -> [arrow actor, label actor]
        self.strikes: list[FreeplayStrike] = []
        self._next_id = 0
        self._init_draw_observers()

    def _init_draw_observers(self) -> None:
        # These observers live on the interactor, so they fire regardless of the
        # camera style. Each no-ops unless the draw tool is armed, and while armed the
        # camera style is switched off (see arm_draw) so a drag draws instead of
        # rotating -- no event-abort needed.
        iren = self.iren.interactor
        iren.AddObserver("LeftButtonPressEvent", self._on_draw_press, 10.0)
        iren.AddObserver("MouseMoveEvent", self._on_draw_move, 10.0)
        iren.AddObserver("LeftButtonReleaseEvent", self._on_draw_release, 10.0)

    # ---- public API ------------------------------------------------------
    def arm_draw(self, on: bool) -> None:
        """Arm/disarm the draw tool. While armed, the camera interactor style is
        removed so a left drag places a strike instead of rotating the view, and the
        cursor is a crosshair; disarming restores the trackball camera."""
        self._draw_armed = bool(on)
        if on:
            self.iren.style = None                 # no camera manipulation while drawing
            self.setCursor(QtCore.Qt.CursorShape.CrossCursor)
        else:
            self.iren.enable_trackball_style()     # restore normal camera control
            self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)

    def set_energy(self, energy_j: float) -> None:
        """Energy assigned to the next strike (set live from the panel)."""
        self._energy_j = float(energy_j)

    def rest_translation(self) -> np.ndarray:
        """The rest-pose lift applied to the bot actor, so callers can map a picked
        world point back to body-local coordinates."""
        return self._rest_matrix[:3, 3].copy()

    def strike_by_id(self, sid: int) -> FreeplayStrike | None:
        return next((s for s in self.strikes if s.id == sid), None)

    def set_bot(self, bot: BotModel) -> None:
        self.clear_arrows()                  # stale arrows don't belong to a new bot
        super().set_bot(bot)

    def update_strike(self, sid: int, energy_j: float, elevation: float) -> None:
        """Re-aim and re-power a placed strike (from the panel editors): keep its
        azimuth, set a new elevation and energy, and redraw its arrow + label."""
        s = self.strike_by_id(sid)
        if s is None:
            return
        s.energy_j = float(energy_j)
        s.direction = direction_from(elevation, s.azimuth_rad)
        self._draw_strike_actors(s)
        self.render()

    def remove_strike(self, sid: int) -> None:
        for actor in self._actors.pop(sid, []):
            if actor is not None:
                self.remove_actor(actor, render=False)
        self.strikes = [s for s in self.strikes if s.id != sid]
        self.render()

    def clear_arrows(self) -> None:
        for actors in self._actors.values():
            for actor in actors:
                if actor is not None:
                    self.remove_actor(actor, render=False)
        self._actors.clear()
        self.strikes.clear()
        self._clear_preview()
        self._drag = None
        self.render()

    # ---- draw interaction ------------------------------------------------
    def _on_draw_press(self, obj, event) -> None:
        if not self._draw_armed or self.bot_actor is None or self._face_part is None:
            return
        x, y = obj.GetEventPosition()
        if not self._pick_bot(x, y):
            self._drag = None                # started off the bot: let it pass
            return
        camera = self.renderer.GetActiveCamera()
        self._drag = _Drag(
            origin=np.asarray(self._picker.GetPickPosition(), dtype=float),
            part=viz.part_at_cell(self._face_part, self._picker.GetCellId()),
            plane_n=np.asarray(camera.GetDirectionOfProjection(), dtype=float))

    def _on_draw_move(self, obj, event) -> None:
        if not self._draw_armed or self._drag is None:
            return
        x, y = obj.GetEventPosition()
        end = self._pixel_to_plane(x, y, self._drag.origin, self._drag.plane_n)
        if end is not None:
            self._show_preview(self._drag.origin, end)

    def _on_draw_release(self, obj, event) -> None:
        if not self._draw_armed or self._drag is None:
            return
        drag, self._drag = self._drag, None
        x, y = obj.GetEventPosition()
        end = self._pixel_to_plane(x, y, drag.origin, drag.plane_n)
        self._clear_preview()
        if end is None:
            self.render()
            return
        vec = end - drag.origin
        length = float(np.linalg.norm(vec))
        if length < 1e-6:
            self.render()
            return                           # a click, not a drag
        # The arrow head sits on the surface; the impact direction (into the bot) is
        # from the drag end toward the picked point.
        direction = (drag.origin - end) / length
        strike = FreeplayStrike(
            id=self._next_id, world_point=drag.origin, part_index=drag.part,
            direction=direction, energy_j=self._energy_j, arrow_len=self._arrow_len())
        self._next_id += 1
        self.strikes.append(strike)
        self._draw_strike_actors(strike)
        self.render()
        self.strike_added.emit(strike)

    # ---- helpers ---------------------------------------------------------
    def _arrow_len(self) -> float:
        """A consistent, clearly visible arrow length tied to the bot's size, so the
        drag only sets direction (not a tiny or huge arrow)."""
        if self.bot is None:
            return 1.0
        lo, hi = self.bot.original.bounds     # (2, 3): min xyz, max xyz
        diag = float(np.linalg.norm(np.asarray(hi) - np.asarray(lo)))
        return max(0.6 * diag, 1e-3)

    def _strike_label(self, strike: FreeplayStrike) -> str:
        mass = self.bot.total_mass() if self.bot is not None else 1.0
        force = force_from_energy(strike.energy_j, max(mass, 1e-9))
        force_txt = f"{force / 1000:.1f} kN" if force >= 1000 else f"{force:.0f} N"
        return f"{force_txt}, {int(round(strike.elevation_deg))}°"

    def set_arrows_visible(self, visible: bool) -> None:
        """Show or hide every strike arrow + label (hidden during the bounce replay,
        where the bot has moved away from them)."""
        for actors in self._actors.values():
            for actor in actors:
                if actor is not None:
                    actor.SetVisibility(bool(visible))
        self.render()

    def _draw_strike_actors(self, strike: FreeplayStrike) -> None:
        """(Re)create the arrow and its force/angle label for one strike."""
        for actor in self._actors.pop(strike.id, []):
            if actor is not None:
                self.remove_actor(actor, render=False)
        head = np.asarray(strike.world_point, dtype=float)
        tail = head - strike.direction * strike.arrow_len
        arrow = self._make_arrow(tail, head, self.ARROW_COLOR)
        label = None
        try:
            label = self.add_point_labels(
                [tail], [self._strike_label(strike)], font_size=11,
                text_color="#7a1008", shape=None, show_points=False,
                always_visible=True, reset_camera=False)
        except Exception:
            logger.debug("strike label failed", exc_info=True)
        self._actors[strike.id] = [arrow, label]

    def _pixel_to_plane(self, x, y, plane_pt, plane_n):
        """Project a screen pixel onto the plane through ``plane_pt`` (the drag
        origin) parallel to the screen, so the drag reads as 'drawing on glass over
        the bot'."""
        near = self._display_to_world(x, y, 0.0)
        far = self._display_to_world(x, y, 1.0)
        return ray_plane_intersect(near, far, plane_pt, plane_n)

    def _display_to_world(self, x, y, z) -> np.ndarray:
        ren = self.renderer
        ren.SetDisplayPoint(float(x), float(y), float(z))
        ren.DisplayToWorld()
        wx, wy, wz, ww = ren.GetWorldPoint()
        if ww != 0.0:
            return np.array([wx / ww, wy / ww, wz / ww])
        return np.array([wx, wy, wz])

    def _make_arrow(self, tail, head, color):
        import pyvista as pv
        vec = np.asarray(head, dtype=float) - np.asarray(tail, dtype=float)
        length = float(np.linalg.norm(vec))
        if length < 1e-9:
            return None
        arrow = pv.Arrow(start=tail, direction=vec, scale=length)
        return self.add_mesh(arrow, color=color, reset_camera=False,
                             show_scalar_bar=False)

    def _show_preview(self, origin, end) -> None:
        self._clear_preview()
        self._preview_actor = self._make_arrow(end, origin, self.PREVIEW_COLOR)
        self.render()

    def _clear_preview(self) -> None:
        if self._preview_actor is not None:
            self.remove_actor(self._preview_actor, render=False)
            self._preview_actor = None
