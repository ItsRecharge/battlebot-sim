"""PyVista rendering helpers shared by the interactive viewport and the report
exporter. This module is free of Qt and MuJoCo so it can render off-screen.

Two damage fields can be displayed on the bot mesh:
- "energy"  : accumulated impact energy (relative), inferno colormap
- "failure" : failure margin (peak stress / yield), turbo colormap; values >= 1
              mean the material would yield.
"""

from __future__ import annotations

import colorsys
import hashlib

import numpy as np
import pyvista as pv

from battlebot_sim.damage.model import DamageResult
from battlebot_sim.mesh.segment import BotModel

FIELD_SPECS = {
    "energy": {"title": "Impact Energy (J, log)", "cmap": "inferno"},
    "failure": {"title": "Failure Margin (stress / yield)", "cmap": "turbo"},
}


def bot_polydata(bot: BotModel) -> pv.PolyData:
    """A PyVista mesh of the bot's original (full-detail) geometry."""
    m = bot.original
    faces = np.hstack(
        [np.full((len(m.faces), 1), 3, dtype=np.int64), m.faces.astype(np.int64)]
    ).ravel()
    return pv.PolyData(np.asarray(m.vertices, dtype=float), faces)


# Neutral grey shown for a part that has no material assigned yet.
UNASSIGNED_COLOR = (0.62, 0.65, 0.69)


def face_part_array(bot: BotModel) -> np.ndarray:
    """An ``(n_faces,)`` array mapping every face of ``bot.original`` to the
    index of the part that owns it — turns a picked VTK cell id into a part."""
    fp = np.zeros(len(bot.original.faces), dtype=np.int64)
    for p in bot.parts:
        fp[p.face_ids] = p.index
    return fp


def part_at_cell(face_part: np.ndarray, cell_id):
    """The part index for a picked cell id, or None if the id is out of range
    (e.g. a click that missed the mesh)."""
    if cell_id is None or cell_id < 0 or cell_id >= len(face_part):
        return None
    return int(face_part[cell_id])


def material_color(name: str) -> tuple[float, float, float]:
    """A deterministic, visually distinct RGB (0-1) colour for a material name.

    Uses a stable hash (not Python's per-process-salted ``hash``) so a material
    maps to the same colour across parts, bots and runs."""
    digest = hashlib.sha1(name.encode("utf-8")).digest()
    hue = int.from_bytes(digest[:2], "big") / 65535.0
    return colorsys.hsv_to_rgb(hue, 0.55, 0.85)


def _to_u8(rgb) -> np.ndarray:
    return np.array([round(float(c) * 255) for c in rgb], dtype=np.uint8)


def face_material_colors(bot: BotModel) -> np.ndarray:
    """An ``(n_faces, 3)`` uint8 RGB array colouring each face by its part's
    assigned material (neutral grey where none is set). Drives the colour-by-
    material view in the viewport and the swatches in the parts table."""
    colors = np.empty((len(bot.original.faces), 3), dtype=np.uint8)
    for p in bot.parts:
        rgb = material_color(p.material.name) if p.material is not None \
            else UNASSIGNED_COLOR
        colors[p.face_ids] = _to_u8(rgb)
    return colors


def face_field_values(result: DamageResult, mode: str) -> np.ndarray:
    """Raw per-face values for the requested field (no vertex smoothing), so the
    heatmap shows crisp, localized hotspots at the mesh's own resolution."""
    fv = result.energy_per_face if mode == "energy" else result.failure_margin_per_face
    return np.asarray(fv, dtype=float)


def attach_field(poly: pv.PolyData, bot: BotModel, result: DamageResult, mode: str):
    """Attach a per-face field to a PolyData for crisp per-cell colouring.

    Returns ``(cmap, clim, title, log_scale)``. Impact energy accumulates with
    every hit and spans orders of magnitude, so it is coloured on a log scale
    (real joules stay on the bar); failure margin stays linear so the yield
    threshold (margin = 1) keeps its meaning.
    """
    vals = face_field_values(result, mode)
    poly.cell_data[mode] = vals
    poly.set_active_scalars(mode, preference="cell")
    spec = FIELD_SPECS[mode]
    if mode == "failure":
        return spec["cmap"], [0.0, max(1.0, float(vals.max()))], spec["title"], False
    vmax = float(vals.max())
    if vmax <= 0.0:                       # nothing took any energy yet
        return spec["cmap"], [0.0, 1e-9], "Impact Energy (J)", False
    vmin = float(vals[vals > 0].min())   # log scale needs a positive floor
    return spec["cmap"], [vmin, vmax], spec["title"], True


def render_heatmap_png(
    bot: BotModel,
    result: DamageResult,
    mode: str,
    path: str,
    size: tuple[int, int] = (960, 720),
) -> str:
    """Render one heatmap to a PNG off-screen. Returns the path written."""
    poly = bot_polydata(bot)
    cmap, clim, title, log_scale = attach_field(poly, bot, result, mode)
    plotter = pv.Plotter(off_screen=True, window_size=list(size))
    plotter.set_background("#c7d0d9", top="#eef2f6")
    plotter.add_mesh(
        poly, scalars=mode, cmap=cmap, clim=clim, log_scale=log_scale,
        preference="cell", interpolate_before_map=False, show_edges=False,
        scalar_bar_args={"title": title, "color": "#1b2733", "n_labels": 5,
                         "fmt": "%.2g", "vertical": True},
    )
    if mode == "failure":
        # Mark the failure threshold (margin = 1) as a labelled contour-ish note.
        plotter.add_text("red >= yield", position="upper_right", font_size=10,
                         color="#1b2733")
    # This is a single-renderer offscreen image, so depth-based effects are safe
    # here (unlike the live two-layer viewport): SSAA + SSAO add realistic depth.
    try:
        plotter.enable_anti_aliasing("ssaa")
        plotter.enable_ssao(radius=0.05)
    except Exception:
        pass
    plotter.add_axes(color="#1b2733")
    plotter.view_isometric()
    plotter.screenshot(path)
    plotter.close()
    return path
