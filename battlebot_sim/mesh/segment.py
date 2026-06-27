"""Load an STL, split it into parts, and aggregate mass properties.

Geometry is kept in SI metres throughout. STL files carry no units, so `load_bot`
accepts a `scale_to_m` factor (e.g. 0.001 for a millimetre STL). Mass properties
(mass, centre of mass, inertia tensor) are derived from per-part volume and the
assigned material density.

Non-watertight parts have an ill-defined volume; for those we fall back to the
convex hull so mass stays positive and finite, and flag the part so the UI can
warn the user.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh

from battlebot_sim.materials.library import Material


def sample_bot_path() -> str:
    """Absolute path to the bundled sample bot STL, resolved in both dev and
    frozen (.exe) runs. Shared by the entry-point self-test and the UI's
    'Load sample bot' action so they can never drift apart."""
    if getattr(sys, "frozen", False):
        base = Path(getattr(sys, "_MEIPASS"))     # PyInstaller unpack dir
    else:
        base = Path(__file__).resolve().parents[2]
    return str(base / "data" / "sample_bots" / "wedge_bot.stl")


def _hull_or_none(mesh: trimesh.Trimesh):
    """Convex hull of a non-watertight part (used for volume/inertia), or None
    when the geometry is too degenerate for a hull.

    Real CAD STL exports routinely contain stray single triangles or collinear
    slivers as separate connected components. Those have <4 non-coplanar points,
    so scipy's Qhull raises ``QhullError`` ("not enough points to construct
    initial simplex"). Such artifacts carry no meaningful volume, so we treat a
    failed hull as "no volume" rather than letting the error abort the load.
    """
    try:
        # Building/repairing the hull of a near-flat sliver integrates moments
        # that divide by ~0; keep that internal float noise out of the output.
        with np.errstate(invalid="ignore", divide="ignore"):
            return mesh.convex_hull
    except Exception:       # scipy QhullError (and friends) on degenerate input
        return None


def _finite_centroid(mesh: trimesh.Trimesh) -> np.ndarray:
    """A finite centroid for a part, even when its area is ~0 (which makes the
    area-weighted ``mesh.centroid`` non-finite). Falls back to the vertex mean."""
    c = np.asarray(mesh.centroid, dtype=float)
    if not np.all(np.isfinite(c)):
        c = np.asarray(mesh.vertices, dtype=float).mean(axis=0)
    return c


def _mass_properties(mesh: trimesh.Trimesh, density: float):
    """Return (mass_kg, center_mass(3,), inertia(3,3)) for `mesh` at `density`.

    Inertia is taken about the mesh's own centre of mass. Falls back to the
    convex hull when the mesh is not watertight (volume otherwise unreliable).
    Degenerate / zero-volume parts yield zero mass with a *finite* centroid so a
    stray sliver can never poison the bot's aggregate centre of mass or inertia
    (``0 * NaN = NaN`` would otherwise corrupt the whole bot).
    """
    source = mesh if mesh.is_watertight else _hull_or_none(mesh)
    if source is None or float(source.volume) <= 0.0:
        return 0.0, _finite_centroid(mesh), np.zeros((3, 3))
    source = source.copy()
    source.density = float(density)
    # A joggled hull of a near-flat sliver can have a tiny positive volume whose
    # moment integration divides by ~0; compute under errstate and discard any
    # non-finite result below rather than emitting a spurious RuntimeWarning.
    with np.errstate(invalid="ignore", divide="ignore"):
        mass = float(source.mass)
        com = np.asarray(source.center_mass, dtype=float)
        inertia = np.asarray(source.moment_inertia, dtype=float)
    if not (np.isfinite(mass) and np.all(np.isfinite(com))
            and np.all(np.isfinite(inertia))):
        return 0.0, _finite_centroid(mesh), np.zeros((3, 3))
    return mass, com, inertia


def _shift_inertia(inertia: np.ndarray, mass: float, offset: np.ndarray) -> np.ndarray:
    """Parallel-axis shift of an inertia tensor by `offset` (com -> new point)."""
    d = np.asarray(offset, dtype=float)
    return inertia + mass * (float(d @ d) * np.eye(3) - np.outer(d, d))


@dataclass
class Part:
    """One connected solid chunk of the bot mesh, with an optional material."""

    index: int
    mesh: trimesh.Trimesh
    face_ids: np.ndarray                 # indices into the original mesh's faces
    name: str = ""
    material: Material | None = None
    is_brace: bool = False
    watertight_fallback: bool = field(init=False, default=False)

    def __post_init__(self):
        if not self.name:
            self.name = f"part_{self.index}"
        self.watertight_fallback = not bool(self.mesh.is_watertight)

    @property
    def volume_m3(self) -> float:
        if self.mesh.is_watertight:
            return abs(float(self.mesh.volume))
        hull = _hull_or_none(self.mesh)
        if hull is None:
            return 0.0
        with np.errstate(invalid="ignore", divide="ignore"):
            return abs(float(hull.volume))

    @property
    def surface_area_m2(self) -> float:
        return float(self.mesh.area)

    @property
    def centroid(self) -> np.ndarray:
        return np.asarray(self.mesh.centroid, dtype=float)

    @property
    def bounds(self) -> np.ndarray:
        return np.asarray(self.mesh.bounds, dtype=float)

    @property
    def mass_kg(self) -> float:
        if self.material is None:
            return 0.0
        return self.volume_m3 * self.material.density

    def mass_properties(self):
        """(mass, com, inertia-about-own-com). Zero mass if no material yet."""
        if self.material is None:
            return 0.0, self.centroid, np.zeros((3, 3))
        return _mass_properties(self.mesh, self.material.density)


def segment_mesh(mesh: trimesh.Trimesh) -> list[Part]:
    """Split a mesh into parts by connected components of the face graph.

    Each disconnected solid chunk becomes one Part. Face indices into the
    original mesh are preserved so damage can be mapped back to the source faces.
    """
    n_faces = len(mesh.faces)
    if n_faces == 0:
        return []

    adjacency = mesh.face_adjacency  # (k, 2) pairs of touching faces
    components = trimesh.graph.connected_components(
        adjacency, nodes=np.arange(n_faces)
    )
    # Order parts largest-first for stable, intuitive numbering.
    components = sorted(components, key=len, reverse=True)

    parts: list[Part] = []
    for i, face_ids in enumerate(components):
        face_ids = np.asarray(face_ids, dtype=np.int64)
        submesh = mesh.submesh([face_ids], append=False, repair=False)[0]
        parts.append(Part(index=i, mesh=submesh, face_ids=face_ids))
    return parts


class BotModel:
    """A segmented bot: the original mesh plus its parts and aggregate properties."""

    def __init__(self, original: trimesh.Trimesh, parts: list[Part]):
        self.original = original
        self.parts = parts

    # ---- editing ---------------------------------------------------------
    def assign_material(self, part_index: int, material: Material) -> None:
        self.parts[part_index].material = material

    def assign_material_to_all(self, material: Material) -> None:
        for p in self.parts:
            p.material = material

    def set_brace(self, part_index: int, is_brace: bool = True) -> None:
        self.parts[part_index].is_brace = is_brace

    def merge(self, indices: list[int]) -> None:
        """Merge several parts into one, keeping the lowest index's material."""
        indices = sorted(set(indices))
        if len(indices) < 2:
            return
        keep = self.parts[indices[0]]
        merged_faces = np.concatenate([self.parts[i].face_ids for i in indices])
        merged_mesh = self.original.submesh([merged_faces], append=True, repair=False)
        new_part = Part(
            index=keep.index,
            mesh=merged_mesh,
            face_ids=merged_faces,
            name=keep.name,
            material=keep.material,
            is_brace=any(self.parts[i].is_brace for i in indices),
        )
        remaining = [p for j, p in enumerate(self.parts) if j not in indices]
        remaining.insert(0, new_part)
        self.parts = self._reindex(remaining)

    @staticmethod
    def _reindex(parts: list[Part]) -> list[Part]:
        for new_i, p in enumerate(parts):
            p.index = new_i
        return parts

    # ---- aggregate mass properties --------------------------------------
    def total_mass(self) -> float:
        return float(sum(p.mass_kg for p in self.parts))

    def center_of_mass(self) -> np.ndarray:
        m_total = self.total_mass()
        if m_total <= 0:
            return np.asarray(self.original.centroid, dtype=float)
        acc = np.zeros(3)
        for p in self.parts:
            m, com, _ = p.mass_properties()
            acc += m * com
        return acc / m_total

    def inertia_tensor(self) -> np.ndarray:
        """Inertia tensor (3x3, kg*m^2) about the bot's centre of mass."""
        com_total = self.center_of_mass()
        inertia = np.zeros((3, 3))
        for p in self.parts:
            m, com, i_local = p.mass_properties()
            if m <= 0:
                continue
            inertia += _shift_inertia(i_local, m, com_total - com)
        return inertia

    def assigned(self) -> bool:
        """True once every part has a material."""
        return all(p.material is not None for p in self.parts)


def segment_scene(scene: trimesh.Scene, scale_to_m: float = 1.0):
    """Build ``(original_mesh, parts)`` from a multi-body scene — one Part per
    placed geometry instance.

    The geometry's name (the CAD body name) becomes the Part name and the
    assembly transform is baked into the vertices, so a 3MF/glTF export keeps
    each body distinct and named instead of collapsing to ``part_0``. Face ids
    index into the concatenated ``original`` so the damage pipeline (which walks
    ``bot.original`` + ``Part.face_ids``) is unchanged.
    """
    meshes: list[trimesh.Trimesh] = []
    parts: list[Part] = []
    offset = 0
    for node in scene.graph.nodes_geometry:
        transform, geom_name = scene.graph[node]
        geom = scene.geometry[geom_name].copy()
        geom.apply_transform(transform)
        if scale_to_m != 1.0:
            geom.apply_scale(float(scale_to_m))
        n = len(geom.faces)
        if n == 0:
            continue
        parts.append(Part(
            index=len(parts), mesh=geom,
            face_ids=np.arange(offset, offset + n, dtype=np.int64),
            name=str(geom_name),
        ))
        meshes.append(geom)
        offset += n
    original = trimesh.util.concatenate(meshes) if meshes else trimesh.Trimesh()
    return original, parts


def load_bot(path: str, scale_to_m: float = 1.0) -> BotModel:
    """Load a bot mesh from `path`, scale into metres, and segment it into parts.

    Two input styles are supported:

    - **Named multi-body** (``.glb`` / ``.gltf``, ``.3mf``): each body becomes a
      Part that keeps its CAD name — the best way to tell parts apart for
      per-part material assignment.
    - **Single mesh** (``.stl``, or an ``.obj`` trimesh merges): split into parts
      by connected components and named ``part_0``, ``part_1`` ...

    STEP/IGES/Parasolid/JT/ACIS need a CAD kernel trimesh does not bundle; export
    3MF or glTF from the CAD tool instead.
    """
    loaded = trimesh.load(path)
    if isinstance(loaded, trimesh.Scene):
        original, parts = segment_scene(loaded, scale_to_m)
        if len(parts) >= 2:
            return BotModel(original=original, parts=parts)
        # One body (no per-part names to keep): fall back to connected
        # components on the already-scaled mesh, matching STL behaviour.
        mesh = original
    else:
        mesh = loaded
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"{path!r} did not load as a mesh or scene")
        if scale_to_m != 1.0:
            mesh.apply_scale(float(scale_to_m))
    if len(mesh.faces) == 0:
        raise ValueError(f"{path!r} contains no faces")
    parts = segment_mesh(mesh)
    return BotModel(original=mesh, parts=parts)
