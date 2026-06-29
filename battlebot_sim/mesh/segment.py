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

from battlebot_sim.logging_setup import get_logger
from battlebot_sim.materials.library import Material
from battlebot_sim.validation import validate_mesh, validate_scale

logger = get_logger(__name__)

# A real combat bot is a handful of parts; a messy CAD/STL export can shatter into
# thousands of disconnected mesh fragments (stray triangles, unwelded seams). Each
# fragment becomes a Part -> a MuJoCo <geom> + a VTK actor + a table row, so an
# unbounded count freezes the UI on load and explodes the MJCF the engine compiles.
# We weld coincident vertices and cap the part count to keep every stage bounded.
DEFAULT_MAX_PARTS = 64


# The bundled sample bot STL is authored in centimetres; load it at this scale so
# the demo and the self-test always treat its units correctly (it ignores the UI
# units dropdown).
SAMPLE_SCALE_TO_M = 0.01


def sample_bot_path() -> str:
    """Absolute path to the bundled sample bot STL, resolved in both dev and
    frozen (.exe) runs. Shared by the entry-point self-test and the UI's
    'Load sample bot' action so they can never drift apart. Load it at
    :data:`SAMPLE_SCALE_TO_M` (the STL is in centimetres)."""
    if getattr(sys, "frozen", False):
        base = Path(sys._MEIPASS)     # PyInstaller unpack dir
    else:
        base = Path(__file__).resolve().parents[2]
    return str(base / "data" / "sample_bots" / "bot_test_1.stl")


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
        logger.debug("convex hull failed for a degenerate component "
                     "(%d verts, %d faces); treating as zero-volume",
                     len(mesh.vertices), len(mesh.faces), exc_info=True)
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


def _weld(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Return a copy with coincident vertices fused and duplicate faces dropped.

    CAD/STL exports routinely emit a *triangle soup* whose touching triangles do
    not share vertex indices. That breaks the face-adjacency graph, so a single
    solid shell shatters into thousands of spurious one-triangle components.
    Welding stitches those seams back together (validated: a 150k-face bot dropped
    from 11,047 components to 586), which is the difference between a load that
    freezes the app and one that finishes promptly.
    """
    m = mesh.copy()
    m.merge_vertices()
    m.update_faces(m.unique_faces())
    m.remove_unreferenced_vertices()
    return m


def _connected_components(mesh: trimesh.Trimesh) -> list[np.ndarray]:
    """Face-index components of the adjacency graph, largest-first."""
    n_faces = len(mesh.faces)
    if n_faces == 0:
        return []
    components = trimesh.graph.connected_components(
        mesh.face_adjacency, nodes=np.arange(n_faces)
    )
    return sorted(
        (np.asarray(c, dtype=np.int64) for c in components),
        key=len, reverse=True,
    )


def _cap_components(
    components: list[np.ndarray], max_parts: int | None
) -> list[np.ndarray]:
    """Keep the ``max_parts - 1`` largest components and fuse the remainder into a
    single aggregate, so a badly-fragmented mesh can never exceed ``max_parts``
    parts. No-op when already within budget (the common, well-formed case)."""
    if max_parts is None or len(components) <= max_parts:
        return components
    head = components[: max_parts - 1]
    tail = np.concatenate(components[max_parts - 1:])
    logger.info(
        "capping %d mesh components to %d parts (largest %d kept, %d fragments "
        "merged into one aggregate part)",
        len(components), max_parts, max_parts - 1, len(components) - (max_parts - 1),
    )
    return [*head, tail]


def _parts_from_components(
    mesh: trimesh.Trimesh, components: list[np.ndarray]
) -> list[Part]:
    """Build one Part per face-index component, preserving original face ids."""
    parts: list[Part] = []
    for i, face_ids in enumerate(components):
        face_ids = np.asarray(face_ids, dtype=np.int64)
        submesh = mesh.submesh([face_ids], append=False, repair=False)[0]
        parts.append(Part(index=i, mesh=submesh, face_ids=face_ids))
    return parts


def segment_mesh(
    mesh: trimesh.Trimesh, max_parts: int | None = DEFAULT_MAX_PARTS
) -> list[Part]:
    """Split a mesh into parts by connected components of the face graph.

    Each disconnected solid chunk becomes one Part. Face indices into the
    original mesh are preserved so damage can be mapped back to the source faces.
    A mesh that fragments past ``max_parts`` has its smallest fragments fused into
    one aggregate part (see :func:`_cap_components`); pass ``max_parts=None`` to
    disable the cap.
    """
    return _parts_from_components(
        mesh, _cap_components(_connected_components(mesh), max_parts)
    )


class BotModel:
    """A segmented bot: the original mesh plus its parts and aggregate properties."""

    def __init__(self, original: trimesh.Trimesh, parts: list[Part],
                 source_fragments: int | None = None):
        self.original = original
        self.parts = parts
        # How many disconnected mesh components the source had *before* the
        # part-count cap fused the small ones. When this exceeds ``len(parts)`` the
        # UI tells the user the bot was simplified for simulation.
        self.source_fragments = (
            len(parts) if source_fragments is None else int(source_fragments)
        )

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


def _cap_named_parts(
    original: trimesh.Trimesh, parts: list[Part], max_parts: int | None
) -> list[Part]:
    """Cap a named multi-body part list, keeping the largest bodies (by face count)
    and fusing the smallest into one aggregate ``misc`` part. Preserves the CAD
    names of the bodies that survive."""
    if max_parts is None or len(parts) <= max_parts:
        return parts
    ordered = sorted(parts, key=lambda p: len(p.face_ids), reverse=True)
    head = ordered[: max_parts - 1]
    tail_faces = np.concatenate([p.face_ids for p in ordered[max_parts - 1:]])
    aggregate = Part(
        index=0,
        mesh=original.submesh([tail_faces], append=True, repair=False),
        face_ids=tail_faces,
        name="misc",
        is_brace=any(p.is_brace for p in ordered[max_parts - 1:]),
    )
    capped = [*head, aggregate]
    logger.info(
        "capping %d named bodies to %d parts (smallest %d merged into 'misc')",
        len(parts), max_parts, len(parts) - (max_parts - 1),
    )
    return BotModel._reindex(capped)


def load_bot(path: str, scale_to_m: float = 1.0,
             max_parts: int | None = DEFAULT_MAX_PARTS) -> BotModel:
    """Load a bot mesh from `path`, scale into metres, and segment it into parts.

    Two input styles are supported:

    - **Named multi-body** (``.glb`` / ``.gltf``, ``.3mf``): each body becomes a
      Part that keeps its CAD name — the best way to tell parts apart for
      per-part material assignment.
    - **Single mesh** (``.stl``, or an ``.obj`` trimesh merges): split into parts
      by connected components and named ``part_0``, ``part_1`` ...

    Vertices are welded (single-mesh path) and the part count is capped at
    ``max_parts`` so a messy export that shatters into thousands of fragments still
    loads and simulates; ``BotModel.source_fragments`` records the pre-cap count.

    STEP/IGES/Parasolid/JT/ACIS need a CAD kernel trimesh does not bundle; export
    3MF or glTF from the CAD tool instead.
    """
    validate_scale(scale_to_m)
    loaded = trimesh.load(path)
    if isinstance(loaded, trimesh.Scene):
        original, parts = segment_scene(loaded, scale_to_m)
        if len(parts) >= 2:
            validate_mesh(original)
            n_fragments = len(parts)
            return BotModel(original=original,
                            parts=_cap_named_parts(original, parts, max_parts),
                            source_fragments=n_fragments)
        # One body (no per-part names to keep): fall back to connected
        # components on the already-scaled mesh, matching STL behaviour.
        mesh = original
    else:
        mesh = loaded
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"{path!r} did not load as a mesh or scene")
        if scale_to_m != 1.0:
            mesh.apply_scale(float(scale_to_m))
    mesh = _weld(mesh)         # stitch triangle-soup seams before splitting
    validate_mesh(mesh)        # finite verts, non-empty, sane bounding box
    components = _connected_components(mesh)
    parts = _parts_from_components(mesh, _cap_components(components, max_parts))
    return BotModel(original=mesh, parts=parts, source_fragments=len(components))
