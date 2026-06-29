"""Closed-form mechanics-of-materials functions for an *absolute* damage verdict.

The damage model used to report a purely *comparative* number: the Hertzian
surface pressure ``p0`` compared straight to the yield strength. That is wrong in
three ways — first yield in contact is subsurface (von Mises), a part is not its
own equivalent sphere, and structural (bending/membrane) stress was ignored
entirely. This module supplies the small, independently testable pieces that fix
all three, each mapping directly to a textbook formula so it can be pinned
against a hand calculation. Everything here is pure (no mesh painting, no Qt) and
SI throughout (newtons, metres, pascals).

References: K. L. Johnson, *Contact Mechanics* (1985), §3.4 (subsurface stresses)
and §4.2 (combined curvature); any mechanics-of-materials text for ``M c / I``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --- 1. Contact: subsurface von Mises onset ---------------------------------


def vm_factor_from_poisson(nu: float) -> float:
    """Hertzian subsurface von-Mises coefficient ``sigma_vm,max / p0``.

    First yield in a Hertzian (sphere/flat) contact is *not* at the surface; the
    maximum von Mises stress sits a little below it, on the contact axis. Along
    that axis the radial and hoop stresses are equal, so von Mises reduces to
    ``|sigma_z - sigma_r|``. Using Johnson's axial solution (compressive
    negative, ``zeta = z / a``)::

        sigma_z / p0 = -1 / (1 + zeta^2)
        sigma_r / p0 = -[ (1 + nu)*(1 - zeta*arctan(1/zeta)) - 1/(2(1+zeta^2)) ]

    this returns ``max_zeta |sigma_z - sigma_r| / p0``. For ``nu = 0.3`` it gives
    ~0.62, i.e. first yield at ``p0 ~= 1.60 * yield`` — the documented constant,
    here *derived* rather than hard-coded.
    """
    z = np.linspace(1.0e-3, 3.0, 6000)
    sz = -1.0 / (1.0 + z**2)
    sr = -((1.0 + nu) * (1.0 - z * np.arctan2(1.0, z)) - 1.0 / (2.0 * (1.0 + z**2)))
    return float(np.max(np.abs(sz - sr)))


def contact_von_mises_peak(p0: float, vm_factor: float) -> float:
    """Peak (subsurface) von Mises stress under a Hertzian contact: ``vm_factor*p0``."""
    return vm_factor * p0


# --- 2. Combined-curvature contact radius -----------------------------------


def effective_contact_radius(r_indenter: float, r_surface: float) -> float:
    """Combined-curvature radius: ``1/R_eff = 1/R_indenter + 1/R_surface``.

    A sharp striker (small ``r_indenter``) on a near-flat plate (large or
    infinite ``r_surface``) gives a small ``R_eff`` and so a high ``p0``. Either
    radius may be ``inf`` to mean "locally flat".
    """
    inv = 1.0 / max(r_indenter, 1.0e-6) + 1.0 / max(r_surface, 1.0e-6)
    return 1.0 / inv if inv > 0.0 else np.inf


# --- 3. Oriented section (beam/plate idealisation of a part) -----------------


@dataclass(frozen=True)
class Section:
    """A part reduced to a rectangular beam/plate for hand-calc bending.

    ``t <= w <= L`` are the part's principal extents (thickness, width, length);
    bending is worst about the *thin* axis, so ``t`` is the smallest extent.
    """

    t: float      # thickness = smallest principal extent (bending axis), m
    w: float      # width     = middle principal extent, m
    L: float      # length    = largest principal extent (default span), m

    @property
    def area(self) -> float:
        """Cross-section area used for the membrane (F/A) term, m^2."""
        return self.w * self.t


def long_axis(verts: np.ndarray) -> np.ndarray:
    """Unit vector of a point cloud's longest principal axis.

    Used to test whether a candidate brace bridges parts on *opposite* sides of
    its span. Falls back to +x for a degenerate cloud.
    """
    v = np.asarray(verts, dtype=float)
    if v.ndim != 2 or v.shape[0] < 2:
        return np.array([1.0, 0.0, 0.0])
    centered = v - v.mean(axis=0)
    try:
        _, vecs = np.linalg.eigh(np.cov(centered, rowvar=False))
        proj = centered @ vecs
        extents = proj.max(axis=0) - proj.min(axis=0)
        return np.asarray(vecs[:, int(np.argmax(extents))], dtype=float)
    except np.linalg.LinAlgError:  # pragma: no cover
        return np.array([1.0, 0.0, 0.0])


def section_from_vertices(verts: np.ndarray, min_extent: float = 1.0e-4) -> Section:
    """Principal extents of a point cloud via PCA (an oriented bounding box).

    Projecting the vertices onto the eigenvectors of their covariance recovers
    the true thickness of a plate even when it is diagonal in world axes (an AABB
    would over-report it). Falls back to the axis-aligned extents if the cloud is
    degenerate. Each extent is clamped to ``min_extent`` to avoid divide-by-zero
    on slivers.
    """
    v = np.asarray(verts, dtype=float)
    if v.ndim != 2 or v.shape[0] < 2:
        return Section(min_extent, min_extent, min_extent)
    centered = v - v.mean(axis=0)
    try:
        # Eigenvectors of the covariance are the principal axes.
        _, vecs = np.linalg.eigh(np.cov(centered, rowvar=False))
        proj = centered @ vecs
        extents = proj.max(axis=0) - proj.min(axis=0)
    except np.linalg.LinAlgError:  # pragma: no cover - covariance ill-conditioned
        extents = centered.max(axis=0) - centered.min(axis=0)
    t, w, L = np.sort(np.maximum(np.abs(extents), min_extent))
    return Section(float(t), float(w), float(L))


# --- 4. Structural stresses --------------------------------------------------


def bending_stress(force: float, span: float, t: float, w: float,
                   bc_factor: float) -> float:
    """Peak fibre bending stress from a transverse impact ``force`` (Pa).

    Bending moment ``M = bc_factor * force * span`` (``bc_factor = 0.25`` is a
    simply-supported central point load, ``M = F L / 4``). Section modulus of a
    rectangular plate about the thin axis is ``Z = w t^2 / 6``, so
    ``sigma_b = M / Z = 6 * bc_factor * force * span / (w * t^2)``.
    """
    z_mod = max(w * t * t / 6.0, 1.0e-18)
    return bc_factor * force * span / z_mod


def membrane_stress(force: float, area: float) -> float:
    """Direct/membrane (crush) stress ``force / area`` (Pa)."""
    return force / max(area, 1.0e-12)


def governing_stress(contact_vm: float, bending: float, membrane: float
                     ) -> tuple[float, str]:
    """Governing equivalent stress and which mode drives it.

    Bending and membrane are collinear normal stresses (they add); the contact
    von Mises peak is an independent near-surface driver. The governing value is
    the larger, a standard conservative hand-calc combination::

        sigma_struct = bending + membrane
        sigma_gov    = max(contact_vm, sigma_struct)
    """
    struct = bending + membrane
    if contact_vm >= struct:
        return contact_vm, "contact"
    return struct, ("bending" if bending >= membrane else "membrane")


# --- 5. Per-part result ------------------------------------------------------


@dataclass
class PartStress:
    """Absolute per-part stress verdict (all stresses in Pa)."""

    part_index: int
    contact_stress: float       # peak subsurface von Mises under contact
    bending_stress: float
    membrane_stress: float
    governing_stress: float
    governing_mode: str         # "contact" | "bending" | "membrane"
    span_used: float            # m
    thickness_used: float       # m
    margin: float               # governing_stress / yield (>= 1 yields)
    yields: bool
    fractures: bool             # governing_stress >= ultimate strength
