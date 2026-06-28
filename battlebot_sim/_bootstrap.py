"""Pin the load order of the heavy native libraries.

The intermittent interpreter crash (0xC0000005 / 0xC0000409) when NumPy, SciPy,
VTK, MuJoCo and Pillow all load in one process is sensitive to the *order* in
which their native extension modules first initialise. Calling
:func:`preload_native_libraries` before anything else touches them pins a single
deterministic order for the whole process.

Wired into the application entry point (:mod:`battlebot_sim.__main__`) only — NOT
the test harness: the per-file test isolation already contains the crash, and
forcing VTK/MuJoCo into otherwise-lightweight test processes would make them more
crash-prone, not less.

This is a time-boxed spike. If pinning the order proves to eliminate the crash in
practice, the ``numpy<2`` pin and the per-file test workaround can be revisited.
Best-effort: a library that fails to import here is skipped (the real import
later will raise with full context).
"""
from __future__ import annotations

# Order matters: BLAS-backed math first, then imaging, then the GL/physics stacks.
_PRELOAD_ORDER = (
    "numpy",
    "scipy",
    "scipy.spatial",          # cKDTree / Qhull
    "scipy.spatial.transform",
    "PIL.Image",
    "vtkmodules.all",         # VTK native libs
    "mujoco",                 # MuJoCo native engine
)


def preload_native_libraries() -> None:
    """Import the heavy native libraries once, in a fixed order. Idempotent."""
    for name in _PRELOAD_ORDER:
        try:
            __import__(name)
        except Exception:
            # Missing/optional here is non-fatal; the real import site reports it.
            pass
