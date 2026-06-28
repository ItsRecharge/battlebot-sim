"""Central logging setup for BattleBot Sim.

Two channels, kept deliberately separate:

* **User-facing** — Qt signals / dialogs (e.g. ``StreamWorker.failed``). Unchanged.
* **Engineering** — the stdlib :mod:`logging` tree, configured here.

Every module under ``battlebot_sim.*`` gets a namespaced logger via
:func:`get_logger` and **never** configures handlers itself. The application
entry point (:mod:`battlebot_sim.__main__`) calls :func:`configure_logging`
exactly once to attach handlers to the package-root logger; submodule records
propagate up to it. The real root logger is left untouched, so importing
BattleBot Sim as a library never hijacks the host application's logging.
"""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

ROOT_LOGGER_NAME = "battlebot_sim"
_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Return a logger for ``name`` (pass ``__name__`` from each module)."""
    return logging.getLogger(name)


def _coerce_level(level: int | str) -> int:
    """Turn ``"DEBUG"`` / ``10`` / junk into a usable numeric level."""
    if isinstance(level, int):
        return level
    numeric = logging.getLevelName(str(level).upper())
    return numeric if isinstance(numeric, int) else logging.INFO


def configure_logging(
    level: int | str = "INFO",
    logfile: str | Path | None = None,
    *,
    force: bool = False,
) -> logging.Logger:
    """Attach handlers to the ``battlebot_sim`` package-root logger.

    Idempotent by design: the GUI launch and the ``--selftest`` path may both
    call this, and a repeat call only updates levels rather than stacking another
    console handler. Pass ``force=True`` to tear down and rebuild handlers.

    Parameters
    ----------
    level:
        Console/file threshold, e.g. ``"DEBUG"`` or ``logging.WARNING``.
    logfile:
        If given, also write to a rotating file at this path (1 MB x 3 backups).
    """
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    numeric = _coerce_level(level)
    logger.setLevel(numeric)

    if logger.handlers and not force:
        for handler in logger.handlers:
            handler.setLevel(numeric)
        return logger

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    fmt = logging.Formatter(_LOG_FORMAT)

    console = logging.StreamHandler()
    console.setLevel(numeric)
    console.setFormatter(fmt)
    logger.addHandler(console)

    if logfile is not None:
        path = Path(logfile)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            path, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setLevel(numeric)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    # Leave logger.propagate at its default (True): the real root logger has no
    # handlers in normal use, so records aren't double-emitted, but pytest's
    # caplog (which captures at the root) still sees them.
    return logger
