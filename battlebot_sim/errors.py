"""Exception types for BattleBot Sim."""
from __future__ import annotations


class ValidationError(ValueError):
    """Raised when user/boundary input is malformed or out of range.

    Subclasses :class:`ValueError` so existing ``except ValueError`` handlers keep
    working, while letting the UI catch this one type and surface a clear message
    through its existing failure channel.
    """
