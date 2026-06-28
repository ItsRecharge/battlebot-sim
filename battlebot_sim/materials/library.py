"""Material model and the preset material library.

Units (SI, consistent throughout the app):
- density:           kg / m^3
- yield_strength:    MPa  (= N / mm^2 = 1e6 Pa)
- ultimate_strength: MPa
- youngs_modulus:    GPa  (= 1e9 Pa)

Stress in the damage model is computed in Pa; helper properties below expose the
Pa values so callers never juggle the unit prefixes by hand.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path


def _data_dir() -> Path:
    """Locate the bundled data/ directory in both dev and frozen (.exe) runs."""
    if getattr(sys, "frozen", False):
        # PyInstaller unpacks bundled data under sys._MEIPASS.
        return Path(sys._MEIPASS) / "data"
    return Path(__file__).resolve().parents[2] / "data"


_DATA_DIR = _data_dir()
_DEFAULT_MATERIALS_FILE = _DATA_DIR / "materials.json"


@dataclass(frozen=True)
class Material:
    """A structural material with the few properties the damage model needs."""

    name: str
    density: float            # kg / m^3
    yield_strength: float     # MPa
    ultimate_strength: float  # MPa
    youngs_modulus: float     # GPa
    category: str = "metal"

    @property
    def yield_pa(self) -> float:
        """Yield strength in pascals."""
        return self.yield_strength * 1.0e6

    @property
    def ultimate_pa(self) -> float:
        """Ultimate strength in pascals."""
        return self.ultimate_strength * 1.0e6

    @property
    def youngs_pa(self) -> float:
        """Young's modulus in pascals."""
        return self.youngs_modulus * 1.0e9

    @classmethod
    def from_dict(cls, d: dict) -> Material:
        return cls(
            name=d["name"],
            density=float(d["density"]),
            yield_strength=float(d["yield_strength"]),
            ultimate_strength=float(d["ultimate_strength"]),
            youngs_modulus=float(d["youngs_modulus"]),
            category=d.get("category", "metal"),
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "density": self.density,
            "yield_strength": self.yield_strength,
            "ultimate_strength": self.ultimate_strength,
            "youngs_modulus": self.youngs_modulus,
            "category": self.category,
        }


class MaterialLibrary:
    """An ordered, name-indexed collection of materials (presets + user-added)."""

    def __init__(self, materials: list[Material] | None = None):
        self._materials: dict[str, Material] = {}
        for m in materials or []:
            self.add(m)

    def add(self, material: Material) -> None:
        """Add or replace a material by name."""
        self._materials[material.name] = material

    def get(self, name: str) -> Material:
        try:
            return self._materials[name]
        except KeyError as exc:
            raise KeyError(f"No material named {name!r} in library") from exc

    def names(self) -> list[str]:
        return list(self._materials.keys())

    def __len__(self) -> int:
        return len(self._materials)

    def __iter__(self):
        return iter(self._materials.values())

    def __contains__(self, name: object) -> bool:
        return name in self._materials

    def save(self, path: str | Path) -> None:
        payload = {"materials": [m.to_dict() for m in self]}
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def from_file(cls, path: str | Path) -> MaterialLibrary:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([Material.from_dict(d) for d in data["materials"]])


def load_default_library() -> MaterialLibrary:
    """Load the bundled preset materials shipped in data/materials.json."""
    return MaterialLibrary.from_file(_DEFAULT_MATERIALS_FILE)
