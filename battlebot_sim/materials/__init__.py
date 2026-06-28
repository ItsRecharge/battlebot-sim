"""Material definitions, the preset library, and weight-class validation."""

from battlebot_sim.materials.assign import NHRL_CLASSES, WeightClass, validate_weight_class
from battlebot_sim.materials.library import Material, MaterialLibrary, load_default_library

__all__ = [
    "Material",
    "MaterialLibrary",
    "load_default_library",
    "WeightClass",
    "NHRL_CLASSES",
    "validate_weight_class",
]
