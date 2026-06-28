"""Tests for the central config: defaults, TOML overlay, and live wiring."""
from __future__ import annotations

import sys
from dataclasses import replace

import numpy as np
import pytest
import trimesh

from battlebot_sim.arena.nhrl import build_arena
from battlebot_sim.config import DEFAULT_CONFIG, load_config
from battlebot_sim.damage.model import compute_damage
from battlebot_sim.materials.assign import NHRL_CLASSES
from battlebot_sim.materials.library import load_default_library
from battlebot_sim.mesh.segment import BotModel, segment_mesh
from battlebot_sim.sim.recorder import ContactEvent, SimTrace

_HAS_TOMLLIB = sys.version_info >= (3, 11)


def test_defaults_match_historical_literals():
    """The defaults must reproduce the constants that used to be hardcoded."""
    d = DEFAULT_CONFIG
    assert d.sim.timestep == 5e-4
    assert d.damage.poisson == 0.3
    assert d.damage.opponent_modulus_pa == 200e9
    assert d.damage.sigma_patch_factor == 8.0
    assert d.damage.sigma_min_frac == 0.025
    assert d.damage.sigma_abs_min == 3.0e-3
    assert d.damage.sigma_part_frac == 0.5
    assert d.damage.stress_sigma_frac == 0.4
    assert d.damage.kernel_radius_sigmas == 3.0
    assert d.brace.k_ref == 5.0e6
    assert d.brace.adjacency_tol == 5e-3
    assert d.brace.transfer == 0.5
    assert d.battery.strike_dv_cap_factor == 1.5
    assert d.battery.contain_restitution == 0.3
    assert d.battery.strike_energy_per_kg == 200.0
    assert d.battery.class_speed == {"3lb": 6.0, "12lb": 8.0, "30lb": 10.0}
    assert d.contact.gravity == -9.81
    assert d.contact.material_friction_bounce["metal"] == (0.6, 0.2)


@pytest.mark.skipif(not _HAS_TOMLLIB, reason="tomllib needs Python 3.11+")
def test_load_config_overlay(tmp_path):
    p = tmp_path / "cfg.toml"
    p.write_text("[damage]\nsigma_patch_factor = 6.0\n\n[sim]\ntimestep = 2.5e-4\n")
    cfg = load_config(p)
    assert cfg.damage.sigma_patch_factor == 6.0                  # overridden
    assert cfg.sim.timestep == 2.5e-4                            # overridden
    assert cfg.damage.poisson == DEFAULT_CONFIG.damage.poisson  # untouched
    assert cfg.brace.transfer == DEFAULT_CONFIG.brace.transfer  # untouched
    # The defaults are not mutated by an overlay.
    assert DEFAULT_CONFIG.damage.sigma_patch_factor == 8.0


@pytest.mark.skipif(not _HAS_TOMLLIB, reason="tomllib needs Python 3.11+")
def test_load_config_rejects_unknown(tmp_path):
    bad_section = tmp_path / "a.toml"
    bad_section.write_text("[nope]\nx = 1\n")
    with pytest.raises(ValueError):
        load_config(bad_section)

    bad_key = tmp_path / "b.toml"
    bad_key.write_text("[damage]\nnot_a_field = 1.0\n")
    with pytest.raises(ValueError):
        load_config(bad_key)


def _cube_bot(library):
    cube = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    bot = BotModel(cube, segment_mesh(cube))
    bot.assign_material_to_all(library.get("Aluminum 6061-T6"))
    return bot


def _one_contact_trace(local_pos, force=2000.0, speed=5.0):
    contact = ContactEvent(
        time=0.0, event="strike",
        pos=np.asarray(local_pos, float), local_pos=np.asarray(local_pos, float),
        normal=np.array([1.0, 0.0, 0.0]),
        normal_force=force, tangential_force=0.0, rel_speed=speed,
        part_index=0, other="opponent_weapon",
    )
    return SimTrace(dt=5e-4, n_parts=1, contacts=[contact])


def test_damage_config_is_live_wiring():
    """A tweaked DamageConfig must actually change the computed result — proof the
    config is threaded through, not a dead parameter."""
    library = load_default_library()
    bot = _cube_bot(library)
    arena = build_arena(NHRL_CLASSES["3lb"])
    trace = _one_contact_trace([0.05, 0.0, 0.0])

    base = compute_damage(trace, bot, arena, library)
    stiffer = compute_damage(
        trace, bot, arena, library,
        cfg=replace(DEFAULT_CONFIG.damage, opponent_modulus_pa=400e9),
    )
    # A stiffer opponent raises the Hertzian peak stress -> the verdict moves.
    assert base.peak_stress_per_face.max() > 0
    assert stiffer.peak_stress_per_face.max() != base.peak_stress_per_face.max()
