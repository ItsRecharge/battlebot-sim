"""Golden-baseline regression oracle for the offline damage pipeline.

Phases 1-4 of the engineering-grade hardening are *behavior-preserving* refactors
(logging, input validation, lifting magic numbers into a config object, etc.).
This module pins the seeded self-test pipeline's scalar outputs so any of those
refactors that accidentally perturbs the numerics fails loudly.

The reference lives in ``tests/data/golden_selftest.json``. Regenerate it ONLY
when a change is *meant* to move the numbers, by running:

    UPDATE_GOLDEN=1 pytest tests/test_golden_selftest.py::test_pipeline_matches_golden

(on Windows PowerShell: ``$env:UPDATE_GOLDEN=1; pytest ...``).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from helpers import run_selftest_pipeline, summarize_result

GOLDEN_PATH = Path(__file__).parent / "data" / "golden_selftest.json"
# A behavior-preserving refactor reproduces the run bit-for-bit on this machine;
# the tolerance only absorbs cross-machine float drift. A regression from changing
# a physics constant shifts values by far more than this.
RTOL = 1e-6
ATOL = 1e-9


def _assert_close(actual: dict, expected: dict) -> None:
    assert actual["n_parts"] == expected["n_parts"]
    assert actual["total_contacts"] == expected["total_contacts"]
    for key in ("energy_sum", "stress_sum", "margin_sum"):
        np.testing.assert_allclose(actual[key], expected[key], rtol=RTOL, atol=ATOL,
                                   err_msg=f"{key} drifted from golden baseline")
    for dict_key in ("part_max_margin", "part_total_energy"):
        a, e = actual[dict_key], expected[dict_key]
        assert a.keys() == e.keys(), f"{dict_key} part set changed"
        for pid in e:
            np.testing.assert_allclose(a[pid], e[pid], rtol=RTOL, atol=ATOL,
                                       err_msg=f"{dict_key}[{pid}] drifted")


def test_pipeline_matches_golden(selftest_result):
    """The seeded pipeline reproduces the committed baseline summaries."""
    summary = summarize_result(*selftest_result)

    if os.environ.get("UPDATE_GOLDEN"):
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True))
        pytest.skip(f"Regenerated golden baseline at {GOLDEN_PATH}")

    assert GOLDEN_PATH.exists(), (
        f"Missing golden baseline {GOLDEN_PATH}; create it with "
        f"UPDATE_GOLDEN=1 pytest {Path(__file__).name}"
    )
    expected = json.loads(GOLDEN_PATH.read_text())
    # JSON keys are strings; normalize the per-part dict keys back to ints.
    for dict_key in ("part_max_margin", "part_total_energy"):
        expected[dict_key] = {int(k): v for k, v in expected[dict_key].items()}
    _assert_close(summary, expected)


def test_pipeline_is_deterministic():
    """Two in-process runs produce bit-identical per-face fields.

    This is the determinism guarantee the golden baseline rests on: same seed +
    same code => same numbers, with no hidden unseeded state or order dependence.
    """
    _, r1 = run_selftest_pipeline()
    _, r2 = run_selftest_pipeline()
    for field in ("energy_per_face", "peak_stress_per_face", "failure_margin_per_face"):
        assert np.array_equal(getattr(r1, field), getattr(r2, field), equal_nan=True), (
            f"{field} differs between two identical runs — pipeline is non-deterministic"
        )
