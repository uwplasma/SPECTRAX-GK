from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np


def _load_driver_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "optimization"
        / "vmec_jax_qa_low_turbulence_optimization.py"
    )
    spec = importlib.util.spec_from_file_location("_vmec_jax_qa_driver_for_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_solved_wout_candidate_gate_passes_valid_qa_branch() -> None:
    driver = _load_driver_module()
    result = SimpleNamespace(
        history={"aspect_final": 5.999233, "iota_final": 0.427011, "qs_final": 2.604013e-2},
    )

    report = driver._build_solved_wout_gate_report(
        result,
        target_aspect=6.0,
        aspect_atol=5.0e-2,
        min_abs_mean_iota=0.41,
        qs_residual_max=5.0e-2,
        iota_profile_floor=0.41,
        iota_profiles=(
            np.asarray([0.0, 0.410131, 0.414]),
            np.asarray([0.410706, 0.414]),
        ),
    )

    assert report["passed"] is True
    assert report["checks"]["aspect"]["passed"] is True
    assert report["checks"]["mean_iota"]["passed"] is True
    assert report["checks"]["quasisymmetry"]["passed"] is True
    assert report["checks"]["iota_profile"]["passed"] is True
    json.dumps(report, allow_nan=False)


def test_solved_wout_candidate_gate_rejects_transport_branch_that_breaks_constraints() -> None:
    driver = _load_driver_module()
    result = SimpleNamespace(
        history={"aspect_final": 5.996817, "iota_final": 0.425028, "qs_final": 1.091236e-1},
    )

    report = driver._build_solved_wout_gate_report(
        result,
        target_aspect=6.0,
        aspect_atol=5.0e-2,
        min_abs_mean_iota=0.41,
        qs_residual_max=5.0e-2,
        iota_profile_floor=0.41,
        iota_profiles=(
            np.asarray([0.0, 0.402043, 0.414]),
            np.asarray([0.402493, 0.414]),
        ),
    )

    assert report["passed"] is False
    assert report["checks"]["aspect"]["passed"] is True
    assert report["checks"]["mean_iota"]["passed"] is True
    assert report["checks"]["quasisymmetry"]["passed"] is False
    assert report["checks"]["iota_profile"]["passed"] is False
    assert "do not promote" in report["next_action"]
    json.dumps(report, allow_nan=False)
