from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import spectraxgk
from spectraxgk.solver_objective_gradients import (
    SOLVER_GEOMETRY_PARAMETER_NAMES,
    SOLVER_OBJECTIVE_NAMES,
    default_solver_geometry_design_params,
    linear_solver_geometry_gradient_report,
    solver_ready_geometry_mapping,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "build_solver_objective_gradient_gate.py"
spec = importlib.util.spec_from_file_location("build_solver_objective_gradient_gate", SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def test_solver_ready_geometry_mapping_validates_contract() -> None:
    theta = jnp.linspace(-jnp.pi, jnp.pi, 8, endpoint=False)
    mapping = solver_ready_geometry_mapping(default_solver_geometry_design_params(), theta)

    assert spectraxgk.solver_ready_geometry_mapping is solver_ready_geometry_mapping
    assert tuple(SOLVER_GEOMETRY_PARAMETER_NAMES) == ("bmag_ripple", "curvature_drift_scale")
    assert mapping["theta"].shape == theta.shape
    assert np.all(np.asarray(mapping["bmag"]) > 0.0)
    with pytest.raises(ValueError, match="length-2"):
        solver_ready_geometry_mapping(jnp.ones(3), theta)


def test_linear_solver_geometry_gradient_report_passes_actual_rhs_gate() -> None:
    report = linear_solver_geometry_gradient_report(fd_step=1.0e-3, rtol=1.0e-1, atol=2.0e-3)

    assert spectraxgk.linear_solver_geometry_gradient_report is linear_solver_geometry_gradient_report
    assert report["passed"] is True
    assert report["source_scope"] == "solver_ready_geometry_contract"
    assert report["linear_growth_gradient_gate"] is True
    assert report["quasilinear_weight_gradient_gate"] is True
    assert report["nonlinear_window_gradient_gate"] is False
    assert report["objective_names"] == list(SOLVER_OBJECTIVE_NAMES)
    assert np.asarray(report["eigenpair_gate"]["jacobian_implicit"]).shape == (
        len(SOLVER_OBJECTIVE_NAMES),
        len(SOLVER_GEOMETRY_PARAMETER_NAMES),
    )


def test_write_solver_objective_gradient_artifacts(tmp_path: Path) -> None:
    payload = {
        "kind": "linear_solver_geometry_gradient_gate",
        "passed": True,
        "parameter_names": ["p0", "p1"],
        "objective_names": ["gamma", "omega"],
        "objective_gates": [
            {
                "objective": "gamma",
                "parameter": "p0",
                "implicit": 1.0,
                "finite_difference": 1.0,
                "abs_error": 0.0,
                "rel_error": 0.0,
                "passed": True,
            },
            {
                "objective": "omega",
                "parameter": "p1",
                "implicit": 2.0,
                "finite_difference": 2.0,
                "abs_error": 0.0,
                "rel_error": 0.0,
                "passed": True,
            },
        ],
        "eigenpair_gate": {
            "atol": 1.0e-6,
            "jacobian_implicit": [[1.0, 0.0], [0.0, 2.0]],
            "jacobian_fd": [[1.0, 0.0], [0.0, 2.0]],
        },
    }

    paths = mod.write_solver_objective_gradient_artifacts(payload, out=tmp_path / "gate.png")

    for path in paths.values():
        assert Path(path).exists()
    saved = json.loads((tmp_path / "gate.json").read_text(encoding="utf-8"))
    assert saved["kind"] == "linear_solver_geometry_gradient_gate"
