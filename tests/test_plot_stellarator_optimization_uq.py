"""Tests for the stellarator optimization UQ plotting tool."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


def _load_tool_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "plot_stellarator_optimization_uq.py"
    spec = importlib.util.spec_from_file_location("plot_stellarator_optimization_uq", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _result(kind: str, scale: float) -> dict[str, object]:
    params = [0.2, 0.1, 0.05, -0.03]
    covariance = [
        [scale * 4.0e-4, scale * 1.0e-4, 0.0, 0.0],
        [scale * 1.0e-4, scale * 3.0e-4, 0.0, 0.0],
        [0.0, 0.0, scale * 2.0e-4, 0.0],
        [0.0, 0.0, 0.0, scale * 1.5e-4],
    ]
    return {
        "objective_kind": kind,
        "parameter_names": [
            "minor_radius_log_shift",
            "vertical_elongation_shift",
            "helical_ripple_amplitude",
            "magnetic_shear_shift",
        ],
        "initial_params": [0.28, 0.46, 0.42, -0.32],
        "final_params": params,
        "initial_objective": 1.0,
        "final_objective": 0.4,
        "gradient_gate": {
            "passed": True,
            "max_abs_error": 1.0e-6,
            "max_rel_error": 1.0e-4,
            "tangent_max_abs_error": 2.0e-6,
            "jacobian_ad": [[0.1, -0.2, 0.3, -0.4]],
            "jacobian_fd": [[0.1000005, -0.199999, 0.300001, -0.400001]],
        },
        "covariance": {
            "covariance": covariance,
            "covariance_std": [0.02, 0.017, 0.014, 0.012],
            "covariance_correlation": [
                [1.0, 0.25, 0.0, 0.0],
                [0.25, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "covariance_eigenvalues": [1.0e-4, 2.0e-4, 3.0e-4, 4.0e-4],
            "uq_ellipse_area_1sigma": 2.0e-3,
            "jacobian_singular_values": [3.0, 1.0, 0.5, 0.1],
            "jacobian_condition_number": 30.0,
            "sensitivity_map_rank": 4,
        },
    }


def test_stellarator_optimization_uq_summary_and_artifacts(tmp_path: Path) -> None:
    mod = _load_tool_module()
    payload = {
        "parameter_names": [
            "minor_radius_log_shift",
            "vertical_elongation_shift",
            "helical_ripple_amplitude",
            "magnetic_shear_shift",
        ],
        "observable_names": ["growth_rate"],
        "results": [
            _result("growth", 1.0),
            _result("quasilinear_flux", 1.2),
            _result("nonlinear_heat_flux", 0.8),
        ],
    }

    summary = mod.build_uq_summary(payload)
    assert summary["kind"] == "stellarator_itg_optimization_uq"
    assert summary["all_gradient_gates_passed"] is True
    assert summary["all_sensitivity_maps_full_rank"] is True
    assert len(summary["results"]) == 3
    assert summary["results"][0]["max_abs_error"] == 1.0e-6

    paths = mod.write_uq_figure(summary, out=tmp_path / "uq.png")

    assert Path(paths["png"]).exists()
    assert Path(paths["pdf"]).exists()
    written = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert written["claim_level"] == "reduced_objective_uq_and_sensitivity_validation_not_full_vmec_gk_optimization"


def test_stellarator_optimization_uq_rejects_bad_gradient_shape() -> None:
    mod = _load_tool_module()
    payload = {
        "parameter_names": ["a", "b"],
        "results": [_result("growth", 1.0)],
    }

    try:
        mod.build_uq_summary(payload)
    except ValueError as exc:
        assert "gradient gate" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected malformed payload to be rejected")
