"""Tests for the reduced QA low-turbulence optimization comparison."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

import spectraxgk
from spectraxgk.qa_low_turbulence import (
    QA_LOW_TURBULENCE_DESIGN_NAMES,
    QA_LOW_TURBULENCE_OBSERVABLE_NAMES,
    QALowTurbulenceConfig,
    qa_low_turbulence_comparison_payload,
    qa_low_turbulence_heat_flux_trace,
    qa_low_turbulence_observable_vector,
)


def _fast_config() -> QALowTurbulenceConfig:
    return QALowTurbulenceConfig(
        steps=20,
        nonlinear_steps=480,
        long_window_min_time=60.0,
        surface_ntheta=12,
        surface_nzeta=14,
        scan_density_gradients=(0.8, 2.2, 3.2),
    )


def _load_tool_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "build_qa_low_turbulence_comparison.py"
    spec = importlib.util.spec_from_file_location("build_qa_low_turbulence_comparison", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_qa_low_turbulence_payload_passes_gradient_and_transport_gates() -> None:
    assert spectraxgk.QALowTurbulenceConfig is QALowTurbulenceConfig
    assert spectraxgk.qa_low_turbulence_comparison_payload is qa_low_turbulence_comparison_payload

    payload = qa_low_turbulence_comparison_payload(_fast_config(), finite_difference_workers=1)
    metrics = payload["comparison_metrics"]
    results = {result["design_name"]: result for result in payload["results"]}

    assert payload["kind"] == "qa_low_turbulence_comparison"
    assert set(results) == set(QA_LOW_TURBULENCE_DESIGN_NAMES)
    assert metrics["passed"] is True
    assert metrics["ad_fd_gates_passed"] is True
    assert metrics["constraints_passed"] is True
    assert metrics["transport_reduction_gate_passed"] is True
    assert metrics["transport_design_heat_flux_mean"] < metrics["control_design_heat_flux_mean"]

    for result in results.values():
        assert result["scalar_gradient_gate"]["passed"] is True
        assert result["residual_gradient_gate"]["passed"] is True
        assert result["observable_gradient_gate"]["passed"] is True
        obs = dict(zip(QA_LOW_TURBULENCE_OBSERVABLE_NAMES, result["final_observables"], strict=True))
        assert abs(obs["aspect"] - 6.0) / 6.0 < 0.03
        assert obs["mean_iota"] >= _fast_config().iota_operating_floor - 2.0e-3
        assert obs["qa_residual"] < 0.03

    assert metrics["long_window_gates_passed"] is True
    assert payload["differentiable_plumbing"]["passed"] is True
    slopes = {
        design["design_name"]: design["density_gradient_scan"]["linear_slope_dQ_d_a_over_Ln"]
        for design in payload["designs"]
    }
    assert abs(slopes["qa_plus_nonlinear_heat_flux"]) < abs(slopes["qa_constraints"])


def test_qa_low_turbulence_observables_and_traces_are_finite() -> None:
    cfg = _fast_config()
    params = np.asarray([0.05, 0.45, 0.02, 0.18])
    observables = np.asarray(qa_low_turbulence_observable_vector(params, cfg), dtype=float)
    times, heat_flux = qa_low_turbulence_heat_flux_trace(
        params,
        cfg,
        density_gradient=2.2,
        temperature_gradient=6.0,
    )

    assert observables.shape == (len(QA_LOW_TURBULENCE_OBSERVABLE_NAMES),)
    assert np.all(np.isfinite(observables))
    assert len(times) == cfg.nonlinear_steps + 1
    assert len(heat_flux) == cfg.nonlinear_steps + 1
    assert float(np.min(np.asarray(heat_flux))) > 0.0


def test_qa_low_turbulence_artifact_tool_writes_json_csv_and_png(tmp_path: Path) -> None:
    mod = _load_tool_module()
    payload = qa_low_turbulence_comparison_payload(_fast_config(), finite_difference_workers=1)
    paths = mod.write_artifacts(payload, tmp_path / "qa_panel.png", write_pdf=False)

    assert Path(paths["png"]).exists()
    assert Path(paths["json"]).exists()
    assert Path(paths["summary_csv"]).exists()
    assert Path(paths["scan_csv"]).exists()
    written = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert written["comparison_metrics"]["passed"] is True
