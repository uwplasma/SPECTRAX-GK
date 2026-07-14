"""Tests for the reduced QA low-turbulence optimization comparison."""

from __future__ import annotations

import json
from pathlib import Path

from support.paths import load_artifact_tool

import numpy as np

import spectraxgk
from spectraxgk.objectives.qa_low_turbulence_artifacts import (
    qa_low_turbulence_comparison_payload,
)
from spectraxgk.objectives.qa_low_turbulence_contracts import (
    QA_LOW_TURBULENCE_DESIGN_NAMES,
    QA_LOW_TURBULENCE_OBSERVABLE_NAMES,
    QALowTurbulenceConfig,
)
from spectraxgk.objectives.qa_low_turbulence_model import (
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
    return load_artifact_tool("build_qa_transport_validation_artifacts")


def _load_time_horizon_tool_module():
    return load_artifact_tool("build_qa_transport_validation_artifacts")


def test_qa_low_turbulence_payload_passes_gradient_and_transport_gates() -> None:
    assert spectraxgk.QALowTurbulenceConfig is QALowTurbulenceConfig
    assert (
        spectraxgk.qa_low_turbulence_comparison_payload
        is qa_low_turbulence_comparison_payload
    )

    payload = qa_low_turbulence_comparison_payload(
        _fast_config(), finite_difference_workers=1
    )
    metrics = payload["comparison_metrics"]
    results = {result["design_name"]: result for result in payload["results"]}

    assert payload["kind"] == "qa_low_turbulence_comparison"
    assert "Q_env" in payload["model_equations"]["nonlinear_envelope"]
    assert set(results) == set(QA_LOW_TURBULENCE_DESIGN_NAMES)
    assert metrics["passed"] is True
    assert metrics["ad_fd_gates_passed"] is True
    assert metrics["constraints_passed"] is True
    assert metrics["transport_reduction_gate_passed"] is True
    assert (
        metrics["transport_design_heat_flux_mean"]
        < metrics["control_design_heat_flux_mean"]
    )

    for result in results.values():
        assert result["scalar_gradient_gate"]["passed"] is True
        assert result["residual_gradient_gate"]["passed"] is True
        assert result["observable_gradient_gate"]["passed"] is True
        assert abs(float(result["final_params"][2])) > 0.08
        obs = dict(
            zip(
                QA_LOW_TURBULENCE_OBSERVABLE_NAMES,
                result["final_observables"],
                strict=True,
            )
        )
        assert abs(obs["aspect"] - 6.0) / 6.0 < 0.03
        assert obs["mean_iota"] >= _fast_config().iota_operating_floor - 2.0e-3
        assert obs["qa_residual"] < 0.03

    assert metrics["long_window_gates_passed"] is True
    assert payload["differentiable_plumbing"]["passed"] is True
    for design in payload["designs"]:
        trace = design["fixed_gradient_trace"]
        assert (
            trace["trace_kind"]
            == "smooth_reduced_nonlinear_envelope_not_full_turbulent_gk"
        )
        assert "Q_env" in trace["trace_equation"]
    slopes = {
        design["design_name"]: design["density_gradient_scan"][
            "linear_slope_dQ_d_a_over_Ln"
        ]
        for design in payload["designs"]
    }
    assert abs(slopes["qa_plus_nonlinear_heat_flux"]) < abs(slopes["qa_constraints"])


def test_qa_low_turbulence_observables_and_traces_are_finite() -> None:
    cfg = _fast_config()
    params = np.asarray([0.05, 0.45, 0.02, 0.18])
    observables = np.asarray(
        qa_low_turbulence_observable_vector(params, cfg), dtype=float
    )
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


def test_qa_low_turbulence_artifact_tool_writes_json_csv_and_png(
    tmp_path: Path,
) -> None:
    mod = _load_tool_module()
    payload = qa_low_turbulence_comparison_payload(
        _fast_config(), finite_difference_workers=1
    )
    paths = mod.write_comparison_artifacts(
        payload, tmp_path / "qa_panel.png", write_pdf=False
    )

    assert Path(paths["png"]).exists()
    assert Path(paths["json"]).exists()
    assert Path(paths["summary_csv"]).exists()
    assert Path(paths["scan_csv"]).exists()
    written = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert written["comparison_metrics"]["passed"] is True


def test_qa_low_turbulence_time_horizon_tool_recommends_t400(tmp_path: Path) -> None:
    mod = _load_time_horizon_tool_module()
    comparison = tmp_path / "comparison.json"
    comparison.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "design_name": "qa_constraints",
                        "final_params": [
                            0.08530009537935257,
                            0.7973108291625977,
                            0.16232633590698242,
                            0.4597696363925934,
                        ],
                    },
                    {
                        "design_name": "qa_plus_nonlinear_heat_flux",
                        "final_params": [
                            0.17546722292900085,
                            1.1668813228607178,
                            0.16293159127235413,
                            0.47783327102661133,
                        ],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    payload = mod.build_time_horizon_payload(
        comparison,
        horizons=(150.0, 200.0, 300.0, 400.0),
        nonlinear_dt=0.20,
    )
    paths = mod.write_horizon_artifacts(
        payload, tmp_path / "horizon", write_pdf=False
    )

    assert payload["passed"] is True
    assert Path(paths["json"]).exists()
    assert Path(paths["csv"]).exists()
    assert Path(paths["png"]).exists()
    for metric in payload["metrics"].values():
        assert metric["passed"] is True
        assert "t=400 is sufficient" in metric["recommendation"]
