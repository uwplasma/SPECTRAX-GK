from __future__ import annotations

from dataclasses import asdict
import importlib.util
import json
from pathlib import Path

from support.paths import REPO_ROOT
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import spectraxgk
import spectraxgk.objectives.stellarator as so
from spectraxgk.validation.autodiff import autodiff_finite_difference_report
from spectraxgk.objectives.stellarator import (
    OBSERVABLE_NAMES,
    PARAMETER_NAMES,
    StellaratorITGOptimizationConfig,
    StellaratorITGOptimizationResult,
    StellaratorITGSampleSet,
    compare_stellarator_itg_objectives,
    default_stellarator_initial_params,
    nonlinear_heat_flux_trace,
    nonlinear_heat_flux_window_metrics,
    optimize_stellarator_itg,
    qa_observable_vector,
    stellarator_itg_density_gradient_scan,
    stellarator_itg_objective,
    stellarator_itg_objective_residual_names,
    stellarator_itg_objective_residual_vector,
    stellarator_itg_portfolio_gate_payload,
    stellarator_itg_portfolio_sensitivity_report,
    stellarator_itg_reduced_portfolio_objective,
    stellarator_itg_residual_sensitivity_report,
    stellarator_itg_sample_objective_table,
    stellarator_itg_vmec_boozer_portfolio_objective_from_state,
    stellarator_itg_vmec_boozer_sample_objective_table_from_state,
)


def _fast_config() -> StellaratorITGOptimizationConfig:
    return StellaratorITGOptimizationConfig(
        nonlinear_dt=0.18,
        nonlinear_steps=96,
        nonlinear_tail_fraction=0.30,
        fd_step=1.0e-4 if bool(jax.config.jax_enable_x64) else 5.0e-3,
    )


def _fd_tolerances() -> tuple[float, float]:
    if bool(jax.config.jax_enable_x64):
        return 5.0e-3, 6.0e-4
    return 5.0e-2, 6.0e-3


def _disable_optional_backend_discovery(monkeypatch) -> None:
    monkeypatch.setattr(
        so,
        "discover_differentiable_geometry_backends",
        lambda: {
            "vmec_jax_available": False,
            "vmec_jax_boundary_api_available": False,
            "booz_xform_jax_available": False,
            "booz_xform_jax_api_available": False,
        },
    )


def _load_stellarator_itg_plotting_module():
    path = (
        REPO_ROOT
        / "examples"
        / "theory_and_demos"
        / "reduced_stellarator_itg"
        / "_stellarator_itg_plotting.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_stellarator_itg_plotting_for_test", path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_reduced_stellarator_itg_development_scripts_are_explicit_workflows() -> None:
    examples = REPO_ROOT / "examples" / "theory_and_demos" / "reduced_stellarator_itg"
    scripts = [
        examples / "stellarator_itg_growth_optimization.py",
        examples / "stellarator_itg_quasilinear_flux_optimization.py",
        examples / "stellarator_itg_nonlinear_heat_flux_optimization.py",
        examples / "compare_stellarator_itg_optimizations.py",
    ]

    for script in scripts:
        text = script.read_text(encoding="utf-8")
        assert (
            "run_stellarator_itg_adam" in text
            or "compare_scripted_stellarator_itg_objectives" in text
        )
        assert "optimize_stellarator_itg" not in text
        assert "QA_optimization.py" in text


def test_public_optimization_examples_exclude_reduced_synthetic_workflows() -> None:
    examples = REPO_ROOT / "examples" / "optimization"
    names = {path.name for path in examples.iterdir() if path.is_file()}

    assert "QA_optimization_linear_ITG.py" in names
    assert "QA_optimization_quasilinear_ITG.py" in names
    assert "QA_optimization_nonlinear_ITG.py" in names
    assert "vmec_jax_qa_low_turbulence_optimization.py" not in names
    assert not any(name.startswith("stellarator_itg_") for name in names)
    assert "_stellarator_itg_plotting.py" not in names
    assert "compare_stellarator_itg_optimizations.py" not in names
    assert (
        REPO_ROOT / "tools" / "campaigns" / "vmec_jax_qa_low_turbulence_optimization.py"
    ).exists()


def test_public_optimization_examples_keep_editable_constant_style() -> None:
    examples = REPO_ROOT / "examples" / "optimization"
    optimizer_scripts = {
        "QA_optimization_linear_ITG.py",
        "QA_optimization_quasilinear_ITG.py",
        "QA_optimization_nonlinear_ITG.py",
    }
    for script in sorted(examples.glob("*.py")):
        text = script.read_text(encoding="utf-8")
        assert "argparse" not in text
        assert "def main(" not in text
        assert "def _main(" not in text
        assert 'if __name__ == "__main__"' not in text
        if script.name in optimizer_scripts:
            assert "SPECTRAX_SURFACES = (0.45, 0.64, 0.78)" in text
            assert "SPECTRAX_ALPHAS = (0.0, np.pi / 4.0)" in text
            assert "SPECTRAX_KY_VALUES = (0.10, 0.30, 0.50)" in text
        elif script.name == "QA_parameter_scan.py":
            assert 'COEFFICIENT = "RBC(1,1)"' in text
            assert 'FRACTIONS = "-0.75,-0.70,-0.65' in text
            assert '0.65,0.70,0.75"' in text
            assert 'SURFACES = "0.45,0.64,0.78"' in text
            assert 'ALPHAS = "0.0,0.7853981633974483"' in text
            assert 'KY_VALUES = "0.10,0.30,0.50"' in text
        elif script.name == "QA_nonlinear_ITG_matched_audit.py":
            assert "BASELINE_ENSEMBLE" in text
            assert "OPTIMIZED_ENSEMBLE" in text
            assert "MIN_RELATIVE_REDUCTION = 0.02" in text
            assert "REQUIRE_UNCERTAINTY_SEPARATION = True" in text
        elif script.name == "QA_nonlinear_ITG_transport_matrix.py":
            assert "BASELINE_VMEC_FILE" in text
            assert "CANDIDATE_VMEC_FILE" in text
            assert 'SURFACES = "0.45,0.64,0.78"' in text
            assert 'ALPHAS = "0.0,pi/4"' in text
            assert 'KY_VALUES = "0.10,0.30,0.50"' in text
            assert 'HORIZONS = "700,1100,1500"' in text
            assert "WINDOW_TMIN = 1100.0" in text
            assert "WINDOW_TMAX = 1500.0" in text
            assert "GPU_SPLITS = 2" in text
        else:
            raise AssertionError(f"unexpected optimization example {script.name}")


def test_stellarator_itg_observable_contract_is_finite_and_exported() -> None:
    assert spectraxgk.STELLARATOR_ITG_PARAMETER_NAMES == PARAMETER_NAMES
    assert spectraxgk.STELLARATOR_ITG_OBSERVABLE_NAMES == OBSERVABLE_NAMES
    assert spectraxgk.optimize_stellarator_itg is optimize_stellarator_itg
    assert (
        spectraxgk.stellarator_itg_density_gradient_scan
        is stellarator_itg_density_gradient_scan
    )
    assert (
        spectraxgk.stellarator_itg_residual_sensitivity_report
        is stellarator_itg_residual_sensitivity_report
    )

    cfg = _fast_config()
    params = default_stellarator_initial_params()
    observables = np.asarray(qa_observable_vector(params, cfg))

    assert observables.shape == (len(OBSERVABLE_NAMES),)
    assert np.all(np.isfinite(observables))
    obs = dict(zip(OBSERVABLE_NAMES, observables, strict=True))
    assert obs["aspect"] > 0.0
    assert obs["kperp_eff2"] > 0.0
    assert obs["growth_rate"] > 0.0
    assert obs["linear_heat_flux_weight"] > 0.0
    assert obs["quasilinear_heat_flux"] > 0.0
    assert obs["nonlinear_heat_flux_mean"] > 0.0


def test_stellarator_itg_density_gradient_scan_is_monotone_and_scoped() -> None:
    cfg = _fast_config()
    params = default_stellarator_initial_params()
    scan = stellarator_itg_density_gradient_scan(
        params,
        cfg,
        density_gradients=(0.8, cfg.reference_density_gradient, 4.8),
    )
    default_obs = dict(
        zip(OBSERVABLE_NAMES, qa_observable_vector(params, cfg), strict=True)
    )

    assert (
        scan["scope"]
        == "reduced_max_mode1_density_gradient_response_not_full_nonlinear_scan"
    )
    assert scan["fixed_temperature_gradient"] == cfg.reference_temperature_gradient
    assert scan["density_gradient_axis"] == [0.8, cfg.reference_density_gradient, 4.8]
    assert np.all(np.diff(scan["heat_flux_mean"]) > 0.0)
    assert np.all(np.diff(scan["growth_rate"]) > 0.0)
    assert scan["linear_slope_dQ_d_a_over_Ln"] > 0.0
    assert scan["heat_flux_mean"][1] == pytest.approx(
        default_obs["nonlinear_heat_flux_mean"]
    )


@pytest.mark.parametrize("density_gradients", [(), (0.8, np.nan), (-0.1, 1.0)])
def test_stellarator_itg_density_gradient_scan_rejects_invalid_axes(
    density_gradients: tuple[float, ...],
) -> None:
    with pytest.raises(ValueError, match="density_gradients"):
        stellarator_itg_density_gradient_scan(
            default_stellarator_initial_params(),
            _fast_config(),
            density_gradients=density_gradients,
        )


def test_stellarator_itg_plotting_artifact_includes_reduced_diagnostics(
    tmp_path: Path,
) -> None:
    plotting = _load_stellarator_itg_plotting_module()
    cfg = _fast_config()
    p0 = np.asarray(default_stellarator_initial_params(), dtype=float)
    p1 = 0.55 * p0
    obs0 = np.asarray(qa_observable_vector(p0, cfg), dtype=float)
    obs1 = np.asarray(qa_observable_vector(p1, cfg), dtype=float)
    payload = {
        "objective_kind": "growth",
        "parameter_names": list(PARAMETER_NAMES),
        "observable_names": list(OBSERVABLE_NAMES),
        "initial_params": p0.tolist(),
        "final_params": p1.tolist(),
        "initial_objective": 1.0,
        "final_objective": 0.5,
        "initial_observables": obs0.tolist(),
        "final_observables": obs1.tolist(),
        "history": [
            {
                "step": 0,
                "objective": 1.0,
                "gradient_norm": 1.0,
                "params": p0.tolist(),
                "observables": obs0.tolist(),
            },
            {
                "step": 1,
                "objective": 0.5,
                "gradient_norm": 0.2,
                "params": p1.tolist(),
                "observables": obs1.tolist(),
            },
        ],
        "gradient_gate": {"passed": True, "max_abs_error": 1.0e-8},
        "covariance": {},
        "nonlinear_trace": None,
        "config": asdict(cfg),
        "backend_info": {
            "vmec_jax_available": True,
            "booz_xform_jax_available": True,
            "vmec_jax_paths": ["/Users/tester/local/vmec_jax"],
            "booz_xform_jax_paths": ["/Users/tester/local/booz_xform_jax"],
        },
        "claim_level": "reduced_linear_or_quasilinear_objective_optimization",
    }
    result = SimpleNamespace(to_dict=lambda: payload)
    out = tmp_path / "stellarator_itg_panel"

    plotting.write_result_artifacts(result, out, title="test panel")

    written = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert "/Users/tester" not in out.with_suffix(".json").read_text(encoding="utf-8")
    assert written["backend_info"] == {
        "vmec_jax_available": True,
        "booz_xform_jax_available": True,
    }
    assert out.with_suffix(".history.csv").exists()
    assert out.with_suffix(".png").exists()
    assert out.with_suffix(".pdf").exists()
    diagnostics = written["reduced_diagnostics"]
    assert (
        diagnostics["claim_level"]
        == "reduced_max_mode1_diagnostics_not_solved_vmec_or_full_nonlinear_scan"
    )
    assert diagnostics["final"]["density_gradient_scan"]["scope"].startswith(
        "reduced_max_mode1"
    )
    assert diagnostics["final"]["fixed_gradient_trace"]["trace_kind"] == (
        "smooth_reduced_nonlinear_envelope_not_full_turbulent_gk"
    )
    assert diagnostics["final"]["surface"]["scope"].endswith(
        "not_solved_vmec_equilibrium"
    )
    assert diagnostics["final"]["lcfs_bmag"]["scope"].startswith(
        "synthetic_reduced_lcfs_bmag"
    )
    assert len(diagnostics["final"]["lcfs_bmag"]["theta"]) >= 64


def test_stellarator_itg_comparison_artifact_has_publication_lcfs_diagnostics(
    tmp_path: Path,
) -> None:
    plotting = _load_stellarator_itg_plotting_module()
    cfg = _fast_config()
    p0 = np.asarray(default_stellarator_initial_params(), dtype=float)
    names = list(OBSERVABLE_NAMES)

    def row(kind: str, scale: float) -> dict[str, object]:
        p1 = scale * p0
        obs0 = np.asarray(qa_observable_vector(p0, cfg), dtype=float)
        obs1 = np.asarray(qa_observable_vector(p1, cfg), dtype=float)
        return {
            "objective_kind": kind,
            "parameter_names": list(PARAMETER_NAMES),
            "observable_names": names,
            "initial_params": p0.tolist(),
            "final_params": p1.tolist(),
            "initial_objective": 1.0,
            "final_objective": 0.5 * scale,
            "initial_observables": obs0.tolist(),
            "final_observables": obs1.tolist(),
            "history": [
                {
                    "step": 0,
                    "objective": 1.0,
                    "gradient_norm": 1.0,
                    "params": p0.tolist(),
                    "observables": obs0.tolist(),
                },
                {
                    "step": 1,
                    "objective": 0.5 * scale,
                    "gradient_norm": 0.2,
                    "params": p1.tolist(),
                    "observables": obs1.tolist(),
                },
            ],
            "gradient_gate": {"passed": True, "max_abs_error": 1.0e-8},
            "covariance": {},
            "nonlinear_trace": None,
            "config": asdict(cfg),
            "backend_info": {},
            "claim_level": "reduced_linear_or_quasilinear_objective_optimization",
        }

    payload = {
        "claim_level": "reduced_objective_optimization_comparison_not_full_production_vmec_gk",
        "production_nonlinear_optimization_claim": False,
        "parameter_names": list(PARAMETER_NAMES),
        "observable_names": names,
        "results": [
            row("growth", 0.70),
            row("quasilinear_flux", 0.55),
            row("nonlinear_heat_flux", 0.45),
        ],
        "backend_info": {},
        "parallel": {"requested_workers": 1},
    }
    out = tmp_path / "stellarator_compare"

    plotting.write_comparison_artifacts(payload, out)

    written = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert written["figure_scope"]["surface_grid"] == {
        "ntheta": 72,
        "nzeta": 72,
        "cmap": "jet",
    }
    assert "synthetic reduced" in written["figure_scope"]["surface_claim"]
    assert "not a solved VMEC LCFS" in written["figure_scope"]["surface_claim"]
    assert out.with_suffix(".png").exists()
    assert out.with_suffix(".pdf").exists()
    plotting_source = REPO_ROOT.joinpath(
        "examples/theory_and_demos/reduced_stellarator_itg/_stellarator_itg_plotting.py"
    ).read_text(encoding="utf-8")
    assert (
        "Reduced synthetic QA ITG optimization-plumbing comparison" in plotting_source
    )
    assert "Synthetic reduced LCFS" in plotting_source
    assert "Synthetic reduced Boozer-LCFS" in plotting_source
    for result in written["results"]:
        diagnostics = result["reduced_diagnostics"]["final"]
        assert diagnostics["surface_summary"]["theta_count"] == 72
        assert diagnostics["surface_summary"]["zeta_count"] == 72
        assert diagnostics["lcfs_bmag_summary"]["theta_count"] == 72
        assert diagnostics["lcfs_bmag_summary"]["zeta_count"] == 72
        assert diagnostics["lcfs_bmag_summary"]["cmap"] == "jet"
        assert "surface" not in diagnostics
        assert "lcfs_bmag" not in diagnostics
        assert diagnostics["fixed_gradient_trace"]["trace_kind"].startswith(
            "smooth_reduced"
        )


def test_stellarator_itg_objectives_have_fd_checked_gradients() -> None:
    cfg = _fast_config()
    params = jnp.asarray([0.18, 0.25, 0.22, -0.16])
    rtol, atol = _fd_tolerances()

    for kind in ("growth", "quasilinear_flux", "nonlinear_heat_flux"):
        value = stellarator_itg_objective(params, kind, cfg)
        residual = stellarator_itg_objective_residual_vector(params, kind, cfg)
        assert float(value) > 0.0
        assert residual.shape == (3 + len(PARAMETER_NAMES) + 1,)
        assert len(stellarator_itg_objective_residual_names(kind)) == residual.size
        np.testing.assert_allclose(
            float(value), float(jnp.dot(residual, residual)), rtol=1.0e-6
        )
        report = autodiff_finite_difference_report(
            lambda x, kind=kind: stellarator_itg_objective(x, kind, cfg),
            params,
            step=cfg.fd_step,
            rtol=rtol,
            atol=atol,
            workers=2,
        )
        assert report["passed"] is True
        assert report["finite_difference_parallel"]["requested_workers"] == 2


def test_quasilinear_residual_sensitivity_report_checks_fd_and_conditioning() -> None:
    cfg = _fast_config()
    params = jnp.asarray([0.16, 0.21, 0.24, -0.18])

    report = stellarator_itg_residual_sensitivity_report(
        params,
        "quasilinear_flux",
        cfg,
        finite_difference_workers=2,
    )

    assert report["passed"] is True
    assert report["objective_kind"] == "quasilinear_flux"
    assert report["finite_difference_gate"]["passed"] is True
    assert report["finite_difference_gate"]["step"] <= cfg.fd_step
    assert (
        report["finite_difference_gate"]["finite_difference_parallel"][
            "requested_workers"
        ]
        == 2
    )
    assert report["conditioning_gate"]["passed"] is True
    assert report["conditioning_gate"]["sensitivity_map_rank"] == len(PARAMETER_NAMES)
    assert report["covariance"]["source"] == "weighted_objective_residual"
    assert report["covariance"]["conditioning_gate"]["passed"] is True
    assert report["residual_names"] == list(
        stellarator_itg_objective_residual_names("quasilinear_flux")
    )


def test_stellarator_itg_sample_portfolio_is_rectangular_and_exported() -> None:
    assert spectraxgk.StellaratorITGSampleSet is StellaratorITGSampleSet
    assert (
        spectraxgk.stellarator_itg_sample_objective_table
        is stellarator_itg_sample_objective_table
    )
    assert (
        spectraxgk.stellarator_itg_reduced_portfolio_objective
        is stellarator_itg_reduced_portfolio_objective
    )
    assert (
        spectraxgk.stellarator_itg_portfolio_sensitivity_report
        is stellarator_itg_portfolio_sensitivity_report
    )
    assert (
        spectraxgk.stellarator_itg_portfolio_gate_payload
        is stellarator_itg_portfolio_gate_payload
    )

    cfg = _fast_config()
    samples = StellaratorITGSampleSet(
        surfaces=(0.55, 0.64),
        alphas=(0.0, 0.7),
        ky_values=(0.1, 0.3, 0.5),
        surface_weights=(1.0, 2.0),
        ky_weights=(1.0, 2.0, 1.0),
    )
    params = jnp.asarray([0.18, 0.25, 0.22, -0.16])
    table = stellarator_itg_sample_objective_table(
        params,
        ("growth", "quasilinear_flux", "nonlinear_heat_flux"),
        cfg,
        samples,
    )
    reduced = stellarator_itg_reduced_portfolio_objective(
        params,
        ("growth", "quasilinear_flux"),
        cfg,
        samples,
        objective_weights=(2.0, 1.0),
    )

    assert samples.n_samples == 12
    assert table.shape == (2, 2, 3, 3)
    assert float(jnp.min(table)) > 0.0
    assert float(reduced) > 0.0
    assert samples.to_dict()["reduction"] == "weighted_mean"


def test_stellarator_itg_portfolio_sensitivity_report_checks_rows_and_scalar() -> None:
    cfg = _fast_config()
    samples = StellaratorITGSampleSet(
        surfaces=(0.55, 0.64),
        alphas=(0.0, 0.7),
        ky_values=(0.15, 0.35),
    )
    params = jnp.asarray([0.16, 0.21, 0.24, -0.18])

    report = stellarator_itg_portfolio_sensitivity_report(
        params,
        ("growth", "quasilinear_flux"),
        cfg,
        samples,
        workers=2,
    )
    inner = report["portfolio_report"]

    assert report["passed"] is True
    assert report["objective_names"] == ["growth", "quasilinear_flux"]
    assert report["sample_set"]["n_samples"] == 8
    assert inner["portfolio_contract"]["row_shape"] == [2, 2, 2, 2]
    assert inner["scalar_gradient_gate"]["passed"] is True
    assert inner["row_jacobian_gate"]["passed"] is True
    assert inner["conditioning_gate"]["sensitivity_map_rank"] == len(PARAMETER_NAMES)
    assert (
        inner["scalar_gradient_gate"]["finite_difference_parallel"]["requested_workers"]
        == 2
    )


def test_stellarator_itg_portfolio_gate_payload_is_json_ready() -> None:
    cfg = _fast_config()
    samples = StellaratorITGSampleSet(
        surfaces=(0.55, 0.64),
        alphas=(0.0, 0.7),
        ky_values=(0.15, 0.35),
    )
    params = jnp.asarray([0.16, 0.21, 0.24, -0.18])

    payload = stellarator_itg_portfolio_gate_payload(
        params,
        ("growth", "quasilinear_flux"),
        cfg,
        samples,
        objective_weights=(2.0, 1.0),
        finite_difference_workers=2,
    )

    assert payload["kind"] == "stellarator_itg_portfolio_gate"
    assert payload["passed"] is True
    assert payload["production_nonlinear_optimization_claim"] is False
    assert payload["sample_set"]["n_samples"] == 8
    assert len(payload["samples"]) == 8
    assert len(payload["base_sample_values"]) == 8
    json.dumps(payload, allow_nan=False)
    objective_table = np.asarray(payload["base_objective_table"], dtype=float)
    objective_tensor = np.asarray(payload["base_objective_tensor"], dtype=float)
    objective_weights = np.asarray(payload["objective_weights"], dtype=float)
    sample_values = np.asarray(payload["base_sample_values"], dtype=float)
    sample_rows = payload["samples"]

    assert objective_table.shape == (8, 2)
    assert objective_tensor.shape == (2, 2, 2, 2)
    np.testing.assert_allclose(objective_table, objective_tensor.reshape((8, 2)))
    np.testing.assert_allclose(sample_values, objective_table @ objective_weights)
    assert [(row["surface"], row["alpha"], row["ky"]) for row in sample_rows] == [
        (surface, alpha, ky)
        for surface in samples.surfaces
        for alpha in samples.alphas
        for ky in samples.ky_values
    ]
    np.testing.assert_allclose(sum(payload["objective_weights"]), 1.0)
    np.testing.assert_allclose(sum(row["weight"] for row in payload["samples"]), 1.0)
    np.testing.assert_allclose(
        payload["base_value"],
        float(np.dot(sample_values, [row["weight"] for row in sample_rows])),
        rtol=1.0e-6,
        atol=1.0e-8,
    )
    assert payload["portfolio_report"]["scalar_gradient_gate"]["passed"] is True
    assert payload["portfolio_report"]["row_jacobian_gate"]["passed"] is True
    assert "real vmec_jax" in payload["next_action"]


def test_stellarator_itg_vmec_boozer_portfolio_wraps_real_table_contract(
    monkeypatch,
) -> None:
    calls: dict[str, object] = {}

    def fake_table(_state, _static, _indata, _wout, **kwargs):  # noqa: ANN001, ANN202
        calls.update(kwargs)
        rows = []
        for index in range(8):
            value = float(index + 1)
            rows.append([value, 0.0, 1.0, 2.0, 0.0, 10.0 * value])
        metadata = [{"sample": index} for index in range(8)]
        return jnp.asarray(rows), metadata

    monkeypatch.setattr(
        so, "vmec_boozer_solver_objective_table_with_metadata_from_state", fake_table
    )
    samples = StellaratorITGSampleSet(
        surfaces=(0.50, 0.70),
        alphas=(0.0, 0.6),
        ky_values=(0.1, 0.3),
    )

    table = stellarator_itg_vmec_boozer_sample_objective_table_from_state(
        "state",
        "static",
        "indata",
        "wout",
        ("growth", "quasilinear_flux"),
        samples,
        ntheta=8,
    )
    reduced = stellarator_itg_vmec_boozer_portfolio_objective_from_state(
        "state",
        "static",
        "indata",
        "wout",
        ("growth", "quasilinear_flux"),
        samples,
        objective_weights=(1.0, 0.0),
        ntheta=8,
    )

    assert spectraxgk.stellarator_itg_vmec_boozer_sample_objective_table_from_state is (
        stellarator_itg_vmec_boozer_sample_objective_table_from_state
    )
    assert spectraxgk.stellarator_itg_vmec_boozer_portfolio_objective_from_state is (
        stellarator_itg_vmec_boozer_portfolio_objective_from_state
    )
    assert table.shape == (2, 2, 2, 2)
    np.testing.assert_allclose(np.asarray(table)[0, 0, 0], (1.0, 10.0))
    assert float(reduced) == pytest.approx(4.5)
    assert calls["torflux_values"] == samples.surfaces
    assert calls["alphas"] == samples.alphas
    assert calls["ky_values"] == samples.ky_values
    assert calls["ntheta"] == 8


def test_nonlinear_heat_flux_window_metrics_use_late_stable_samples() -> None:
    cfg = _fast_config()
    times, heat_flux = nonlinear_heat_flux_trace(
        default_stellarator_initial_params(), cfg
    )
    metrics = nonlinear_heat_flux_window_metrics(
        times,
        heat_flux,
        tail_fraction=cfg.nonlinear_tail_fraction,
    )

    assert times.shape == heat_flux.shape == (cfg.nonlinear_steps + 1,)
    assert int(metrics["start_index"]) < cfg.nonlinear_steps - 1
    assert float(metrics["mean"]) > 0.0
    assert float(metrics["cv"]) < 0.15
    assert float(metrics["trend"]) < 0.35


def test_optimize_stellarator_itg_reduces_nonlinear_window_objective(
    monkeypatch,
) -> None:
    _disable_optional_backend_discovery(monkeypatch)
    cfg = _fast_config()

    result = optimize_stellarator_itg("nonlinear_heat_flux", config=cfg)

    assert result.objective_kind == "nonlinear_heat_flux"
    assert result.final_objective < 0.20 * result.initial_objective
    assert result.gradient_gate["passed"] is True
    assert result.covariance["source"] == "weighted_objective_residual"
    assert result.covariance["residual_sensitivity_passed"] is True
    assert result.covariance["residual_jacobian_gate"]["passed"] is True
    assert result.covariance["conditioning_gate"]["passed"] is True
    assert len(result.covariance["residual_names"]) == 3 + len(PARAMETER_NAMES) + 1
    assert result.covariance["sensitivity_map_rank"] == len(PARAMETER_NAMES)
    assert result.nonlinear_trace is not None
    assert result.nonlinear_trace["final_window"]["cv"] < 0.05
    assert result.nonlinear_trace["final_window"]["trend"] < 0.15
    serialized = result.to_dict()
    assert (
        serialized["claim_level"]
        == "reduced_nonlinear_window_estimator_optimization_not_transport_average"
    )
    assert serialized["nonlinear_transport_scope"]["transport_average_gate"] is False
    assert (
        serialized["nonlinear_transport_scope"][
            "production_nonlinear_optimization_claim"
        ]
        is False
    )

    initial = dict(zip(OBSERVABLE_NAMES, result.initial_observables, strict=True))
    final = dict(zip(OBSERVABLE_NAMES, result.final_observables, strict=True))
    assert final["growth_rate"] < initial["growth_rate"]
    assert final["quasilinear_heat_flux"] < initial["quasilinear_heat_flux"]
    assert final["nonlinear_heat_flux_mean"] < initial["nonlinear_heat_flux_mean"]


def test_compare_stellarator_itg_objectives_payload_is_json_ready(monkeypatch) -> None:
    _disable_optional_backend_discovery(monkeypatch)
    cfg = _fast_config()

    payload = compare_stellarator_itg_objectives(
        ("growth",), config=cfg, workers=2, finite_difference_workers=2
    )

    assert (
        payload["claim_level"]
        == "reduced_objective_optimization_comparison_not_full_production_vmec_gk"
    )
    assert payload["production_nonlinear_optimization_claim"] is False
    assert payload["parameter_names"] == list(PARAMETER_NAMES)
    assert payload["observable_names"] == list(OBSERVABLE_NAMES)
    assert payload["parallel"]["requested_workers"] == 2
    assert payload["parallel"]["finite_difference_workers"] == 2
    assert len(payload["results"]) == 1
    result = payload["results"][0]
    assert result["objective_kind"] == "growth"
    assert result["final_objective"] < result["initial_objective"]
    assert result["gradient_gate"]["passed"] is True
    assert (
        result["gradient_gate"]["finite_difference_parallel"]["requested_workers"] == 2
    )


def test_compare_stellarator_itg_objectives_parallel_preserves_order(
    monkeypatch,
) -> None:
    _disable_optional_backend_discovery(monkeypatch)

    def fake_optimize(kind, initial_params=None, config=None, **kwargs):  # noqa: ANN001, ANN202
        idx = {"growth": 1.0, "quasilinear_flux": 2.0, "nonlinear_heat_flux": 3.0}[kind]
        return StellaratorITGOptimizationResult(
            objective_kind=kind,
            parameter_names=PARAMETER_NAMES,
            observable_names=OBSERVABLE_NAMES,
            initial_params=(0.0, 0.0, 0.0, 0.0),
            final_params=(idx, idx, idx, idx),
            initial_objective=idx + 1.0,
            final_objective=idx,
            initial_observables=tuple(0.0 for _ in OBSERVABLE_NAMES),
            final_observables=tuple(idx for _ in OBSERVABLE_NAMES),
            history=(),
            gradient_gate={
                "passed": True,
                "finite_difference_parallel": {
                    "requested_workers": kwargs["finite_difference_workers"],
                    "executor": kwargs["finite_difference_executor"],
                },
            },
            covariance={"source": "test"},
            nonlinear_trace=None,
            config={},
            backend_info={},
        )

    monkeypatch.setattr(so, "optimize_stellarator_itg", fake_optimize)
    payload = compare_stellarator_itg_objectives(
        ("growth", "quasilinear_flux", "nonlinear_heat_flux"),
        workers=3,
        finite_difference_workers=2,
    )

    assert [row["objective_kind"] for row in payload["results"]] == [
        "growth",
        "quasilinear_flux",
        "nonlinear_heat_flux",
    ]
    assert [row["final_objective"] for row in payload["results"]] == [1.0, 2.0, 3.0]
    assert payload["parallel"]["effective_workers"] == 3
