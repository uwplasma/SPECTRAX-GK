from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import spectraxgk
import spectraxgk.stellarator_optimization as so
from spectraxgk.autodiff_validation import autodiff_finite_difference_report
from spectraxgk.stellarator_optimization import (
    OBSERVABLE_NAMES,
    PARAMETER_NAMES,
    StellaratorITGOptimizationConfig,
    StellaratorITGOptimizationResult,
    compare_stellarator_itg_objectives,
    default_stellarator_initial_params,
    nonlinear_heat_flux_trace,
    nonlinear_heat_flux_window_metrics,
    optimize_stellarator_itg,
    qa_observable_vector,
    stellarator_itg_objective,
    stellarator_itg_objective_residual_names,
    stellarator_itg_objective_residual_vector,
    stellarator_itg_residual_sensitivity_report,
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


def test_stellarator_itg_observable_contract_is_finite_and_exported() -> None:
    assert spectraxgk.STELLARATOR_ITG_PARAMETER_NAMES == PARAMETER_NAMES
    assert spectraxgk.STELLARATOR_ITG_OBSERVABLE_NAMES == OBSERVABLE_NAMES
    assert spectraxgk.optimize_stellarator_itg is optimize_stellarator_itg
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
        np.testing.assert_allclose(float(value), float(jnp.dot(residual, residual)), rtol=1.0e-6)
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
    assert report["finite_difference_gate"]["finite_difference_parallel"]["requested_workers"] == 2
    assert report["conditioning_gate"]["passed"] is True
    assert report["conditioning_gate"]["sensitivity_map_rank"] == len(PARAMETER_NAMES)
    assert report["covariance"]["source"] == "weighted_objective_residual"
    assert report["covariance"]["conditioning_gate"]["passed"] is True
    assert report["residual_names"] == list(stellarator_itg_objective_residual_names("quasilinear_flux"))


def test_nonlinear_heat_flux_window_metrics_use_late_stable_samples() -> None:
    cfg = _fast_config()
    times, heat_flux = nonlinear_heat_flux_trace(default_stellarator_initial_params(), cfg)
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


def test_optimize_stellarator_itg_reduces_nonlinear_window_objective(monkeypatch) -> None:
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

    initial = dict(zip(OBSERVABLE_NAMES, result.initial_observables, strict=True))
    final = dict(zip(OBSERVABLE_NAMES, result.final_observables, strict=True))
    assert final["growth_rate"] < initial["growth_rate"]
    assert final["quasilinear_heat_flux"] < initial["quasilinear_heat_flux"]
    assert final["nonlinear_heat_flux_mean"] < initial["nonlinear_heat_flux_mean"]


def test_compare_stellarator_itg_objectives_payload_is_json_ready(monkeypatch) -> None:
    _disable_optional_backend_discovery(monkeypatch)
    cfg = _fast_config()

    payload = compare_stellarator_itg_objectives(("growth",), config=cfg, workers=2, finite_difference_workers=2)

    assert payload["parameter_names"] == list(PARAMETER_NAMES)
    assert payload["observable_names"] == list(OBSERVABLE_NAMES)
    assert payload["parallel"]["requested_workers"] == 2
    assert payload["parallel"]["finite_difference_workers"] == 2
    assert len(payload["results"]) == 1
    result = payload["results"][0]
    assert result["objective_kind"] == "growth"
    assert result["final_objective"] < result["initial_objective"]
    assert result["gradient_gate"]["passed"] is True
    assert result["gradient_gate"]["finite_difference_parallel"]["requested_workers"] == 2


def test_compare_stellarator_itg_objectives_parallel_preserves_order(monkeypatch) -> None:
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
