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
    compare_stellarator_itg_objectives,
    default_stellarator_initial_params,
    nonlinear_heat_flux_trace,
    nonlinear_heat_flux_window_metrics,
    optimize_stellarator_itg,
    qa_observable_vector,
    stellarator_itg_objective,
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
        assert float(value) > 0.0
        report = autodiff_finite_difference_report(
            lambda x, kind=kind: stellarator_itg_objective(x, kind, cfg),
            params,
            step=cfg.fd_step,
            rtol=rtol,
            atol=atol,
        )
        assert report["passed"] is True


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
    assert result.covariance["sensitivity_map_rank"] >= 3
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

    payload = compare_stellarator_itg_objectives(("growth",), config=cfg)

    assert payload["parameter_names"] == list(PARAMETER_NAMES)
    assert payload["observable_names"] == list(OBSERVABLE_NAMES)
    assert len(payload["results"]) == 1
    result = payload["results"][0]
    assert result["objective_kind"] == "growth"
    assert result["final_objective"] < result["initial_objective"]
    assert result["gradient_gate"]["passed"] is True
