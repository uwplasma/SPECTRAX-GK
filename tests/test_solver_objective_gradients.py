from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import spectraxgk
from spectraxgk.solver_objective_gradients import (
    SOLVER_GEOMETRY_PARAMETER_NAMES,
    SOLVER_OBJECTIVE_NAMES,
    TINY_OBJECTIVE_NAMES,
    VMEC_BOOZER_FREQUENCY_OBJECTIVE_NAMES,
    VMEC_BOOZER_NONLINEAR_WINDOW_OBJECTIVE_NAMES,
    VMEC_BOOZER_QUASILINEAR_OBJECTIVE_NAMES,
    VMEC_BOOZER_STATE_PARAMETER_NAMES,
    _objective_gate_rows,
    _reduced_nonlinear_window_metrics_from_linear_observables,
    _vmec_boozer_state_parameter_name,
    default_solver_geometry_design_params,
    linear_solver_geometry_gradient_report,
    mode21_vmec_boozer_linear_frequency_gradient_report,
    mode21_vmec_boozer_nonlinear_window_gradient_report,
    mode21_vmec_boozer_quasilinear_gradient_report,
    solver_ready_geometry_mapping,
    tiny_differentiable_objective_gradient_report,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "build_solver_objective_gradient_gate.py"
spec = importlib.util.spec_from_file_location(
    "build_solver_objective_gradient_gate", SCRIPT
)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def test_solver_ready_geometry_mapping_validates_contract() -> None:
    theta = jnp.linspace(-jnp.pi, jnp.pi, 8, endpoint=False)
    mapping = solver_ready_geometry_mapping(
        default_solver_geometry_design_params(), theta
    )

    assert spectraxgk.solver_ready_geometry_mapping is solver_ready_geometry_mapping
    assert tuple(SOLVER_GEOMETRY_PARAMETER_NAMES) == (
        "bmag_ripple",
        "curvature_drift_scale",
    )
    assert mapping["theta"].shape == theta.shape
    assert np.all(np.asarray(mapping["bmag"]) > 0.0)
    with pytest.raises(ValueError, match="length-2"):
        solver_ready_geometry_mapping(jnp.ones(3), theta)


def test_tiny_differentiable_objective_gradient_report_is_finite_and_conditioned() -> None:
    x64_enabled = bool(jax.config.read("jax_enable_x64"))
    fd_step = 2.0e-5 if x64_enabled else 1.0e-3
    rtol = 5.0e-4 if x64_enabled else 1.0e-2
    atol = 1.0e-7 if x64_enabled else 1.0e-4

    report = tiny_differentiable_objective_gradient_report(
        fd_step=fd_step,
        rtol=rtol,
        atol=atol,
    )

    assert report["passed"] is True
    assert report["source_scope"] == "solver_ready_geometry_contract"
    assert report["observable_names"] == list(TINY_OBJECTIVE_NAMES)
    assert report["parameter_names"] == list(SOLVER_GEOMETRY_PARAMETER_NAMES)
    assert report["finite_flags"]["autodiff_jacobian"] is True
    assert report["finite_flags"]["finite_difference_jacobian"] is True
    assert report["conditioning"]["jacobian_shape"] == [
        len(TINY_OBJECTIVE_NAMES),
        len(SOLVER_GEOMETRY_PARAMETER_NAMES),
    ]
    assert report["conditioning"]["sensitivity_map_rank"] == 2
    assert np.isfinite(float(report["tangent_ad_norm"]))
    assert len(report["gradient_checks"]) == (
        len(TINY_OBJECTIVE_NAMES) * len(SOLVER_GEOMETRY_PARAMETER_NAMES)
    )
    json.dumps(report, allow_nan=False)

    with pytest.raises(ValueError, match="length-2"):
        tiny_differentiable_objective_gradient_report(jnp.ones(3))


def test_vmec_boozer_state_parameter_name_tracks_default_and_explicit_modes() -> None:
    assert (
        _vmec_boozer_state_parameter_name(17, 1, default_mid_surface=17)
        == "Rcos_mid_surface_m1"
    )
    assert (
        _vmec_boozer_state_parameter_name(11, 2, default_mid_surface=17)
        == "Rcos_r11_m2"
    )


def test_linear_solver_geometry_gradient_report_passes_actual_rhs_gate() -> None:
    report = linear_solver_geometry_gradient_report(
        fd_step=1.0e-3, rtol=1.0e-1, atol=2.0e-3
    )

    assert (
        spectraxgk.linear_solver_geometry_gradient_report
        is linear_solver_geometry_gradient_report
    )
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

    with pytest.raises(ValueError, match="length-2"):
        linear_solver_geometry_gradient_report(jnp.ones(3))


def test_reduced_nonlinear_window_metrics_are_smooth_and_fd_checked() -> None:
    x0 = jnp.asarray([0.22, 0.75, 1.10], dtype=jnp.float32)

    def metric_fn(x: jnp.ndarray) -> jnp.ndarray:
        return _reduced_nonlinear_window_metrics_from_linear_observables(
            x[0],
            x[1],
            x[2],
            dt=0.08,
            steps=18,
            tail_fraction=0.40,
        )

    metrics = np.asarray(metric_fn(x0))
    assert metrics.shape == (3,)
    assert np.all(np.isfinite(metrics))
    assert metrics[0] > 0.0
    assert metrics[1] >= 0.0

    ad = np.asarray(jax.jacfwd(metric_fn)(x0))
    fd_columns = []
    step = 1.0e-3
    eye = np.eye(3, dtype=np.float32)
    for direction in eye:
        forward = np.asarray(metric_fn(x0 + step * jnp.asarray(direction)))
        backward = np.asarray(metric_fn(x0 - step * jnp.asarray(direction)))
        fd_columns.append((forward - backward) / (2.0 * step))
    fd = np.stack(fd_columns, axis=1)
    np.testing.assert_allclose(ad, fd, rtol=2.0e-2, atol=5.0e-3)

    with pytest.raises(ValueError, match="steps"):
        _reduced_nonlinear_window_metrics_from_linear_observables(
            0.1, 1.0, 1.0, steps=3
        )
    with pytest.raises(ValueError, match="tail_fraction"):
        _reduced_nonlinear_window_metrics_from_linear_observables(
            0.1, 1.0, 1.0, tail_fraction=0.0
        )


def test_objective_gate_rows_are_json_ready_and_gate_each_parameter() -> None:
    report = {
        "jacobian_implicit": [[1.0, 2.0], [0.0, 1.1]],
        "jacobian_fd": [[1.0, 2.2], [0.0, 1.0]],
    }

    rows = _objective_gate_rows(
        report,
        parameter_names=("p0", "p1"),
        objective_names=("gamma", "ql"),
        rtol=0.05,
        atol=1.0e-6,
    )

    assert len(rows) == 4
    assert rows[0]["objective"] == "gamma"
    assert rows[0]["parameter"] == "p0"
    assert rows[0]["passed"] is True
    assert rows[1]["passed"] is False
    assert rows[1]["rel_error"] == pytest.approx(abs(2.0 - 2.2) / 2.2)
    assert all(isinstance(row["passed"], bool) for row in rows)


def test_mode21_vmec_boozer_frequency_gate_exports_and_scope() -> None:
    assert spectraxgk.mode21_vmec_boozer_linear_frequency_gradient_report is (
        mode21_vmec_boozer_linear_frequency_gradient_report
    )
    assert spectraxgk.mode21_vmec_boozer_quasilinear_gradient_report is (
        mode21_vmec_boozer_quasilinear_gradient_report
    )
    assert spectraxgk.mode21_vmec_boozer_nonlinear_window_gradient_report is (
        mode21_vmec_boozer_nonlinear_window_gradient_report
    )
    assert tuple(VMEC_BOOZER_STATE_PARAMETER_NAMES) == ("Rcos_mid_surface_m1",)
    assert tuple(VMEC_BOOZER_FREQUENCY_OBJECTIVE_NAMES) == ("gamma", "omega")
    assert tuple(VMEC_BOOZER_QUASILINEAR_OBJECTIVE_NAMES) == (
        "gamma",
        "omega",
        "kperp_eff2",
        "linear_heat_flux_weight",
        "mixing_length_heat_flux_proxy",
    )
    assert tuple(VMEC_BOOZER_NONLINEAR_WINDOW_OBJECTIVE_NAMES) == (
        "gamma",
        "omega",
        "kperp_eff2",
        "linear_heat_flux_weight",
        "mixing_length_heat_flux_proxy",
        "nonlinear_window_heat_flux_mean",
        "nonlinear_window_heat_flux_cv",
        "nonlinear_window_heat_flux_trend",
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

    paths = mod.write_solver_objective_gradient_artifacts(
        payload, out=tmp_path / "gate.png"
    )

    for path in paths.values():
        assert Path(path).exists()
    saved = json.loads((tmp_path / "gate.json").read_text(encoding="utf-8"))
    assert saved["kind"] == "linear_solver_geometry_gradient_gate"
