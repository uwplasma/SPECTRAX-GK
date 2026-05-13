from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import spectraxgk
import spectraxgk.solver_objective_gradients as sog
from spectraxgk.solver_objective_gradients import (
    SOLVER_GEOMETRY_PARAMETER_NAMES,
    SOLVER_OBJECTIVE_NAMES,
    SolverScalarObjective,
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
    solver_objective_branch_gradient_report,
    solver_objective_vector_from_geometry,
    solver_scalar_objective_from_vector,
    solver_ready_geometry_mapping,
    tiny_differentiable_objective_gradient_report,
    vmec_boozer_aggregate_scalar_objective_finite_difference_report,
    vmec_boozer_aggregate_scalar_objective_from_state,
    vmec_boozer_scalar_objective_finite_difference_report,
    vmec_boozer_scalar_objective_from_state,
    vmec_boozer_scalar_objective_line_search_report,
    vmec_boozer_solver_objective_table_from_state,
    vmec_boozer_solver_objective_vector_from_state,
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


def test_solver_objective_vector_from_geometry_is_finite_and_exported() -> None:
    theta = jnp.linspace(-jnp.pi, jnp.pi, 4, endpoint=False)
    geom = spectraxgk.flux_tube_geometry_from_mapping(
        solver_ready_geometry_mapping(default_solver_geometry_design_params(), theta),
        validate_finite=False,
    )

    vector = solver_objective_vector_from_geometry(
        geom,
        n_laguerre=1,
        n_hermite=1,
        ny=4,
        selected_ky_index=1,
    )

    assert spectraxgk.solver_objective_vector_from_geometry is solver_objective_vector_from_geometry
    assert vector.shape == (len(SOLVER_OBJECTIVE_NAMES),)
    assert np.all(np.isfinite(np.asarray(vector)))
    with pytest.raises(ValueError, match="selected_ky_index"):
        solver_objective_vector_from_geometry(geom, selected_ky_index=99)
    with pytest.raises(ValueError, match="positive"):
        solver_objective_vector_from_geometry(geom, n_laguerre=0)


def test_solver_scalar_objective_selector_aliases_and_errors() -> None:
    vector = jnp.asarray([1.0, -0.5, 2.0, 3.0, 4.0, 5.0])

    assert spectraxgk.SolverScalarObjective is SolverScalarObjective
    assert spectraxgk.solver_scalar_objective_from_vector is solver_scalar_objective_from_vector
    assert float(solver_scalar_objective_from_vector(vector, "growth")) == pytest.approx(1.0)
    assert float(solver_scalar_objective_from_vector(vector, "gamma")) == pytest.approx(1.0)
    assert float(solver_scalar_objective_from_vector(vector, "frequency")) == pytest.approx(-0.5)
    assert float(solver_scalar_objective_from_vector(vector, "quasilinear_flux")) == pytest.approx(5.0)
    with pytest.raises(ValueError, match="unknown solver objective"):
        solver_scalar_objective_from_vector(vector, "bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="length"):
        solver_scalar_objective_from_vector(jnp.ones(2), "growth")


def test_solver_objective_branch_gradient_report_gates_public_evaluator() -> None:
    report = solver_objective_branch_gradient_report(
        fd_step=1.0e-3,
        rtol=1.0e-1,
        atol=2.0e-3,
        n_laguerre=2,
        n_hermite=1,
    )

    assert (
        spectraxgk.solver_objective_branch_gradient_report
        is solver_objective_branch_gradient_report
    )
    assert report["passed"] is True
    assert report["source_scope"] == "solver_ready_geometry_contract"
    assert report["value_evaluator_finite"] is True
    assert report["branch_continuity_gate"] is True
    assert report["ad_fd_gate"] is True
    assert len(report["branch_rows"]) == 2 * len(SOLVER_GEOMETRY_PARAMETER_NAMES)
    assert np.asarray(report["value_evaluator_objectives"]).shape == (
        len(SOLVER_OBJECTIVE_NAMES),
    )
    assert np.asarray(report["eigenpair_gate"]["jacobian_implicit"]).shape == (
        len(SOLVER_OBJECTIVE_NAMES),
        len(SOLVER_GEOMETRY_PARAMETER_NAMES),
    )
    with pytest.raises(ValueError, match="length-2"):
        solver_objective_branch_gradient_report(jnp.ones(3))


def test_vmec_boozer_solver_objective_vector_from_state_splits_options(
    monkeypatch,
) -> None:
    calls: dict[str, object] = {}

    def fake_geometry(state, static, indata, wout, **kwargs):  # noqa: ANN001, ANN202
        calls["geometry"] = (state, static, indata, wout, kwargs)
        return "geom"

    def fake_objective(geom, **kwargs):  # noqa: ANN001, ANN202
        calls["objective"] = (geom, kwargs)
        return jnp.arange(len(SOLVER_OBJECTIVE_NAMES), dtype=jnp.float32)

    monkeypatch.setattr(sog, "flux_tube_geometry_from_vmec_boozer_state", fake_geometry)
    monkeypatch.setattr(sog, "solver_objective_vector_from_geometry", fake_objective)

    vector = vmec_boozer_solver_objective_vector_from_state(
        "state",
        "static",
        "indata",
        "wout",
        surface_index=2,
        ntheta=8,
        mboz=21,
        nboz=21,
        selected_ky_index=1,
        n_laguerre=2,
    )

    assert spectraxgk.vmec_boozer_solver_objective_vector_from_state is (
        vmec_boozer_solver_objective_vector_from_state
    )
    assert np.asarray(vector).tolist() == list(range(len(SOLVER_OBJECTIVE_NAMES)))
    assert calls["geometry"][4] == {
        "surface_index": 2,
        "ntheta": 8,
        "mboz": 21,
        "nboz": 21,
    }
    assert calls["objective"] == ("geom", {"selected_ky_index": 1, "n_laguerre": 2})
    with pytest.raises(TypeError, match="unknown VMEC/Boozer objective options"):
        vmec_boozer_solver_objective_vector_from_state(
            "state",
            "static",
            "indata",
            "wout",
            unexpected=True,
        )


def test_vmec_boozer_scalar_objective_from_state_uses_vector_selector(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        sog,
        "vmec_boozer_solver_objective_vector_from_state",
        lambda *_args, **_kwargs: jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    )

    value = vmec_boozer_scalar_objective_from_state(
        "state",
        "static",
        "indata",
        "wout",
        objective="quasilinear_flux",
        ntheta=4,
    )

    assert spectraxgk.vmec_boozer_scalar_objective_from_state is vmec_boozer_scalar_objective_from_state
    assert float(value) == pytest.approx(6.0)


def test_vmec_boozer_solver_objective_table_samples_surfaces_alphas_and_ky(
    monkeypatch,
) -> None:
    geometry_calls: list[dict[str, object]] = []
    objective_calls: list[tuple[str, dict[str, object]]] = []

    def fake_geometry(_state, _static, _indata, _wout, **kwargs):  # noqa: ANN001, ANN202
        geometry_calls.append(dict(kwargs))
        return f"geom-{len(geometry_calls)}"

    def fake_objective(geom, **kwargs):  # noqa: ANN001, ANN202
        objective_calls.append((geom, dict(kwargs)))
        ky = float(kwargs["selected_ky_index"])
        geom_index = float(str(geom).split("-")[-1])
        return jnp.asarray([geom_index + ky, 0.0, 1.0, 2.0, 0.0, 3.0])

    monkeypatch.setattr(sog, "flux_tube_geometry_from_vmec_boozer_state", fake_geometry)
    monkeypatch.setattr(sog, "solver_objective_vector_from_geometry", fake_objective)

    table = vmec_boozer_solver_objective_table_from_state(
        "state",
        "static",
        "indata",
        "wout",
        surface_indices=[1, 3],
        alphas=[0.0, 0.5],
        selected_ky_indices=[1, 2],
        ntheta=8,
        n_laguerre=2,
    )

    assert spectraxgk.vmec_boozer_solver_objective_table_from_state is (
        vmec_boozer_solver_objective_table_from_state
    )
    assert np.asarray(table).shape == (8, len(SOLVER_OBJECTIVE_NAMES))
    assert len(geometry_calls) == 4
    assert len(objective_calls) == 8
    assert geometry_calls[0] == {"surface_index": 1, "alpha": 0.0, "ntheta": 8}
    assert objective_calls[0] == ("geom-1", {"n_laguerre": 2, "selected_ky_index": 1})
    assert objective_calls[1] == ("geom-1", {"n_laguerre": 2, "selected_ky_index": 2})
    with pytest.raises(TypeError, match="selected_ky_indices"):
        vmec_boozer_solver_objective_table_from_state(
            "state",
            "static",
            "indata",
            "wout",
            selected_ky_index=1,
            selected_ky_indices=[1, 2],
        )


def test_vmec_boozer_aggregate_scalar_objective_from_state_reductions(
    monkeypatch,
) -> None:
    table = jnp.asarray(
        [
            [1.0, 0.0, 1.0, 2.0, 0.0, 10.0],
            [2.0, 0.0, 1.0, 2.0, 0.0, 20.0],
            [4.0, 0.0, 1.0, 2.0, 0.0, 40.0],
        ]
    )
    monkeypatch.setattr(
        sog,
        "vmec_boozer_solver_objective_table_from_state",
        lambda *_args, **_kwargs: table,
    )

    mean_value = vmec_boozer_aggregate_scalar_objective_from_state(
        "state",
        "static",
        "indata",
        "wout",
        objective="growth",
    )
    weighted_value = vmec_boozer_aggregate_scalar_objective_from_state(
        "state",
        "static",
        "indata",
        "wout",
        objective="quasilinear_flux",
        reduction="weighted_mean",
        weights=[1.0, 1.0, 2.0],
    )
    max_value = vmec_boozer_aggregate_scalar_objective_from_state(
        "state",
        "static",
        "indata",
        "wout",
        objective="growth",
        reduction="max",
    )

    assert spectraxgk.vmec_boozer_aggregate_scalar_objective_from_state is (
        vmec_boozer_aggregate_scalar_objective_from_state
    )
    assert float(mean_value) == pytest.approx(7.0 / 3.0)
    assert float(weighted_value) == pytest.approx(27.5)
    assert float(max_value) == pytest.approx(4.0)
    with pytest.raises(ValueError, match="weights"):
        vmec_boozer_aggregate_scalar_objective_from_state(
            "state",
            "static",
            "indata",
            "wout",
            reduction="weighted_mean",
            weights=[1.0],
        )
    with pytest.raises(ValueError, match="reduction"):
        vmec_boozer_aggregate_scalar_objective_from_state(
            "state",
            "static",
            "indata",
            "wout",
            reduction="median",  # type: ignore[arg-type]
        )


def test_vmec_boozer_scalar_objective_finite_difference_report(
    monkeypatch,
) -> None:
    @dataclass(frozen=True)
    class FakeState:
        Rcos: jnp.ndarray

    fake_state = FakeState(Rcos=jnp.zeros((5, 3), dtype=jnp.float32))
    monkeypatch.setattr(
        sog,
        "_load_vmec_jax_example_state_bundle",
        lambda case_name: {
            "case_name": case_name,
            "input_path": "input.test",
            "wout_path": "wout.test",
            "state": fake_state,
            "static": "static",
            "indata": "indata",
            "wout": "wout",
        },
    )

    def fake_vector(state, *_args, **_kwargs):  # noqa: ANN001, ANN202
        coeff = float(np.asarray(state.Rcos[2, 1]))
        return jnp.asarray([1.0 + 3.0 * coeff, 0.0, 2.0, 4.0, 0.5, 5.0 + coeff])

    monkeypatch.setattr(sog, "vmec_boozer_solver_objective_vector_from_state", fake_vector)

    report = vmec_boozer_scalar_objective_finite_difference_report(
        case_name="case",
        objective="growth",
        base_delta=0.1,
        perturbation_step=1.0e-3,
        response_atol=1.0e-6,
        ntheta=4,
    )

    assert spectraxgk.vmec_boozer_scalar_objective_finite_difference_report is (
        vmec_boozer_scalar_objective_finite_difference_report
    )
    assert report["passed"] is True
    assert report["source_scope"] == "mode21_vmec_boozer_state"
    assert report["parameter_name"] == "Rcos_mid_surface_m1"
    assert report["base_delta"] == pytest.approx(0.1)
    assert report["central_derivative"] == pytest.approx(3.0, rel=1.0e-4)
    assert report["response_resolved"] is True
    assert report["finite_difference_consistent"] is True
    assert report["curvature_ratio"] < 1.0e-3
    assert report["options"] == {"ntheta": 4}

    with pytest.raises(ValueError, match="perturbation_step"):
        vmec_boozer_scalar_objective_finite_difference_report(perturbation_step=0.0)
    with pytest.raises(ValueError, match="max_curvature_ratio"):
        vmec_boozer_scalar_objective_finite_difference_report(max_curvature_ratio=-1.0)
    with pytest.raises(ValueError, match="radial_index"):
        vmec_boozer_scalar_objective_finite_difference_report(radial_index=99)


def test_vmec_boozer_aggregate_scalar_objective_finite_difference_report(
    monkeypatch,
) -> None:
    @dataclass(frozen=True)
    class FakeState:
        Rcos: jnp.ndarray

    fake_state = FakeState(Rcos=jnp.zeros((5, 3), dtype=jnp.float32))
    monkeypatch.setattr(
        sog,
        "_load_vmec_jax_example_state_bundle",
        lambda case_name: {
            "case_name": case_name,
            "input_path": "input.multi",
            "wout_path": "wout.multi",
            "state": fake_state,
            "static": "static",
            "indata": "indata",
            "wout": "wout",
        },
    )

    def fake_table(state, *_args, **kwargs):  # noqa: ANN001, ANN202
        coeff = float(np.asarray(state.Rcos[2, 1]))
        ky_indices = tuple(kwargs["selected_ky_indices"])
        rows = []
        for ky in ky_indices:
            rows.append([1.0 + 2.0 * coeff + float(ky), 0.0, 1.0, 3.0, 0.0, 5.0 + coeff])
        return jnp.asarray(rows)

    monkeypatch.setattr(sog, "vmec_boozer_solver_objective_table_from_state", fake_table)

    report = vmec_boozer_aggregate_scalar_objective_finite_difference_report(
        case_name="case",
        objective="growth",
        reduction="weighted_mean",
        weights=[1.0, 3.0],
        selected_ky_indices=[1, 2],
        base_delta=0.1,
        perturbation_step=1.0e-3,
        response_atol=1.0e-6,
        ntheta=4,
    )

    assert spectraxgk.vmec_boozer_aggregate_scalar_objective_finite_difference_report is (
        vmec_boozer_aggregate_scalar_objective_finite_difference_report
    )
    assert report["passed"] is True
    assert report["source_scope"] == "mode21_vmec_boozer_state_multi_point"
    assert report["n_samples"] == 2
    assert report["parameter_name"] == "Rcos_mid_surface_m1"
    assert report["base_value"] == pytest.approx(2.95)
    assert report["central_derivative"] == pytest.approx(2.0, rel=1.0e-4)
    assert report["curvature_ratio"] < 1.0e-3
    assert report["samples"] == [
        {"surface_index": None, "alpha": 0.0, "selected_ky_index": 1, "weight": 0.25},
        {"surface_index": None, "alpha": 0.0, "selected_ky_index": 2, "weight": 0.75},
    ]
    assert report["options"] == {"ntheta": 4}

    with pytest.raises(ValueError, match="perturbation_step"):
        vmec_boozer_aggregate_scalar_objective_finite_difference_report(perturbation_step=0.0)
    with pytest.raises(ValueError, match="weights"):
        vmec_boozer_aggregate_scalar_objective_finite_difference_report(
            selected_ky_indices=[1, 2],
            weights=[1.0],
        )


def test_vmec_boozer_scalar_objective_line_search_report_accepts_safe_updates(
    monkeypatch,
) -> None:
    calls: list[float] = []

    def fake_fd_report(**kwargs):  # noqa: ANN003, ANN202
        delta = float(kwargs.get("base_delta", 0.0))
        calls.append(delta)
        value = 1.0 + 2.0 * delta
        return {
            "passed": True,
            "base_value": value,
            "central_derivative": 2.0,
            "curvature_ratio": 0.0,
        }

    monkeypatch.setattr(
        sog,
        "vmec_boozer_scalar_objective_finite_difference_report",
        fake_fd_report,
    )

    report = vmec_boozer_scalar_objective_line_search_report(
        objective="growth",
        update_step=0.05,
        max_steps=2,
        ntheta=4,
    )

    assert spectraxgk.vmec_boozer_scalar_objective_line_search_report is (
        vmec_boozer_scalar_objective_line_search_report
    )
    assert report["passed"] is True
    assert report["accepted_steps"] == 2
    assert report["final_delta"] == pytest.approx(-0.10)
    assert report["final_objective"] < report["initial_objective"]
    assert all(row["accepted"] for row in report["history"])
    assert calls[:2] == [0.0, -0.05]


def test_vmec_boozer_scalar_objective_line_search_report_fails_closed(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        sog,
        "vmec_boozer_scalar_objective_finite_difference_report",
        lambda **_kwargs: {
            "passed": False,
            "base_value": 1.0,
            "central_derivative": 2.0,
            "curvature_ratio": 9.0,
        },
    )

    report = vmec_boozer_scalar_objective_line_search_report(max_steps=1)

    assert report["passed"] is False
    assert report["accepted_steps"] == 0
    assert report["stop_reason"] == "finite_difference_gate_failed"
    with pytest.raises(ValueError, match="max_steps"):
        vmec_boozer_scalar_objective_line_search_report(max_steps=0)
    with pytest.raises(ValueError, match="update_step"):
        vmec_boozer_scalar_objective_line_search_report(update_step=0.0)
    with pytest.raises(ValueError, match="min_improvement"):
        vmec_boozer_scalar_objective_line_search_report(min_improvement=-1.0)


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
