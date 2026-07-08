"""Tests for VMEC artifact reports outside the VMEC/Boozer aggregate suite."""

from __future__ import annotations

import json
from pathlib import Path

from netCDF4 import Dataset
import numpy as np
import pytest

from support.paths import load_artifact_tool



# External VMEC replicate ensemble assertions
def _build_external_vmec_replicate_ensemble_write_output(
    path: Path, offset: float
) -> None:
    netcdf4 = pytest.importorskip("netCDF4")
    t = np.linspace(0.0, 100.0, 101)
    q = 10.0 + offset + 0.02 * np.sin(2.0 * np.pi * t / 20.0)
    with netcdf4.Dataset(path, "w") as root:
        root.createDimension("time", t.size)
        root.createDimension("s", 1)
        grids = root.createGroup("Grids")
        diagnostics = root.createGroup("Diagnostics")
        grids.createVariable("time", "f8", ("time",))[:] = t
        diagnostics.createVariable("HeatFlux_st", "f8", ("time", "s"))[:, :] = q[
            :, None
        ]


def test_replicate_ensemble_tool_builds_trace_reports_and_plot(tmp_path: Path) -> None:
    mod = load_artifact_tool("build_external_vmec_replicate_ensemble")
    outputs = [
        tmp_path / "demo_nonlinear_t100_n64_seed31.out.nc",
        tmp_path / "demo_nonlinear_t100_n64_seed32.out.nc",
        tmp_path / "demo_nonlinear_t100_n64_dt0p04.out.nc",
    ]
    for path, offset in zip(outputs, (-0.05, 0.05, 0.0)):
        _build_external_vmec_replicate_ensemble_write_output(path, offset)
    out_dir = tmp_path / "artifacts"

    rc = mod.main(
        [
            *[str(path) for path in outputs],
            "--out-dir",
            str(out_dir),
            "--case",
            "demo_replicate_window",
            "--tmin",
            "50",
            "--tmax",
            "100",
            "--baseline-seed",
            "22",
            "--baseline-dt",
            "0.05",
            "--artifact-prefix",
            "docs/_static/demo_replicates",
            "--bootstrap-samples",
            "32",
        ]
    )

    assert rc == 0
    readiness = json.loads((out_dir / "replicate_ensemble_readiness.json").read_text())
    ensemble = json.loads((out_dir / "replicate_ensemble_gate.json").read_text())
    summary = json.loads(
        (out_dir / "demo_nonlinear_t100_n64_seed31_transport_window.json").read_text()
    )
    assert readiness["passed"] is True
    assert ensemble["passed"] is True
    assert summary["nonlinear_artifact"] == "demo_nonlinear_t100_n64_seed31.out.nc"
    assert len(list(out_dir.glob("*_heat_flux_trace.csv"))) == 3
    assert (out_dir / "replicate_ensemble_gate.png").exists()
    assert readiness["observed_artifacts"][0]["source_artifact"].startswith(
        "docs/_static/demo_replicates/"
    )


def test_replicate_ensemble_tool_can_collect_failed_diagnostic_points(
    tmp_path: Path,
) -> None:
    mod = load_artifact_tool("build_external_vmec_replicate_ensemble")
    outputs = [
        tmp_path / "diagnostic_nonlinear_t100_n64_seed31.out.nc",
        tmp_path / "diagnostic_nonlinear_t100_n64_seed32.out.nc",
        tmp_path / "diagnostic_nonlinear_t100_n64_dt0p04.out.nc",
    ]
    for path, offset in zip(outputs, (-4.0, 0.0, 4.0)):
        _build_external_vmec_replicate_ensemble_write_output(path, offset)

    common_args = [
        *[str(path) for path in outputs],
        "--case",
        "diagnostic_landscape_point",
        "--tmin",
        "50",
        "--tmax",
        "100",
        "--baseline-seed",
        "22",
        "--baseline-dt",
        "0.05",
        "--bootstrap-samples",
        "32",
        "--max-mean-rel-spread",
        "0.01",
    ]
    strict_dir = tmp_path / "strict"
    relaxed_dir = tmp_path / "relaxed"

    strict_rc = mod.main([*common_args, "--out-dir", str(strict_dir)])
    relaxed_rc = mod.main(
        [*common_args, "--out-dir", str(relaxed_dir), "--allow-failed-gates"]
    )

    assert strict_rc == 1
    assert relaxed_rc == 0
    ensemble = json.loads((relaxed_dir / "replicate_ensemble_gate.json").read_text())
    assert ensemble["passed"] is False
    assert (relaxed_dir / "replicate_ensemble_gate.png").exists()


def test_replicate_ensemble_tool_handles_requested_window_outside_trace(
    tmp_path: Path,
) -> None:
    mod = load_artifact_tool("build_external_vmec_replicate_ensemble")
    outputs = [
        tmp_path / "short_nonlinear_t100_n64_seed31.out.nc",
        tmp_path / "short_nonlinear_t100_n64_seed32.out.nc",
        tmp_path / "short_nonlinear_t100_n64_dt0p04.out.nc",
    ]
    for path, offset in zip(outputs, (-0.05, 0.05, 0.0)):
        _build_external_vmec_replicate_ensemble_write_output(path, offset)
    out_dir = tmp_path / "outside_window"

    rc = mod.main(
        [
            *[str(path) for path in outputs],
            "--out-dir",
            str(out_dir),
            "--case",
            "outside_requested_window",
            "--tmin",
            "200",
            "--tmax",
            "300",
            "--bootstrap-samples",
            "16",
            "--allow-failed-gates",
        ]
    )

    assert rc == 0
    ensemble = json.loads((out_dir / "replicate_ensemble_gate.json").read_text())
    report = json.loads(
        next(
            (out_dir / "nonlinear_window_convergence_reports").glob("*seed31*")
        ).read_text()
    )
    assert ensemble["passed"] is False
    assert ensemble["statistics"]["n_finite_means"] == 0
    assert ensemble["statistics"]["ensemble_mean"] is None
    assert report["window"]["n_finite_late"] == 0
    assert (out_dir / "replicate_ensemble_gate.png").exists()


def test_replicate_ensemble_tool_parses_joint_seed_timestep_variant(
    tmp_path: Path,
) -> None:
    mod = load_artifact_tool("build_external_vmec_replicate_ensemble")
    variant = mod._variant_from_path(
        tmp_path / "demo_nonlinear_t100_n64_seed32_dt0p04.out.nc",
        baseline_seed=22,
        baseline_dt=0.05,
    )

    assert variant == {
        "variant_axis": "seed_timestep",
        "variant_label": "seed32_dt0p04",
        "seed": 32,
        "dt": 0.04,
        "variant": {"seed": 32, "timestep": 0.04},
    }


def test_replicate_ensemble_tool_parses_timestep_variant_with_device_suffix(
    tmp_path: Path,
) -> None:
    mod = load_artifact_tool("build_external_vmec_replicate_ensemble")
    variant = mod._variant_from_path(
        tmp_path / "demo_nonlinear_t250_n48_dt0p01_gpu.out.nc",
        baseline_seed=22,
        baseline_dt=0.05,
    )

    assert variant == {
        "variant_axis": "timestep",
        "variant_label": "dt0p01",
        "seed": 22,
        "dt": 0.01,
        "variant": {"seed": 22, "timestep": 0.01},
    }


def test_replicate_ensemble_tool_ignores_protocol_dt_in_case_slug(
    tmp_path: Path,
) -> None:
    mod = load_artifact_tool("build_external_vmec_replicate_ensemble")

    seed_variant = mod._variant_from_path(
        tmp_path / "solovev_reference_repair_dt002_amp1em5_n48_seed31.out.nc",
        baseline_seed=22,
        baseline_dt=0.02,
    )
    timestep_variant = mod._variant_from_path(
        tmp_path / "solovev_reference_repair_dt002_amp1em5_n48_dt0p01_gpu.out.nc",
        baseline_seed=22,
        baseline_dt=0.02,
    )

    assert seed_variant == {
        "variant_axis": "seed",
        "variant_label": "seed31",
        "seed": 31,
        "dt": 0.02,
        "variant": {"seed": 31, "timestep": 0.02},
    }
    assert timestep_variant == {
        "variant_axis": "timestep",
        "variant_label": "dt0p01",
        "seed": 22,
        "dt": 0.01,
        "variant": {"seed": 22, "timestep": 0.01},
    }


# VMEC-JAX boundary-chain collection assertions
def _build_vmec_jax_boundary_chain_collection_probe(
    path: Path, *, index: int, exact_ok: bool, growth_ok: bool
) -> None:
    path.write_text(
        json.dumps(
            {
                "kind": "vmec_jax_boundary_chain_probe",
                "index": index,
                "name": f"coeff{index}",
                "summary": {
                    "kind": "vmec_jax_boundary_chain_summary",
                    "finite": True,
                    "classification": (
                        "exact_fd_and_frozen_axis_replay_consistent"
                        if exact_ok
                        else "frozen_axis_replay_consistent_but_exact_fd_branch_sensitive"
                    ),
                    "passes": {
                        "final_state_matches_exact_fd": True,
                        "frozen_axis_matches_exact_fd": exact_ok,
                        "frozen_axis_jvp_vjp_consistent": True,
                    },
                    "errors": {
                        "frozen_axis_vs_exact_fd_rel": 0.02 if exact_ok else 0.4,
                    },
                    "metrics": {
                        "exact_fd_cost_gradient": 0.1,
                        "frozen_axis_replay_cost_gradient": 0.1 if exact_ok else 0.2,
                    },
                },
                "growth_branch_locality": {
                    "enabled": True,
                    "passed": growth_ok,
                    "classification": (
                        "all_samples_dominant_growth_branch_locally_consistent"
                        if growth_ok
                        else "growth_branch_locality_failed_or_incomplete"
                    ),
                },
            }
        ),
        encoding="utf-8",
    )


def test_build_collection_payload_counts_growth_branch_status(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _build_vmec_jax_boundary_chain_collection_probe(
        first, index=1, exact_ok=True, growth_ok=True
    )
    _build_vmec_jax_boundary_chain_collection_probe(
        second, index=2, exact_ok=False, growth_ok=False
    )

    payload = load_artifact_tool(
        "build_vmec_jax_boundary_chain_collection"
    ).build_collection_payload([first, second])

    assert payload["finite"] is True
    assert (
        payload["classification"]
        == "mixed_exact_fd_consistency_with_branch_sensitive_modes"
    )
    assert payload["counts"]["n_exact_fd_consistent"] == 1
    assert payload["counts"]["n_growth_branch_locality_checked"] == 2
    assert payload["counts"]["n_growth_branch_locality_passed"] == 1
    assert payload["rows"][0]["growth_branch_locality_passed"] is True
    assert payload["rows"][1]["growth_branch_locality_passed"] is False
    assert payload["probe_jsons"] == [str(first), str(second)]
    assert "not a nonlinear transport optimization claim" in payload["claim_scope"]


def test_build_collection_main_writes_json(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    out = tmp_path / "collection.json"
    _build_vmec_jax_boundary_chain_collection_probe(
        first, index=1, exact_ok=True, growth_ok=True
    )

    rc = load_artifact_tool("build_vmec_jax_boundary_chain_collection").main(
        ["--probe-json", str(first), "--out-json", str(out)]
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 0
    assert payload["counts"]["n_total"] == 1
    assert payload["counts"]["n_growth_branch_locality_passed"] == 1


# VMEC optimization candidate-screen assertions
def _build_vmec_optimization_candidate_screen_gate_write_spectrum(
    path: Path, rows: list[tuple[float, float, float, float, float]]
) -> None:
    lines = ["ky,gamma,omega,kperp_eff2,heat_flux_weight_total"]
    lines.extend(
        f"{ky},{gamma},{omega},{kperp},{heat}" for ky, gamma, omega, kperp, heat in rows
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_candidate_screen_rejects_nonpositive_kperp_even_with_large_growth(
    tmp_path: Path,
) -> None:
    mod = load_artifact_tool("build_vmec_optimization_candidate_screen_gate")
    spectrum = tmp_path / "bad.csv"
    _build_vmec_optimization_candidate_screen_gate_write_spectrum(
        spectrum,
        [
            (0.1, 1.2, -0.5, -0.7, 0.1),
            (0.2, 0.8, -0.4, -0.1, 0.2),
            (0.3, 0.4, -0.2, -0.2, 0.3),
        ],
    )

    row = mod.summarize_spectrum(label="bad_metric", spectrum_path=spectrum)

    assert row["passed"] is False
    assert row["status"] == "invalid_metric_nonpositive_kperp2"
    assert "nonpositive_effective_kperp2" in row["blockers"]
    assert row["max_gamma"] == 1.2


def test_candidate_screen_accepts_positive_metric_launch_candidate(
    tmp_path: Path,
) -> None:
    mod = load_artifact_tool("build_vmec_optimization_candidate_screen_gate")
    spectrum = tmp_path / "good.csv"
    _build_vmec_optimization_candidate_screen_gate_write_spectrum(
        spectrum,
        [
            (0.1, 0.01, -0.5, 0.7, 0.1),
            (0.2, 0.04, -0.4, 0.8, 0.2),
            (0.3, 0.03, -0.2, 0.9, 0.3),
        ],
    )

    report = mod.build_report([("good", spectrum)])

    assert report["passed"] is True
    assert report["n_launch_candidates"] == 1
    assert report["rows"][0]["status"] == "nonlinear_launch_candidate"


def test_candidate_screen_tool_writes_fail_closed_artifacts(tmp_path: Path) -> None:
    mod = load_artifact_tool("build_vmec_optimization_candidate_screen_gate")
    spectrum = tmp_path / "marginal.csv"
    _build_vmec_optimization_candidate_screen_gate_write_spectrum(
        spectrum,
        [
            (0.1, -0.01, -0.5, 0.7, 0.1),
            (0.2, 0.01, -0.4, 0.8, 0.2),
            (0.3, 0.015, -0.2, 0.9, 0.3),
        ],
    )
    out = tmp_path / "screen.json"

    assert mod.main(["--spectrum", f"marginal:{spectrum}", "--out", str(out)]) == 2
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert payload["rows"][0]["status"] == "marginal_or_incomplete_screen"
    assert out.with_suffix(".csv").exists()


# VMEC state-control bracket sweep assertions
def _build_vmec_state_control_bracket_sweep_status_gate(
    path: Path, *, alpha: float, parameter: str, response: float, passed: bool
) -> None:
    path.write_text(
        json.dumps(
            {
                "passed": passed,
                "blockers": [] if passed else ["fd_response_resolved"],
                "delta_parameter": alpha,
                "parameter_name": parameter,
                "config": {
                    "min_fd_response_fraction": 0.03,
                    "max_fd_asymmetry_rel": 0.5,
                    "max_gradient_uncertainty_rel": 0.5,
                },
                "metrics": {
                    "response_fraction": response,
                    "fd_asymmetry_rel": 0.2 if passed else 2.0,
                    "gradient_uncertainty_rel": 0.1 if passed else 4.0,
                    "baseline_window_mean": 1.0,
                    "plus_window_mean": 1.1,
                    "minus_window_mean": 0.9,
                },
            }
        ),
        encoding="utf-8",
    )


def test_build_vmec_state_control_bracket_sweep_status(tmp_path: Path) -> None:
    mod = load_artifact_tool("build_vmec_state_control_bracket_sweep_status")
    gate_a = tmp_path / "gate_a.json"
    gate_b = tmp_path / "gate_b.json"
    run_summary = tmp_path / "summary.json"
    out_prefix = tmp_path / "status"
    _build_vmec_state_control_bracket_sweep_status_gate(
        gate_a,
        alpha=0.003,
        parameter="state_control_rsin_mid_surface_m1",
        response=0.004,
        passed=False,
    )
    _build_vmec_state_control_bracket_sweep_status_gate(
        gate_b,
        alpha=0.01,
        parameter="state_control_zcos_mid_surface_m1",
        response=0.04,
        passed=True,
    )
    run_summary.write_text(
        json.dumps(
            {"successes": 36, "failures": [], "started_at": 1.0, "finished_at": 4.5}
        )
    )

    report = mod.build_bracket_sweep_status(
        [gate_b, gate_a], run_summary=run_summary, out_prefix=out_prefix
    )

    assert report["passed"] is False
    assert report["summary"]["central_fd_gates_passed"] == 1
    assert report["summary"]["central_fd_gates_total"] == 2
    assert report["summary"]["nonlinear_runs_completed"] == 36
    assert report["rows"][0]["alpha_delta"] == 0.003
    assert out_prefix.with_suffix(".json").exists()
    assert out_prefix.with_suffix(".csv").exists()
    assert out_prefix.with_suffix(".png").exists()
    assert out_prefix.with_suffix(".pdf").exists()


# VMEC state-to-input mapping assertions
def _build_vmec_state_to_input_mapping_response_controls() -> list[dict[str, object]]:
    return [
        {"state_parameter": "Rsin_mid_surface_m1"},
        {"state_parameter": "Zcos_mid_surface_m1"},
    ]


def _build_vmec_state_to_input_mapping_response_directions() -> list[dict[str, object]]:
    return [
        {
            "coefficient": "RBC(1,1)",
            "coefficient_slug": "rbc_1_1",
            "delta_parameter": 0.1,
        },
        {
            "coefficient": "ZBS(1,1)",
            "coefficient_slug": "zbs_1_1",
            "delta_parameter": 0.2,
        },
    ]


def _build_vmec_state_to_input_mapping_response_sample(
    value: tuple[float, float],
) -> dict[str, float]:
    return {
        "Rsin_mid_surface_m1": value[0],
        "Zcos_mid_surface_m1": value[1],
    }


def test_mapping_report_fails_closed_for_zero_symmetric_response() -> None:
    mod = load_artifact_tool("build_vmec_state_to_input_mapping_response")
    samples = {
        "RBC(1,1)": {
            "baseline": _build_vmec_state_to_input_mapping_response_sample((0.0, 0.0)),
            "plus_delta": _build_vmec_state_to_input_mapping_response_sample(
                (0.0, 0.0)
            ),
            "minus_delta": _build_vmec_state_to_input_mapping_response_sample(
                (0.0, 0.0)
            ),
        },
        "ZBS(1,1)": {
            "baseline": _build_vmec_state_to_input_mapping_response_sample((0.0, 0.0)),
            "plus_delta": _build_vmec_state_to_input_mapping_response_sample(
                (0.0, 0.0)
            ),
            "minus_delta": _build_vmec_state_to_input_mapping_response_sample(
                (0.0, 0.0)
            ),
        },
    }

    report = mod.mapping_report_from_samples(
        case="zero",
        admitted_state_controls=_build_vmec_state_to_input_mapping_response_controls(),
        input_directions=_build_vmec_state_to_input_mapping_response_directions(),
        samples=samples,
    )

    assert report["passed"] is False
    assert report["jacobian"]["rank"] == 0
    assert report["jacobian"]["condition_number"] is None
    assert "zero_state_response" in report["blockers"]
    assert all(
        "state_control_not_observed" in row["blockers"] for row in report["controls"]
    )
    json.dumps(report, allow_nan=False)


def test_mapping_report_passes_conditioned_square_response() -> None:
    mod = load_artifact_tool("build_vmec_state_to_input_mapping_response")
    samples = {
        "RBC(1,1)": {
            "baseline": _build_vmec_state_to_input_mapping_response_sample((0.0, 0.0)),
            "plus_delta": _build_vmec_state_to_input_mapping_response_sample(
                (0.1, 0.0)
            ),
            "minus_delta": _build_vmec_state_to_input_mapping_response_sample(
                (-0.1, 0.0)
            ),
        },
        "ZBS(1,1)": {
            "baseline": _build_vmec_state_to_input_mapping_response_sample((0.0, 0.0)),
            "plus_delta": _build_vmec_state_to_input_mapping_response_sample(
                (0.0, 0.2)
            ),
            "minus_delta": _build_vmec_state_to_input_mapping_response_sample(
                (0.0, -0.2)
            ),
        },
    }

    report = mod.mapping_report_from_samples(
        case="identity",
        admitted_state_controls=_build_vmec_state_to_input_mapping_response_controls(),
        input_directions=_build_vmec_state_to_input_mapping_response_directions(),
        samples=samples,
    )

    assert report["passed"] is True
    assert report["jacobian"]["rank"] == 2
    assert report["jacobian"]["matrix"] == [[1.0, 0.0], [0.0, 1.0]]
    assert [row["passed"] for row in report["controls"]] == [True, True]


def test_mapping_report_rejects_missing_states_and_bad_deltas() -> None:
    mod = load_artifact_tool("build_vmec_state_to_input_mapping_response")
    directions = _build_vmec_state_to_input_mapping_response_directions()
    directions[0]["delta_parameter"] = float("nan")
    with pytest.raises(ValueError, match="delta_parameter"):
        mod.mapping_report_from_samples(
            case="bad",
            admitted_state_controls=_build_vmec_state_to_input_mapping_response_controls(),
            input_directions=directions,
            samples={},
        )

    with pytest.raises(ValueError, match="missing baseline"):
        mod.mapping_report_from_samples(
            case="missing",
            admitted_state_controls=_build_vmec_state_to_input_mapping_response_controls(),
            input_directions=_build_vmec_state_to_input_mapping_response_directions(),
            samples={
                "RBC(1,1)": {
                    "baseline": _build_vmec_state_to_input_mapping_response_sample(
                        (0.0, 0.0)
                    )
                }
            },
        )


# External VMEC time-horizon gate assertions
def _plot_external_vmec_time_horizon_gate_write_gate(
    tmp_path: Path, name: str, means: tuple[float, float], *, passed: bool = True
) -> Path:
    payload = {
        "kind": "external_vmec_nonlinear_grid_convergence_gate",
        "passed": passed,
        "common_window": {"max_pairwise_heat_flux_symmetric_relative_difference": 0.05},
        "least_windows": {"max_pairwise_heat_flux_symmetric_relative_difference": 0.04},
        "runs": [
            {
                "label": "n64",
                "common_window": {"heat_flux_mean": means[0]},
                "least_trending_window": {"heat_flux_mean": means[0] * 0.99},
            },
            {
                "label": "n80",
                "common_window": {"heat_flux_mean": means[1]},
                "least_trending_window": {"heat_flux_mean": means[1] * 1.01},
            },
        ],
    }
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_time_horizon_gate_passes_for_stable_high_grid_means(tmp_path: Path) -> None:
    mod = load_artifact_tool("plot_external_vmec_time_horizon_gate")
    first = _plot_external_vmec_time_horizon_gate_write_gate(
        tmp_path, "t250", (10.0, 10.4)
    )
    second = _plot_external_vmec_time_horizon_gate_write_gate(
        tmp_path, "t350", (10.2, 10.5)
    )

    paths = mod.write_time_horizon_panel(
        [(250.0, first), (350.0, second)],
        out=tmp_path / "horizon.png",
        case="synthetic time horizon",
    )

    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert payload["kind"] == "external_vmec_time_horizon_gate"
    assert payload["passed"] is True
    assert payload["promotion_gate"]["passed"] is False
    assert (
        payload["claim_level"]
        == "passed_high_grid_time_horizon_candidate_not_replicated_holdout"
    )
    assert Path(paths["png"]).exists()
    assert Path(paths["pdf"]).exists()
    assert Path(paths["csv"]).exists()


def test_time_horizon_gate_fails_large_horizon_shift(tmp_path: Path) -> None:
    mod = load_artifact_tool("plot_external_vmec_time_horizon_gate")
    first = _plot_external_vmec_time_horizon_gate_write_gate(
        tmp_path, "t250", (10.0, 10.0)
    )
    second = _plot_external_vmec_time_horizon_gate_write_gate(
        tmp_path, "t350", (14.0, 14.0)
    )

    payload = mod.build_time_horizon_payload(
        [(250.0, first), (350.0, second)],
        case="synthetic time horizon",
    )

    failed = {
        gate["metric"] for gate in payload["gate_report"]["gates"] if not gate["passed"]
    }
    assert payload["passed"] is False
    assert (
        payload["claim_level"]
        == "negative_time_horizon_result_not_transport_validation"
    )
    assert "common_window_time_horizon_relative_change" in failed


def test_time_horizon_gate_fails_when_input_grid_gate_failed(tmp_path: Path) -> None:
    mod = load_artifact_tool("plot_external_vmec_time_horizon_gate")
    first = _plot_external_vmec_time_horizon_gate_write_gate(
        tmp_path, "t250", (10.0, 10.4)
    )
    second = _plot_external_vmec_time_horizon_gate_write_gate(
        tmp_path, "t350", (10.2, 10.5), passed=False
    )

    payload = mod.build_time_horizon_payload(
        [(250.0, first), (350.0, second)],
        case="synthetic time horizon",
    )

    failed = {
        gate["metric"] for gate in payload["gate_report"]["gates"] if not gate["passed"]
    }
    assert payload["passed"] is False
    assert "failed_grid_gate_count" in failed


# VMEC-JAX equilibrium inventory assertions
def _plot_vmec_jax_equilibrium_inventory_write_wout(
    path: Path,
    *,
    nfp: int,
    ntor: int,
    aspect: float,
    iota_edge: float,
    aminor: float = 0.3,
    rmajor: float = 1.2,
    volume: float = 2.0,
) -> None:
    with Dataset(path, "w") as ds:
        ds.createDimension("radius", 3)
        ds.createVariable("nfp", "i4").assignValue(nfp)
        ds.createVariable("ns", "i4").assignValue(3)
        ds.createVariable("mpol", "i4").assignValue(4)
        ds.createVariable("ntor", "i4").assignValue(ntor)
        ds.createVariable("aspect", "f8").assignValue(aspect)
        ds.createVariable("Aminor_p", "f8").assignValue(aminor)
        ds.createVariable("Rmajor_p", "f8").assignValue(rmajor)
        ds.createVariable("volume_p", "f8").assignValue(volume)
        ds.createVariable("betatotal", "f8").assignValue(0.01)
        iota = ds.createVariable("iotaf", "f8", ("radius",))
        iota[:] = [0.4, 0.5, iota_edge]
        pres = ds.createVariable("presf", "f8", ("radius",))
        pres[:] = [1.0, 0.5, 0.0]


def test_vmec_jax_inventory_report_and_figure_are_replayable(tmp_path: Path) -> None:
    mod = load_artifact_tool("plot_vmec_jax_equilibrium_inventory")
    _plot_vmec_jax_equilibrium_inventory_write_wout(
        tmp_path / "wout_circular_tokamak.nc", nfp=1, ntor=0, aspect=3.0, iota_edge=0.25
    )
    _plot_vmec_jax_equilibrium_inventory_write_wout(
        tmp_path / "wout_nfp4_QH_warm_start.nc",
        nfp=4,
        ntor=2,
        aspect=7.0,
        iota_edge=-1.1,
    )

    report = mod.build_inventory(tmp_path)
    paths = mod.write_inventory_figure(report, out=tmp_path / "inventory.png")

    assert report["kind"] == "vmec_jax_equilibrium_inventory"
    assert report["claim_level"] == "equilibrium_selection_not_transport_validation"
    assert report["n_equilibria"] == 2
    assert report["family_counts"]["axisymmetric"] == 1
    assert "wout_nfp4_QH_warm_start.nc" in report["recommended_next_linear_portfolio"]
    assert Path(paths["png"]).exists()
    assert Path(paths["pdf"]).exists()
    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert payload["rows"][0]["validation_role"].startswith("external_vmec_fixture")


def test_vmec_jax_inventory_defers_degenerate_reference_scales(tmp_path: Path) -> None:
    mod = load_artifact_tool("plot_vmec_jax_equilibrium_inventory")
    _plot_vmec_jax_equilibrium_inventory_write_wout(
        tmp_path / "wout_nfp4_QH_warm_start.nc",
        nfp=4,
        ntor=2,
        aspect=7.0,
        iota_edge=-1.1,
    )
    _plot_vmec_jax_equilibrium_inventory_write_wout(
        tmp_path / "wout_LandremanPaul2021_QA_lowres.nc",
        nfp=2,
        ntor=8,
        aspect=0.0,
        iota_edge=0.4,
        aminor=0.0,
        rmajor=0.0,
        volume=0.0,
    )

    report = mod.build_inventory(tmp_path)
    degenerate = next(
        row
        for row in report["rows"]
        if row["name"] == "wout_LandremanPaul2021_QA_lowres.nc"
    )

    assert degenerate["reference_scale_valid"] is False
    assert (
        degenerate["geometry_contract_status"]
        == "deferred_degenerate_vmec_reference_scale"
    )
    assert degenerate["candidate_score"] == 0.0
    assert (
        "wout_LandremanPaul2021_QA_lowres.nc"
        not in report["recommended_next_linear_portfolio"]
    )
