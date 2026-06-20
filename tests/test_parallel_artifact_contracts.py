from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib  # type: ignore[no-redef]


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "docs" / "_static"


def _load_json(name: str) -> dict:
    return json.loads((STATIC / name).read_text(encoding="utf-8"))


def _load_toml(path: Path) -> dict:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _load_parallel_checker():
    path = ROOT / "tools" / "check_parallel_scaling_artifacts.py"
    spec = importlib.util.spec_from_file_location(
        "check_parallel_scaling_artifacts", path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_nonlinear_sharding_source_artifacts(
    tmp_path: Path, rows: list[dict]
) -> None:
    split_names = {
        "cpu": "nonlinear_sharding_strong_scaling_cpu_large.json",
        "gpu": "nonlinear_sharding_strong_scaling_gpu_xlarge.json",
    }
    for backend, name in split_names.items():
        backend_rows = [row for row in rows if row.get("backend") == backend]
        (tmp_path / name).write_text(
            json.dumps({"rows": backend_rows}),
            encoding="utf-8",
        )


def _assert_positive_stats(stats: dict) -> None:
    assert stats["min"] > 0.0
    assert stats["median"] > 0.0
    assert stats["mean"] > 0.0
    assert stats["max"] > 0.0
    assert stats["min"] <= stats["median"] <= stats["max"]
    assert stats["std"] >= 0.0


def _assert_worker_timing_payload(row: dict) -> None:
    assert row["error"] is None
    assert row["timed_wall_s"] > 0.0
    assert row["wall_s"] > 0.0
    assert row["strong_speedup_vs_1_device"] > 0.0
    assert row["parallel_efficiency"] > 0.0
    assert len(row["worker_stats"]) == row["actual_workers"]
    for worker in row["worker_stats"]:
        assert worker["samples_s"]
        assert all(sample > 0.0 for sample in worker["samples_s"])
        _assert_positive_stats(worker["stats_s"])


def test_parallel_manifests_track_current_cpu_gpu_scaling_artifacts() -> None:
    required = {
        "docs/_static/independent_ky_scan_scaling_large.json",
        "docs/_static/independent_ky_scan_scaling_large.csv",
        "docs/_static/independent_ky_scan_scaling_large.png",
        "docs/_static/independent_ky_scan_scaling_cpu_large.json",
        "docs/_static/independent_ky_scan_scaling_cpu_large.csv",
        "docs/_static/independent_ky_scan_scaling_cpu_large.png",
        "docs/_static/independent_ky_scan_scaling_gpu_large.json",
        "docs/_static/independent_ky_scan_scaling_gpu_large.csv",
        "docs/_static/independent_ky_scan_scaling_gpu_large.png",
        "docs/_static/quasilinear_uq_ensemble_scaling_large.json",
        "docs/_static/quasilinear_uq_ensemble_scaling_large.csv",
        "docs/_static/quasilinear_uq_ensemble_scaling_large.png",
        "docs/_static/quasilinear_uq_ensemble_scaling_cpu_large.json",
        "docs/_static/quasilinear_uq_ensemble_scaling_cpu_large.csv",
        "docs/_static/quasilinear_uq_ensemble_scaling_cpu_large.png",
        "docs/_static/quasilinear_uq_ensemble_scaling_gpu_large.json",
        "docs/_static/quasilinear_uq_ensemble_scaling_gpu_large.csv",
        "docs/_static/quasilinear_uq_ensemble_scaling_gpu_large.png",
        "docs/_static/parallelization_completion_status.json",
        "docs/_static/parallelization_completion_status.png",
        "docs/_static/nonlinear_sharding_strong_scaling_large.json",
        "docs/_static/nonlinear_sharding_strong_scaling_large.csv",
        "docs/_static/nonlinear_sharding_strong_scaling_large.png",
        "docs/_static/nonlinear_sharding_strong_scaling_cpu_large.json",
        "docs/_static/nonlinear_sharding_strong_scaling_cpu_large.csv",
        "docs/_static/nonlinear_sharding_strong_scaling_cpu_large.png",
        "docs/_static/nonlinear_sharding_strong_scaling_gpu_xlarge.json",
        "docs/_static/nonlinear_sharding_strong_scaling_gpu_xlarge.csv",
        "docs/_static/nonlinear_sharding_strong_scaling_gpu_xlarge.png",
        "docs/_static/nonlinear_sharding_production_speedup_gate.json",
        "docs/_static/nonlinear_sharding_production_speedup_gate.csv",
        "docs/_static/nonlinear_device_z_pencil_transport_gpu2_observable_split_profile.json",
        "docs/_static/nonlinear_device_z_pencil_transport_gpu2_observable_split_profile.csv",
        "docs/_static/nonlinear_device_z_pencil_transport_gpu2_observable_split_profile.png",
        "docs/_static/linear_rhs_parallel_slices_sweep.json",
        "docs/_static/linear_rhs_parallel_slices_sweep.csv",
        "docs/_static/linear_rhs_parallel_slices_sweep.png",
    }

    performance = _load_toml(ROOT / "tools" / "performance_optimization_manifest.toml")
    parallel_lane = next(
        lane for lane in performance["lanes"] if lane["name"] == "parallel_scaling"
    )
    validation = _load_toml(ROOT / "tools" / "validation_coverage_manifest.toml")
    validation_paths = {
        path
        for module in validation["modules"]
        if module["module"]
        in {"spectraxgk.parallel.__init__", "spectraxgk.parallel.state"}
        for path in module["artifact_paths"]
    }

    assert required <= set(parallel_lane["artifact_paths"])
    assert required <= validation_paths
    for artifact in required:
        assert (ROOT / artifact).exists(), artifact


def test_parallelization_completion_status_scopes_production_and_diagnostic_lanes() -> (
    None
):
    payload = _load_json("parallelization_completion_status.json")

    assert payload["kind"] == "parallelization_completion_status"
    assert payload["passed"] is True
    assert payload["production_completion_percent"] == 100.0
    assert "Release production parallelization is closed" in payload["claim_scope"]
    lanes = {lane["lane"]: lane for lane in payload["lanes"]}
    assert lanes["independent_ky_scan"]["status"] == "production_closed"
    assert lanes["quasilinear_uq_ensemble"]["status"] == "production_closed"
    assert (
        lanes["independent_ky_scan"]["source_contract"]["claim_separation_passed"]
        is True
    )
    assert lanes["independent_ky_scan"]["source_contract"]["input_backends"] == [
        "cpu",
        "gpu",
    ]
    assert (
        lanes["quasilinear_uq_ensemble"]["source_contract"]["claim_separation_passed"]
        is True
    )
    assert lanes["independent_ky_scan"]["best_speedups"]["cpu"] >= 5.0
    assert lanes["independent_ky_scan"]["best_speedups"]["gpu"] >= 1.5
    assert (
        lanes["whole_state_nonlinear_sharding"]["status"]
        == "diagnostic_closed_not_production"
    )
    assert (
        lanes["whole_state_nonlinear_sharding"]["source_contract"][
            "claim_separation_passed"
        ]
        is True
    )
    assert lanes["fft_axis_domain"]["status"] == "diagnostic_identity_closed"


def test_nonlinear_domain_parallel_identity_gate_is_scoped_and_fail_closed() -> None:
    payload = _load_json("nonlinear_domain_parallel_identity_gate.json")

    assert payload["case"] == "Nonlinear state-domain decomposition identity gate"
    assert payload["gate"]["identity_passed"] is True
    assert payload["gate"]["decomposed_path_enabled"] is True
    assert payload["gated_state_matches_serial"] is True
    assert payload["gated_state_matches_decomposed"] is True
    assert payload["gate"]["max_abs_error"] <= payload["gate"]["atol"]
    assert payload["gate"]["max_rel_error"] <= payload["gate"]["rtol"]
    assert payload["transport_window"]["gate"]["identity_passed"] is True
    assert payload["transport_window"]["gate"]["decomposed_path_enabled"] is True
    assert (
        payload["transport_window"]["gate"]["max_abs_state_error"]
        <= payload["gate"]["atol"]
    )
    assert (
        payload["transport_window"]["gate"]["mass_trace_max_abs_error"]
        <= payload["gate"]["atol"]
    )
    assert (
        payload["transport_window"]["gate"]["free_energy_trace_max_abs_error"]
        <= payload["gate"]["atol"]
    )
    assert (
        payload["transport_window"]["gate"]["flux_proxy_trace_max_abs_error"]
        <= payload["gate"]["atol"]
    )
    assert {row["metric"] for row in payload["transport_window"]["metrics"]} == {
        "mass_trace",
        "free_energy_trace",
        "boundary_flux_proxy_trace",
    }
    assert all(
        row["identity_passed"] is True for row in payload["transport_window"]["metrics"]
    )
    assert "no production routing or speedup claim" in payload["claim_scope"]
    assert (STATIC / "nonlinear_domain_parallel_identity_gate.png").exists()


def test_nonlinear_spectral_communication_identity_gate_is_scoped_and_fail_closed() -> (
    None
):
    payload = _load_json("nonlinear_spectral_communication_identity_gate.json")

    assert payload["case"] == "Nonlinear spectral decomposition identity gate"
    assert payload["kind"] == "nonlinear_spectral_communication_identity_gate"
    assert payload["gate"]["identity_passed"] is True
    assert payload["gate"]["decomposed_path_enabled"] is True
    assert payload["gate"]["communication_identity_passed"] is True
    assert payload["gate"]["rhs_identity_passed"] is True
    assert payload["gate"]["integrator_identity_passed"] is True
    assert payload["gate"]["pencil_rhs_identity_passed"] is True
    assert payload["gate"]["pencil_transport_window_identity_passed"] is True
    assert payload["communication_gate"]["fft_max_abs_error"] <= payload["gate"]["atol"]
    assert (
        payload["communication_gate"]["bracket_max_abs_error"]
        <= payload["gate"]["atol"]
    )
    assert (
        payload["communication_gate"]["field_max_abs_error"] <= payload["gate"]["atol"]
    )
    assert payload["rhs_gate"]["rhs_max_abs_error"] <= payload["gate"]["atol"]
    assert (
        payload["integrator_gate"]["final_state_max_abs_error"]
        <= payload["gate"]["atol"]
    )
    assert (
        payload["integrator_gate"]["flux_proxy_trace_max_abs_error"]
        <= payload["gate"]["atol"]
    )
    assert payload["pencil_rhs_gate"]["rhs_max_abs_error"] <= payload["gate"]["atol"]
    assert (
        payload["pencil_transport_window_gate"]["final_state_max_abs_error"]
        <= payload["gate"]["atol"]
    )
    assert all(row["identity_passed"] is True for row in payload["rows"])
    assert {row["operator"] for row in payload["rows"]} == {
        "fft_forward_inverse",
        "nonlinear_bracket",
        "spectral_field_solve_layout",
        "logical_sharded_rhs",
        "logical_integrator_final_state",
        "logical_integrator_flux_proxy_trace",
        "pencil_fused_rhs",
        "pencil_physical_transport_window",
    }
    assert "pencil fused-bracket" in payload["claim_scope"]
    assert "physical transport-window identity gate" in payload["claim_scope"]
    assert (
        "no production distributed FFT routing or speedup claim"
        in payload["claim_scope"]
    )
    assert (STATIC / "nonlinear_spectral_communication_identity_gate.png").exists()


def test_parallel_scaling_artifact_checker_validates_tracked_large_run_evidence() -> (
    None
):
    mod = _load_parallel_checker()

    summary = mod.validate_all()

    assert summary["n_families"] == 4
    assert summary["n_json_artifacts"] == 12
    assert summary["n_sidecars"] == 35
    assert summary["manifest_checked"] is True
    assert {family["name"] for family in summary["families"]} == {
        "independent_ky_scan",
        "quasilinear_uq_ensemble",
        "nonlinear_sharding",
        "linear_rhs_parallel_slices",
    }
    assert (
        summary["production_gate"]["name"]
        == "nonlinear_sharding_production_speedup_gate"
    )
    assert summary["production_gate"]["gate_passed"] is False
    assert summary["production_gate"]["status"] == "diagnostic_only"
    assert summary["production_gate"]["production_candidate_backends"] == ["cpu"]
    assert summary["observable_split"]["name"] == "device_z_pencil_observable_split"
    assert summary["observable_split"]["production_speedup_claim_allowed"] is False
    assert summary["observable_split"]["max_observable_gate_overhead_vs_compute"] > 1.0


def test_parallel_scaling_artifact_checker_validates_observable_split() -> None:
    mod = _load_parallel_checker()

    summary = mod.validate_device_z_pencil_observable_split(STATIC)

    assert summary["json"] == mod.OBSERVABLE_SPLIT_JSON
    assert summary["production_speedup_claim_allowed"] is False
    assert summary["max_speedup_vs_serial"] < summary["min_speedup"]
    assert summary["max_observable_gate_overhead_vs_compute"] > 10.0


def test_parallel_scaling_artifact_checker_rejects_promoted_observable_split(
    tmp_path: Path,
) -> None:
    mod = _load_parallel_checker()
    payload = _load_json(mod.OBSERVABLE_SPLIT_JSON)
    payload["summary"]["max_speedup_vs_serial"] = payload["min_speedup"]
    (tmp_path / mod.OBSERVABLE_SPLIT_JSON).write_text(
        json.dumps(payload), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="no longer below gate"):
        mod.validate_device_z_pencil_observable_split(tmp_path, check_sidecars=False)


def test_parallel_scaling_artifact_checker_rejects_failed_identity_gate(
    tmp_path: Path,
) -> None:
    mod = _load_parallel_checker()
    family = mod.ArtifactFamily(
        name="bad_identity",
        combined="bad_identity.json",
        split=(),
        expected_combined_kind="bad_identity",
        expected_split_kind=None,
        identity_claim_phrase="identity-only",
        split_identity_claim_phrase=None,
        timing_fields=("serial_median_s",),
        error_fields=("max_abs_error",),
        row_identity_key="identity_passed",
        combined_has_inputs=False,
    )
    (tmp_path / "bad_identity.json").write_text(
        json.dumps(
            {
                "kind": "bad_identity",
                "identity_passed": False,
                "claim_scope": "identity-only local test",
                "rows": [
                    {
                        "requested_devices": 1,
                        "identity_passed": True,
                        "serial_median_s": 1.0,
                        "max_abs_error": 0.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="identity_passed must be true"):
        mod.validate_family(tmp_path, family, check_sidecars=False)


def test_parallel_scaling_artifact_checker_rejects_tiny_problem_metadata(
    tmp_path: Path,
) -> None:
    mod = _load_parallel_checker()
    family = mod.ArtifactFamily(
        name="tiny",
        combined="tiny.json",
        split=(),
        expected_combined_kind="tiny_scaling",
        expected_split_kind=None,
        identity_claim_phrase="identity-only",
        split_identity_claim_phrase=None,
        timing_fields=("serial_median_s",),
        error_fields=("max_abs_error",),
        row_identity_key="identity_passed",
        combined_has_inputs=False,
        min_grid=(("Ny", 64), ("Nz", 32)),
        min_steps=100,
    )
    (tmp_path / "tiny.json").write_text(
        json.dumps(
            {
                "kind": "tiny_scaling",
                "identity_passed": True,
                "claim_scope": "identity-only local test",
                "grid": {"Ny": 16, "Nz": 32},
                "time": {"steps": 100},
                "rows": [
                    {
                        "requested_devices": 1,
                        "identity_passed": True,
                        "serial_median_s": 1.0,
                        "max_abs_error": 0.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "tiny.csv").write_text(
        "requested_devices,identity_passed,serial_median_s,max_abs_error\n"
        "1,true,1.0,0.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="grid Ny=16 is below required 64"):
        mod.validate_family(tmp_path, family, check_sidecars=False)


def test_parallel_scaling_artifact_checker_accepts_profile_source_contract(
    tmp_path: Path,
) -> None:
    mod = _load_parallel_checker()
    family = mod.ArtifactFamily(
        name="profile_contract",
        combined="profile_contract.json",
        split=(),
        expected_combined_kind="profile_contract",
        expected_split_kind=None,
        identity_claim_phrase="identity-only",
        split_identity_claim_phrase=None,
        timing_fields=("serial_median_s",),
        error_fields=("max_abs_error",),
        row_identity_key="identity_passed",
        combined_has_inputs=False,
    )
    row = {
        "requested_devices": 1,
        "actual_devices": 1,
        "backend": "gpu",
        "identity_passed": True,
        "serial_median_s": 1.0,
        "max_abs_error": 0.0,
        "source_contract_version": 1,
        "profile_command": "python tools/profile_nonlinear_sharding.py --sharding kx",
        "profile_command_argv": [
            "python",
            "tools/profile_nonlinear_sharding.py",
            "--sharding",
            "kx",
        ],
        "source_artifact": "docs/_static/profile.json",
        "software_versions": {
            "python": "3.11.0",
            "spectraxgk": "test",
            "jax": "0.test",
            "jaxlib": "0.test",
            "numpy": "2.test",
        },
        "timing_warmup_repeat": {"warmups": 0, "repeats": 2},
        "profile_backend": "gpu",
        "profile_device_count": 1,
        "profile_sharding_axis": "kx",
    }
    (tmp_path / "profile_contract.json").write_text(
        json.dumps(
            {
                "kind": "profile_contract",
                "identity_passed": True,
                "claim_scope": "identity-only local test",
                "rows": [row],
            }
        ),
        encoding="utf-8",
    )

    summary = mod.validate_family(tmp_path, family, check_sidecars=False)

    assert summary["n_combined_rows"] == 1


def test_parallel_scaling_artifact_checker_rejects_stale_profile_source_contract(
    tmp_path: Path,
) -> None:
    mod = _load_parallel_checker()
    family = mod.ArtifactFamily(
        name="profile_contract",
        combined="profile_contract.json",
        split=(),
        expected_combined_kind="profile_contract",
        expected_split_kind=None,
        identity_claim_phrase="identity-only",
        split_identity_claim_phrase=None,
        timing_fields=("serial_median_s",),
        error_fields=("max_abs_error",),
        row_identity_key="identity_passed",
        combined_has_inputs=False,
    )
    row = {
        "requested_devices": 1,
        "actual_devices": 1,
        "backend": "cpu",
        "identity_passed": True,
        "serial_median_s": 1.0,
        "max_abs_error": 0.0,
        "source_contract_version": 1,
        "profile_command": "python tools/profile_nonlinear_sharding.py --sharding kx",
        "profile_command_argv": [
            "python",
            "tools/profile_nonlinear_sharding.py",
            "--sharding",
            "kx",
        ],
        "source_artifact": "docs/_static/profile.json",
        "software_versions": {
            "python": "3.11.0",
            "spectraxgk": "test",
            "jax": "0.test",
            "jaxlib": "0.test",
            "numpy": "2.test",
        },
        "timing_warmup_repeat": {"warmups": 0, "repeats": 2},
        "profile_backend": "gpu",
        "profile_device_count": 1,
        "profile_sharding_axis": "kx",
    }
    (tmp_path / "profile_contract.json").write_text(
        json.dumps(
            {
                "kind": "profile_contract",
                "identity_passed": True,
                "claim_scope": "identity-only local test",
                "rows": [row],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="profile_backend must match row backend"):
        mod.validate_family(tmp_path, family, check_sidecars=False)


def test_parallel_scaling_artifact_checker_rejects_stale_production_gate(
    tmp_path: Path,
) -> None:
    mod = _load_parallel_checker()
    payload = {
        "kind": "nonlinear_sharding_production_speedup_gate",
        "claim_scope": (
            "Whole-state nonlinear sharding gate; otherwise keep it as a "
            "diagnostic identity/profiler artifact."
        ),
        "gate_passed": True,
        "production_speedup_claim_allowed": True,
        "status": "production_speedup_candidate",
        "required_backends": ["cpu", "gpu"],
        "min_devices": 2,
        "min_speedup_vs_1_device": 1.2,
        "min_parallel_efficiency": 0.5,
        "identity_atol": 1.0e-5,
        "identity_rtol": 1.0e-5,
        "best_candidates": {
            "cpu": {
                "backend": "cpu",
                "requested_devices": 2,
                "actual_devices": 2,
                "source": "docs/_static/nonlinear_sharding_strong_scaling_cpu_large.json",
                "strong_speedup_vs_1_device": 1.3,
            },
            "gpu": None,
        },
        "blockers": [],
        "rows": [
            {
                "backend": "cpu",
                "requested_devices": 2,
                "actual_devices": 2,
                "source": "docs/_static/nonlinear_sharding_strong_scaling_cpu_large.json",
                "state_sharding_active": True,
                "identity_gate_pass": True,
                "strong_speedup_vs_1_device": 1.3,
                "parallel_efficiency": 0.65,
                "max_abs_state_error": 0.0,
                "max_rel_state_error": 0.0,
                "candidate_passed": True,
                "classification": "production_candidate",
                "blockers": [],
            },
            {
                "backend": "gpu",
                "requested_devices": 2,
                "actual_devices": 2,
                "source": "docs/_static/nonlinear_sharding_strong_scaling_gpu_xlarge.json",
                "state_sharding_active": True,
                "identity_gate_pass": True,
                "strong_speedup_vs_1_device": 0.8,
                "parallel_efficiency": 0.4,
                "max_abs_state_error": 0.0,
                "max_rel_state_error": 0.0,
                "candidate_passed": False,
                "classification": "identity_preserving_regression",
                "blockers": [
                    "speedup_below_threshold",
                    "parallel_efficiency_below_threshold",
                ],
            },
        ],
    }
    _write_nonlinear_sharding_source_artifacts(tmp_path, payload["rows"])
    (tmp_path / mod.PRODUCTION_GATE_JSON).write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError, match="blockers do not match missing backend candidates"
    ):
        mod.validate_nonlinear_sharding_production_gate(tmp_path, check_sidecars=False)


def test_parallel_scaling_artifact_checker_rejects_stale_production_gate_source_row(
    tmp_path: Path,
) -> None:
    mod = _load_parallel_checker()
    source_row = {
        "backend": "cpu",
        "requested_devices": 2,
        "actual_devices": 2,
        "best_spec": "kx",
        "state_sharding_active": True,
        "identity_gate_pass": True,
        "strong_speedup_vs_1_device": 1.30,
        "max_abs_state_error": 0.0,
        "max_rel_state_error": 0.0,
    }
    gate_row = {
        **source_row,
        "source": "docs/_static/nonlinear_sharding_strong_scaling_cpu_large.json",
        "strong_speedup_vs_1_device": 1.40,
        "parallel_efficiency": 0.70,
        "candidate_passed": True,
        "classification": "production_candidate",
        "blockers": [],
    }
    _write_nonlinear_sharding_source_artifacts(
        tmp_path,
        [
            source_row,
            {
                "backend": "gpu",
                "requested_devices": 1,
                "actual_devices": 1,
                "best_spec": "auto",
                "state_sharding_active": False,
                "identity_gate_pass": True,
                "strong_speedup_vs_1_device": 1.0,
                "max_abs_state_error": 0.0,
                "max_rel_state_error": 0.0,
            },
        ],
    )
    payload = {
        "kind": "nonlinear_sharding_production_speedup_gate",
        "claim_scope": (
            "Whole-state nonlinear sharding gate; otherwise keep it as a "
            "diagnostic identity/profiler artifact."
        ),
        "gate_passed": True,
        "production_speedup_claim_allowed": True,
        "status": "production_speedup_candidate",
        "required_backends": ["cpu"],
        "min_devices": 2,
        "min_speedup_vs_1_device": 1.2,
        "min_parallel_efficiency": 0.5,
        "identity_atol": 1.0e-5,
        "identity_rtol": 1.0e-5,
        "best_candidates": {"cpu": gate_row},
        "blockers": [],
        "rows": [gate_row],
    }
    (tmp_path / mod.PRODUCTION_GATE_JSON).write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="stale relative to source artifact"):
        mod.validate_nonlinear_sharding_production_gate(tmp_path, check_sidecars=False)


def test_parallel_scaling_artifact_checker_rejects_stale_production_gate_blocker_report(
    tmp_path: Path,
) -> None:
    mod = _load_parallel_checker()
    rows = [
        {
            "backend": "cpu",
            "requested_devices": 2,
            "actual_devices": 2,
            "best_spec": "kx",
            "state_sharding_active": True,
            "identity_gate_pass": True,
            "strong_speedup_vs_1_device": 1.30,
            "parallel_efficiency": 0.65,
            "max_abs_state_error": 0.0,
            "max_rel_state_error": 0.0,
            "source": "docs/_static/nonlinear_sharding_strong_scaling_cpu_large.json",
            "candidate_passed": True,
            "classification": "production_candidate",
            "blockers": [],
        },
        {
            "backend": "gpu",
            "requested_devices": 2,
            "actual_devices": 2,
            "best_spec": "kx",
            "state_sharding_active": True,
            "identity_gate_pass": True,
            "strong_speedup_vs_1_device": 0.80,
            "parallel_efficiency": 0.40,
            "max_abs_state_error": 0.0,
            "max_rel_state_error": 0.0,
            "source": "docs/_static/nonlinear_sharding_strong_scaling_gpu_xlarge.json",
            "candidate_passed": False,
            "classification": "identity_preserving_regression",
            "blockers": [
                "speedup_below_threshold",
                "parallel_efficiency_below_threshold",
            ],
        },
    ]
    _write_nonlinear_sharding_source_artifacts(tmp_path, rows)
    payload = {
        "kind": "nonlinear_sharding_production_speedup_gate",
        "claim_scope": (
            "Whole-state nonlinear sharding gate; otherwise keep it as a "
            "diagnostic identity/profiler artifact."
        ),
        "gate_passed": False,
        "production_speedup_claim_allowed": False,
        "status": "diagnostic_only",
        "required_backends": ["cpu", "gpu"],
        "min_devices": 2,
        "min_speedup_vs_1_device": 1.2,
        "min_parallel_efficiency": 0.5,
        "identity_atol": 1.0e-5,
        "identity_rtol": 1.0e-5,
        "best_candidates": {"cpu": rows[0], "gpu": None},
        "blockers": ["gpu_production_speedup_candidate_missing"],
        "backend_blocker_report": {
            "cpu": {
                "row_count": 1,
                "candidate_row_count": 1,
                "passing_candidate_count": 1,
                "production_speedup_candidate_missing": False,
                "identity_evidence_complete": True,
                "active_identity_evidence_complete": True,
                "classification_counts": {"production_candidate": 1},
                "candidate_blocker_counts": {},
                "primary_blockers": [],
                "claim_scope": "stale",
            },
            "gpu": {
                "row_count": 1,
                "candidate_row_count": 1,
                "passing_candidate_count": 1,
                "production_speedup_candidate_missing": False,
                "identity_evidence_complete": True,
                "active_identity_evidence_complete": True,
                "classification_counts": {"identity_preserving_regression": 1},
                "candidate_blocker_counts": {},
                "primary_blockers": [],
                "claim_scope": "stale",
            },
        },
        "rows": rows,
    }
    (tmp_path / mod.PRODUCTION_GATE_JSON).write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="backend_blocker_report"):
        mod.validate_nonlinear_sharding_production_gate(tmp_path, check_sidecars=False)


def test_independent_ky_scaling_artifact_preserves_order_and_identity_scope() -> None:
    payload = _load_json("independent_ky_scan_scaling_large.json")

    assert payload["kind"] == "independent_ky_scan_scaling_combined"
    assert payload["identity_passed"] is True
    assert "not a nonlinear domain-decomposition" in payload["claim_scope"]
    for backend in {"cpu", "gpu"}:
        rows = sorted(
            (row for row in payload["rows"] if row["backend"] == backend),
            key=lambda row: row["requested_devices"],
        )
        assert rows
        reference_ky = rows[0]["ky"]
        reference_gamma = rows[0]["gamma"]
        reference_omega = rows[0]["omega"]
        for row in rows:
            assert row["identity_gate_pass"] is True
            assert row["actual_workers"] <= row["requested_devices"]
            assert row["ky"] == reference_ky
            assert row["gamma"] == reference_gamma
            assert row["omega"] == reference_omega
            assert row["max_gamma_rel_error"] == 0.0
            assert row["max_omega_abs_error"] == 0.0


def test_independent_ky_split_artifacts_are_large_solver_backed_identity_profiles() -> (
    None
):
    for backend, artifact in {
        "cpu": "independent_ky_scan_scaling_cpu_large.json",
        "gpu": "independent_ky_scan_scaling_gpu_large.json",
    }.items():
        payload = _load_json(artifact)

        assert payload["kind"] == "independent_ky_scan_strong_scaling"
        assert payload["backend"] == backend
        assert payload["identity_passed"] is True
        assert "not a nonlinear domain-decomposition" in payload["claim_scope"]
        assert len(payload["ky"]) >= 8
        assert payload["warmups"] >= 1
        assert payload["repeats"] >= 1
        assert payload["time"]["steps"] >= 200
        assert payload["grid"]["Ny"] >= 96
        assert payload["grid"]["Nz"] >= 64
        assert payload["grid"]["Nl"] >= 4
        assert payload["grid"]["Nm"] >= 8

        rows = sorted(payload["rows"], key=lambda row: row["requested_devices"])
        assert rows[0]["requested_devices"] == 1
        assert rows[-1]["requested_devices"] > 1
        reference = rows[0]
        for row in rows:
            assert row["identity_gate_pass"] is True
            assert row["actual_workers"] <= row["requested_devices"]
            assert row["ky"] == reference["ky"]
            assert row["gamma"] == reference["gamma"]
            assert row["omega"] == reference["omega"]
            assert row["max_gamma_abs_error"] == 0.0
            assert row["max_gamma_rel_error"] == 0.0
            assert row["max_omega_abs_error"] == 0.0
            _assert_worker_timing_payload(row)


def test_quasilinear_uq_scaling_artifact_preserves_member_order_and_identity_scope() -> (
    None
):
    payload = _load_json("quasilinear_uq_ensemble_scaling_large.json")

    assert payload["kind"] == "quasilinear_uq_ensemble_scaling_combined"
    assert payload["identity_passed"] is True
    assert (
        "not a promoted absolute nonlinear heat-flux predictor"
        in payload["claim_scope"]
    )
    for backend in {"cpu", "gpu"}:
        rows = sorted(
            (row for row in payload["rows"] if row["backend"] == backend),
            key=lambda row: row["requested_devices"],
        )
        assert rows
        reference_gradients = [member["R_over_LTi"] for member in rows[0]["members"]]
        reference_flux = [member["heat_flux_proxy"] for member in rows[0]["members"]]
        for row in rows:
            gradients = [member["R_over_LTi"] for member in row["members"]]
            flux = [member["heat_flux_proxy"] for member in row["members"]]
            assert row["identity_gate_pass"] is True
            assert row["actual_workers"] <= row["requested_devices"]
            assert gradients == reference_gradients
            assert flux == reference_flux
            assert row["max_heat_flux_proxy_rel_error"] == 0.0
            assert row["max_gamma_abs_error"] == 0.0


def test_quasilinear_uq_split_artifacts_are_large_solver_backed_identity_profiles() -> (
    None
):
    for backend, artifact in {
        "cpu": "quasilinear_uq_ensemble_scaling_cpu_large.json",
        "gpu": "quasilinear_uq_ensemble_scaling_gpu_large.json",
    }.items():
        payload = _load_json(artifact)

        assert payload["kind"] == "quasilinear_uq_ensemble_scaling"
        assert payload["backend"] == backend
        assert payload["identity_passed"] is True
        assert (
            "not an absolute nonlinear heat-flux validation claim"
            in payload["claim_scope"]
        )
        assert len(payload["gradients"]) >= 6
        assert len(payload["ky"]) >= 5
        assert payload["warmups"] >= 1
        assert payload["repeats"] >= 1
        assert payload["time"]["steps"] >= 1000
        assert payload["grid"]["Ny"] >= 64
        assert payload["grid"]["Nz"] >= 64
        assert payload["grid"]["Nl"] >= 3
        assert payload["grid"]["Nm"] >= 6

        rows = sorted(payload["rows"], key=lambda row: row["requested_devices"])
        assert rows[0]["requested_devices"] == 1
        assert rows[-1]["requested_devices"] > 1
        reference_gradients = [member["R_over_LTi"] for member in rows[0]["members"]]
        reference_flux = [member["heat_flux_proxy"] for member in rows[0]["members"]]
        for row in rows:
            gradients = [member["R_over_LTi"] for member in row["members"]]
            flux = [member["heat_flux_proxy"] for member in row["members"]]
            assert row["identity_gate_pass"] is True
            assert row["actual_workers"] <= row["requested_devices"]
            assert gradients == reference_gradients
            assert flux == reference_flux
            assert row["max_heat_flux_proxy_abs_error"] == 0.0
            assert row["max_heat_flux_proxy_rel_error"] == 0.0
            assert row["max_gamma_abs_error"] == 0.0
            _assert_worker_timing_payload(row)


def test_nonlinear_whole_state_scaling_artifact_is_identity_only_not_speedup_claim() -> (
    None
):
    payload = _load_json("nonlinear_sharding_strong_scaling_large.json")

    assert payload["kind"] == "nonlinear_sharding_strong_scaling_combined"
    assert payload["identity_passed"] is True
    assert payload["speedup_passed"] is False
    assert payload["status"] == "diagnostic_identity_only"
    assert payload["speedup_blockers"]
    assert "not a production speedup claim" in payload["claim_scope"]
    assert {row["backend"] for row in payload["rows"]} == {"cpu", "gpu"}
    for row in payload["rows"]:
        assert row["identity_gate_pass"] is True
        assert row["max_abs_state_error"] == 0.0
        assert row["max_rel_state_error"] == 0.0
        assert row["strong_speedup_vs_1_device"] > 0.0
        assert row["parallel_median_s"] > 0.0
        assert row["best_spec"] in {"auto", "ky", "kx"}
        if row["actual_devices"] < 2:
            assert row["state_sharding_active"] is False


def test_nonlinear_strong_scaling_split_artifacts_embed_profiler_payloads() -> None:
    artifacts = {
        "cpu": (
            "nonlinear_sharding_strong_scaling_cpu_large.json",
            {"Nx": 24, "Ny_requested": 48, "Nz": 96, "Nl": 4, "Nm": 8},
        ),
        "gpu": (
            "nonlinear_sharding_strong_scaling_gpu_xlarge.json",
            {"Nx": 48, "Ny_requested": 96, "Nz": 128, "Nl": 4, "Nm": 8},
        ),
    }

    for backend, (artifact, expected_grid) in artifacts.items():
        payload = _load_json(artifact)

        assert payload["kind"] == "nonlinear_sharding_strong_scaling_sweep"
        assert payload["backend"] == backend
        assert payload["grid"] == expected_grid
        assert payload["steps"] >= 8
        assert payload["identity_passed"] is True
        assert "not as a broad production speedup claim" in payload["claim_scope"]
        assert set(payload["profiles"]) == {
            str(row["requested_devices"]) for row in payload["rows"]
        }

        for row in payload["rows"]:
            assert row["error"] is None
            assert row["identity_gate_pass"] is True
            assert row["actual_devices"] <= row["requested_devices"]
            assert row["max_abs_state_error"] == 0.0
            assert row["max_rel_state_error"] == 0.0
            assert row["parallel_median_s"] > 0.0
            assert row["serial_median_s"] > 0.0
            assert row["same_process_speedup"] > 0.0
            assert row["strong_speedup_vs_1_device"] > 0.0
            assert row["best_spec"] in {"auto", "ky", "kx"}
            if row["actual_devices"] >= 2:
                assert row["state_sharding_active"] is True

            profile = payload["profiles"][str(row["requested_devices"])]
            assert profile["_profile_json"] == row["profile_json"]
            assert profile["default_backend"] == backend
            assert profile["device_count"] == row["actual_devices"]
            assert profile["state_shape"] == row["state_shape"]
            assert profile["identity_gate_pass"] is True
            assert "Do not use as a published runtime claim" in profile["claim_scope"]
            assert set(profile["profiler_trace"]) >= {"requested", "path", "error"}
            assert profile["profiler_trace"]["error"] is None
            _assert_positive_stats(profile["serial_stats_s"])

            best = profile["best_identity_preserving_candidate"]
            assert best["spec"] == row["best_spec"]
            assert best["identity_gate_pass"] is True
            result = profile["sharded_results"][row["best_spec"]]
            assert result["identity_gate_pass"] is True
            assert result["max_abs_state_error"] == row["max_abs_state_error"]
            assert result["max_rel_state_error"] == row["max_rel_state_error"]
            _assert_positive_stats(result["stats_s"])


def test_parallel_docs_keep_speedup_claims_tied_to_current_artifacts() -> None:
    docs = "\n".join(
        (ROOT / path).read_text(encoding="utf-8")
        for path in (
            "docs/parallelization.rst",
            "docs/performance.rst",
            "docs/testing.rst",
        )
    )

    for artifact in (
        "independent_ky_scan_scaling_large.json",
        "quasilinear_uq_ensemble_scaling_large.json",
        "nonlinear_sharding_strong_scaling_large.json",
        "linear_rhs_parallel_slices_sweep.json",
    ):
        assert artifact in docs
    compact_docs = " ".join(docs.split())
    assert "Large-run scaling acceptance checklist" in compact_docs
    assert "fresh profiler artifacts for the exact workload" in compact_docs
    assert "not a production nonlinear speedup claim" in compact_docs
