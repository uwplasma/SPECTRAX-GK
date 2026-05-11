from __future__ import annotations

import json
from pathlib import Path

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
        "docs/_static/independent_ky_scan_scaling_large.pdf",
        "docs/_static/independent_ky_scan_scaling_cpu_large.json",
        "docs/_static/independent_ky_scan_scaling_cpu_large.csv",
        "docs/_static/independent_ky_scan_scaling_cpu_large.png",
        "docs/_static/independent_ky_scan_scaling_cpu_large.pdf",
        "docs/_static/independent_ky_scan_scaling_gpu_large.json",
        "docs/_static/independent_ky_scan_scaling_gpu_large.csv",
        "docs/_static/independent_ky_scan_scaling_gpu_large.png",
        "docs/_static/independent_ky_scan_scaling_gpu_large.pdf",
        "docs/_static/quasilinear_uq_ensemble_scaling_large.json",
        "docs/_static/quasilinear_uq_ensemble_scaling_large.csv",
        "docs/_static/quasilinear_uq_ensemble_scaling_large.png",
        "docs/_static/quasilinear_uq_ensemble_scaling_large.pdf",
        "docs/_static/quasilinear_uq_ensemble_scaling_cpu_large.json",
        "docs/_static/quasilinear_uq_ensemble_scaling_cpu_large.csv",
        "docs/_static/quasilinear_uq_ensemble_scaling_cpu_large.png",
        "docs/_static/quasilinear_uq_ensemble_scaling_cpu_large.pdf",
        "docs/_static/quasilinear_uq_ensemble_scaling_gpu_large.json",
        "docs/_static/quasilinear_uq_ensemble_scaling_gpu_large.csv",
        "docs/_static/quasilinear_uq_ensemble_scaling_gpu_large.png",
        "docs/_static/quasilinear_uq_ensemble_scaling_gpu_large.pdf",
        "docs/_static/nonlinear_sharding_strong_scaling_large.json",
        "docs/_static/nonlinear_sharding_strong_scaling_large.csv",
        "docs/_static/nonlinear_sharding_strong_scaling_large.png",
        "docs/_static/nonlinear_sharding_strong_scaling_large.pdf",
        "docs/_static/nonlinear_sharding_strong_scaling_cpu_large.json",
        "docs/_static/nonlinear_sharding_strong_scaling_cpu_large.csv",
        "docs/_static/nonlinear_sharding_strong_scaling_cpu_large.png",
        "docs/_static/nonlinear_sharding_strong_scaling_cpu_large.pdf",
        "docs/_static/nonlinear_sharding_strong_scaling_gpu_xlarge.json",
        "docs/_static/nonlinear_sharding_strong_scaling_gpu_xlarge.csv",
        "docs/_static/nonlinear_sharding_strong_scaling_gpu_xlarge.png",
        "docs/_static/nonlinear_sharding_strong_scaling_gpu_xlarge.pdf",
        "docs/_static/linear_rhs_parallel_slices_sweep.json",
        "docs/_static/linear_rhs_parallel_slices_sweep.csv",
        "docs/_static/linear_rhs_parallel_slices_sweep.png",
        "docs/_static/linear_rhs_parallel_slices_sweep.pdf",
    }

    performance = _load_toml(ROOT / "tools" / "performance_optimization_manifest.toml")
    parallel_lane = next(
        lane for lane in performance["lanes"] if lane["name"] == "parallel_scaling"
    )
    validation = _load_toml(ROOT / "tools" / "validation_coverage_manifest.toml")
    validation_paths = {
        path
        for module in validation["modules"]
        if module["module"] in {"spectraxgk.parallel", "spectraxgk.sharding"}
        for path in module["artifact_paths"]
    }

    assert required <= set(parallel_lane["artifact_paths"])
    assert required <= validation_paths
    for artifact in required:
        assert (ROOT / artifact).exists(), artifact


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
