from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "build_open_research_lane_status.py"
spec = importlib.util.spec_from_file_location("build_open_research_lane_status", SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def _write_json(root: Path, relative: str, payload: dict[str, object]) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_status_payload_keeps_open_lanes_scoped(tmp_path: Path) -> None:
    _write_json(
        tmp_path,
        "docs/_static/w7x_zonal_reference_compare.json",
        {
            "validation_status": "open",
            "gate_report": {
                "gates": [
                    {"metric": "time_coverage", "passed": True},
                    {"metric": "residual_kx070", "passed": False},
                ]
            },
        },
    )
    _write_json(
        tmp_path,
        "docs/_static/w7x_zonal_recurrence_sweep_kx070.json",
        {
            "rows": [
                {
                    "label": "low",
                    "mean_abs_error": 0.4,
                    "tail_std": 0.2,
                    "reference_tail_std": 0.05,
                },
                {
                    "label": "best",
                    "mean_abs_error": 0.1,
                    "tail_std": 0.1,
                    "reference_tail_std": 0.05,
                    "hermite_tail_at_tmax": 0.2,
                },
            ]
        },
    )
    _write_json(
        tmp_path,
        "docs/_static/w7x_zonal_hypercollision_probe_kx070.json",
        {
            "validation_status": "open",
            "rows": [
                {
                    "label": "const nuhm0.01",
                    "mean_abs_error": 0.3,
                    "tail_std": 0.12,
                    "reference_tail_std": 0.03,
                    "hermite_tail_at_tmax": 0.22,
                    "free_energy_at_tmax_over_initial": 0.75,
                },
                {
                    "label": "const nuhm0.03",
                    "mean_abs_error": 0.28,
                    "tail_std": 0.13,
                    "reference_tail_std": 0.03,
                    "hermite_tail_at_tmax": 0.10,
                    "free_energy_at_tmax_over_initial": 0.60,
                },
            ],
        },
    )
    _write_json(
        tmp_path,
        "docs/_static/w7x_zonal_mixedlm_resolution_kx070.json",
        {
            "rows": [
                {
                    "label": "Nl16 Nm64 mixedLM",
                    "mean_abs_error": 0.27,
                    "tail_std": 0.12,
                    "reference_tail_std": 0.03,
                    "hermite_tail_at_tmax": 1.5e-4,
                    "Nl": 16,
                    "Nm": 64,
                },
                {
                    "label": "Nl24 Nm96 mixedLM",
                    "mean_abs_error": 0.26,
                    "tail_std": 0.11,
                    "reference_tail_std": 0.03,
                    "hermite_tail_at_tmax": 1.2e-5,
                    "laguerre_tail_at_tmax": 8.2e-3,
                    "Nl": 24,
                    "Nm": 96,
                },
            ]
        },
    )
    _write_json(
        tmp_path,
        "docs/_static/w7x_fluctuation_spectrum_panel.json",
        {
            "source_gate_passed": True,
            "time_samples": 8,
            "time_min": 1.0,
            "time_max": 4.0,
            "dominant_phi_ky": 0.2,
            "dominant_heat_flux_ky": 0.4,
        },
    )
    _write_json(
        tmp_path,
        "docs/_static/w7x_tem_extension_status.json",
        {
            "rows": [
                {"lane": "W7-X nonlinear fluctuation spectrum", "status": "closed"},
                {"lane": "TEM / kinetic-electron linear parity", "status": "open"},
            ]
        },
    )
    _write_json(
        tmp_path,
        "docs/_static/quasilinear_validated_calibration_inputs.json",
        {"passed": True},
    )
    _write_json(
        tmp_path,
        "docs/_static/quasilinear_stellarator_train_holdout_report.json",
        {
            "passed": False,
            "points": [
                {"case": "cyclone", "split": "train"},
                {"case": "w7x", "split": "holdout"},
            ],
        },
    )
    _write_json(
        tmp_path,
        "docs/_static/external_vmec_circular_t250_high_grid_convergence_gate.json",
        {"gate_report": {"passed": False}},
    )
    _write_json(
        tmp_path,
        "docs/_static/external_vmec_cth_like_grid_convergence_gate.json",
        {"gate_report": {"passed": False}},
    )
    _write_json(
        tmp_path,
        "docs/_static/external_vmec_cth_like_modified_high_grid_admission_gate.json",
        {"promotion_gate": {"passed": False}},
    )
    _write_json(
        tmp_path,
        "docs/_static/qa_ess_zbs10_rel7p5_control_mean_tmin600_t1100_gate.json",
        {
            "passed": True,
            "summary": {
                "common_pair_count": 21,
                "combined_response_uncertainty_rel": 0.311,
                "independent_response_mean": -0.356,
            },
        },
    )
    _write_json(
        tmp_path,
        "docs/_static/external_vmec_dshape_t250_high_grid_convergence_gate.json",
        {"gate_report": {"passed": True}},
    )
    _write_json(
        tmp_path,
        "docs/_static/external_vmec_itermodel_t350_high_grid_convergence_gate.json",
        {"gate_report": {"passed": True}},
    )
    _write_json(
        tmp_path,
        "docs/_static/external_vmec_qh_grid_convergence_gate.json",
        {"gate_report": {"passed": False}},
    )
    _write_json(
        tmp_path,
        "docs/_static/external_vmec_qh_high_grid_convergence_gate.json",
        {"gate_report": {"passed": False}},
    )
    _write_json(
        tmp_path,
        "docs/_static/differentiable_geometry_bridge.json",
        {
            "sensitivity": {"max_abs_ad_fd_error": 1.0e-8},
            "geometry_inverse_design_report": {"final_residual_norm": 1.0e-12},
            "uq": {"sensitivity_map_rank": 2},
            "backend_info": {"vmec_jax_available": True},
            "booz_xform_jax_api_available": True,
        },
    )
    _write_json(
        tmp_path,
        "docs/_static/nonlinear_sharding_profile_office_gpu.json",
        {
            "identity_gate_pass": True,
            "engineering_speedup": 0.8,
            "device_count": 2,
            "default_backend": "gpu",
            "best_identity_preserving_candidate": {
                "spec": "kx",
                "engineering_speedup_median": 1.03,
            },
        },
    )
    _write_json(
        tmp_path,
        "docs/_static/nonlinear_rhs_profile.json",
        {
            "fastest_full_rhs_label": "GPU spectral",
            "spectral_speedups": {
                "cpu": {
                    "full_rhs_grid_over_spectral": 1.11,
                    "nonlinear_bracket_grid_over_spectral": 1.66,
                },
                "gpu": {
                    "full_rhs_grid_over_spectral": 1.64,
                    "nonlinear_bracket_grid_over_spectral": 2.20,
                },
            },
        },
    )
    _write_json(
        tmp_path,
        "docs/_static/nonlinear_rhs_profile_miller.json",
        {
            "rows": {
                "CPU grid": {"seconds": {"full_rhs": 0.32}},
                "GPU grid": {"seconds": {"full_rhs": 0.013}},
                "GPU spectral": {"seconds": {"full_rhs": 0.015}},
            }
        },
    )
    _write_json(
        tmp_path,
        "docs/_static/nonlinear_rhs_profile_stellarator_runtime.json",
        {
            "rows": {
                "W7-X CPU": {"seconds": {"full_rhs": 0.31}},
                "W7-X GPU": {"seconds": {"full_rhs": 0.027}},
                "HSX CPU": {"seconds": {"full_rhs": 0.31}},
                "HSX GPU": {"seconds": {"full_rhs": 0.027}},
            }
        },
    )
    _write_json(
        tmp_path,
        "docs/_static/full_nonlinear_rhs_trace_summary.json",
        {"warm_seconds": 0.316},
    )
    _write_json(
        tmp_path,
        "docs/_static/full_nonlinear_rhs_trace_gpu_summary.json",
        {"warm_seconds": 0.0128},
    )

    payload = mod.build_status_payload(tmp_path)
    lanes = {row["lane"]: row for row in payload["lanes"]}

    assert payload["summary"] == {
        "n_lanes": 5,
        "n_closed": 1,
        "n_partial": 2,
        "n_open": 2,
        "n_blocked": 0,
    }
    assert lanes["W7-X zonal long-window recurrence/damping"]["status"] == "open"
    assert lanes["W7-X zonal long-window recurrence/damping"]["key_metrics"][
        "failed_reference_gates"
    ] == ["residual_kx070"]
    assert (
        lanes["W7-X zonal long-window recurrence/damping"]["key_metrics"][
            "best_bounded_candidate"
        ]["label"]
        == "best"
    )
    hyper = lanes["W7-X zonal long-window recurrence/damping"]["key_metrics"][
        "best_constant_hypercollision_probe"
    ]
    assert hyper["label"] == "const nuhm0.03"
    assert hyper["validation_status"] == "open"
    mixed_lm = lanes["W7-X zonal long-window recurrence/damping"]["key_metrics"][
        "best_mixed_lm_resolution_probe"
    ]
    assert mixed_lm["label"] == "Nl24 Nm96 mixedLM"
    assert mixed_lm["tail_std_ratio"] == 0.11 / 0.03
    high_moment = lanes["W7-X zonal long-window recurrence/damping"]["key_metrics"][
        "stable_high_moment_mixed_lm_probe"
    ]
    assert high_moment["label"] == "Nl24 Nm96 mixedLM"
    assert high_moment["Nl"] == 24
    assert high_moment["Nm"] == 96
    assert high_moment["laguerre_tail"] == 8.2e-3
    assert (
        "docs/_static/w7x_zonal_mixedlm_resolution_kx070.json"
        in lanes["W7-X zonal long-window recurrence/damping"]["primary_artifacts"]
    )
    assert lanes["Nonlinear holdouts for quasilinear absolute-flux promotion"][
        "claim_level"
    ] == ("diagnostic_calibration_dataset_not_absolute_flux")
    assert lanes["W7-X fluctuation spectrum and TEM/multi-flux extension"][
        "key_metrics"
    ]["open_extension_rows"] == ["TEM / kinetic-electron linear parity"]
    assert (
        lanes["Nonlinear holdouts for quasilinear absolute-flux promotion"][
            "key_metrics"
        ]["cth_like_external_vmec_converged"]
        is False
    )
    assert (
        lanes["Nonlinear holdouts for quasilinear absolute-flux promotion"][
            "key_metrics"
        ]["cth_like_external_vmec_high_grid_admitted"]
        is False
    )
    holdout_metrics = lanes[
        "Nonlinear holdouts for quasilinear absolute-flux promotion"
    ]["key_metrics"]
    assert holdout_metrics["circular_external_vmec_t250_converged"] is False
    assert holdout_metrics["qh_external_vmec_low_to_mid_grid_converged"] is False
    assert holdout_metrics["qh_external_vmec_mid_to_high_grid_converged"] is False
    assert holdout_metrics["dshape_external_vmec_t250_converged"] is True
    assert holdout_metrics["itermodel_external_vmec_t350_converged"] is True
    assert (
        holdout_metrics[
            "variance_reduced_nonlinear_gradient_control_mean_passed"
        ]
        is True
    )
    assert (
        holdout_metrics["variance_reduced_nonlinear_gradient_common_pairs"]
        == 21
    )
    assert (
        holdout_metrics["variance_reduced_nonlinear_gradient_uncertainty_rel"]
        == 0.311
    )
    profiler = lanes["Profiler-backed nonlinear hot-path optimization"]
    assert profiler["status"] == "closed"
    assert profiler["key_metrics"]["release_performance_gate"] is True
    assert "docs/_static/nonlinear_rhs_profile.json" in profiler["primary_artifacts"]
    assert (
        "docs/_static/nonlinear_rhs_profile_stellarator_runtime.json"
        in profiler["primary_artifacts"]
    )
    assert profiler["key_metrics"]["best_identity_candidate"] == "kx"
    assert profiler["key_metrics"]["rhs_fastest_full_label"] == "GPU spectral"
    assert profiler["key_metrics"]["rhs_gpu_full_grid_over_spectral"] == 1.64
    assert profiler["key_metrics"]["rhs_gpu_bracket_grid_over_spectral"] == 2.20
    assert profiler["key_metrics"]["miller_gpu_grid_full_rhs"] == 0.013
    assert profiler["key_metrics"]["w7x_gpu_full_rhs"] == 0.027


def test_build_status_payload_accepts_cth_like_high_grid_admission(tmp_path: Path) -> None:
    """CTH-like can be admitted through the scoped high-grid gate without full-grid convergence."""

    _write_json(
        tmp_path,
        "docs/_static/quasilinear_stellarator_train_holdout_report.json",
        {"passed": False, "points": []},
    )
    _write_json(
        tmp_path,
        "docs/_static/external_vmec_cth_like_grid_convergence_gate.json",
        {"gate_report": {"passed": False}},
    )
    _write_json(
        tmp_path,
        "docs/_static/external_vmec_cth_like_modified_high_grid_admission_gate.json",
        {"promotion_gate": {"passed": True}},
    )

    payload = mod.build_status_payload(tmp_path)
    lanes = {row["lane"]: row for row in payload["lanes"]}
    metrics = lanes["Nonlinear holdouts for quasilinear absolute-flux promotion"][
        "key_metrics"
    ]

    assert metrics["cth_like_external_vmec_full_grid_converged"] is False
    assert metrics["cth_like_external_vmec_high_grid_admitted"] is True
    assert metrics["cth_like_external_vmec_converged"] is True


def test_static_open_lane_status_keeps_deferred_w7x_zonal_and_tem_explicit() -> None:
    payload = mod.build_status_payload(ROOT)
    json.dumps(mod._json_clean(payload), allow_nan=False)

    assert payload["kind"] == "open_research_lane_status"
    assert (
        payload["claim_scope"]
        == "post_v1_5_development_tracking_no_unvalidated_promotion"
    )
    assert set(payload["summary"]) == {
        "n_lanes",
        "n_closed",
        "n_partial",
        "n_open",
        "n_blocked",
    }
    for lane in payload["lanes"]:
        assert {
            "lane",
            "status",
            "claim_level",
            "primary_artifacts",
            "key_metrics",
            "next_action",
        }.issubset(lane)
        assert lane["status"] in mod.STATUS_ORDER
        assert isinstance(lane["primary_artifacts"], list)
        assert isinstance(lane["key_metrics"], dict)

    lanes = {row["lane"]: row for row in payload["lanes"]}
    zonal = lanes["W7-X zonal long-window recurrence/damping"]
    zonal_metrics = zonal["key_metrics"]
    failed = set(zonal_metrics["failed_reference_gates"])

    assert zonal["status"] == "open"
    assert zonal["claim_level"] == "open_physical_closure_not_normalization"
    assert "residual_kx070" in failed
    assert any(metric.startswith("tail_envelope_std_kx") for metric in failed)
    assert (
        zonal_metrics["best_constant_hypercollision_probe"]["validation_status"]
        == "open"
    )
    assert zonal_metrics["stable_high_moment_mixed_lm_probe"]["Nl"] >= 24
    assert "docs/_static/w7x_zonal_reference_compare.json" in zonal["primary_artifacts"]
    assert (
        "docs/_static/w7x_zonal_mixedlm_resolution_kx070.json"
        in zonal["primary_artifacts"]
    )

    tem = lanes["W7-X fluctuation spectrum and TEM/multi-flux extension"]
    tem_metrics = tem["key_metrics"]
    time_min, time_max = tem_metrics["time_window"]
    open_rows = set(tem_metrics["open_extension_rows"])

    assert tem["status"] == "partial"
    assert tem["claim_level"] == "validated_simulation_spectrum_tem_extension_open"
    assert tem_metrics["time_samples"] > 0
    assert time_max > time_min >= 0.0
    assert {
        "TEM / kinetic-electron linear parity",
        "W7-X multi-flux-tube and multi-surface scan",
        "W7-X kinetic-electron/TEM nonlinear window",
    }.issubset(open_rows)
    assert "docs/_static/w7x_tem_extension_status.json" in tem["primary_artifacts"]

    holdouts = lanes["Nonlinear holdouts for quasilinear absolute-flux promotion"]
    assert (
        "docs/_static/qa_ess_zbs10_rel7p5_control_mean_tmin600_t1100_gate.json"
        in holdouts["primary_artifacts"]
    )
    assert (
        holdouts["key_metrics"][
            "variance_reduced_nonlinear_gradient_control_mean_passed"
        ]
        is True
    )
    assert (
        holdouts["key_metrics"][
            "variance_reduced_nonlinear_gradient_common_pairs"
        ]
        == 21
    )
    assert (
        holdouts["key_metrics"][
            "variance_reduced_nonlinear_gradient_uncertainty_rel"
        ]
        < 0.5
    )


def test_write_status_artifacts_writes_all_formats(tmp_path: Path) -> None:
    payload = {
        "kind": "open_research_lane_status",
        "lanes": [
            {
                "lane": "Profiler-backed nonlinear hot-path optimization",
                "status": "partial",
                "claim_level": "profile_identity_artifact_no_speedup_claim",
                "primary_artifacts": ["profile.json"],
                "key_metrics": {"engineering_speedup": 0.75},
                "next_action": "Collect matched profiler traces.",
            }
        ],
        "summary": {
            "n_lanes": 1,
            "n_closed": 0,
            "n_partial": 1,
            "n_open": 0,
            "n_blocked": 0,
        },
    }

    paths = mod.write_status_artifacts(payload, out_png=tmp_path / "status.png")

    for path in paths.values():
        assert Path(path).exists()
    assert (
        json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))["summary"][
            "n_partial"
        ]
        == 1
    )
