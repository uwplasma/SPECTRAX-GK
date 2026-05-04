from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "build_manuscript_readiness_status.py"
spec = importlib.util.spec_from_file_location("build_manuscript_readiness_status", SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def _write_json(root: Path, relative: str, payload: dict[str, object]) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_manuscript_status_closes_negative_ql_and_defers_zonal_tem(tmp_path: Path) -> None:
    _write_json(tmp_path, "docs/_static/quasilinear_validated_calibration_inputs.json", {"passed": True})
    _write_json(
        tmp_path,
        "docs/_static/quasilinear_stellarator_train_holdout_report.json",
        {
            "passed": False,
            "by_split": {"holdout": {"mean_abs_relative_error": 2.5}},
            "points": [
                {"case": "cyclone", "split": "train"},
                {"case": "w7x", "split": "holdout"},
                {"case": "hsx", "split": "holdout"},
            ],
        },
    )
    for name in (
        "quasilinear_saturation_rule_sweep",
        "quasilinear_shape_aware_saturation",
        "quasilinear_candidate_uncertainty",
    ):
        _write_json(
            tmp_path,
            f"docs/_static/{name}.json",
            {"promotion_gate": {"passed": False, "null_training_mean_mean_abs_relative_error": 0.2}},
        )
    _write_json(
        tmp_path,
        "docs/_static/quasilinear_dataset_sufficiency.json",
        {
            "promotion_gate": {"passed": False, "blockers": ["minimum_total_electrostatic_cases"]},
            "requirements": {
                "current_total_cases": 4,
                "min_total_electrostatic_cases": 6,
                "current_explicit_train_geometries": 1,
                "min_explicit_train_geometries": 2,
            },
        },
    )
    _write_json(
        tmp_path,
        "docs/_static/differentiable_geometry_bridge.json",
        {"sensitivity": {"max_abs_ad_fd_error": 1.0e-9}, "uq": {"sensitivity_map_rank": 2}},
    )
    _write_json(
        tmp_path,
        "docs/_static/vmec_boozer_parity_matrix.json",
        {"minimum_boozer_mode_count": 21, "summary": {"all_equal_arc_passed": True, "n_cases": 3}},
    )
    _write_json(
        tmp_path,
        "docs/_static/stellarator_itg_optimization_comparison.json",
        {
            "results": [
                {"initial_objective": 1.0, "final_objective": 0.2, "gradient_gate": {"passed": True}},
                {"initial_objective": 2.0, "final_objective": 0.5, "gradient_gate": {"passed": True}},
            ]
        },
    )
    _write_json(
        tmp_path,
        "docs/_static/stellarator_itg_optimization_uq.json",
        {
            "claim_level": "reduced_objective_uq_and_sensitivity_validation_not_full_vmec_gk_optimization",
            "all_gradient_gates_passed": True,
            "all_sensitivity_maps_full_rank": True,
        },
    )
    _write_json(
        tmp_path,
        "docs/_static/nonlinear_sharding_profile_office_gpu.json",
        {
            "identity_gate_pass": True,
            "engineering_speedup": 0.8,
            "best_identity_preserving_candidate": {"spec": "kx", "engineering_speedup_median": 1.03},
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
        "docs/_static/solver_objective_gradient_gate.json",
        {
            "passed": True,
            "source_scope": "solver_ready_geometry_contract",
            "linear_growth_gradient_gate": True,
            "quasilinear_weight_gradient_gate": True,
        },
    )
    for name in (
        "vmec_boozer_solver_frequency_gradient_gate",
        "vmec_boozer_quasilinear_gradient_gate",
        "vmec_boozer_nonlinear_window_gradient_gate",
        "vmec_boozer_li383_nonlinear_window_gradient_gate",
    ):
        _write_json(
            tmp_path,
            f"docs/_static/{name}.json",
            {
                "passed": True,
                "source_scope": "mode21_vmec_boozer_state",
                "nonlinear_window_gradient_gate": name.endswith("nonlinear_window_gradient_gate"),
                "eigenpair_gate": {"max_rel_error": 1.0e-3},
            },
        )
    _write_json(
        tmp_path,
        "docs/_static/vmec_boozer_gradient_holdout_matrix.json",
        {
            "passed": True,
            "summary": {
                "n_cases": 2,
                "max_relative_error": 4.9e-3,
                "all_gates_passed": True,
            },
        },
    )
    _write_json(
        tmp_path,
        "docs/_static/nonlinear_window_fd_audit.json",
        {
            "passed": True,
            "startup_nonlinear_plumbing_fd_path_gate": True,
            "transport_average_gate": False,
            "production_nonlinear_observable_fd_path_gate": False,
            "production_nonlinear_window_gradient_gate": False,
            "metrics": {
                "response_fraction": 0.11,
                "repeatability_relative_error": 0.0,
                "max_window_cv": 0.09,
                "max_window_trend": 0.31,
            },
        },
    )
    _write_json(
        tmp_path,
        "docs/_static/vmec_boozer_nonlinear_window_fd_audit.json",
        {
            "passed": True,
            "vmec_boozer_startup_nonlinear_plumbing_fd_path_gate": True,
            "transport_average_gate": False,
            "vmec_boozer_production_nonlinear_observable_fd_path_gate": False,
            "production_nonlinear_window_gradient_gate": False,
            "metrics": {
                "response_fraction": 0.04,
                "derivative_asymmetry": 2.8,
            },
        },
    )

    payload = mod.build_manuscript_readiness_payload(tmp_path)
    lanes = {lane["lane"]: lane for lane in payload["lanes"]}

    assert payload["summary"]["n_deferred"] == 2
    assert lanes["Quasilinear diagnostics and saturation-model selection"]["status"] == "closed"
    assert lanes["Quasilinear diagnostics and saturation-model selection"]["key_metrics"]["absolute_flux_promoted"] is False
    assert (
        lanes["Quasilinear diagnostics and saturation-model selection"]["key_metrics"][
            "dataset_sufficiency_promotion_passed"
        ]
        is False
    )
    assert lanes["VMEC/Boozer differentiable geometry parity"]["status"] == "closed"
    assert lanes["Reduced differentiable stellarator ITG optimization"]["status"] == "closed"
    assert lanes["Production solver-objective geometry gradients"]["status"] == "closed"
    assert (
        lanes["Production solver-objective geometry gradients"]["key_metrics"][
            "multi_equilibrium_gradient_holdout_matrix"
        ]
        is True
    )
    assert (
        lanes["Production solver-objective geometry gradients"]["key_metrics"][
            "full_vmec_boozer_reduced_nonlinear_window_gradient_gate"
        ]
        is True
    )
    assert (
        lanes["Production solver-objective geometry gradients"]["key_metrics"][
            "reduced_nonlinear_window_gradient_gate"
        ]
        is True
    )
    assert (
        lanes["Production solver-objective geometry gradients"]["key_metrics"][
            "multi_equilibrium_reduced_nonlinear_window_gradient_gate"
        ]
        is True
    )
    assert (
        lanes["Production solver-objective geometry gradients"]["key_metrics"][
            "production_nonlinear_window_gradient_gate"
        ]
        is False
    )
    assert (
        lanes["Production solver-objective geometry gradients"]["key_metrics"][
            "startup_nonlinear_plumbing_fd_path_gate"
        ]
        is True
    )
    assert (
        lanes["Production solver-objective geometry gradients"]["key_metrics"][
            "startup_nonlinear_plumbing_response_fraction"
        ]
        == 0.11
    )
    assert (
        lanes["Production solver-objective geometry gradients"]["key_metrics"][
            "nonlinear_transport_average_gate"
        ]
        is False
    )
    assert (
        "docs/_static/nonlinear_window_fd_audit.json"
        in lanes["Production solver-objective geometry gradients"]["primary_artifacts"]
    )
    assert (
        "docs/_static/nonlinear_window_fd_audit.png"
        in lanes["Production solver-objective geometry gradients"]["primary_artifacts"]
    )
    assert (
        lanes["Production solver-objective geometry gradients"]["key_metrics"][
            "vmec_boozer_startup_nonlinear_plumbing_fd_path_gate"
        ]
        is True
    )
    assert (
        lanes["Production solver-objective geometry gradients"]["key_metrics"][
            "vmec_boozer_startup_nonlinear_response_fraction"
        ]
        == 0.04
    )
    assert (
        lanes["Production solver-objective geometry gradients"]["key_metrics"][
            "vmec_boozer_nonlinear_transport_average_gate"
        ]
        is False
    )
    assert (
        "docs/_static/vmec_boozer_nonlinear_window_fd_audit.json"
        in lanes["Production solver-objective geometry gradients"]["primary_artifacts"]
    )
    assert lanes["W7-X zonal recurrence/damping"]["status"] == "deferred"
    assert lanes["TEM / kinetic-electron stellarator extension"]["status"] == "deferred"
    profiler = lanes["Profiler-backed nonlinear performance claims"]
    assert profiler["status"] == "partial"
    assert "docs/_static/nonlinear_rhs_profile.json" in profiler["primary_artifacts"]
    assert profiler["key_metrics"]["best_identity_candidate"] == "kx"
    assert profiler["key_metrics"]["rhs_fastest_full_label"] == "GPU spectral"
    assert profiler["key_metrics"]["rhs_gpu_full_grid_over_spectral"] == 1.64
    assert profiler["key_metrics"]["rhs_cpu_bracket_grid_over_spectral"] == 1.66


def test_write_manuscript_readiness_artifacts_writes_all_formats(tmp_path: Path) -> None:
    payload = {
        "kind": "manuscript_readiness_status",
        "lanes": [
            {
                "lane": "Production solver-objective geometry gradients",
                "status": "open",
                "claim_level": "required",
                "primary_artifacts": [],
                "key_metrics": {},
                "next_action": "Add gates.",
            }
        ],
        "summary": {"active_fraction_closed": 0.0},
    }

    paths = mod.write_manuscript_readiness_artifacts(payload, out=tmp_path / "status.png")

    for path in paths.values():
        assert Path(path).exists()
    saved = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert saved["kind"] == "manuscript_readiness_status"
