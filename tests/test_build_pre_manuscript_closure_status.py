from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "build_pre_manuscript_closure_status.py"
spec = importlib.util.spec_from_file_location("build_pre_manuscript_closure_status", SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def _write_json(root: Path, relative: str, payload: dict[str, object]) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_all_pass_fixture(root: Path) -> None:
    _write_json(
        root,
        "docs/_static/quasilinear_stellarator_train_holdout_report.json",
        {
            "passed": True,
            "by_split": {
                "train": {"n": 3, "mean_abs_relative_error": 0.12},
                "holdout": {"n": 10, "mean_abs_relative_error": 0.22},
            },
            "points": [],
        },
    )
    _write_json(
        root,
        "docs/_static/quasilinear_candidate_uncertainty.json",
        {"promotion_gate": {"passed": True, "accepted_candidates": ["candidate"]}},
    )
    _write_json(
        root,
        "docs/_static/quasilinear_model_selection_status.json",
        {
            "passed": True,
            "accepted_candidates": ["candidate"],
            "promotion_gate": {"passed": True, "blockers": []},
            "metrics": {
                "candidate_mean_abs_relative_error": 0.2,
                "candidate_prediction_interval_coverage": 0.9,
            },
        },
    )
    _write_json(
        root,
        "docs/_static/quasilinear_dataset_sufficiency.json",
        {
            "requirements": {
                "checks": {
                    "validated_input_gates": True,
                    "minimum_total_electrostatic_cases": True,
                    "minimum_holdout_geometries": True,
                    "minimum_explicit_train_geometries": True,
                }
            }
        },
    )
    _write_json(root, "docs/_static/quasilinear_promotion_guardrails.json", {"passed": True})
    _write_json(root, "docs/_static/quasilinear_holdout_gap_report.json", {"promotion_gate": {"passed": True, "blockers": []}})
    _write_json(
        root,
        "docs/_static/quasilinear_error_anatomy.json",
        {
            "case_count": 12,
            "holdout_count": 10,
            "candidate_mean_abs_relative_error": 0.28,
            "promotion_gate": {"passed": False, "blockers": ["declared_stress_outliers_deferred"]},
            "frozen_ledger_policy": {"additional_holdout_collection_active": False},
            "core_portfolio_gate": {
                "passed": True,
                "transport_gate": 0.35,
                "interval_coverage_gate": 0.75,
                "core_case_count": 10,
                "core_holdout_count": 8,
                "core_mean_abs_relative_error": 0.28,
                "core_holdout_mean_abs_relative_error": 0.27,
                "core_max_abs_relative_error": 0.57,
                "core_prediction_interval_coverage": 1.0,
                "core_spearman": 0.74,
                "core_holdout_spearman": 0.73,
                "core_pairwise_order_accuracy": 0.75,
                "core_holdout_pairwise_order_accuracy": 0.75,
                "screening_gate_passed": False,
                "excluded_cases": [
                    {"case": "solovev_reference_repair_dt002_amp1em5_n48_t250"},
                    {"case": "shaped_tokamak_pressure_external_vmec_t650_high_grid_window"},
                ],
            },
        },
    )

    _write_json(
        root,
        "docs/_static/production_nonlinear_optimization_guard.json",
        {
            "passed": True,
            "summary": {
                "qualifying_matched_optimized_transport_audits": 3,
                "qualifying_optimized_equilibrium_ensembles": 3,
                "qualifying_replicated_holdout_ensembles": 4,
            },
        },
    )
    _write_json(
        root,
        "docs/_static/vmec_jax_qa_transport_optimization_status.json",
        {"summary": {"long_window_nonlinear_audit_passed": True}},
    )

    _write_json(
        root,
        "docs/_static/nonlinear_sharding_strong_scaling_large.json",
        {
            "identity_passed": True,
            "speedup_passed": True,
            "rows": [
                {"backend": "cpu", "speedup": 2.0},
                {"backend": "gpu", "speedup": 1.8},
            ],
        },
    )
    _write_json(root, "docs/_static/nonlinear_sharding_production_speedup_gate.json", {"passed": True})
    _write_json(root, "docs/_static/nonlinear_domain_parallel_identity_gate.json", {"gate": {"identity_passed": True}})
    _write_json(root, "docs/_static/nonlinear_spectral_communication_identity_gate.json", {"gate": {"identity_passed": True}})
    _write_json(
        root,
        "docs/_static/nonlinear_spectral_domain_routing_profile.json",
        {
            "identity_passed": True,
            "strong_speedup_vs_serial": 1.8,
            "speedup_gate_passed": True,
            "production_speedup_claim_allowed": False,
        },
    )
    _write_json(root, "docs/_static/parallel_decomposition_status.json", {"passed": True})

    _write_json(root, "docs/_static/vmec_boozer_quasilinear_gradient_gate.json", {"passed": True})
    _write_json(root, "docs/_static/vmec_boozer_nonlinear_window_gradient_gate.json", {"passed": True})
    _write_json(root, "docs/_static/vmec_boozer_gradient_holdout_matrix.json", {"passed": True})
    _write_json(root, "docs/_static/vmec_boozer_aggregate_alpha_holdout_gate.json", {"passed": True})
    _write_json(root, "docs/_static/vmec_boozer_aggregate_surface_holdout_gate.json", {"passed": True})
    _write_json(root, "docs/_static/vmec_boozer_second_equilibrium_aggregate_gate.json", {"passed": True})
    _write_json(
        root,
        "docs/_static/vmec_boozer_aggregate_holdout_promotion_gate.json",
        {
            "passed": True,
            "promotion_gate": {"passed": True, "blockers": []},
            "holdout_artifacts": [{"qualifies_for_promotion": True}],
        },
    )


def test_current_repository_pre_manuscript_lanes_fail_closed() -> None:
    payload = mod.build_status_payload(ROOT)
    lanes = {lane["lane"]: lane for lane in payload["lanes"]}

    assert payload["kind"] == "pre_manuscript_closure_status"
    assert payload["summary"]["ready_for_manuscript_drafting"] is False
    assert len(lanes) == 4
    assert lanes["Scoped core quasilinear heat-flux diagnostic"]["passed"] is True
    assert lanes["Production nonlinear domain-decomposition speedup"]["passed"] is False
    ql_lane = lanes["Scoped core quasilinear heat-flux diagnostic"]
    assert ql_lane["status"] == "closed"
    assert ql_lane["completion_percent"] == 100.0
    assert ql_lane["key_metrics"]["full_universal_promotion_passed"] is False
    assert ql_lane["key_metrics"]["core_case_count"] == 10
    assert ql_lane["key_metrics"]["core_holdout_count"] == 8
    assert 0.27 < ql_lane["key_metrics"]["core_mean_abs_relative_error"] < 0.29
    assert 0.27 < ql_lane["key_metrics"]["core_holdout_mean_abs_relative_error"] < 0.29
    assert ql_lane["key_metrics"]["core_prediction_interval_coverage"] == 1.0
    assert ql_lane["key_metrics"]["core_screening_gate_passed"] is False
    assert set(ql_lane["key_metrics"]["declared_stress_outliers"]) == {
        "solovev_reference_repair_dt002_amp1em5_n48_t250",
        "shaped_tokamak_pressure_external_vmec_t650_high_grid_window",
    }
    assert "universal absolute-flux runtime predictor" in ql_lane["next_action"]
    assert ql_lane["required_next_artifacts"] == []
    assert (
        "gpu_domain_speedup_below_1p5"
        in lanes["Production nonlinear domain-decomposition speedup"]["blockers"]
    )
    domain_lane = lanes["Production nonlinear domain-decomposition speedup"]
    assert domain_lane["completion_percent"] == 65.0
    assert domain_lane["key_metrics"]["routed_domain_timing_identity_passed"] is True
    assert domain_lane["key_metrics"]["routed_domain_timing_speedup_gate_passed"] is False


def test_all_pass_fixture_closes_pre_manuscript_dashboard(tmp_path: Path) -> None:
    _write_all_pass_fixture(tmp_path)

    payload = mod.build_status_payload(tmp_path)

    assert payload["summary"]["ready_for_manuscript_drafting"] is True
    assert payload["summary"]["n_closed"] == 4
    assert payload["summary"]["mean_completion_percent"] == 100.0
    assert all(lane["passed"] for lane in payload["lanes"])
    vmec_lane = {
        lane["lane"]: lane for lane in payload["lanes"]
    }["VMEC/Boozer holdout optimization"]
    assert "closed for the current pre-manuscript gate" in vmec_lane["next_action"]
    assert all(
        "production-scope held-out surface or field-line artifact" not in item
        for item in vmec_lane["required_next_artifacts"]
    )


def test_write_pre_manuscript_artifacts(tmp_path: Path) -> None:
    _write_all_pass_fixture(tmp_path)
    payload = mod.build_status_payload(tmp_path)

    paths = mod.write_status_artifacts(payload, out=tmp_path / "pre_status.png")

    for path in paths.values():
        assert Path(path).exists()
    saved = json.loads((tmp_path / "pre_status.json").read_text(encoding="utf-8"))
    assert saved["kind"] == "pre_manuscript_closure_status"
