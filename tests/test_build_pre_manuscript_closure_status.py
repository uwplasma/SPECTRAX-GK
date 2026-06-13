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
                "holdout": {"n": 9, "mean_abs_relative_error": 0.22},
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
    assert lanes["Universal absolute quasilinear heat-flux prediction"]["passed"] is False
    assert lanes["Production nonlinear domain-decomposition speedup"]["passed"] is False
    ql_lane = lanes["Universal absolute quasilinear heat-flux prediction"]
    assert "twelve-case ledger frozen" in ql_lane["next_action"]
    assert "Add independent converged nonlinear holdouts" not in ql_lane["next_action"]
    assert all("additional independent converged nonlinear holdout" not in item for item in ql_lane["required_next_artifacts"])
    assert (
        "holdout_mean_abs_relative_error_exceeds_0.35"
        in ql_lane["blockers"]
    )
    assert (
        "gpu_domain_speedup_below_1p5"
        in lanes["Production nonlinear domain-decomposition speedup"]["blockers"]
    )


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
