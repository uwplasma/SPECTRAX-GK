from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "check_vmec_boozer_aggregate_holdout_gate.py"
spec = importlib.util.spec_from_file_location("check_vmec_boozer_aggregate_holdout_gate", SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _aggregate_payload() -> dict[str, object]:
    return {
        "kind": "vmec_boozer_aggregate_scalar_objective_finite_difference_report",
        "passed": True,
        "claim_scope": "reduced aggregate objective plumbing",
        "samples": [
            {"surface_index": None, "alpha": 0.0, "selected_ky_index": 1, "weight": 0.5},
            {"surface_index": None, "alpha": 0.0, "selected_ky_index": 2, "weight": 0.5},
        ],
    }


def _line_search_payload() -> dict[str, object]:
    return {
        "kind": "vmec_boozer_aggregate_scalar_objective_line_search_report",
        "passed": True,
        "samples": [
            {"surface_index": None, "alpha": 0.0, "selected_ky_index": 1, "weight": 0.5},
            {"surface_index": None, "alpha": 0.0, "selected_ky_index": 2, "weight": 0.5},
        ],
    }


def test_aggregate_holdout_gate_blocks_without_surface_or_field_line_holdout(tmp_path: Path) -> None:
    aggregate = _write_json(tmp_path / "aggregate.json", _aggregate_payload())
    line_search = _write_json(tmp_path / "line_search.json", _line_search_payload())

    report = mod.check_vmec_boozer_aggregate_holdout_gate(
        aggregate_artifact=aggregate,
        line_search_artifact=line_search,
    )

    assert report["passed"] is False
    assert report["promotion_gate"]["blockers"] == ["passed_holdout_surface_or_field_line_artifact"]
    assert report["training_sample_summary"]["alphas"] == ["0"]


def test_aggregate_holdout_gate_rejects_ky_only_holdout(tmp_path: Path) -> None:
    aggregate = _write_json(tmp_path / "aggregate.json", _aggregate_payload())
    line_search = _write_json(tmp_path / "line_search.json", _line_search_payload())
    ky_only = _write_json(
        tmp_path / "ky_only.json",
        {
            "passed": True,
            "claim_level": "passed_grid_convergence_candidate_for_transport_holdout",
            "samples": [
                {"surface_index": None, "alpha": 0.0, "selected_ky_index": 7},
            ],
        },
    )

    report = mod.check_vmec_boozer_aggregate_holdout_gate(
        aggregate_artifact=aggregate,
        line_search_artifact=line_search,
        holdout_artifacts=(ky_only,),
    )

    assert report["passed"] is False
    assert report["holdout_artifacts"][0]["passed"] is True
    assert report["holdout_artifacts"][0]["heldout_surface_or_field_line"] is False
    assert "k_y-only" in report["promotion_gate"]["requirements"][3]


def test_aggregate_holdout_gate_accepts_passed_field_line_holdout(tmp_path: Path) -> None:
    aggregate = _write_json(tmp_path / "aggregate.json", _aggregate_payload())
    line_search = _write_json(tmp_path / "line_search.json", _line_search_payload())
    holdout = _write_json(
        tmp_path / "alpha_holdout.json",
        {
            "promotion_gate": {"passed": True},
            "claim_level": "passed_grid_convergence_candidate_for_transport_holdout",
            "samples": [
                {"surface_index": None, "alpha": 0.75, "selected_ky_index": 1},
            ],
        },
    )

    report = mod.check_vmec_boozer_aggregate_holdout_gate(
        aggregate_artifact=aggregate,
        line_search_artifact=line_search,
        holdout_artifacts=(holdout,),
    )

    assert report["passed"] is True
    assert report["promotion_gate"]["blockers"] == []
    assert report["holdout_artifacts"][0]["qualifies_for_promotion"] is True
    assert report["gates"][-1]["detail"].endswith("held-out field-line alpha=0.75")


def test_aggregate_holdout_gate_rejects_non_promotable_holdout_scope(tmp_path: Path) -> None:
    aggregate = _write_json(tmp_path / "aggregate.json", _aggregate_payload())
    line_search = _write_json(tmp_path / "line_search.json", _line_search_payload())
    startup_holdout = _write_json(
        tmp_path / "startup_holdout.json",
        {
            "passed": True,
            "claim_level": "startup_transient_nonlinear_plumbing_fd_audit_not_transport_average",
            "transport_average_gate": False,
            "heldout_samples": [
                {"surface_index": 3, "alpha": 0.0, "selected_ky_index": 1},
            ],
        },
    )

    report = mod.check_vmec_boozer_aggregate_holdout_gate(
        aggregate_artifact=aggregate,
        line_search_artifact=line_search,
        holdout_artifacts=(startup_holdout,),
    )

    assert report["passed"] is False
    assert report["holdout_artifacts"][0]["n_samples"] == 1
    assert report["holdout_artifacts"][0]["heldout_surface_or_field_line"] is True
    assert report["holdout_artifacts"][0]["qualifies_for_promotion"] is False
    assert "transport_average_gate_false" in report["holdout_artifacts"][0]["claim_scope_blockers"]


def test_aggregate_holdout_gate_main_writes_json(tmp_path: Path) -> None:
    aggregate = _write_json(tmp_path / "aggregate.json", _aggregate_payload())
    line_search = _write_json(tmp_path / "line_search.json", _line_search_payload())
    out = tmp_path / "report.json"

    result = mod.main(
        [
            "--aggregate-artifact",
            str(aggregate),
            "--line-search-artifact",
            str(line_search),
            "--json-out",
            str(out),
        ]
    )

    assert result == 0
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["kind"] == "vmec_boozer_aggregate_holdout_promotion_gate"
    assert saved["passed"] is False
