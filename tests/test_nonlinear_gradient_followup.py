from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

from spectraxgk.nonlinear_gradient_followup import (
    NonlinearGradientFollowupConfig,
    nonlinear_gradient_followup_plan,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "plan_nonlinear_gradient_followup.py"


def _load_tool_module():
    spec = importlib.util.spec_from_file_location("plan_nonlinear_gradient_followup", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _ensemble(state: str, means: tuple[float, float, float] = (1.0, 1.1, 0.9)) -> dict[str, object]:
    return {
        "kind": "nonlinear_window_ensemble_report",
        "passed": True,
        "n_reports": 3,
        "rows": [
            {
                "late_mean": value,
                "source_artifact": f"{state}_nonlinear_t900_n64_{label}_heat_flux_trace.csv",
                "summary_artifact": f"{state}_nonlinear_t900_n64_{label}_transport_window.json",
            }
            for value, label in zip(means, ("seed31", "seed32", "dt0p04"))
        ],
    }


def _artifact(
    *,
    response: float = 0.08,
    asymmetry: float = 0.30,
    uncertainty: float = 0.56,
    passed: bool = False,
) -> dict[str, object]:
    return {
        "kind": "nonlinear_turbulence_gradient_central_fd_gate",
        "claim_level": "production_long_window_nonlinear_turbulence_gradient_candidate",
        "parameter_name": "rbc_1_1",
        "passed": passed,
        "metrics": {
            "response_fraction": response,
            "fd_asymmetry_rel": asymmetry,
            "gradient_uncertainty_rel": uncertainty,
        },
        "source_ensembles": {
            "baseline": _ensemble("baseline"),
            "plus": _ensemble("plus"),
            "minus": _ensemble("minus"),
        },
    }


def test_followup_plan_adds_only_targeted_replicates_for_local_noisy_candidate() -> None:
    report = nonlinear_gradient_followup_plan(
        [_artifact()],
        labels=["rbc"],
        config=NonlinearGradientFollowupConfig(sem_safety_factor=1.0),
    )

    candidate = report["candidate_actions"][0]
    assert report["passed"] is False
    assert report["summary"]["planned_run_count"] == 3
    assert candidate["action"] == "add_matched_nominal_seed_replicates"
    assert candidate["estimated_required_replicates_per_state"] == 4
    assert candidate["extra_replicates_per_state"] == 1
    assert {row["state"] for row in report["planned_runs"]} == {
        "baseline",
        "plus_delta",
        "minus_delta",
    }
    assert {row["variant_label"] for row in report["planned_runs"]} == {"seed33"}


def test_followup_plan_does_not_add_replicates_for_nonlocal_or_unresolved_candidates() -> None:
    report = nonlinear_gradient_followup_plan(
        [
            _artifact(asymmetry=0.75, uncertainty=0.20),
            _artifact(response=0.01, asymmetry=0.20, uncertainty=0.20),
        ],
        labels=["nonlocal", "unresolved"],
    )

    assert report["summary"]["planned_run_count"] == 0
    assert report["candidate_actions"][0]["action"] == "shrink_bracket_or_replace_control"
    assert (
        report["candidate_actions"][1]["action"]
        == "replace_control_or_increase_checked_bracket"
    )
    assert "smaller-bracket" in report["next_action"]


def test_followup_plan_freezes_passed_candidate_and_validates_config() -> None:
    report = nonlinear_gradient_followup_plan([_artifact(passed=True)])

    assert report["passed"] is True
    assert report["summary"]["promoted_candidate_count"] == 1
    assert report["candidate_actions"][0]["action"] == "freeze_promoted_candidate"

    with pytest.raises(ValueError, match="sem_safety_factor"):
        nonlinear_gradient_followup_plan(
            [_artifact()],
            config=NonlinearGradientFollowupConfig(sem_safety_factor=0.0),
        )
    with pytest.raises(ValueError, match="max_extra_replicates_per_state"):
        nonlinear_gradient_followup_plan(
            [_artifact()],
            config=NonlinearGradientFollowupConfig(max_extra_replicates_per_state=-1),
        )
    with pytest.raises(ValueError, match="max_gradient_uncertainty_rel"):
        nonlinear_gradient_followup_plan(
            [_artifact()],
            config=NonlinearGradientFollowupConfig(max_gradient_uncertainty_rel=0.0),
        )
    with pytest.raises(ValueError, match="paths length"):
        nonlinear_gradient_followup_plan([_artifact()], paths=[None, None])
    with pytest.raises(ValueError, match="labels length"):
        nonlinear_gradient_followup_plan([_artifact()], labels=["one", "two"])


def test_followup_plan_recovers_missing_replicate_metadata() -> None:
    artifact = _artifact()
    artifact["source_ensembles"] = {}

    report = nonlinear_gradient_followup_plan([artifact])

    assert report["summary"]["planned_run_count"] == 0
    assert report["candidate_actions"][0]["action"] == "recover_replicate_metadata"
    assert report["candidate_actions"][0]["estimated_required_replicates_per_state"] is None


def test_followup_plan_handles_scalar_pass_without_artifact_pass_and_empty_inputs() -> None:
    scalar_ok = _artifact(response=0.08, asymmetry=0.3, uncertainty=0.2, passed=False)

    report = nonlinear_gradient_followup_plan([scalar_ok])
    empty = nonlinear_gradient_followup_plan([])
    unresolved = nonlinear_gradient_followup_plan([_artifact(response=0.01)])

    assert report["candidate_actions"][0]["action"] == "no_followup_needed"
    assert report["next_action"].startswith("inspect artifacts")
    assert empty["summary"]["candidate_count"] == 0
    assert empty["next_action"].startswith("inspect artifacts")
    assert unresolved["next_action"].startswith("choose controls")


def test_followup_plan_covers_fallback_metadata_and_metric_sources() -> None:
    artifact = {
        "kind": "nonlinear_turbulence_gradient_central_fd_gate",
        "claim_level": "production_long_window_nonlinear_turbulence_gradient_candidate",
        "parameter_name": "fallback_control",
        "conditioning": {
            "response_fraction": 1,
            "fd_asymmetry_rel": 0.2,
            "gradient_relative_uncertainty": 0.8,
        },
        "nonlinear_turbulence_gradient_gate": {"passed": False},
        "source_ensembles": {
            "baseline": {
                "rows": [
                    "not-a-row",
                    {"path": "baseline_without_seed_label.csv"},
                    {"source_artifact": "baseline_dt0p04.csv"},
                ],
            },
            "plus": {"rows": [{"summary_artifact": "plus_dt0p04.json"}]},
            "junk": {"rows": "not-a-sequence"},
            "bad": "not-an-ensemble",
        },
    }

    report = nonlinear_gradient_followup_plan(
        [artifact],
        config=NonlinearGradientFollowupConfig(
            sem_safety_factor=1.0,
            max_extra_replicates_per_state=1,
        ),
    )

    candidate = report["candidate_actions"][0]
    assert candidate["action"] == "add_matched_nominal_seed_replicates"
    assert candidate["current_replicates_per_state"] == 1
    assert {row["state"] for row in candidate["planned_runs"]} == {"baseline", "plus_delta"}
    assert {row["variant_label"] for row in candidate["planned_runs"]} == {"seed31"}


def test_followup_plan_handles_missing_numeric_metrics() -> None:
    report = nonlinear_gradient_followup_plan(
        [
            {
                "metrics": "not-a-mapping",
                "source_ensembles": {"baseline": {"n_reports": "bad"}},
            }
        ]
    )

    candidate = report["candidate_actions"][0]
    assert candidate["action"] == "replace_control_or_increase_checked_bracket"
    assert candidate["metrics"] == {
        "response_fraction": None,
        "fd_asymmetry_rel": None,
        "gradient_uncertainty_rel": None,
    }


def test_plan_nonlinear_gradient_followup_tool_writes_json(tmp_path: Path) -> None:
    tool = _load_tool_module()
    artifact = tmp_path / "candidate.json"
    out = tmp_path / "plan.json"
    artifact.write_text(json.dumps(_artifact()), encoding="utf-8")

    rc = tool.main(
        [
            str(artifact),
            "--case",
            "tool_case",
            "--json-out",
            str(out),
            "--sem-safety-factor",
            "1.0",
        ]
    )

    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["case"] == "tool_case"
    assert payload["summary"]["planned_run_count"] == 3
