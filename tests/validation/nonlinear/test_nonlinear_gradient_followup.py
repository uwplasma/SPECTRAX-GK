from __future__ import annotations

import json
from pathlib import Path

from support.paths import REPO_ROOT, load_artifact_tool, load_campaign_tool

import pytest

from tools.campaigns.nonlinear_gradient_followup import (
    nonlinear_gradient_candidate_design_report,
)
from tools.campaigns.nonlinear_gradient_followup import (
    nonlinear_gradient_composite_control_report,
)
from tools.campaigns.nonlinear_gradient_followup import (
    NonlinearGradientCandidateDesignConfig,
    NonlinearGradientCompositeControlConfig,
    NonlinearGradientControlMeanGateConfig,
    NonlinearGradientControlVariateCampaignConfig,
    NonlinearGradientFollowupConfig,
    NonlinearGradientVarianceReductionConfig,
)
from tools.campaigns.nonlinear_gradient_followup import (
    nonlinear_gradient_followup_plan,
)
from tools.campaigns.nonlinear_gradient_followup import (
    nonlinear_gradient_control_mean_gate,
    nonlinear_gradient_control_variate_campaign_plan,
    nonlinear_gradient_variance_reduction_plan,
)


ROOT = REPO_ROOT


def _load_tool_module():
    return load_campaign_tool("design_nonlinear_gradient")


def _ensemble(
    state: str, means: tuple[float, float, float] = (1.0, 1.1, 0.9)
) -> dict[str, object]:
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


def test_followup_plan_adds_only_targeted_replicates_for_local_noisy_candidate() -> (
    None
):
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


def test_followup_plan_does_not_add_replicates_for_nonlocal_or_unresolved_candidates() -> (
    None
):
    report = nonlinear_gradient_followup_plan(
        [
            _artifact(asymmetry=0.75, uncertainty=0.20),
            _artifact(response=0.01, asymmetry=0.20, uncertainty=0.20),
        ],
        labels=["nonlocal", "unresolved"],
    )

    assert report["summary"]["planned_run_count"] == 0
    assert (
        report["candidate_actions"][0]["action"] == "shrink_bracket_or_replace_control"
    )
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
    assert (
        report["candidate_actions"][0]["estimated_required_replicates_per_state"]
        is None
    )


def test_followup_plan_handles_scalar_pass_without_artifact_pass_and_empty_inputs() -> (
    None
):
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
    assert {row["state"] for row in candidate["planned_runs"]} == {
        "baseline",
        "plus_delta",
    }
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


def test_design_nonlinear_gradient_followup_plan_writes_json(tmp_path: Path) -> None:
    tool = _load_tool_module()
    artifact = tmp_path / "candidate.json"
    out = tmp_path / "plan.json"
    artifact.write_text(json.dumps(_artifact()), encoding="utf-8")

    rc = tool.main(
        [
            "followup-plan",
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


def test_design_nonlinear_gradient_followup_plan_hydrates_compact_ensembles(
    tmp_path: Path,
) -> None:
    tool = _load_tool_module()
    artifact_payload = _artifact(uncertainty=0.56)
    for state in ("baseline", "plus", "minus"):
        ensemble = tmp_path / f"{state}_ensemble.json"
        ensemble.write_text(json.dumps(_ensemble(state)), encoding="utf-8")
        artifact_payload["source_ensembles"][state] = {
            "n_reports": 3,
            "path": str(ensemble),
        }
    artifact = tmp_path / "candidate.json"
    out = tmp_path / "plan.json"
    artifact.write_text(json.dumps(artifact_payload), encoding="utf-8")

    rc = tool.main(
        [
            "followup-plan",
            str(artifact),
            "--case",
            "hydrated_case",
            "--json-out",
            str(out),
            "--sem-safety-factor",
            "1.0",
        ]
    )

    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["summary"]["planned_run_count"] == 3
    assert {row["variant_label"] for row in payload["planned_runs"]} == {"seed33"}


def test_candidate_design_reports_infeasible_rbc_like_followup() -> None:
    artifact = _artifact(response=0.072, asymmetry=0.475, uncertainty=0.683)
    artifact["source_ensembles"] = {
        "baseline": _ensemble("baseline", means=(15.0, 15.2, 15.4, 15.6, 15.8)),
        "plus": _ensemble("plus", means=(14.4, 14.7, 15.0, 15.2, 15.5)),
        "minus": _ensemble("minus", means=(15.8, 16.0, 16.2, 16.3, 16.4)),
    }
    for ensemble in artifact["source_ensembles"].values():
        ensemble["n_reports"] = 5

    report = nonlinear_gradient_candidate_design_report(
        [artifact],
        labels=["rbc11"],
        config=NonlinearGradientCandidateDesignConfig(max_extra_replicates_per_state=4),
    )

    row = report["candidates"][0]
    assert report["passed"] is False
    assert report["next_action"].startswith("design a better-conditioned")
    assert row["action"] == "design_better_conditioned_control_or_variance_reduction"
    assert row["gate_status"] == {
        "response_ok": True,
        "locality_ok": True,
        "uncertainty_ok": False,
    }
    assert row["uncertainty_required_bracket_scale"] == pytest.approx(1.366)
    assert row["locality_safe_bracket_scale_limit"] == pytest.approx(1.0)
    assert row["bracket_only_feasible"] is False
    assert row["estimated_extra_replicates_at_locality_limit"] > 4


def test_candidate_design_distinguishes_bracket_ready_replicate_ready_and_invalid() -> (
    None
):
    bracket_ready = _artifact(response=0.08, asymmetry=0.20, uncertainty=0.60)
    replicate_ready = _artifact(response=0.08, asymmetry=0.48, uncertainty=0.54)
    promoted = _artifact(response=0.08, asymmetry=0.20, uncertainty=0.20, passed=True)
    report = nonlinear_gradient_candidate_design_report(
        [bracket_ready, replicate_ready, promoted],
        labels=["bracket", "replicate", "promoted"],
        config=NonlinearGradientCandidateDesignConfig(
            sem_safety_factor=1.0,
            max_extra_replicates_per_state=2,
        ),
    )

    assert [row["action"] for row in report["candidates"]] == [
        "run_checked_larger_bracket",
        "add_limited_replicates_with_locality_cap",
        "freeze_promoted_candidate",
    ]
    assert report["summary"]["bracket_ready_count"] == 1
    assert report["summary"]["replica_ready_count"] == 1
    assert report["summary"]["promoted_candidate_count"] == 1

    with pytest.raises(ValueError, match="max_checked_bracket_scale"):
        nonlinear_gradient_candidate_design_report(
            [bracket_ready],
            config=NonlinearGradientCandidateDesignConfig(
                max_checked_bracket_scale=0.9
            ),
        )
    with pytest.raises(ValueError, match="paths length"):
        nonlinear_gradient_candidate_design_report([bracket_ready], paths=[None, None])

    validation_cases = [
        (
            "max_gradient_uncertainty_rel",
            NonlinearGradientCandidateDesignConfig(max_gradient_uncertainty_rel=0.0),
        ),
        (
            "max_fd_asymmetry_rel",
            NonlinearGradientCandidateDesignConfig(max_fd_asymmetry_rel=0.0),
        ),
        (
            "max_window_mean_rel_spread",
            NonlinearGradientCandidateDesignConfig(max_window_mean_rel_spread=0.0),
        ),
        (
            "max_window_sem_rel",
            NonlinearGradientCandidateDesignConfig(max_window_sem_rel=0.0),
        ),
        (
            "min_fd_response_fraction",
            NonlinearGradientCandidateDesignConfig(min_fd_response_fraction=0.0),
        ),
        (
            "sem_safety_factor",
            NonlinearGradientCandidateDesignConfig(sem_safety_factor=0.0),
        ),
        (
            "max_extra_replicates_per_state",
            NonlinearGradientCandidateDesignConfig(max_extra_replicates_per_state=-1),
        ),
        (
            "locality_safety_factor",
            NonlinearGradientCandidateDesignConfig(locality_safety_factor=0.0),
        ),
    ]
    for message, config in validation_cases:
        with pytest.raises(ValueError, match=message):
            nonlinear_gradient_candidate_design_report([bracket_ready], config=config)

    with pytest.raises(ValueError, match="labels length"):
        nonlinear_gradient_candidate_design_report(
            [bracket_ready], labels=["one", "two"]
        )


def test_candidate_design_next_actions_and_metric_edge_cases() -> None:
    bracket_ready = _artifact(response=1, asymmetry=0.20, uncertainty=0.60)
    replicate_ready = _artifact(response=0.08, asymmetry=0.48, uncertainty=0.54)
    unresolved = _artifact(response=0.01, asymmetry=0.2, uncertainty=0.2)
    nonlocal_artifact = _artifact(response=0.08, asymmetry=0.7, uncertainty=0.2)
    inspect = _artifact(response=0.08, asymmetry=0.2, uncertainty=0.2, passed=False)

    assert nonlinear_gradient_candidate_design_report([bracket_ready])[
        "next_action"
    ].startswith("run a bounded")
    assert nonlinear_gradient_candidate_design_report([replicate_ready])[
        "next_action"
    ].startswith("combine")
    actions = nonlinear_gradient_candidate_design_report(
        [unresolved, nonlocal_artifact, inspect]
    )["candidates"]
    assert [row["action"] for row in actions] == [
        "increase_checked_bracket_or_replace_control",
        "shrink_or_replace_nonlocal_control",
        "inspect_pass_flag",
    ]

    zero_asymmetry = _artifact(response=0.08, asymmetry=0.0, uncertainty=0.6)
    missing_asymmetry = {
        "metrics": {"response_fraction": 0.08, "gradient_uncertainty_rel": 0.6}
    }
    edge_report = nonlinear_gradient_candidate_design_report(
        [zero_asymmetry, missing_asymmetry]
    )
    assert edge_report["candidates"][0]["locality_safe_bracket_scale_limit"] is None
    assert edge_report["candidates"][1]["usable_bracket_scale_for_estimate"] == 1.2
    assert nonlinear_gradient_candidate_design_report([])["next_action"].startswith(
        "inspect"
    )

    no_replicates = nonlinear_gradient_candidate_design_report(
        [
            {
                "metrics": {
                    "response_fraction": 0.08,
                    "fd_asymmetry_rel": 0.2,
                    "gradient_uncertainty_rel": 0.6,
                }
            }
        ]
    )
    assert (
        no_replicates["candidates"][0]["estimated_required_replicates_no_bracket"]
        is None
    )

    no_runs = nonlinear_gradient_followup_plan(
        [_artifact()],
        config=NonlinearGradientFollowupConfig(max_extra_replicates_per_state=0),
    )
    assert no_runs["candidate_actions"][0]["planned_run_count"] == 0


def test_candidate_design_identifies_limiting_spread_state_for_variance_reduction() -> (
    None
):
    artifact = _artifact(response=0.032, asymmetry=0.044, uncertainty=1.81)
    source_ensembles = artifact["source_ensembles"]
    assert isinstance(source_ensembles, dict)
    for state, spread in (("baseline", 0.108), ("plus", 0.196), ("minus", 0.057)):
        ensemble = source_ensembles[state]
        assert isinstance(ensemble, dict)
        ensemble["passed"] = spread <= 0.15
        ensemble["statistics"] = {
            "n_reports": 4,
            "mean_rel_spread": spread,
            "combined_sem_rel": 0.05,
        }

    report = nonlinear_gradient_candidate_design_report(
        [artifact], labels=["zbs10_rel7p5"]
    )
    row = report["candidates"][0]

    assert row["action"] == "design_variance_reduction_for_limiting_state"
    assert row["variance_reduction"]["limiting_state"] == "plus"
    assert row["variance_reduction"]["failed_spread_states"] == ["plus"]
    assert row["variance_reduction"]["max_mean_rel_spread"] == pytest.approx(0.196)
    assert "paired-seed" in row["recommendation"]
    assert report["next_action"].startswith("target paired-seed")


def test_variance_reduction_plan_quantifies_paired_seed_response() -> None:
    artifact = _artifact(response=0.032, asymmetry=0.044, uncertainty=1.81)
    source_ensembles = artifact["source_ensembles"]
    assert isinstance(source_ensembles, dict)
    values = {
        "baseline": {"seed31": 15.4, "seed32": 16.9, "seed33": 17.2, "dt0p04": 15.7},
        "plus": {"seed31": 18.1, "seed32": 15.6, "seed33": 15.5, "dt0p04": 14.9},
        "minus": {"seed31": 17.1, "seed32": 16.2, "seed33": 16.5, "dt0p04": 16.4},
    }
    for state, labeled_values in values.items():
        source_ensembles[state] = {
            "passed": state != "plus",
            "statistics": {
                "n_reports": 4,
                "mean_rel_spread": 0.196 if state == "plus" else 0.05,
                "combined_sem_rel": 0.05,
            },
            "rows": [
                {
                    "late_mean": value,
                    "source_artifact": f"{state}_nonlinear_t900_n64_{label}_heat_flux_trace.csv",
                }
                for label, value in labeled_values.items()
            ],
        }

    report = nonlinear_gradient_variance_reduction_plan(
        artifact,
        config=NonlinearGradientVarianceReductionConfig(max_extra_paired_seeds=1),
    )

    assert report["passed"] is False
    assert report["action"] == "estimate_control_mean_or_redesign_observable"
    assert report["summary"]["common_pair_count"] == 4
    assert report["summary"]["common_with_baseline_count"] == 4
    assert report["summary"]["paired_response_uncertainty_rel"] > 0.5
    assert (
        report["summary"]["best_control_variate"] == "plus_minus_midpoint_common_mode"
    )
    midpoint = report["control_variate_candidates"][1]
    assert midpoint["adjusted_response_uncertainty_rel"] < 0.5
    assert midpoint["control_sample_std"] > 0.0
    assert midpoint["adjusted_response_sample_std"] > 0.0
    assert "control_mean_not_independently_known" in midpoint["blockers"]
    assert report["variance_reduction"]["limiting_state"] == "plus"
    assert report["pair_rows"][0]["label"] == "dt0p04"

    allowed = nonlinear_gradient_variance_reduction_plan(
        artifact,
        config=NonlinearGradientVarianceReductionConfig(
            require_known_control_mean=False
        ),
    )
    assert allowed["action"] == "use_control_variate_response_estimator"
    assert allowed["passed"] is True

    validation_cases = [
        (
            "max_paired_response_uncertainty_rel",
            NonlinearGradientVarianceReductionConfig(
                max_paired_response_uncertainty_rel=0.0
            ),
        ),
        (
            "max_control_variate_uncertainty_rel",
            NonlinearGradientVarianceReductionConfig(
                max_control_variate_uncertainty_rel=0.0
            ),
        ),
        (
            "min_control_variate_sem_reduction",
            NonlinearGradientVarianceReductionConfig(
                min_control_variate_sem_reduction=-0.1
            ),
        ),
        (
            "sem_safety_factor",
            NonlinearGradientVarianceReductionConfig(sem_safety_factor=0.0),
        ),
        (
            "min_common_pairs",
            NonlinearGradientVarianceReductionConfig(min_common_pairs=0),
        ),
        (
            "max_extra_paired_seeds",
            NonlinearGradientVarianceReductionConfig(max_extra_paired_seeds=-1),
        ),
    ]
    for message, config in validation_cases:
        with pytest.raises(ValueError, match=message):
            nonlinear_gradient_variance_reduction_plan(artifact, config=config)


def test_control_variate_campaign_plan_requires_independent_control_mean() -> None:
    artifact = json.loads(
        (
            ROOT
            / "docs"
            / "_static"
            / "qa_ess_zbs10_rel7p5_nonlinear_gradient_zbs_1_0_central_fd_gradient_gate.json"
        ).read_text(encoding="utf-8")
    )
    variance = nonlinear_gradient_variance_reduction_plan(
        artifact,
        case="qa_ess_zbs10_rel7p5_variance_reduction_plan",
    )

    plan = nonlinear_gradient_control_variate_campaign_plan(
        variance,
        case="qa_ess_zbs10_rel7p5_control_variate_campaign_plan",
    )

    assert plan["passed"] is True
    assert plan["action"] == "launch_independent_control_mean_campaign"
    assert plan["candidate_name"] == "plus_minus_midpoint_common_mode"
    assert plan["summary"]["required_independent_control_mean_pairs"] == 21
    assert plan["summary"]["planned_new_run_count"] == 42
    assert plan["summary"]["predicted_combined_uncertainty_rel"] <= 0.5
    assert plan["planned_pairs"][0]["variant_label"] == "seed34"
    assert plan["postprocess_contract"]["promotion_rule"].startswith("do not promote")

    blocked = nonlinear_gradient_control_variate_campaign_plan(
        variance,
        config=NonlinearGradientControlVariateCampaignConfig(max_control_mean_pairs=8),
    )
    assert blocked["passed"] is False
    assert blocked["action"] == "redesign_observable_or_raise_control_mean_budget"
    assert "control_mean_pair_budget_exceeded" in blocked["blockers"]

    validation_cases = [
        (
            "target_response_uncertainty_rel",
            NonlinearGradientControlVariateCampaignConfig(
                target_response_uncertainty_rel=0.0
            ),
        ),
        (
            "sem_safety_factor",
            NonlinearGradientControlVariateCampaignConfig(sem_safety_factor=0.0),
        ),
        (
            "min_control_mean_pairs",
            NonlinearGradientControlVariateCampaignConfig(min_control_mean_pairs=0),
        ),
        (
            "max_control_mean_pairs",
            NonlinearGradientControlVariateCampaignConfig(
                min_control_mean_pairs=4, max_control_mean_pairs=3
            ),
        ),
        (
            "first_new_seed",
            NonlinearGradientControlVariateCampaignConfig(first_new_seed=-1),
        ),
    ]
    for message, config in validation_cases:
        with pytest.raises(ValueError, match=message):
            nonlinear_gradient_control_variate_campaign_plan(variance, config=config)


def _control_ensemble(state: str, *, passed: bool = True) -> dict[str, object]:
    rows = []
    for idx, seed in enumerate(range(34, 55)):
        control = 16.28 + 0.02 * ((idx % 3) - 1)
        response = -0.52 + 0.01 * ((idx % 5) - 2)
        if state == "plus":
            value = control + 0.5 * response
        else:
            value = control - 0.5 * response
        rows.append(
            {
                "late_mean": value,
                "source_artifact": f"{state}_seed{seed}_heat_flux_trace.csv",
                "summary_artifact": f"{state}_seed{seed}_transport_window.json",
                "variant": {"seed": seed, "timestep": 0.05},
            }
        )
    return {
        "kind": "nonlinear_window_ensemble_report",
        "passed": passed,
        "n_reports": len(rows),
        "rows": rows,
        "statistics": {
            "n_reports": len(rows),
            "ensemble_mean": sum(float(row["late_mean"]) for row in rows) / len(rows),
            "combined_sem": 0.02,
            "combined_sem_rel": 0.001,
            "mean_rel_spread": 0.002,
        },
    }


def test_control_mean_gate_combines_independent_control_uncertainty() -> None:
    artifact = json.loads(
        (
            ROOT
            / "docs"
            / "_static"
            / "qa_ess_zbs10_rel7p5_nonlinear_gradient_zbs_1_0_central_fd_gradient_gate.json"
        ).read_text(encoding="utf-8")
    )
    variance = nonlinear_gradient_variance_reduction_plan(artifact)

    gate = nonlinear_gradient_control_mean_gate(
        variance,
        plus_ensemble=_control_ensemble("plus"),
        minus_ensemble=_control_ensemble("minus"),
        plus_path="plus.json",
        minus_path="minus.json",
    )

    assert gate["passed"] is True
    assert gate["candidate_name"] == "plus_minus_midpoint_common_mode"
    assert gate["summary"]["common_pair_count"] == 21
    assert gate["summary"]["combined_response_uncertainty_rel"] < 0.5
    assert gate["pair_rows"][0]["label"] == "seed34"

    blocked = nonlinear_gradient_control_mean_gate(
        variance,
        plus_ensemble=_control_ensemble("plus", passed=False),
        minus_ensemble=_control_ensemble("minus"),
    )
    assert blocked["passed"] is False
    assert "plus_control_ensemble_failed" in blocked["blockers"]

    validation_cases = [
        (
            "target_response_uncertainty_rel",
            NonlinearGradientControlMeanGateConfig(target_response_uncertainty_rel=0.0),
        ),
        (
            "min_control_mean_pairs",
            NonlinearGradientControlMeanGateConfig(min_control_mean_pairs=0),
        ),
    ]
    for message, config in validation_cases:
        with pytest.raises(ValueError, match=message):
            nonlinear_gradient_control_mean_gate(
                variance,
                plus_ensemble=_control_ensemble("plus"),
                minus_ensemble=_control_ensemble("minus"),
                config=config,
            )


def test_variance_reduction_plan_tool_writes_artifacts(tmp_path: Path) -> None:
    module = load_artifact_tool("build_nonlinear_gradient_evidence")

    artifact = tmp_path / "candidate.json"
    payload = _artifact(response=0.032, asymmetry=0.044, uncertainty=1.81)
    source_ensembles = payload["source_ensembles"]
    assert isinstance(source_ensembles, dict)
    for state in ("baseline", "plus", "minus"):
        ensemble = source_ensembles[state]
        assert isinstance(ensemble, dict)
        ensemble["statistics"] = {
            "n_reports": 3,
            "mean_rel_spread": 0.18 if state == "plus" else 0.04,
            "combined_sem_rel": 0.05,
        }
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    out_prefix = tmp_path / "variance_plan"

    assert (
        module.main(["variance-plan", str(artifact), "--out-prefix", str(out_prefix)])
        == 0
    )
    report = json.loads(out_prefix.with_suffix(".json").read_text(encoding="utf-8"))
    assert report["kind"] == "nonlinear_turbulence_gradient_variance_reduction_plan"
    assert out_prefix.with_suffix(".csv").exists()
    assert out_prefix.with_suffix(".png").exists()
    assert out_prefix.with_suffix(".pdf").exists()


def test_control_variate_campaign_plan_tool_writes_artifacts(tmp_path: Path) -> None:
    module = load_campaign_tool("design_nonlinear_gradient")

    out_prefix = tmp_path / "cv_campaign"
    source = (
        ROOT / "docs" / "_static" / "qa_ess_zbs10_rel7p5_variance_reduction_plan.json"
    )
    assert (
        module.main(
            ["control-variate-campaign", str(source), "--out-prefix", str(out_prefix)]
        )
        == 0
    )

    report = json.loads(out_prefix.with_suffix(".json").read_text(encoding="utf-8"))
    rows = out_prefix.with_suffix(".csv").read_text(encoding="utf-8").splitlines()
    assert (
        report["kind"] == "nonlinear_turbulence_gradient_control_variate_campaign_plan"
    )
    assert report["action"] == "launch_independent_control_mean_campaign"
    assert rows[0].startswith("pair_index,variant_label")
    assert out_prefix.with_suffix(".png").exists()
    assert out_prefix.with_suffix(".pdf").exists()


def test_control_mean_gate_matches_seed_from_artifact_basename() -> None:
    artifact = json.loads(
        (
            ROOT
            / "docs"
            / "_static"
            / "qa_ess_zbs10_rel7p5_nonlinear_gradient_zbs_1_0_central_fd_gradient_gate.json"
        ).read_text(encoding="utf-8")
    )
    variance = nonlinear_gradient_variance_reduction_plan(artifact)
    plus = _control_ensemble("plus")
    minus = _control_ensemble("minus")
    for ensemble in (plus, minus):
        for row in ensemble["rows"]:
            row.pop("variant", None)
            row["source_artifact"] = f"/tmp/interim_seed34_42/{row['source_artifact']}"
            row["summary_artifact"] = (
                f"/tmp/interim_seed34_42/{row['summary_artifact']}"
            )

    gate = nonlinear_gradient_control_mean_gate(
        variance,
        plus_ensemble=plus,
        minus_ensemble=minus,
    )

    assert gate["passed"] is True
    assert gate["summary"]["common_pair_count"] == 21
    assert gate["pair_rows"][1]["label"] == "seed35"


def test_control_mean_gate_tool_writes_artifacts(tmp_path: Path) -> None:
    module = load_artifact_tool("build_nonlinear_gradient_evidence")

    plus = tmp_path / "plus.json"
    minus = tmp_path / "minus.json"
    plus.write_text(json.dumps(_control_ensemble("plus")), encoding="utf-8")
    minus.write_text(json.dumps(_control_ensemble("minus")), encoding="utf-8")
    source = (
        ROOT / "docs" / "_static" / "qa_ess_zbs10_rel7p5_variance_reduction_plan.json"
    )
    out_prefix = tmp_path / "control_mean_gate"

    assert (
        module.main(
            [
                "control-mean",
                "--variance-report",
                str(source),
                "--plus-ensemble",
                str(plus),
                "--minus-ensemble",
                str(minus),
                "--out-prefix",
                str(out_prefix),
            ]
        )
        == 0
    )

    report = json.loads(out_prefix.with_suffix(".json").read_text(encoding="utf-8"))
    assert report["kind"] == "nonlinear_turbulence_gradient_control_mean_gate"
    assert report["passed"] is True
    assert out_prefix.with_suffix(".csv").exists()
    assert out_prefix.with_suffix(".png").exists()
    assert out_prefix.with_suffix(".pdf").exists()


def test_design_nonlinear_gradient_next_campaign_subcommand_writes_artifacts(
    tmp_path: Path,
) -> None:
    module = load_campaign_tool("design_nonlinear_gradient")

    artifact = tmp_path / "candidate.json"
    out_prefix = tmp_path / "design"
    artifact.write_text(
        json.dumps(_artifact(response=0.072, asymmetry=0.475, uncertainty=0.683)),
        encoding="utf-8",
    )

    assert (
        module.main(["next-campaign", str(artifact), "--out-prefix", str(out_prefix)])
        == 0
    )
    payload = json.loads(out_prefix.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["kind"] == "nonlinear_turbulence_gradient_candidate_design_report"
    assert payload["summary"]["candidate_count"] == 1
    csv_bytes = out_prefix.with_suffix(".csv").read_bytes()
    assert b"\r" not in csv_bytes
    assert out_prefix.with_suffix(".png").exists()
    assert out_prefix.with_suffix(".pdf").exists()


def test_composite_control_report_builds_descent_direction_and_blocks_bad_controls() -> (
    None
):
    rbc = _artifact(response=0.072, asymmetry=0.475, uncertainty=0.683)
    rbc["parameter_name"] = "rbc_1_1"
    rbc["metrics"]["central_gradient"] = -186.95
    rbc["paired_replicate_diagnostics"] = {
        "central_gradient_uncertainty_rel": 0.247,
        "same_sign_fraction": 1.0,
    }
    zbs = _artifact(response=0.095, asymmetry=0.605, uncertainty=0.355)
    zbs["parameter_name"] = "zbs_1_1"
    zbs["metrics"]["central_gradient"] = 270.7
    zbs["paired_replicate_diagnostics"] = {"same_sign_fraction": 1.0}

    report = nonlinear_gradient_composite_control_report(
        [rbc, zbs], labels=["rbc", "zbs"]
    )

    assert report["passed"] is False
    assert report["summary"]["admissible_control_count"] == 1
    assert report["controls"][0]["coefficient"] == "RBC(1,1)"
    assert report["controls"][0]["weight"] == pytest.approx(1.0)
    assert "single-control bracket" in report["next_action"]
    assert report["candidates"][1]["blockers"] == ["nonlocal_finite_difference_bracket"]

    two_control = _artifact(response=0.12, asymmetry=0.2, uncertainty=0.3)
    two_control["parameter_name"] = "zbs_1_1"
    two_control["metrics"]["central_gradient"] = 93.475
    ready = nonlinear_gradient_composite_control_report([rbc, two_control])
    assert ready["passed"] is True
    assert ready["controls"][0]["control_argument"] == "RBC(1,1):1"
    assert ready["controls"][1]["control_argument"] == "ZBS(1,1):-0.5"
    assert (
        "write_vmec_boundary_profile_perturbation_inputs.py"
        in ready["write_profile_direction_command_template"]
    )


def test_composite_control_report_validates_config_and_metadata() -> None:
    artifact = _artifact(response=0.08, asymmetry=0.2, uncertainty=0.2)
    artifact["parameter_name"] = "not_a_vmec_coefficient"
    artifact["metrics"]["central_gradient"] = 1.0
    artifact["paired_replicate_diagnostics"] = {"same_sign_fraction": 0.5}

    report = nonlinear_gradient_composite_control_report([artifact])
    assert report["controls"] == []
    assert (
        "parameter_not_vmec_boundary_coefficient" in report["candidates"][0]["blockers"]
    )
    assert "paired_replicate_sign_not_robust" in report["candidates"][0]["blockers"]

    validation_cases = [
        (
            "max_gradient_uncertainty_rel",
            NonlinearGradientCompositeControlConfig(max_gradient_uncertainty_rel=0.0),
        ),
        (
            "max_fd_asymmetry_rel",
            NonlinearGradientCompositeControlConfig(max_fd_asymmetry_rel=0.0),
        ),
        (
            "min_fd_response_fraction",
            NonlinearGradientCompositeControlConfig(min_fd_response_fraction=0.0),
        ),
        (
            "min_same_sign_fraction",
            NonlinearGradientCompositeControlConfig(min_same_sign_fraction=0.0),
        ),
        (
            "min_same_sign_fraction",
            NonlinearGradientCompositeControlConfig(min_same_sign_fraction=1.1),
        ),
        ("min_controls", NonlinearGradientCompositeControlConfig(min_controls=0)),
        (
            "default_relative_delta",
            NonlinearGradientCompositeControlConfig(default_relative_delta=0.0),
        ),
        ("max_weight_abs", NonlinearGradientCompositeControlConfig(max_weight_abs=0.0)),
    ]
    for message, config in validation_cases:
        with pytest.raises(ValueError, match=message):
            nonlinear_gradient_composite_control_report([artifact], config=config)

    with pytest.raises(ValueError, match="paths length"):
        nonlinear_gradient_composite_control_report([artifact], paths=[None, None])
    with pytest.raises(ValueError, match="labels length"):
        nonlinear_gradient_composite_control_report([artifact], labels=["one", "two"])


def test_design_nonlinear_gradient_composite_control_subcommand_writes_artifacts(
    tmp_path: Path,
) -> None:
    module = load_campaign_tool("design_nonlinear_gradient")

    artifact = _artifact(response=0.072, asymmetry=0.475, uncertainty=0.683)
    artifact["parameter_name"] = "rbc_1_1"
    artifact["metrics"]["central_gradient"] = -186.95
    candidate_path = tmp_path / "candidate.json"
    out_prefix = tmp_path / "composite"
    candidate_path.write_text(json.dumps(artifact), encoding="utf-8")

    assert (
        module.main(
            [
                "composite-control",
                str(candidate_path),
                "--out-prefix",
                str(out_prefix),
                "--min-controls",
                "1",
            ]
        )
        == 0
    )
    payload = json.loads(out_prefix.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["kind"] == "nonlinear_turbulence_gradient_composite_control_design"
    assert payload["passed"] is True
    assert payload["controls"][0]["control_argument"] == "RBC(1,1):1"
    assert out_prefix.with_suffix(".csv").exists()
    assert out_prefix.with_suffix(".png").exists()
    assert out_prefix.with_suffix(".pdf").exists()
