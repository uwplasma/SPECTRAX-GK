from __future__ import annotations

import json

import spectraxgk
import spectraxgk.validation.stellarator.transport_audit as transport_audit
import spectraxgk.validation.stellarator.transport_campaign as transport_campaign
import spectraxgk.validation.stellarator.transport_landscape as transport_landscape
import spectraxgk.validation.stellarator.transport_prelaunch as transport_prelaunch
from spectraxgk.validation.stellarator.transport_audit import (
    build_nonlinear_audit_redesign_report,
)
from spectraxgk.validation.stellarator.transport_campaign import (
    build_nonlinear_campaign_admission_report,
)
from spectraxgk.validation.stellarator.transport_landscape import (
    build_nonlinear_landscape_admission_report,
)
from spectraxgk.validation.stellarator.transport_policies import (
    VMECJAXNonlinearAuditPolicy,
    VMECJAXNonlinearCampaignPolicy,
    VMECJAXReducedPrelaunchPolicy,
    VMECJAXTransportAdmissionPolicy,
)
from spectraxgk.validation.stellarator.transport_prelaunch import (
    build_reduced_nonlinear_audit_prelaunch_report,
)
from spectraxgk.validation.stellarator.transport_samples import (
    candidate_transport_metric,
    transport_objective_sample_summary,
)
from spectraxgk.validation.stellarator.transport_selection import (
    build_transport_admission_report,
    select_admitted_transport_candidate,
)


def _candidate(
    label: str,
    *,
    objective: float,
    weight: float | None = None,
    passed: bool = True,
    authoritative: bool = True,
    baseline: bool = False,
) -> dict[str, object]:
    return {
        "label": label,
        "baseline": baseline,
        "transport_weight": weight,
        "passed": passed and authoritative,
        "gate_reported_passed": passed,
        "gate_is_authoritative": authoritative,
        "gate_checks": {
            "aspect": passed,
            "mean_iota": True,
            "quasisymmetry": passed,
            "iota_profile": passed,
        },
        "objective_final": objective,
    }


def test_transport_metric_prefers_explicit_transport_metric_over_total_objective() -> None:
    metric = candidate_transport_metric(
        {
            "objective_final": 4.0,
            "spectrax_objective_final": 2.0,
            "transport_objective_final": 1.0,
        }
    )

    assert metric["available"] is True
    assert metric["source"] == "transport_objective_final"
    assert metric["value"] == 1.0
    assert metric["uses_total_objective_proxy"] is False


def test_transport_admission_selects_largest_physical_improving_weight() -> None:
    summaries = [
        _candidate("baseline", objective=1.0, baseline=True),
        _candidate("low", objective=0.8, weight=0.001),
        _candidate("high", objective=0.7, weight=0.005),
        _candidate("failed", objective=0.1, weight=0.01, passed=False),
    ]

    report = build_transport_admission_report(summaries)

    assert report["transport_candidate_admitted"] is True
    assert report["promoted_candidate"]["label"] == "high"
    assert report["promoted_candidate"]["transport_weight"] == 0.005
    assert report["admitted_transport_candidates"] == ["low", "high"]


def test_transport_admission_blocks_worse_transport_metric_even_if_gate_passes() -> None:
    summaries = [
        _candidate("baseline", objective=1.0, baseline=True),
        _candidate("worse", objective=1.1, weight=0.001),
    ]

    report = build_transport_admission_report(summaries)
    worse = report["candidates"][1]

    assert report["transport_candidate_admitted"] is False
    assert report["promoted_candidate"]["label"] == "baseline"
    assert worse["relative_transport_improvement"] < 0.0
    assert "insufficient_transport_improvement" in worse["admission_blockers"]


def test_transport_admission_blocks_non_authoritative_gate() -> None:
    summaries = [
        _candidate("baseline", objective=1.0, baseline=True),
        _candidate("legacy", objective=0.5, weight=0.001, authoritative=False),
    ]

    report = build_transport_admission_report(summaries)
    legacy = report["candidates"][1]

    assert report["transport_candidate_admitted"] is False
    assert "non_authoritative_gate" in legacy["admission_blockers"]
    assert report["promoted_candidate"]["label"] == "baseline"


def test_transport_admission_can_require_stronger_relative_improvement() -> None:
    policy = VMECJAXTransportAdmissionPolicy(minimum_relative_improvement=0.25)
    summaries = [
        _candidate("baseline", objective=1.0, baseline=True),
        _candidate("small", objective=0.9, weight=0.001),
        _candidate("large", objective=0.7, weight=0.002),
    ]

    report = build_transport_admission_report(summaries, policy=policy)

    assert report["admitted_transport_candidates"] == ["large"]
    assert report["promoted_candidate"]["label"] == "large"


def test_transport_admission_exports_public_api() -> None:
    assert spectraxgk.VMECJAXTransportAdmissionPolicy is VMECJAXTransportAdmissionPolicy
    assert spectraxgk.build_transport_admission_report is build_transport_admission_report
    assert spectraxgk.candidate_transport_metric is candidate_transport_metric
    assert spectraxgk.select_admitted_transport_candidate is select_admitted_transport_candidate


def _matched_comparison(
    *,
    relative_reduction: float,
    z_score: float,
    passed: bool,
) -> dict[str, object]:
    return {
        "kind": "matched_nonlinear_transport_comparison",
        "case": "qa_projected_transport_step1e3",
        "passed": passed,
        "baseline": {"passed": True, "ensemble_mean": 9.833},
        "candidate": {"passed": True, "ensemble_mean": 9.891},
        "statistics": {
            "relative_reduction": relative_reduction,
            "uncertainty_z_score": z_score,
        },
    }


def _ensemble(mean: float, sem: float, *, passed: bool = True, n_reports: int = 3) -> dict[str, object]:
    return {
        "case": f"ensemble_mean_{mean}",
        "passed": passed,
        "statistics": {
            "ensemble_mean": mean,
            "combined_sem": sem,
            "combined_sem_rel": sem / abs(mean),
            "n_reports": n_reports,
        },
    }


def test_nonlinear_landscape_admission_selects_uncertainty_resolved_candidate() -> None:
    report = build_nonlinear_landscape_admission_report(
        _ensemble(8.554362366164424, 0.11951503416978174),
        [
            _ensemble(6.274543846475065, 0.04213243251063571),
            _ensemble(6.42653555490751, 0.04399590111876854),
        ],
        candidate_labels=("+3%", "+6%"),
        policy=VMECJAXNonlinearAuditPolicy(
            minimum_relative_reduction=0.02,
            minimum_uncertainty_z_score=2.0,
            maximum_combined_sem_rel=0.05,
            minimum_replicate_count=3,
        ),
    )

    assert report["passed"] is True
    assert report["selected_candidate"]["label"] == "+3%"
    assert report["selected_candidate"]["relative_reduction"] > 0.26
    assert report["selected_candidate"]["uncertainty_z_score"] > 17.0
    assert all(row["admitted"] for row in report["candidates"])
    assert spectraxgk.build_nonlinear_landscape_admission_report is build_nonlinear_landscape_admission_report
    assert (
        build_nonlinear_landscape_admission_report
        is transport_landscape.build_nonlinear_landscape_admission_report
    )
    json.dumps(report, allow_nan=False)


def test_nonlinear_landscape_admission_fails_closed_for_noisy_or_unresolved_candidates() -> None:
    report = build_nonlinear_landscape_admission_report(
        _ensemble(8.0, 0.5),
        [
            _ensemble(7.95, 0.5),
            _ensemble(6.0, 2.0, n_reports=2),
            _ensemble(5.0, 0.1, passed=False),
        ],
        policy=VMECJAXNonlinearAuditPolicy(
            minimum_relative_reduction=0.02,
            minimum_uncertainty_z_score=2.0,
            maximum_combined_sem_rel=0.2,
            minimum_replicate_count=3,
        ),
    )

    assert report["passed"] is False
    assert report["selected_candidate"] is None
    blockers = [set(row["admission_blockers"]) for row in report["candidates"]]
    assert "insufficient_relative_reduction" in blockers[0]
    assert "insufficient_uncertainty_separation" in blockers[0]
    assert "candidate_combined_sem_rel_too_large" in blockers[1]
    assert "candidate_insufficient_replicates" in blockers[1]
    assert "candidate_ensemble_failed" in blockers[2]


def test_nonlinear_landscape_admission_validates_candidate_labels() -> None:
    try:
        build_nonlinear_landscape_admission_report(
            _ensemble(8.0, 0.1),
            [_ensemble(7.0, 0.1)],
            candidate_labels=("one", "two"),
        )
    except ValueError as exc:
        assert "same length" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("mismatched candidate labels were accepted")


def test_reduced_nonlinear_audit_prelaunch_passes_calibrated_landscape_margin() -> None:
    baseline = 0.06558065223919245
    candidate = 0.06251277500404685

    report = build_reduced_nonlinear_audit_prelaunch_report(
        baseline_metric=baseline,
        candidate_metric=candidate,
        objective_sample_set={
            "surfaces": [0.45, 0.64, 0.78],
            "alphas": [0.0, 0.7853981633974483],
            "ky_values": [0.1, 0.3, 0.5],
        },
        baseline_sample_statistics={
            "weighted_mean": 0.06777885259618041,
            "weighted_standard_error": 0.015344998342625694,
        },
        candidate_sample_statistics={
            "weighted_mean": 0.06450805792574345,
            "weighted_standard_error": 0.014457225619392737,
        },
        failed_reference_relative_reduction=0.022876,
        policy=VMECJAXReducedPrelaunchPolicy(minimum_relative_reduction=0.04),
    )

    assert report["passed"] is True
    assert report["relative_reduced_reduction"] > 0.046
    assert report["required_relative_reduced_reduction"] == 0.04
    assert report["blockers"] == []
    assert report["gates"][0]["passed"] is True
    assert report["reduced_cross_sample_statistics"]["passed"] is True
    assert report["gates"][2]["metric"] == "reduced_cross_sample_dispersion"
    assert spectraxgk.VMECJAXReducedPrelaunchPolicy is VMECJAXReducedPrelaunchPolicy
    assert (
        spectraxgk.build_reduced_nonlinear_audit_prelaunch_report
        is build_reduced_nonlinear_audit_prelaunch_report
    )
    assert (
        build_reduced_nonlinear_audit_prelaunch_report
        is transport_prelaunch.build_reduced_nonlinear_audit_prelaunch_report
    )


def test_reduced_nonlinear_audit_prelaunch_blocks_weak_failed_transfer_margin() -> None:
    report = build_reduced_nonlinear_audit_prelaunch_report(
        baseline_metric=0.08010670290,
        candidate_metric=0.07827418221,
        objective_sample_set={
            "surfaces": [0.45, 0.64, 0.78],
            "alphas": [0.0, 0.7853981633974483],
            "ky_values": [0.1, 0.3, 0.5],
        },
        failed_reference_relative_reduction=0.022876,
        policy=VMECJAXReducedPrelaunchPolicy(
            minimum_relative_reduction=0.04,
            failed_reference_safety_factor=1.5,
        ),
    )

    assert report["passed"] is False
    assert "insufficient_reduced_margin_for_nonlinear_audit" in report["blockers"]
    assert report["relative_reduced_reduction"] < report["required_relative_reduced_reduction"]


def test_reduced_prelaunch_blocks_excessive_reduced_cross_sample_spread() -> None:
    report = build_reduced_nonlinear_audit_prelaunch_report(
        baseline_metric=0.06558065223919245,
        candidate_metric=0.06251277500404685,
        objective_sample_set={
            "surfaces": [0.45, 0.64, 0.78],
            "alphas": [0.0, 0.7853981633974483],
            "ky_values": [0.1, 0.3, 0.5],
        },
        baseline_sample_statistics={
            "weighted_mean": 0.067,
            "weighted_standard_error": 0.03,
        },
        candidate_sample_statistics={
            "weighted_mean": 0.064,
            "weighted_standard_error": 0.04,
        },
        policy=VMECJAXReducedPrelaunchPolicy(
            minimum_relative_reduction=0.04,
            maximum_cross_sample_sem_rel=0.35,
        ),
    )

    assert report["passed"] is False
    assert "candidate_cross_sample_sem_rel_too_large" in report["blockers"]
    assert report["gates"][2]["passed"] is False


def test_campaign_admission_combines_reduced_and_replicated_landscape_gates() -> None:
    prelaunch = build_reduced_nonlinear_audit_prelaunch_report(
        baseline_metric=0.06558065223919245,
        candidate_metric=0.06251277500404685,
        objective_sample_set={
            "surfaces": [0.45, 0.64, 0.78],
            "alphas": [0.0, 0.7853981633974483],
            "ky_values": [0.1, 0.3, 0.5],
        },
        baseline_sample_statistics={
            "weighted_mean": 0.06777885259618041,
            "weighted_standard_error": 0.015344998342625694,
        },
        candidate_sample_statistics={
            "weighted_mean": 0.06450805792574345,
            "weighted_standard_error": 0.014457225619392737,
        },
        policy=VMECJAXReducedPrelaunchPolicy(minimum_relative_reduction=0.04),
    )
    landscape = build_nonlinear_landscape_admission_report(
        _ensemble(8.554362366164424, 0.11951503416978174),
        [_ensemble(6.274543846475065, 0.04213243251063571)],
        candidate_labels=("+3% RBC(0,1)",),
        policy=VMECJAXNonlinearAuditPolicy(
            minimum_relative_reduction=0.02,
            minimum_uncertainty_z_score=2.0,
            maximum_combined_sem_rel=0.05,
            minimum_replicate_count=3,
        ),
    )

    report = build_nonlinear_campaign_admission_report(
        reduced_prelaunch_report=prelaunch,
        landscape_admission_report=landscape,
    )

    assert report["campaign_admitted"] is True
    assert report["blockers"] == []
    assert report["selected_landscape_candidate"]["label"] == "+3% RBC(0,1)"
    assert report["claim_scope"].startswith("next nonlinear optimizer-campaign admission")
    assert spectraxgk.VMECJAXNonlinearCampaignPolicy is VMECJAXNonlinearCampaignPolicy
    assert (
        spectraxgk.build_nonlinear_campaign_admission_report
        is build_nonlinear_campaign_admission_report
    )
    assert (
        build_nonlinear_campaign_admission_report
        is transport_campaign.build_nonlinear_campaign_admission_report
    )
    json.dumps(report, allow_nan=False)


def test_campaign_admission_fails_closed_without_cross_sample_gate_or_landscape_margin() -> None:
    prelaunch = build_reduced_nonlinear_audit_prelaunch_report(
        baseline_metric=1.0,
        candidate_metric=0.95,
        objective_sample_set={
            "surfaces": [0.45, 0.64, 0.78],
            "alphas": [0.0, 0.7853981633974483],
            "ky_values": [0.1, 0.3, 0.5],
        },
        policy=VMECJAXReducedPrelaunchPolicy(minimum_relative_reduction=0.04),
    )
    landscape = build_nonlinear_landscape_admission_report(
        _ensemble(8.0, 0.3),
        [_ensemble(7.4, 0.3)],
        candidate_labels=("weak",),
        policy=VMECJAXNonlinearAuditPolicy(minimum_relative_reduction=0.02),
    )

    report = build_nonlinear_campaign_admission_report(
        reduced_prelaunch_report=prelaunch,
        landscape_admission_report=landscape,
        policy=VMECJAXNonlinearCampaignPolicy(
            minimum_landscape_relative_reduction=0.10,
            minimum_landscape_uncertainty_z_score=3.0,
        ),
    )

    assert report["campaign_admitted"] is False
    assert "reduced_cross_sample_statistics_missing" in report["blockers"]
    assert "selected_landscape_reduction_too_small" in report["blockers"]
    assert "selected_landscape_uncertainty_separation_too_small" in report["blockers"]


def test_transport_sample_summary_requires_surface_alpha_and_ky_coverage() -> None:
    summary = transport_objective_sample_summary({"surfaces": [0.5], "alphas": [0.0], "ky_values": [0.3]})

    assert summary["passed"] is False
    assert summary["sample_count"] == 1
    assert "insufficient_surface_coverage" in summary["blockers"]
    assert "insufficient_field_line_coverage" in summary["blockers"]
    assert "insufficient_ky_coverage" in summary["blockers"]


def test_nonlinear_audit_redesign_blocks_negative_transfer_and_recommends_multisample_design() -> None:
    report = build_nonlinear_audit_redesign_report(
        _matched_comparison(relative_reduction=-0.00585, z_score=-0.20, passed=False),
        objective_sample_set={"surfaces": [0.64], "alphas": [0.0], "ky_values": [0.3]},
    )

    assert report["nonlinear_audit_promoted"] is False
    assert report["requires_objective_redesign"] is True
    assert "insufficient_matched_reduction" in report["blockers"]
    assert "insufficient_uncertainty_separation" in report["blockers"]
    assert "insufficient_total_sample_count" in report["blockers"]
    assert report["recommended_sample_set"]["sample_count"] == 18
    assert report["gates"][0]["passed"] is False
    json.dumps(report, allow_nan=False)


def test_nonlinear_audit_redesign_promotes_only_when_audit_and_sample_coverage_pass() -> None:
    policy = VMECJAXNonlinearAuditPolicy(
        minimum_relative_reduction=0.02,
        minimum_uncertainty_z_score=1.0,
        minimum_surface_count=3,
        minimum_alpha_count=2,
        minimum_ky_count=3,
        minimum_sample_count=12,
    )
    sample_set = {
        "surfaces": [0.45, 0.64, 0.78],
        "alphas": [0.0, 0.7853981633974483],
        "ky_values": [0.1, 0.3, 0.5],
    }

    report = build_nonlinear_audit_redesign_report(
        _matched_comparison(relative_reduction=0.08, z_score=2.5, passed=True),
        objective_sample_set=sample_set,
        policy=policy,
    )

    assert report["nonlinear_audit_promoted"] is True
    assert report["requires_objective_redesign"] is False
    assert report["blockers"] == []
    assert report["objective_sample_summary"]["sample_count"] == 18
    assert all(gate["passed"] for gate in report["gates"])
    assert spectraxgk.VMECJAXNonlinearAuditPolicy is VMECJAXNonlinearAuditPolicy
    assert spectraxgk.build_nonlinear_audit_redesign_report is build_nonlinear_audit_redesign_report
    assert (
        build_nonlinear_audit_redesign_report
        is transport_audit.build_nonlinear_audit_redesign_report
    )
    assert spectraxgk.transport_objective_sample_summary is transport_objective_sample_summary


def test_transport_sample_summary_rejects_ky_values_not_supported_by_single_solver_grid() -> None:
    summary = transport_objective_sample_summary(
        {
            "surfaces": [0.45, 0.64, 0.78],
            "alphas": [0.0, 0.7853981633974483],
            "ky_values": [0.19, 0.3, 0.476],
        }
    )

    assert summary["passed"] is False
    assert "ky_values_not_single_grid_compatible" in summary["blockers"]
