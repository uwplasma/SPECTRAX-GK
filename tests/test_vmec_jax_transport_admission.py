from __future__ import annotations

import json

import spectraxgk
from spectraxgk.vmec_jax_transport_admission import (
    VMECJAXNonlinearAuditPolicy,
    VMECJAXTransportAdmissionPolicy,
    build_nonlinear_audit_redesign_report,
    build_transport_admission_report,
    candidate_transport_metric,
    select_admitted_transport_candidate,
    transport_objective_sample_summary,
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
        "ky_values": [0.19, 0.3, 0.476],
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
    assert spectraxgk.transport_objective_sample_summary is transport_objective_sample_summary
