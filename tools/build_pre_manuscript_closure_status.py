#!/usr/bin/env python3
"""Build a strict pre-manuscript closure dashboard.

This artifact is stricter than the release/readiness dashboards. It tracks the
four lanes that must close before manuscript drafting starts:

* universal absolute quasilinear heat-flux prediction,
* broad end-to-end nonlinear turbulent-flux stellarator optimization,
* production nonlinear domain-decomposition speedup, and
* VMEC/Boozer held-out optimization promotion.

Release-safe scoped diagnostics can be green while this dashboard remains open.
That fail-closed split prevents reduced, startup, or single-candidate evidence
from being promoted into broader manuscript claims.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import textwrap
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from spectraxgk.plotting import set_plot_style  # type: ignore[import-untyped]  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "docs" / "_static" / "pre_manuscript_closure_status.png"

STATUS_COLORS = {
    "closed": "#2a9d8f",
    "partial": "#e9c46a",
    "open": "#f4a261",
    "blocked": "#d1495b",
}
STATUS_ORDER = {"closed": 0, "partial": 1, "open": 2, "blocked": 3}

QL_ABSOLUTE_ERROR_GATE = 0.35
QL_CANDIDATE_SOFT_ERROR_GATE = 0.50
MIN_BROAD_MATCHED_OPTIMIZATION_AUDITS = 3
MIN_BROAD_OPTIMIZED_EQUILIBRIUM_ENSEMBLES = 3
MIN_BROAD_REPLICATED_HOLDOUT_ENSEMBLES = 4
MIN_DOMAIN_CPU_SPEEDUP = 1.5
MIN_DOMAIN_GPU_SPEEDUP = 1.5


def _read_json(root: Path, relative: str) -> dict[str, Any] | None:
    path = root / relative
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{relative} must contain a JSON object")
    return payload


def _finite_float(value: object, default: float | None = None) -> float | None:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, np.generic):
        return _json_clean(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _as_dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: object) -> list[Any]:
    return value if isinstance(value, list) else []


def _gate_bool(payload: dict[str, Any] | None, *path: str, default: bool = False) -> bool:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return bool(current) if current is not None else default


def _count_split(report: dict[str, Any] | None, split: str) -> int:
    points = _as_list((report or {}).get("points"))
    return sum(
        1
        for point in points
        if isinstance(point, dict) and str(point.get("split")) == split
    )


def _ratio_score(value: int, target: int, weight: float) -> float:
    if target <= 0:
        return 0.0
    return min(float(value) / float(target), 1.0) * weight


def _bool_score(passed: bool, weight: float) -> float:
    return weight if passed else 0.0


def _normalize_blockers(blockers: list[str]) -> list[str]:
    """Return stable, human-readable blocker identifiers."""

    replacements = {
        "dataset_sufficiency_passed": "dataset_sufficiency_gate_failed",
        "candidate_uncertainty_passed": "candidate_uncertainty_gate_failed",
        "required_candidate_accepted": "required_candidate_not_accepted",
        "required_candidate_transport_error": "required_candidate_transport_error_gate_failed",
        "passed_holdout_surface_or_field_line_artifact": "production_scope_heldout_surface_or_field_line_artifact_missing",
    }
    normalized = [replacements.get(str(item), str(item)) for item in blockers]
    return sorted(set(normalized))


def _max_speedup(rows: list[Any], backend: str) -> float | None:
    values: list[float] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("backend", "")).lower() != backend.lower():
            continue
        speed = _finite_float(row.get("speedup"))
        if speed is None:
            speed = _finite_float(row.get("warm_speedup"))
        if speed is not None:
            values.append(speed)
    return max(values) if values else None


def _lane_status(passed: bool, blockers: list[str], completion: float) -> str:
    if passed:
        return "closed"
    if blockers and completion < 55.0:
        return "blocked"
    return "partial" if completion >= 35.0 else "open"


def _universal_ql_lane(root: Path) -> dict[str, Any]:
    ql_report = _read_json(root, "docs/_static/quasilinear_stellarator_train_holdout_report.json")
    ql_uncertainty = _read_json(root, "docs/_static/quasilinear_candidate_uncertainty.json")
    ql_model = _read_json(root, "docs/_static/quasilinear_model_selection_status.json")
    ql_dataset = _read_json(root, "docs/_static/quasilinear_dataset_sufficiency.json")
    ql_guardrails = _read_json(root, "docs/_static/quasilinear_promotion_guardrails.json")
    ql_gap = _read_json(root, "docs/_static/quasilinear_holdout_gap_report.json")

    report_by_split = _as_dict((ql_report or {}).get("by_split"))
    holdout_stats = _as_dict(report_by_split.get("holdout"))
    holdout_error = _finite_float(holdout_stats.get("mean_abs_relative_error"))
    holdouts = int(holdout_stats.get("n") or _count_split(ql_report, "holdout"))
    train = int(_as_dict(report_by_split.get("train")).get("n") or _count_split(ql_report, "train"))
    model_metrics = _as_dict((ql_model or {}).get("metrics"))
    candidate_mean_error = _finite_float(model_metrics.get("candidate_mean_abs_relative_error"))
    candidate_coverage = _finite_float(model_metrics.get("candidate_prediction_interval_coverage"))
    uncertainty_gate = _as_dict((ql_uncertainty or {}).get("promotion_gate"))
    model_gate = _as_dict((ql_model or {}).get("promotion_gate"))
    dataset_requirements = _as_dict((ql_dataset or {}).get("requirements"))
    dataset_checks = _as_dict(dataset_requirements.get("checks"))
    accepted_candidates = _as_list((ql_model or {}).get("accepted_candidates")) or _as_list(uncertainty_gate.get("accepted_candidates"))

    validated_inputs = bool(dataset_checks.get("validated_input_gates", False)) or bool(ql_report)
    dataset_volume = bool(
        dataset_checks.get("minimum_total_electrostatic_cases", False)
        and dataset_checks.get("minimum_holdout_geometries", False)
        and dataset_checks.get("minimum_explicit_train_geometries", False)
    )
    guardrails_passed = bool((ql_guardrails or {}).get("passed", False))
    holdout_coverage = holdouts >= 8 and train >= 2
    candidate_soft_skill = candidate_mean_error is not None and candidate_mean_error <= QL_CANDIDATE_SOFT_ERROR_GATE
    uncertainty_passed = bool(uncertainty_gate.get("passed", False))
    model_selection_passed = bool((ql_model or {}).get("passed", False)) or bool(model_gate.get("passed", False))
    absolute_report_passed = bool((ql_report or {}).get("passed", False))
    absolute_error_passed = holdout_error is not None and holdout_error <= QL_ABSOLUTE_ERROR_GATE
    accepted_runtime_candidate = bool(accepted_candidates)

    blockers: list[str] = []
    if not absolute_report_passed:
        blockers.append("absolute_train_holdout_report_failed")
    if not absolute_error_passed:
        blockers.append("holdout_mean_abs_relative_error_exceeds_0.35")
    if not model_selection_passed:
        blockers.extend(str(item) for item in _as_list(model_gate.get("blockers")))
        if not blockers or "model_selection_not_passed" not in blockers:
            blockers.append("model_selection_not_passed")
    if not uncertainty_passed:
        blockers.append("candidate_uncertainty_gate_failed")
    if not accepted_runtime_candidate:
        blockers.append("no_accepted_absolute_flux_candidate")

    completion = (
        _bool_score(validated_inputs, 12.0)
        + _bool_score(dataset_volume, 13.0)
        + _bool_score(guardrails_passed, 10.0)
        + _bool_score(holdout_coverage, 15.0)
        + _bool_score(candidate_soft_skill, 10.0)
        + _bool_score(uncertainty_passed, 10.0)
        + _bool_score(model_selection_passed, 10.0)
        + _bool_score(absolute_report_passed and absolute_error_passed, 15.0)
        + _bool_score(accepted_runtime_candidate, 5.0)
    )
    passed = bool(
        absolute_report_passed
        and absolute_error_passed
        and model_selection_passed
        and uncertainty_passed
        and accepted_runtime_candidate
    )

    return {
        "lane": "Universal absolute quasilinear heat-flux prediction",
        "status": _lane_status(passed, blockers, completion),
        "passed": passed,
        "completion_percent": round(completion, 1),
        "claim_level": "blocked_absolute_flux_prediction" if not passed else "universal_absolute_flux_prediction_ready",
        "primary_artifacts": [
            "docs/_static/quasilinear_stellarator_train_holdout_report.json",
            "docs/_static/quasilinear_candidate_uncertainty.json",
            "docs/_static/quasilinear_model_selection_status.json",
            "docs/_static/quasilinear_dataset_sufficiency.json",
            "docs/_static/quasilinear_holdout_gap_report.json",
            "docs/_static/quasilinear_promotion_guardrails.json",
        ],
        "key_metrics": {
            "train_cases": train,
            "holdout_cases": holdouts,
            "holdout_mean_abs_relative_error": holdout_error,
            "transport_mean_relative_error_gate": QL_ABSOLUTE_ERROR_GATE,
            "candidate_mean_abs_relative_error": candidate_mean_error,
            "candidate_soft_error_gate": QL_CANDIDATE_SOFT_ERROR_GATE,
            "candidate_prediction_interval_coverage": candidate_coverage,
            "accepted_candidates": accepted_candidates,
            "dataset_volume_passed": dataset_volume,
            "guardrails_passed": guardrails_passed,
            "holdout_gap_blockers": _as_list(_as_dict((ql_gap or {}).get("promotion_gate")).get("blockers")),
        },
        "blockers": _normalize_blockers(blockers),
        "required_next_artifacts": [
            "at least one additional independent converged nonlinear holdout outside the current residual-dominated families",
            "a saturation/model candidate whose leave-one-geometry-out mean relative error and interval coverage pass the strict uncertainty gate",
            "a train/holdout absolute-flux report with holdout mean relative error <= 0.35",
            "a promotion-guardrail report proving the candidate is not just a scoped diagnostic",
        ],
        "next_action": (
            "Add independent converged nonlinear holdouts and a better saturation/transport-amplitude model; "
            "do not expose a runtime/TOML absolute-flux predictor until all promotion gates pass."
        ),
    }


def _broad_nonlinear_optimization_lane(root: Path) -> dict[str, Any]:
    guard = _read_json(root, "docs/_static/production_nonlinear_optimization_guard.json")
    vmec_holdout = _read_json(root, "docs/_static/vmec_boozer_aggregate_holdout_promotion_gate.json")
    qa_status = _read_json(root, "docs/_static/vmec_jax_qa_transport_optimization_status.json")

    summary = _as_dict((guard or {}).get("summary"))
    matched = int(summary.get("qualifying_matched_optimized_transport_audits") or 0)
    optimized = int(summary.get("qualifying_optimized_equilibrium_ensembles") or 0)
    replicated = int(summary.get("qualifying_replicated_holdout_ensembles") or 0)
    scoped_guard_passed = bool((guard or {}).get("passed", False))
    holdout_promotion_passed = bool((vmec_holdout or {}).get("passed", False))
    qa_long_window_anchor = bool(_as_dict((qa_status or {}).get("summary")).get("long_window_nonlinear_audit_passed", False))

    blockers: list[str] = []
    if matched < MIN_BROAD_MATCHED_OPTIMIZATION_AUDITS:
        blockers.append("need_at_least_three_matched_optimized_transport_audits")
    if optimized < MIN_BROAD_OPTIMIZED_EQUILIBRIUM_ENSEMBLES:
        blockers.append("need_at_least_three_optimized_equilibrium_ensembles")
    if replicated < MIN_BROAD_REPLICATED_HOLDOUT_ENSEMBLES:
        blockers.append("need_at_least_four_replicated_holdout_ensembles")
    if not holdout_promotion_passed:
        blockers.append("vmec_boozer_production_scope_holdout_missing")
    if not scoped_guard_passed:
        blockers.append("scoped_production_nonlinear_guard_failed")

    completion = (
        _bool_score(scoped_guard_passed, 25.0)
        + _bool_score(qa_long_window_anchor, 10.0)
        + _ratio_score(matched, MIN_BROAD_MATCHED_OPTIMIZATION_AUDITS, 20.0)
        + _ratio_score(optimized, MIN_BROAD_OPTIMIZED_EQUILIBRIUM_ENSEMBLES, 15.0)
        + _ratio_score(replicated, MIN_BROAD_REPLICATED_HOLDOUT_ENSEMBLES, 15.0)
        + _bool_score(holdout_promotion_passed, 15.0)
    )
    passed = bool(not blockers)

    return {
        "lane": "Broad end-to-end nonlinear turbulent-flux stellarator optimization",
        "status": _lane_status(passed, blockers, completion),
        "passed": passed,
        "completion_percent": round(completion, 1),
        "claim_level": "single_candidate_scoped_positive_audit_not_broad_optimization" if not passed else "broad_nonlinear_turbulent_flux_optimization_ready",
        "primary_artifacts": [
            "docs/_static/production_nonlinear_optimization_guard.json",
            "docs/_static/qa_no_ess_to_optimized_nonlinear_audit.json",
            "docs/_static/optimized_equilibrium_replicates/optimized_equilibrium_replicate_t700_ensemble_gate.json",
            "docs/_static/vmec_jax_qa_transport_optimization_status.json",
            "docs/_static/vmec_boozer_aggregate_holdout_promotion_gate.json",
        ],
        "key_metrics": {
            "scoped_guard_passed": scoped_guard_passed,
            "qa_long_window_anchor_passed": qa_long_window_anchor,
            "qualifying_matched_optimized_transport_audits": matched,
            "min_qualifying_matched_optimized_transport_audits": MIN_BROAD_MATCHED_OPTIMIZATION_AUDITS,
            "qualifying_optimized_equilibrium_ensembles": optimized,
            "min_qualifying_optimized_equilibrium_ensembles": MIN_BROAD_OPTIMIZED_EQUILIBRIUM_ENSEMBLES,
            "qualifying_replicated_holdout_ensembles": replicated,
            "min_qualifying_replicated_holdout_ensembles": MIN_BROAD_REPLICATED_HOLDOUT_ENSEMBLES,
            "vmec_boozer_holdout_promotion_passed": holdout_promotion_passed,
        },
        "blockers": _normalize_blockers(blockers),
        "required_next_artifacts": [
            "matched long post-transient baseline-vs-optimized audits for at least three independent optimized equilibria",
            "surface/alpha held-out VMEC/Boozer production-scope nonlinear transport artifact",
            "replicated seed/timestep ensembles for each optimized equilibrium and selected holdout",
            "running-mean, block/SEM, spread, and finite-flux gates for all promoted windows",
        ],
        "next_action": (
            "Move from one scoped QA positive audit to a multi-equilibrium, multi-surface/multi-alpha campaign; "
            "only count long post-transient replicated transport windows, not reduced/startup objectives."
        ),
    }


def _domain_decomposition_lane(root: Path) -> dict[str, Any]:
    combined = _read_json(root, "docs/_static/nonlinear_sharding_strong_scaling_large.json")
    production = _read_json(root, "docs/_static/nonlinear_sharding_production_speedup_gate.json")
    domain_identity = _read_json(root, "docs/_static/nonlinear_domain_parallel_identity_gate.json")
    spectral_identity = _read_json(root, "docs/_static/nonlinear_spectral_communication_identity_gate.json")
    decomposition_status = _read_json(root, "docs/_static/parallel_decomposition_status.json")

    rows = _as_list((combined or {}).get("rows"))
    cpu_speedup = _max_speedup(rows, "cpu")
    gpu_speedup = _max_speedup(rows, "gpu")
    identity_passed = bool((combined or {}).get("identity_passed", False))
    strong_scaling_speedup_passed = bool((combined or {}).get("speedup_passed", False))
    production_passed = bool((production or {}).get("passed", False)) or str((production or {}).get("status", "")).lower() == "production_speedup"
    domain_identity_passed = _gate_bool(domain_identity, "gate", "identity_passed")
    spectral_identity_passed = _gate_bool(spectral_identity, "gate", "identity_passed")
    decomposition_contract_passed = bool((decomposition_status or {}).get("passed", False))
    cpu_speedup_passed = cpu_speedup is not None and cpu_speedup >= MIN_DOMAIN_CPU_SPEEDUP
    gpu_speedup_passed = gpu_speedup is not None and gpu_speedup >= MIN_DOMAIN_GPU_SPEEDUP

    blockers: list[str] = []
    if not production_passed:
        blockers.append("production_speedup_gate_not_passed")
    if not strong_scaling_speedup_passed:
        blockers.extend(str(item) for item in _as_list((combined or {}).get("speedup_blockers")))
        blockers.append("combined_strong_scaling_speedup_not_passed")
    if not gpu_speedup_passed:
        blockers.append("gpu_domain_speedup_below_1p5")
    if not cpu_speedup_passed:
        blockers.append("cpu_domain_speedup_below_1p5")

    completion = (
        _bool_score(domain_identity_passed, 15.0)
        + _bool_score(spectral_identity_passed, 15.0)
        + _bool_score(identity_passed, 15.0)
        + _bool_score(decomposition_contract_passed, 10.0)
        + _bool_score(cpu_speedup_passed, 15.0)
        + _bool_score(gpu_speedup_passed, 15.0)
        + _bool_score(strong_scaling_speedup_passed, 10.0)
        + _bool_score(production_passed, 5.0)
    )
    passed = bool(
        domain_identity_passed
        and spectral_identity_passed
        and identity_passed
        and cpu_speedup_passed
        and gpu_speedup_passed
        and strong_scaling_speedup_passed
        and production_passed
    )

    return {
        "lane": "Production nonlinear domain-decomposition speedup",
        "status": _lane_status(passed, blockers, completion),
        "passed": passed,
        "completion_percent": round(completion, 1),
        "claim_level": "identity_and_profiler_diagnostic_not_production_speedup" if not passed else "production_nonlinear_domain_speedup_ready",
        "primary_artifacts": [
            "docs/_static/nonlinear_domain_parallel_identity_gate.json",
            "docs/_static/nonlinear_spectral_communication_identity_gate.json",
            "docs/_static/nonlinear_sharding_strong_scaling_large.json",
            "docs/_static/nonlinear_sharding_production_speedup_gate.json",
            "docs/_static/parallel_decomposition_status.json",
        ],
        "key_metrics": {
            "domain_identity_passed": domain_identity_passed,
            "spectral_identity_passed": spectral_identity_passed,
            "combined_identity_passed": identity_passed,
            "parallel_decomposition_contract_passed": decomposition_contract_passed,
            "cpu_best_speedup": cpu_speedup,
            "cpu_speedup_gate": MIN_DOMAIN_CPU_SPEEDUP,
            "gpu_best_speedup": gpu_speedup,
            "gpu_speedup_gate": MIN_DOMAIN_GPU_SPEEDUP,
            "strong_scaling_speedup_passed": strong_scaling_speedup_passed,
            "production_gate_passed": production_passed,
        },
        "blockers": _normalize_blockers(blockers),
        "required_next_artifacts": [
            "real communication-aware nonlinear domain decomposition in the production RHS/integrator path",
            "transport-window identity gates comparing serial and decomposed trajectories on a nonlinear case",
            "large-grid CPU and multi-GPU strong-scaling artifacts with speedup >= 1.5 on each backend",
            "profiler traces proving the communication overhead is below the saved RHS work",
        ],
        "next_action": (
            "Keep independent-work batching as the production path; for nonlinear domains, implement a real decomposed "
            "RHS/integrator route and require identity plus CPU/GPU speedup before any manuscript claim."
        ),
    }


def _vmec_boozer_holdout_lane(root: Path) -> dict[str, Any]:
    promotion = _read_json(root, "docs/_static/vmec_boozer_aggregate_holdout_promotion_gate.json")
    alpha = _read_json(root, "docs/_static/vmec_boozer_aggregate_alpha_holdout_gate.json")
    surface = _read_json(root, "docs/_static/vmec_boozer_aggregate_surface_holdout_gate.json")
    second_eq = _read_json(root, "docs/_static/vmec_boozer_second_equilibrium_aggregate_gate.json")
    gradient_matrix = _read_json(root, "docs/_static/vmec_boozer_gradient_holdout_matrix.json")
    ql_grad = _read_json(root, "docs/_static/vmec_boozer_quasilinear_gradient_gate.json")
    nonlinear_window_grad = _read_json(root, "docs/_static/vmec_boozer_nonlinear_window_gradient_gate.json")

    promotion_passed = bool((promotion or {}).get("passed", False))
    alpha_passed = bool((alpha or {}).get("passed", False))
    surface_passed = bool((surface or {}).get("passed", False))
    second_eq_passed = bool((second_eq or {}).get("passed", False))
    gradient_matrix_passed = bool((gradient_matrix or {}).get("passed", False))
    ql_grad_passed = bool((ql_grad or {}).get("passed", False))
    nonlinear_grad_passed = bool((nonlinear_window_grad or {}).get("passed", False))
    promotion_gate = _as_dict((promotion or {}).get("promotion_gate"))
    holdout_artifacts = _as_list((promotion or {}).get("holdout_artifacts"))
    qualifying_production_holdouts = sum(
        1
        for item in holdout_artifacts
        if isinstance(item, dict) and bool(item.get("qualifies_for_promotion", False))
    )
    blockers = [str(item) for item in _as_list(promotion_gate.get("blockers"))]
    if not promotion_passed and "aggregate_holdout_promotion_gate_failed" not in blockers:
        blockers.append("aggregate_holdout_promotion_gate_failed")
    if qualifying_production_holdouts <= 0:
        blockers.append("no_production_scope_heldout_surface_or_alpha_artifact")

    completion = (
        _bool_score(ql_grad_passed, 12.0)
        + _bool_score(nonlinear_grad_passed, 12.0)
        + _bool_score(gradient_matrix_passed, 14.0)
        + _bool_score(alpha_passed, 13.0)
        + _bool_score(surface_passed, 13.0)
        + _bool_score(second_eq_passed, 14.0)
        + _ratio_score(qualifying_production_holdouts, 1, 10.0)
        + _bool_score(promotion_passed, 12.0)
    )
    passed = bool(promotion_passed and qualifying_production_holdouts >= 1)
    required_next_artifacts = (
        [
            "VMEC/Boozer nonlinear transport-gradient or robust finite-difference gate on the held-out split",
            "second-equilibrium heldout nonlinear transport validation before broad geometry-optimization claims",
            "same-WOUT provenance linking optimizer state, Boozer transform, SPECTRAX-GK input, and nonlinear audit output",
        ]
        if passed
        else [
            "production-scope held-out surface or field-line artifact using long post-transient nonlinear transport, not reduced growth/QL objectives",
            "VMEC/Boozer nonlinear transport-gradient or robust finite-difference gate on the held-out split",
            "second-equilibrium heldout nonlinear transport validation before broad geometry-optimization claims",
            "same-WOUT provenance linking optimizer state, Boozer transform, SPECTRAX-GK input, and nonlinear audit output",
        ]
    )
    next_action = (
        "VMEC/Boozer held-out nonlinear transport is closed for the current pre-manuscript gate; extend to nonlinear "
        "transport-gradient and second-equilibrium nonlinear transport before broader optimization claims."
        if passed
        else (
            "Promote the existing reduced alpha/surface/second-equilibrium gates only as plumbing; add a true production-scope "
            "heldout nonlinear transport artifact before claiming VMEC/Boozer optimization closure."
        )
    )

    return {
        "lane": "VMEC/Boozer holdout optimization",
        "status": _lane_status(passed, blockers, completion),
        "passed": passed,
        "completion_percent": round(completion, 1),
        "claim_level": "reduced_holdout_plumbing_not_production_optimization" if not passed else "vmec_boozer_holdout_optimization_ready",
        "primary_artifacts": [
            "docs/_static/vmec_boozer_quasilinear_gradient_gate.json",
            "docs/_static/vmec_boozer_nonlinear_window_gradient_gate.json",
            "docs/_static/vmec_boozer_gradient_holdout_matrix.json",
            "docs/_static/vmec_boozer_aggregate_alpha_holdout_gate.json",
            "docs/_static/vmec_boozer_aggregate_surface_holdout_gate.json",
            "docs/_static/vmec_boozer_second_equilibrium_aggregate_gate.json",
            "docs/_static/vmec_boozer_aggregate_holdout_promotion_gate.json",
        ],
        "key_metrics": {
            "vmec_boozer_quasilinear_gradient_gate_passed": ql_grad_passed,
            "vmec_boozer_nonlinear_window_gradient_gate_passed": nonlinear_grad_passed,
            "gradient_holdout_matrix_passed": gradient_matrix_passed,
            "reduced_alpha_holdout_passed": alpha_passed,
            "reduced_surface_holdout_passed": surface_passed,
            "second_equilibrium_reduced_gate_passed": second_eq_passed,
            "qualifying_production_holdout_artifacts": qualifying_production_holdouts,
            "promotion_gate_passed": promotion_passed,
            "promotion_gate_blockers": _as_list(promotion_gate.get("blockers")),
        },
        "blockers": _normalize_blockers(blockers),
        "required_next_artifacts": required_next_artifacts,
        "next_action": next_action,
    }


def build_status_payload(root: Path = REPO_ROOT) -> dict[str, Any]:
    """Return a JSON-ready strict pre-manuscript closure payload."""

    root = Path(root)
    lanes = [
        _universal_ql_lane(root),
        _broad_nonlinear_optimization_lane(root),
        _domain_decomposition_lane(root),
        _vmec_boozer_holdout_lane(root),
    ]
    ready = all(bool(lane["passed"]) for lane in lanes)
    mean_completion = float(np.mean([float(lane["completion_percent"]) for lane in lanes]))
    return {
        "kind": "pre_manuscript_closure_status",
        "claim_scope": (
            "strict manuscript-blocking closure gates; release-safe diagnostics and scoped optimization evidence "
            "do not close these lanes unless the listed production or absolute-prediction gates pass"
        ),
        "status_order": STATUS_ORDER,
        "lanes": lanes,
        "summary": {
            "ready_for_manuscript_drafting": ready,
            "n_lanes": len(lanes),
            "n_closed": sum(1 for lane in lanes if lane["status"] == "closed"),
            "n_partial": sum(1 for lane in lanes if lane["status"] == "partial"),
            "n_open": sum(1 for lane in lanes if lane["status"] == "open"),
            "n_blocked": sum(1 for lane in lanes if lane["status"] == "blocked"),
            "mean_completion_percent": round(mean_completion, 1),
            "blocking_lanes": [lane["lane"] for lane in lanes if not lane["passed"]],
        },
    }


def write_status_artifacts(payload: dict[str, Any], *, out: Path = DEFAULT_OUT) -> dict[str, str]:
    """Write JSON/CSV/PNG/PDF companions for the closure payload."""

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json_path = out.with_suffix(".json")
    csv_path = out.with_suffix(".csv")
    pdf_path = out.with_suffix(".pdf")
    json_path.write_text(
        json.dumps(_json_clean(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    fieldnames = [
        "lane",
        "status",
        "completion_percent",
        "claim_level",
        "blockers",
        "next_action",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for lane in payload["lanes"]:
            writer.writerow(
                {
                    "lane": lane["lane"],
                    "status": lane["status"],
                    "completion_percent": lane["completion_percent"],
                    "claim_level": lane["claim_level"],
                    "blockers": ";".join(lane["blockers"]),
                    "next_action": lane["next_action"],
                }
            )

    set_plot_style()
    lanes = list(payload["lanes"])
    y = np.arange(len(lanes))
    completion = [float(lane["completion_percent"]) for lane in lanes]
    colors = [STATUS_COLORS.get(str(lane["status"]), "#777777") for lane in lanes]
    labels = [textwrap.fill(str(lane["lane"]), width=34) for lane in lanes]

    fig, ax = plt.subplots(figsize=(12.0, 5.2))
    ax.barh(y, completion, color=colors, edgecolor="#333333", alpha=0.95)
    ax.set_yticks(y, labels)
    ax.set_xlim(0.0, 100.0)
    ax.set_xlabel("strict pre-manuscript closure (%)")
    ax.set_title("Pre-manuscript closure status: remaining blocking lanes")
    ax.grid(axis="x", alpha=0.25)
    ax.invert_yaxis()
    for yi, lane, value in zip(y, lanes, completion, strict=True):
        blockers = lane.get("blockers", [])
        blocker_text = "closed" if lane.get("passed") else f"{len(blockers)} blocker(s)"
        ax.text(
            min(value + 1.2, 98.0),
            float(yi),
            f"{value:.1f}% | {lane['status']} | {blocker_text}",
            va="center",
            ha="left" if value < 82 else "right",
            fontsize=8.0,
            color="#222222",
        )
    summary = payload.get("summary", {})
    caption = (
        f"Ready for manuscript drafting: {summary.get('ready_for_manuscript_drafting')}. "
        f"Mean strict closure: {float(summary.get('mean_completion_percent', 0.0)):.1f}%. "
        "Release-safe scoped diagnostics remain separate from these four promotion gates."
    )
    fig.text(0.5, 0.025, caption, fontsize=8.4, color="#333333", ha="center")
    fig.subplots_adjust(left=0.34, right=0.97, top=0.86, bottom=0.18)
    fig.savefig(out, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    return {"png": str(out), "pdf": str(pdf_path), "json": str(json_path), "csv": str(csv_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--json-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_status_payload(args.root)
    if args.json_only:
        print(json.dumps(_json_clean(payload), indent=2, sort_keys=True))
        return 0
    paths = write_status_artifacts(payload, out=args.out)
    print(json.dumps(paths, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
