#!/usr/bin/env python3
"""Audit quasilinear absolute-flux promotion metadata and docs scope."""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


DEFAULT_REPORT_PATTERNS = (
    str(ROOT / "docs/_static/quasilinear_*train_holdout_report.json"),
    str(ROOT / "docs/_static/quasilinear_saturation_rule_sweep.json"),
    str(ROOT / "docs/_static/quasilinear_shape_aware_saturation.json"),
    str(ROOT / "docs/_static/quasilinear_candidate_uncertainty.json"),
    str(ROOT / "docs/_static/quasilinear_candidate_regularization_sweep.json"),
    str(ROOT / "docs/_static/quasilinear_dataset_sufficiency.json"),
    str(ROOT / "docs/_static/quasilinear_validated_calibration_inputs.json"),
    str(ROOT / "docs/_static/manuscript_readiness_status.json"),
)
DEFAULT_DOCS = (
    ROOT / "README.md",
    ROOT / "docs/quasilinear.rst",
    ROOT / "docs/manuscript_figures.rst",
    ROOT / "docs/testing.rst",
)
DEFAULT_OUT = ROOT / "docs/_static/quasilinear_promotion_guardrails.json"
DEFAULT_MANUSCRIPT_INDEX = ROOT / "docs/manuscript_figures.rst"
DEFAULT_MANUSCRIPT_FIGURE_BASES = (
    ROOT / "docs/_static/quasilinear_stellarator_train_holdout",
    ROOT / "docs/_static/quasilinear_saturation_rule_sweep",
    ROOT / "docs/_static/quasilinear_shape_aware_saturation",
    ROOT / "docs/_static/quasilinear_candidate_uncertainty",
    ROOT / "docs/_static/quasilinear_candidate_regularization_sweep",
    ROOT / "docs/_static/quasilinear_dataset_sufficiency",
    ROOT / "docs/_static/quasilinear_model_selection_status",
    ROOT / "docs/_static/quasilinear_stellarator_usefulness",
    ROOT / "docs/_static/quasilinear_screening_skill",
    ROOT / "docs/_static/quasilinear_holdout_gap_report",
)

PROMOTED_CLAIM = "calibrated_absolute_flux"
REQUIRED_SPLITS = {"train", "holdout"}
REQUIRED_POINT_FIELDS = (
    "case",
    "split",
    "geometry",
    "electron_model",
    "saturation_rule",
    "nonlinear_artifact",
    "quasilinear_artifact",
)
DOC_SCOPE_MARKERS = (
    "not a runtime/toml absolute-flux predictor",
    "not a calibrated absolute-flux claim",
    "absolute-flux prediction not promoted",
    "not a promoted absolute nonlinear heat-flux model",
    "not a validated transport model",
)


def _finite_number(value: Any) -> bool:
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def nonlinear_window_stats_promotion_ready(
    stats: object,
) -> tuple[bool, list[str]]:
    """Return whether serialized nonlinear-window metadata supports promotion.

    This duplicates the lightweight schema check from
    ``spectraxgk.validation.quasilinear.window_promotion`` so the CI repo-hygiene job can run before
    installing JAX and the rest of the runtime stack.
    """

    failures: list[str] = []
    if not isinstance(stats, dict):
        return False, ["missing nonlinear_window_stats object"]
    if stats.get("kind") != "nonlinear_window_convergence_report":
        failures.append("unexpected nonlinear_window_stats kind")
    if not bool(stats.get("passed", False)):
        failures.append("nonlinear window convergence report did not pass")
    provenance = stats.get("provenance")
    if (
        not isinstance(provenance, dict)
        or not str(provenance.get("source_artifact", "")).strip()
    ):
        failures.append("missing nonlinear source_artifact provenance")
    statistics = stats.get("statistics")
    if not isinstance(statistics, dict):
        failures.append("missing statistics object")
        statistics = {}
    for field in (
        "late_mean",
        "sem",
        "block_bootstrap_sem",
        "running_mean_rel_drift",
    ):
        if not _finite_number(statistics.get(field)):
            failures.append(f"missing/non-finite statistics.{field}")
    window = stats.get("window")
    if not isinstance(window, dict):
        failures.append("missing window object")
        window = {}
    for field in ("transient_cutoff", "late_tmin", "late_tmax"):
        if not _finite_number(window.get(field)):
            failures.append(f"missing/non-finite window.{field}")
    raw_transient_fraction = window.get("transient_fraction", 0.0)
    has_declared_cutoff = _finite_number(window.get("input_tmin")) or (
        _finite_number(raw_transient_fraction)
        and float(raw_transient_fraction) > 0.0
    )
    if not has_declared_cutoff:
        failures.append("missing declared transient cutoff policy")
    n_finite_late = window.get("n_finite_late", 0)
    if not _finite_number(n_finite_late) or int(float(n_finite_late)) <= 0:
        failures.append("window has no finite late samples")
    gate_report = stats.get("gate_report")
    if not isinstance(gate_report, dict) or not bool(gate_report.get("passed", False)):
        failures.append("missing passed gate_report")
    return not failures, failures
SCOPED_NON_ABSOLUTE_MARKERS = (
    "calibration_dataset",
    "model_development",
    "model-selection",
    "model_selection",
    "no_absolute_flux_promotion",
    "not_runtime",
    "not runtime",
    "not a runtime",
    "not a transport model",
    "not validated transport",
    "not_validated_transport",
    "not_runtime_absolute_flux",
    "not_runtime_option",
    "candidate",
    "sufficiency",
    "scoped",
)
MANUSCRIPT_QL_LANE = "Quasilinear diagnostics and saturation-model selection"
MANUSCRIPT_NON_ABSOLUTE_CLAIM_MARKERS = (
    "negative_absolute_flux_promotion",
    "not_runtime_flux_predictor",
    "not_runtime",
    "not runtime",
    "not absolute",
)


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _expand_patterns(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(str(pattern))
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(pattern))
    return sorted(set(paths))


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return data


def _nonnegative_finite(value: object) -> bool:
    number = _float_or_none(value)
    return number is not None and number >= 0.0


def _gate(metric: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"metric": metric, "passed": bool(passed), "detail": detail}


def _has_scope_marker(text: str, markers: tuple[str, ...]) -> bool:
    low = _normalized_text(text)
    return any(marker in low for marker in markers)


def _normalized_text(text: str) -> str:
    """Normalize docs text so RST markup and line wraps do not hide scope markers."""

    return " ".join(text.replace("``", "").lower().split())


def _line_overclaims_absolute_flux(line: str) -> bool:
    """Return true for positive absolute-flux-predictor wording.

    The docs intentionally contain negative phrases such as "not a runtime/TOML
    absolute-flux predictor"; those are accepted when the negation is close to
    the claim phrase.
    """

    low = line.lower()
    if "absolute-flux" not in low and "absolute flux" not in low:
        return False
    if not re.search(r"\b(promoted|validated|calibrated|production|runtime)\b", low):
        return False
    if re.search(r"\b(no|not|without|blocked|fail|failed|fails|open|deferred)\b", low):
        return False
    return bool(re.search(r"\b(predictor|model|claim|transport)\b", low))


def _audit_calibration_report(
    path: Path, data: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    gates: list[dict[str, Any]] = []
    claim = str(data.get("claim_level", ""))
    points = data.get("points", [])
    by_split = data.get("by_split", {})
    metadata = data.get("metadata", {})
    promoted = claim == PROMOTED_CLAIM

    points_are_list = isinstance(points, list)
    gates.append(
        _gate(
            "calibration_points_are_list",
            points_are_list,
            f"{_repo_relative(path)} points field",
        )
    )
    split_counts = {split: 0 for split in REQUIRED_SPLITS}
    point_failures: list[str] = []
    if points_are_list:
        for idx, point in enumerate(points):
            if not isinstance(point, dict):
                point_failures.append(f"point {idx} is not an object")
                continue
            split = str(point.get("split", ""))
            if split in split_counts:
                split_counts[split] += 1
                missing = [
                    field
                    for field in REQUIRED_POINT_FIELDS
                    if not str(point.get(field, "")).strip()
                ]
                if missing:
                    point_failures.append(
                        f"{point.get('case', idx)} missing {','.join(missing)}"
                    )
                for field in (
                    "predicted_heat_flux",
                    "observed_heat_flux",
                    "raw_predicted_heat_flux",
                    "calibration_scale",
                ):
                    if field in point and not _finite_number(point.get(field)):
                        point_failures.append(
                            f"{point.get('case', idx)} has non-finite {field}"
                        )
                if not _nonnegative_finite(point.get("observed_heat_flux_std")):
                    point_failures.append(
                        f"{point.get('case', idx)} has missing/non-finite window std"
                    )

    gates.append(
        _gate(
            "train_holdout_point_metadata",
            points_are_list
            and not point_failures
            and all(split_counts[split] > 0 for split in REQUIRED_SPLITS),
            "; ".join(point_failures)
            or f"train={split_counts['train']} holdout={split_counts['holdout']}",
        )
    )

    holdout_window_failures: list[str] = []
    holdout_window_passes = 0
    if points_are_list:
        for idx, point in enumerate(points):
            if not isinstance(point, dict) or str(point.get("split", "")) != "holdout":
                continue
            ready, failures = nonlinear_window_stats_promotion_ready(
                point.get("nonlinear_window_stats")
            )
            if ready:
                holdout_window_passes += 1
            else:
                label = str(point.get("case", idx))
                holdout_window_failures.extend(
                    f"{label}: {failure}" for failure in failures
                )

    holdout_gate = _float_or_none(data.get("holdout_mean_rel_gate"))
    has_holdout_gate = holdout_gate is not None and holdout_gate > 0.0
    holdout_metrics = by_split.get("holdout", {}) if isinstance(by_split, dict) else {}
    holdout_mean = (
        holdout_metrics.get("mean_abs_relative_error")
        if isinstance(holdout_metrics, dict)
        else None
    )
    holdout_passes = (
        has_holdout_gate
        and (holdout_mean_value := _float_or_none(holdout_mean)) is not None
        and holdout_gate is not None
        and holdout_mean_value <= holdout_gate
    )
    if promoted:
        gates.append(
            _gate(
                "promoted_report_passed",
                bool(data.get("passed", False)),
                "promoted reports must pass",
            )
        )
        gates.append(
            _gate(
                "promoted_holdout_gate",
                holdout_passes,
                f"holdout_mean={holdout_mean} gate={data.get('holdout_mean_rel_gate')}",
            )
        )
        gates.append(
            _gate(
                "promoted_calibration_policy_metadata",
                isinstance(metadata, dict)
                and bool(
                    metadata.get("calibration_policy")
                    or metadata.get("heat_flux_scale_fit")
                ),
                "promoted reports must serialize calibration policy or scale fit metadata",
            )
        )
        gates.append(
            _gate(
                "promoted_holdout_window_convergence",
                not holdout_window_failures
                and holdout_window_passes == split_counts["holdout"],
                "; ".join(holdout_window_failures)
                or f"converged_holdout_windows={holdout_window_passes}",
            )
        )
    else:
        gates.append(
            _gate(
                "unpromoted_report_not_absolute_flux",
                not bool(data.get("passed", False)) or claim != PROMOTED_CLAIM,
                f"claim_level={claim} passed={data.get('passed')}",
            )
        )

    summary = {
        "artifact": _repo_relative(path),
        "kind": data.get("kind"),
        "claim_level": claim,
        "passed": data.get("passed"),
        "n_train": split_counts["train"],
        "n_holdout": split_counts["holdout"],
        "holdout_mean_abs_relative_error": holdout_mean,
        "holdout_mean_rel_gate": data.get("holdout_mean_rel_gate"),
    }
    return gates, summary


def _audit_input_validation(
    path: Path, data: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    validation = data.get("input_validation")
    if not isinstance(validation, dict):
        return [], None
    cases = validation.get("cases", [])
    rows = cases if isinstance(cases, list) else []
    required_rows = [
        row
        for row in rows
        if isinstance(row, dict) and bool(row.get("required", False))
    ]
    failed = [
        str(row.get("case", "unknown"))
        for row in required_rows
        if not bool(row.get("passed", False))
    ]
    gates = [
        _gate(
            "input_validation_passed",
            bool(validation.get("passed", False)) and not failed,
            "; ".join(failed)
            or f"{len(required_rows)} required nonlinear summaries passed",
        )
    ]
    return gates, {
        "artifact": _repo_relative(path),
        "kind": data.get("kind"),
        "required_cases": len(required_rows),
        "failed_required_cases": failed,
    }


def _audit_promotion_gate(
    path: Path, data: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    promotion = data.get("promotion_gate")
    if not isinstance(promotion, dict):
        return [], None
    claim = str(data.get("claim_level", ""))
    notes = str(data.get("notes", ""))
    text = f"{claim} {notes}"
    gates = [
        _gate(
            "non_absolute_promotion_scope",
            claim == PROMOTED_CLAIM
            or _has_scope_marker(text, SCOPED_NON_ABSOLUTE_MARKERS),
            f"claim_level={claim}",
        )
    ]
    if claim == PROMOTED_CLAIM:
        gates.append(
            _gate(
                "absolute_claim_requires_transport_gate",
                bool(promotion.get("passed", False)),
                "absolute-flux claim must carry a passed promotion gate",
            )
        )
    return gates, {
        "artifact": _repo_relative(path),
        "kind": data.get("kind"),
        "claim_level": claim,
        "promotion_gate_passed": bool(promotion.get("passed", False)),
        "accepted": promotion.get(
            "accepted_candidates", promotion.get("accepted_rules", [])
        ),
        "blockers": promotion.get("blockers", []),
    }


def _audit_manuscript_readiness(
    path: Path, data: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if data.get("kind") != "manuscript_readiness_status":
        return [], None

    lanes = data.get("lanes", [])
    ql_lane = None
    if isinstance(lanes, list):
        for lane in lanes:
            if isinstance(lane, dict) and lane.get("lane") == MANUSCRIPT_QL_LANE:
                ql_lane = lane
                break

    gates = [
        _gate(
            "manuscript_ql_lane_present",
            ql_lane is not None,
            f"{_repo_relative(path)} contains {MANUSCRIPT_QL_LANE!r}",
        )
    ]
    if ql_lane is None:
        return gates, {
            "artifact": _repo_relative(path),
            "kind": data.get("kind"),
            "ql_lane_present": False,
        }

    claim_level = str(ql_lane.get("claim_level", ""))
    status = str(ql_lane.get("status", ""))
    metrics = ql_lane.get("key_metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    artifacts = ql_lane.get("primary_artifacts", [])
    if not isinstance(artifacts, list):
        artifacts = []

    absolute_flux_promoted = bool(metrics.get("absolute_flux_promoted", False))
    accepted_candidates = metrics.get("accepted_uq_candidates", [])
    candidate_selected = bool(
        metrics.get("uq_candidate_promotion_passed", False)
        or metrics.get("dataset_sufficiency_promotion_passed", False)
        or accepted_candidates
    )
    claim_is_scoped = _has_scope_marker(
        claim_level, MANUSCRIPT_NON_ABSOLUTE_CLAIM_MARKERS
    )
    gates.extend(
        [
            _gate(
                "manuscript_ql_not_absolute_flux",
                not absolute_flux_promoted and claim_level != PROMOTED_CLAIM,
                f"claim_level={claim_level} absolute_flux_promoted={absolute_flux_promoted}",
            ),
            _gate(
                "manuscript_ql_closed_scope_is_non_absolute",
                status != "closed" or claim_is_scoped,
                f"status={status} claim_level={claim_level}",
            ),
            _gate(
                "manuscript_ql_candidate_scope_not_runtime",
                not candidate_selected or claim_is_scoped,
                f"candidate_selected={candidate_selected} claim_level={claim_level}",
            ),
            _gate(
                "manuscript_ql_guardrail_artifact_listed",
                "docs/_static/quasilinear_promotion_guardrails.json" in artifacts,
                "manuscript QL lane should list the promotion guardrail audit artifact",
            ),
        ]
    )
    return gates, {
        "artifact": _repo_relative(path),
        "kind": data.get("kind"),
        "ql_lane_present": True,
        "ql_status": status,
        "ql_claim_level": claim_level,
        "absolute_flux_promoted": absolute_flux_promoted,
        "accepted_uq_candidates": accepted_candidates,
    }


def _json_text(data: dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, allow_nan=False)


def _has_non_absolute_json_scope(data: dict[str, Any]) -> bool:
    claim = str(data.get("claim_level", ""))
    notes = str(data.get("notes", ""))
    promotion = data.get("promotion_gate")
    promotion_text = json.dumps(promotion, sort_keys=True) if promotion else ""
    return (
        claim != PROMOTED_CLAIM
        and (
            not bool(data.get("passed", True))
            or _has_scope_marker(
                f"{claim} {notes} {promotion_text}", SCOPED_NON_ABSOLUTE_MARKERS
            )
        )
    )


def _gate_metric_passed(data: dict[str, Any], metric: str) -> bool:
    gate_report = data.get("gate_report")
    gates = gate_report.get("gates", []) if isinstance(gate_report, dict) else []
    return any(
        isinstance(gate, dict)
        and gate.get("metric") == metric
        and bool(gate.get("passed", False))
        for gate in gates
    )


def _audit_failed_baseline_contract(
    sidecar: Path, data: dict[str, Any]
) -> tuple[bool, str]:
    """Check that QL model-development sidecars serialize failed baselines.

    This intentionally audits metadata shape rather than physics correctness.
    The plotting tools own the numerical values; this guardrail ensures the
    manuscript-facing JSON keeps enough information to state failed baselines
    and non-promotion blockers explicitly.
    """

    name = sidecar.stem
    promotion = data.get("promotion_gate")
    promotion = promotion if isinstance(promotion, dict) else {}
    metrics = data.get("metrics")
    metrics = metrics if isinstance(metrics, dict) else {}

    if name == "quasilinear_stellarator_train_holdout":
        source = data.get("source")
        source_path = ROOT / str(source) if source else None
        return (
            bool(source)
            and source_path is not None
            and source_path.exists()
            and data.get("claim_level") == "calibration_dataset"
            and data.get("passed") is False
            and _finite_number(data.get("mean_abs_relative_error")),
            f"source={source} passed={data.get('passed')} claim_level={data.get('claim_level')}",
        )

    if name == "quasilinear_saturation_rule_sweep":
        return (
            promotion.get("passed") is False
            and promotion.get("accepted_rules") == []
            and bool(promotion.get("requires_beating_training_mean_null", False))
            and isinstance(data.get("null_training_mean_baseline"), dict)
            and _finite_number(
                promotion.get("best_rule_holdout_mean_abs_relative_error")
            )
            and _finite_number(
                promotion.get("null_training_mean_holdout_mean_abs_relative_error")
            ),
            (
                f"accepted_rules={promotion.get('accepted_rules')} "
                f"best={promotion.get('best_rule_holdout_mean_abs_relative_error')} "
                f"null={promotion.get('null_training_mean_holdout_mean_abs_relative_error')}"
            ),
        )

    if name == "quasilinear_shape_aware_saturation":
        return (
            promotion.get("passed") is False
            and bool(promotion.get("requires_beating_linear_weight_baseline", False))
            and bool(promotion.get("requires_beating_training_mean_null", False))
            and _finite_number(
                metrics.get("baseline_linear_weight_mean_abs_relative_error")
            )
            and _finite_number(metrics.get("null_training_mean_mean_abs_relative_error"))
            and _finite_number(metrics.get("shape_aware_mean_abs_relative_error")),
            (
                f"shape={metrics.get('shape_aware_mean_abs_relative_error')} "
                f"linear={metrics.get('baseline_linear_weight_mean_abs_relative_error')} "
                f"null={metrics.get('null_training_mean_mean_abs_relative_error')}"
            ),
        )

    if name == "quasilinear_candidate_uncertainty":
        candidates = data.get("candidates")
        candidates = candidates if isinstance(candidates, dict) else {}
        linear_weight = candidates.get("linear_weight", {})
        linear_state = candidates.get("linear_state_ridge", {})
        spectral = candidates.get("spectral_envelope_ridge", {})
        accepted = promotion.get("accepted_candidates", [])
        return (
            promotion.get("passed") is False
            and accepted == []
            and "linear_weight" not in accepted
            and bool(promotion.get("requires_beating_linear_weight_baseline", False))
            and bool(promotion.get("requires_beating_training_mean_null", False))
            and isinstance(data.get("null_training_mean_baseline"), dict)
            and _finite_number(linear_weight.get("mean_abs_relative_error"))
            and _finite_number(spectral.get("mean_abs_relative_error"))
            and float(spectral["mean_abs_relative_error"]) > float(
                promotion.get("transport_mean_relative_error_gate", 0.35)
            )
            and "spectral_envelope_ridge" not in accepted,
            (
                f"accepted={accepted} "
                f"spectral_error={spectral.get('mean_abs_relative_error')} "
                f"linear_weight_error={linear_weight.get('mean_abs_relative_error')} "
                f"linear_state_failures={linear_state.get('eligibility_failures')}"
            ),
        )

    if name == "quasilinear_candidate_regularization_sweep":
        rows = data.get("rows")
        rows = rows if isinstance(rows, list) else []
        accepted = promotion.get("accepted_lambdas", [])
        gate = float(promotion.get("transport_mean_relative_error_gate", 0.35))
        best_error = _float_or_none(data.get("best_mean_abs_relative_error"))
        best_coverage = _float_or_none(data.get("best_prediction_interval_coverage"))
        return (
            data.get("kind") == "quasilinear_candidate_regularization_sweep"
            and "not_runtime_flux_predictor" in str(data.get("claim_level", ""))
            and promotion.get("passed") is False
            and accepted == []
            and "best_regularization_transport_error_above_gate"
            in promotion.get("blockers", [])
            and best_error is not None
            and best_error > gate
            and best_coverage is not None
            and best_coverage >= float(data.get("interval_coverage_gate", 0.0))
            and any(
                isinstance(row, dict)
                and row.get("lambda") == data.get("best_lambda")
                and row.get("transport_gate_passed") is False
                for row in rows
            ),
            (
                f"accepted_lambdas={accepted} best_lambda={data.get('best_lambda')} "
                f"best_error={data.get('best_mean_abs_relative_error')} "
                f"coverage={data.get('best_prediction_interval_coverage')} "
                f"blockers={promotion.get('blockers')}"
            ),
        )

    if name == "quasilinear_dataset_sufficiency":
        requirements = data.get("candidate_requirements")
        requirements = requirements if isinstance(requirements, list) else []
        linear_state_rows = [
            row
            for row in requirements
            if isinstance(row, dict) and row.get("candidate") == "linear_state_ridge"
        ]
        downstream = data.get("downstream_gates")
        downstream = downstream if isinstance(downstream, dict) else {}
        simple_sweep = downstream.get("saturation_rule_sweep")
        simple_sweep = simple_sweep if isinstance(simple_sweep, dict) else {}
        downstream_skill_blocked = (
            "downstream_candidate_skill_gates_not_passed"
            in promotion.get("blockers", [])
            and bool(promotion.get("requires_downstream_candidate_skill_gates", False))
        )
        return (
            bool(linear_state_rows)
            and simple_sweep.get("accepted") == []
            and promotion.get("passed") is False
            and downstream_skill_blocked,
            (
                f"linear_state_data_volume_passed="
                f"{linear_state_rows[0].get('data_volume_passed') if linear_state_rows else None} "
                f"saturation_rule_accepted={simple_sweep.get('accepted')} "
                f"blockers={promotion.get('blockers', [])}"
            ),
        )

    if name == "quasilinear_model_selection_status":
        reports = data.get("calibration_reports")
        reports = reports if isinstance(reports, list) else []
        promotion_gate = data.get("promotion_gate")
        promotion_gate = promotion_gate if isinstance(promotion_gate, dict) else {}
        return (
            data.get("passed") is False
            and data.get("accepted_candidates", []) == []
            and "required_candidate_transport_error" in promotion_gate.get("blockers", [])
            and _finite_number(
                data.get("metrics", {}).get("linear_weight_mean_abs_relative_error")
                if isinstance(data.get("metrics"), dict)
                else None
            )
            and _finite_number(
                data.get("metrics", {}).get("null_training_mean_mean_abs_relative_error")
                if isinstance(data.get("metrics"), dict)
                else None
            )
            and _gate_metric_passed(data, "absolute_flux_not_promoted")
            and all(
                isinstance(report, dict)
                and report.get("claim_level") != PROMOTED_CLAIM
                for report in reports
            ),
            (
                f"accepted={data.get('accepted_candidates')} "
                f"blockers={promotion_gate.get('blockers')} "
                f"absolute_flux_not_promoted={_gate_metric_passed(data, 'absolute_flux_not_promoted')}"
            ),
        )

    if name == "quasilinear_stellarator_usefulness":
        models = data.get("models")
        models = models if isinstance(models, dict) else {}
        positive_ml = models.get("positive_mixing_length")
        positive_ml = positive_ml if isinstance(positive_ml, dict) else {}
        spectral = models.get("spectral_envelope_ridge")
        spectral = spectral if isinstance(spectral, dict) else {}
        rows = data.get("rows")
        rows = rows if isinstance(rows, list) else []
        by_case = {
            str(row.get("case")): row
            for row in rows
            if isinstance(row, dict) and "case" in row
        }
        statuses = data.get("stellarator_status")
        statuses = statuses if isinstance(statuses, dict) else {}
        qa = statuses.get("QA")
        qa = qa if isinstance(qa, dict) else {}
        qh = statuses.get("QH")
        qh = qh if isinstance(qh, dict) else {}
        hsx = by_case.get("hsx_nonlinear_window", {})
        w7x = by_case.get("w7x_nonlinear_window", {})
        simple_rule_fails_stellarators = (
            positive_ml.get("accepted") is False
            and _finite_number(positive_ml.get("holdout_mean_abs_relative_error"))
            and float(positive_ml["holdout_mean_abs_relative_error"]) > 1.0
            and hsx.get("positive_mixing_length_prediction") == 0.0
            and w7x.get("positive_mixing_length_prediction") == 0.0
            and _finite_number(hsx.get("observed_heat_flux"))
            and _finite_number(w7x.get("observed_heat_flux"))
            and float(hsx["observed_heat_flux"]) > 0.0
            and float(w7x["observed_heat_flux"]) > 0.0
        )
        return (
            spectral.get("accepted") is False
            and _finite_number(spectral.get("mean_abs_relative_error"))
            and float(spectral["mean_abs_relative_error"]) > 0.35
            and simple_rule_fails_stellarators
            and "audit only" in str(qa.get("status", "")).lower()
            and qh.get("high_grid_gate_passed") is False,
            (
                f"spectral={spectral.get('mean_abs_relative_error')} "
                f"positive_ml={positive_ml.get('holdout_mean_abs_relative_error')} "
                f"hsx_ml={hsx.get('positive_mixing_length_prediction')} "
                f"w7x_ml={w7x.get('positive_mixing_length_prediction')} "
                f"qa={qa.get('status')} qh_passed={qh.get('high_grid_gate_passed')}"
            ),
        )

    if name == "quasilinear_screening_skill":
        gates = data.get("gates")
        gates = gates if isinstance(gates, dict) else {}
        models = data.get("models")
        models = models if isinstance(models, list) else []
        by_model = {
            str(row.get("model")): row
            for row in models
            if isinstance(row, dict) and "model" in row
        }
        spectral = by_model.get("spectral_envelope_ridge", {})
        simple = by_model.get("positive_mixing_length", {})
        no_absolute_promotion = (
            gates.get("mean_error_gate_models") == []
            and gates.get("accepted_absolute_flux_models") == []
            and gates.get("absolute_flux_promotion_passed") is False
            and simple.get("screening_gate_passed") is False
        )
        scoped_screening_pass = (
            gates.get("accepted_screening_models") == ["spectral_envelope_ridge"]
            and gates.get("accepted_holdout_screening_models") == ["spectral_envelope_ridge"]
            and gates.get("holdout_screening_correlation_passed") is True
            and spectral.get("screening_gate_passed") is True
            and spectral.get("holdout_screening_gate_passed") is True
            and _finite_number(spectral.get("spearman"))
            and float(spectral["spearman"]) >= 0.75
            and _finite_number(spectral.get("holdout_spearman"))
            and float(spectral["holdout_spearman"]) >= 0.75
        )
        fail_closed_screening = (
            gates.get("accepted_screening_models") == []
            and gates.get("accepted_holdout_screening_models") == []
            and gates.get("screening_correlation_passed") is False
            and gates.get("holdout_screening_correlation_passed") is False
            and spectral.get("screening_gate_passed") is False
            and spectral.get("holdout_screening_gate_passed") is False
            and _finite_number(spectral.get("spearman"))
            and _finite_number(spectral.get("holdout_spearman"))
            and float(spectral["spearman"]) < float(gates.get("spearman_gate", 0.75))
            and float(spectral["holdout_spearman"]) < float(gates.get("spearman_gate", 0.75))
        )
        return (
            no_absolute_promotion and (scoped_screening_pass or fail_closed_screening),
            (
                f"screening={gates.get('accepted_screening_models')} "
                f"holdout_screening={gates.get('accepted_holdout_screening_models')} "
                f"mean_error={gates.get('mean_error_gate_models')} "
                f"absolute={gates.get('accepted_absolute_flux_models')} "
                f"spectral_spearman={spectral.get('spearman')} "
                f"spectral_holdout_spearman={spectral.get('holdout_spearman')} "
                f"simple_screening={simple.get('screening_gate_passed')}"
            ),
        )

    if name == "quasilinear_holdout_gap_report":
        status = data.get("calibration_status")
        status = status if isinstance(status, dict) else {}
        screening = data.get("screening_skill_status")
        screening = screening if isinstance(screening, dict) else {}
        screening_reqs = data.get("screening_promotion_requirements")
        screening_reqs = screening_reqs if isinstance(screening_reqs, dict) else {}
        blockers = promotion.get("blockers", [])
        screening_passed = (
            screening.get("screening_correlation_passed") is True
            and screening.get("holdout_screening_correlation_passed") is True
        )
        screening_fail_closed = (
            screening.get("screening_correlation_passed") is False
            and screening.get("holdout_screening_correlation_passed") is False
            and "full_portfolio_screening_correlation_passed" in screening_reqs.get("blockers", [])
            and "heldout_screening_correlation_passed" in screening_reqs.get("blockers", [])
        )
        return (
            promotion.get("passed") is False
            and "absolute_flux_predictor_not_promoted" in blockers
            and "absolute_requirement:holdout_mean_abs_relative_error" in blockers
            and status.get("absolute_flux_promoted") is False
            and status.get("passed") is False
            and _finite_number(status.get("holdout_mean_abs_relative_error"))
            and _finite_number(status.get("holdout_mean_rel_gate"))
            and float(status["holdout_mean_abs_relative_error"])
            > float(status["holdout_mean_rel_gate"])
            and (screening_passed or screening_fail_closed)
            and screening_reqs.get("screening_promoted") is False
            and (
                "full_portfolio_screening_correlation_passed"
                in screening_reqs.get("blockers", [])
            )
            and (
                "heldout_screening_correlation_passed"
                in screening_reqs.get("blockers", [])
            ),
            (
                f"blockers={blockers} "
                f"holdout_mean={status.get('holdout_mean_abs_relative_error')} "
                f"screening={screening.get('screening_correlation_passed')} "
                f"holdout_screening={screening.get('holdout_screening_correlation_passed')}"
            ),
        )

    return True, "no specialized baseline contract for this sidecar"


def _audit_manuscript_figures(
    figure_bases: list[str | Path], index_path: str | Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    gates: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    index = Path(index_path)
    index_text = index.read_text(encoding="utf-8") if index.exists() else ""
    normalized_index = _normalized_text(index_text)
    gates.append(
        _gate(
            "manuscript_figure_index_exists",
            index.exists(),
            _repo_relative(index),
        )
    )

    for raw_base in figure_bases:
        base = Path(raw_base)
        if base.suffix:
            base = base.with_suffix("")
        png = base.with_suffix(".png")
        json_sidecar = base.with_suffix(".json")
        rel_png = _repo_relative(png)
        rel_json = _repo_relative(json_sidecar)
        rel_base = _repo_relative(base)
        png_exists = png.exists()
        json_exists = json_sidecar.exists()
        gates.append(
            _gate(
                f"ql_figure_png_exists:{rel_base}",
                png_exists,
                rel_png,
            )
        )
        gates.append(
            _gate(
                f"ql_figure_json_sidecar_exists:{rel_base}",
                json_exists,
                rel_json,
            )
        )

        index_mentions_png = rel_png in index_text
        png_pos = index_text.find(rel_png)
        nearby_index_text = (
            index_text[max(0, png_pos - 240) : png_pos + len(rel_png) + 320]
            if png_pos >= 0
            else ""
        )
        index_mentions_json = rel_json in index_text or "json" in nearby_index_text.lower()
        gates.append(
            _gate(
                f"ql_figure_index_mentions_png:{rel_base}",
                index_mentions_png,
                rel_png,
            )
        )
        gates.append(
            _gate(
                f"ql_figure_index_mentions_json_sidecar:{rel_base}",
                index_mentions_json,
                rel_json if rel_json in index_text else "nearby index text mentions JSON companion",
            )
        )

        sidecar_summary: dict[str, Any] = {
            "figure": rel_png,
            "json_sidecar": rel_json,
            "png_exists": png_exists,
            "json_exists": json_exists,
            "index_mentions_png": index_mentions_png,
            "index_mentions_json_sidecar": index_mentions_json,
        }
        if json_exists:
            data = _load_json(json_sidecar)
            claim_level = str(data.get("claim_level", ""))
            kind = str(data.get("kind", ""))
            json_lines = _json_text(data).splitlines()
            runtime_overclaims = [
                line for line in json_lines if _line_overclaims_absolute_flux(line)
            ]
            scoped = _has_non_absolute_json_scope(data)
            baseline_passed, baseline_detail = _audit_failed_baseline_contract(
                json_sidecar, data
            )
            gates.extend(
                [
                    _gate(
                        f"ql_figure_sidecar_has_kind:{rel_base}",
                        bool(kind),
                        f"kind={kind}",
                    ),
                    _gate(
                        f"ql_figure_sidecar_claim_scoped:{rel_base}",
                        scoped,
                        f"claim_level={claim_level} passed={data.get('passed')}",
                    ),
                    _gate(
                        f"ql_figure_sidecar_no_runtime_absolute_overclaim:{rel_base}",
                        not runtime_overclaims,
                        "; ".join(runtime_overclaims[:2])
                        or "no runtime absolute-flux overclaim in JSON sidecar",
                    ),
                    _gate(
                        f"ql_figure_failed_baselines_explicit:{rel_base}",
                        baseline_passed,
                        baseline_detail,
                    ),
                ]
            )
            sidecar_summary.update(
                {
                    "kind": kind,
                    "claim_level": claim_level,
                    "passed": data.get("passed"),
                    "claim_scoped": scoped,
                    "failed_baselines_explicit": baseline_passed,
                    "failed_baseline_detail": baseline_detail,
                    "runtime_absolute_overclaim_lines": runtime_overclaims,
                }
            )
        rows.append(sidecar_summary)

    gates.append(
        _gate(
            "manuscript_index_has_non_absolute_ql_scope",
            "no runtime/toml absolute-flux predictor" in normalized_index
            and "absolute-flux runtime promotion remains blocked" in normalized_index,
            "manuscript index states no runtime/TOML absolute-flux predictor and blocked runtime promotion",
        )
    )
    return gates, rows


def _audit_docs(paths: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    gates: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        lower = _normalized_text(text)
        scope_present = any(marker in lower for marker in DOC_SCOPE_MARKERS)
        text_lines = text.splitlines()
        overclaim_lines = []
        for idx, line in enumerate(text_lines):
            context = " ".join(
                text_lines[max(0, idx - 1) : min(len(text_lines), idx + 2)]
            )
            if _line_overclaims_absolute_flux(line) and _line_overclaims_absolute_flux(
                context
            ):
                overclaim_lines.append(f"{idx + 1}: {line.strip()}")
        gates.append(
            _gate(
                f"doc_scope_marker:{_repo_relative(path)}",
                scope_present,
                "contains explicit non-promotion wording"
                if scope_present
                else "missing absolute-flux non-promotion wording",
            )
        )
        gates.append(
            _gate(
                f"doc_no_absolute_flux_overclaim:{_repo_relative(path)}",
                not overclaim_lines,
                "; ".join(overclaim_lines[:3])
                or "no positive absolute-flux predictor wording found",
            )
        )
        rows.append(
            {
                "doc": _repo_relative(path),
                "has_scope_marker": scope_present,
                "overclaim_lines": overclaim_lines,
            }
        )
    return gates, rows


def build_guardrail_audit(
    report_patterns: list[str],
    doc_paths: list[str | Path],
    figure_bases: list[str | Path] | None = None,
    figure_index_path: str | Path = DEFAULT_MANUSCRIPT_INDEX,
) -> dict[str, Any]:
    report_paths = _expand_patterns(report_patterns)
    gates: list[dict[str, Any]] = []
    report_rows: list[dict[str, Any]] = []
    input_rows: list[dict[str, Any]] = []
    promotion_rows: list[dict[str, Any]] = []
    manuscript_rows: list[dict[str, Any]] = []
    figure_rows: list[dict[str, Any]] = []

    for path in report_paths:
        data = _load_json(path)
        if data.get("kind") == "quasilinear_calibration_report":
            report_gates, summary = _audit_calibration_report(path, data)
            gates.extend(report_gates)
            report_rows.append(summary)
        input_gates, input_summary = _audit_input_validation(path, data)
        gates.extend(input_gates)
        if input_summary is not None:
            input_rows.append(input_summary)
        promotion_gates, promotion_summary = _audit_promotion_gate(path, data)
        gates.extend(promotion_gates)
        if promotion_summary is not None:
            promotion_rows.append(promotion_summary)
        manuscript_gates, manuscript_summary = _audit_manuscript_readiness(path, data)
        gates.extend(manuscript_gates)
        if manuscript_summary is not None:
            manuscript_rows.append(manuscript_summary)

    doc_gates, doc_rows = _audit_docs([Path(path) for path in doc_paths])
    gates.extend(doc_gates)
    figure_gates, figure_rows = _audit_manuscript_figures(
        figure_bases
        if figure_bases is not None
        else list(DEFAULT_MANUSCRIPT_FIGURE_BASES),
        figure_index_path,
    )
    gates.extend(figure_gates)
    passed = all(bool(gate["passed"]) for gate in gates)
    failed = [gate for gate in gates if not bool(gate["passed"])]
    return {
        "kind": "quasilinear_promotion_guardrail_audit",
        "claim_level": "validation_metadata_and_docs_guardrail_not_physics_claim",
        "passed": passed,
        "gate_report": {
            "case": "quasilinear_absolute_flux_promotion_guardrails",
            "source": "tracked quasilinear JSON artifacts and documentation scope checks",
            "passed": passed,
            "max_abs_error": 0.0 if passed else 1.0,
            "max_rel_error": 0.0 if passed else 1.0,
            "gates": gates,
        },
        "summary": {
            "n_reports_scanned": len(report_paths),
            "n_calibration_reports": len(report_rows),
            "n_input_validation_reports": len(input_rows),
            "n_promotion_gate_reports": len(promotion_rows),
            "n_manuscript_readiness_reports": len(manuscript_rows),
            "n_doc_checks": len(doc_rows),
            "n_manuscript_figure_checks": len(figure_rows),
            "n_failed_gates": len(failed),
        },
        "calibration_reports": report_rows,
        "input_validation_reports": input_rows,
        "promotion_gate_reports": promotion_rows,
        "manuscript_readiness_reports": manuscript_rows,
        "manuscript_figure_provenance": figure_rows,
        "doc_checks": doc_rows,
        "notes": (
            "Fast metadata guardrail only: it verifies finite nonlinear window statistics, "
            "train/holdout provenance, absolute-flux promotion gates, manuscript-readiness QL scope, "
            "manuscript figure JSON sidecars, explicit failed-baseline metadata, "
            "and conservative README/docs wording. "
            "It does not replace nonlinear convergence simulations."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        action="append",
        dest="reports",
        default=None,
        help="Report JSON path or glob.",
    )
    parser.add_argument(
        "--doc",
        action="append",
        dest="docs",
        default=None,
        help="Documentation file to scope-check.",
    )
    parser.add_argument(
        "--figure-base",
        action="append",
        dest="figure_bases",
        default=None,
        help=(
            "Manuscript QL model-development figure base to audit. "
            "Defaults to the tracked quasilinear manuscript figure set."
        ),
    )
    parser.add_argument(
        "--figure-index",
        type=Path,
        default=DEFAULT_MANUSCRIPT_INDEX,
        help="Manuscript figure index file to check for figure/sidecar references.",
    )
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_guardrail_audit(
        args.reports or list(DEFAULT_REPORT_PATTERNS),
        args.docs or [str(path) for path in DEFAULT_DOCS],
        args.figure_bases,
        args.figure_index,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.out_json}")
    print(
        "quasilinear_promotion_guardrails_passed={passed} failed_gates={n_failed}".format(
            passed=payload["passed"],
            n_failed=payload["summary"]["n_failed_gates"],
        )
    )
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
