#!/usr/bin/env python3
"""Build a quasilinear absolute-flux holdout gap report from tracked artifacts.

The report is intentionally a readiness/gap artifact. It consumes only frozen
JSON metadata and must not be read as promoting a runtime absolute-flux
predictor.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from spectraxgk.artifacts.plotting import set_plot_style  # noqa: E402
from spectraxgk.validation.quasilinear.holdout_admission import (  # noqa: E402
    external_vmec_holdout_admission_status,
)


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "docs" / "_static"
DEFAULT_OUT = STATIC / "quasilinear_holdout_gap_report.png"
DEFAULT_MODEL_SELECTION = STATIC / "quasilinear_model_selection_status.json"
DEFAULT_TRAIN_HOLDOUT = STATIC / "quasilinear_stellarator_train_holdout_report.json"
DEFAULT_WINDOW_STATS = STATIC / "nonlinear_window_statistics.json"
DEFAULT_DATASET_SUFFICIENCY = STATIC / "quasilinear_dataset_sufficiency.json"
DEFAULT_SCREENING_SKILL = STATIC / "quasilinear_screening_skill.json"
DEFAULT_EXTERNAL_GLOB = str(STATIC / "external_vmec_*_convergence_gate.json")
DEFAULT_NEXT_CANDIDATES = 4
DEFAULT_MIN_ABSOLUTE_PROMOTION_HOLDOUTS = 9
DEFAULT_MIN_EXTERNAL_VMEC_HOLDOUT_FAMILIES = 4
DEFAULT_MIN_NONAXISYMMETRIC_EXTERNAL_HOLDOUT_FAMILIES = 1
CLAIM_LEVEL = "holdout_gap_report_no_absolute_flux_promotion"

CSV_FIELDS = (
    "section",
    "rank",
    "status",
    "case",
    "geometry",
    "split",
    "gate_case",
    "gate_passed",
    "raw_gate_passed",
    "promotion_gate_passed",
    "claim_level",
    "claim_level_acceptable",
    "admitted_for_calibration",
    "negative_evidence",
    "source_artifact",
    "observed_heat_flux",
    "predicted_heat_flux",
    "absolute_relative_error",
    "holdout_mean_abs_relative_error",
    "holdout_mean_rel_gate",
    "next_best_score",
    "failed_gates",
    "admission_blockers",
    "reason",
)

GATE_LABELS = {
    "common_window_max_relative_slope_per_time": "common slope",
    "least_window_max_relative_slope_per_time": "least-window slope",
    "common_window_max_heat_flux_cv": "common CV",
    "least_window_max_heat_flux_cv": "least-window CV",
    "common_window_pairwise_heat_flux_symmetric_relative_difference": "common grid shift",
    "least_window_pairwise_heat_flux_symmetric_relative_difference": "least-window grid shift",
    "common_window_min_samples_deficit": "common samples",
    "least_window_min_samples_deficit": "least-window samples",
}


class ArtifactError(ValueError):
    """Raised when an input artifact does not match the expected JSON shape."""


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


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ArtifactError(f"{path} does not contain a JSON object")
    return payload


def _repo_relative_path(path: str | Path | None) -> str:
    if path in (None, ""):
        return ""
    raw = Path(str(path))
    try:
        return raw.resolve().relative_to(ROOT.resolve()).as_posix()
    except (OSError, ValueError):
        text = str(path)
        root = str(ROOT.resolve())
        return text[len(root) + 1 :] if text.startswith(root + "/") else text


def _repo_relative_pattern(pattern: str) -> str:
    root = str(ROOT.resolve())
    return pattern[len(root) + 1 :] if pattern.startswith(root + "/") else pattern


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _relative_error(observed: Any, predicted: Any, *, floor: float = 1.0e-12) -> float | None:
    obs = _finite_float(observed)
    pred = _finite_float(predicted)
    if obs is None or pred is None:
        return None
    return abs(pred - obs) / max(abs(obs), float(floor))


def _case_label(case: str) -> str:
    label = case
    label = label.replace("_external_vmec", " VMEC")
    label = label.replace("_nonlinear_window", "")
    label = label.replace("_long_window", "")
    label = label.replace("_window", "")
    label = label.replace("_", " ")
    return label.title().replace("Vmec", "VMEC").replace("W7X", "W7-X")


def _short_label(label: str, *, max_chars: int = 46) -> str:
    """Return a compact plot label without changing machine-readable JSON."""

    text = " ".join(str(label).split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "..."


def _external_family(case: str, artifact: str = "") -> str:
    text = f"{case} {artifact}".lower()
    if "updown" in text or "up-down" in text:
        return "updown_asym_external_vmec"
    if "itermodel" in text:
        return "itermodel_external_vmec"
    if "li383" in text:
        return "li383_external_vmec"
    if "qi_stel" in text or "quasi-isodynamic" in text or "nfp3_qi" in text:
        return "qi_external_vmec"
    if "qa" in text and ("landremanpaul" in text or "quasi-axisymmetric" in text):
        return "qa_external_vmec"
    if "dshape" in text or "d-shaped" in text or "d shaped" in text:
        return "dshape_external_vmec"
    if "circular" in text:
        return "circular_external_vmec"
    if "shaped_tokamak" in text or "shaped tokamak" in text:
        return "shaped_tokamak_external_vmec"
    if "cth" in text:
        return "cth_like_external_vmec"
    if "qh" in text or "nfp4" in text:
        return "qh_external_vmec"
    if "basic_non_stellsym" in text or "non_stellsym" in text:
        return "non_stellsym_external_vmec"
    if "purely_toroidal" in text:
        return "purely_toroidal_external_vmec"
    if "solovev" in text:
        return "solovev_external_vmec"
    return "external_vmec"


def _is_external_vmec_family(family: str) -> bool:
    return str(family).endswith("external_vmec")


def _is_nonaxisymmetric_external_vmec_family(family: str) -> bool:
    text = str(family).lower()
    return _is_external_vmec_family(text) and any(
        marker in text
        for marker in (
            "cth",
            "li383",
            "non_stellsym",
            "qa_",
            "qh_",
            "qi_",
        )
    )


def _gate_limit(gate: dict[str, Any]) -> float | None:
    atol = _finite_float(gate.get("atol"))
    rtol = _finite_float(gate.get("rtol"))
    reference = abs(_finite_float(gate.get("reference")) or 0.0)
    if atol is None and rtol is None:
        return None
    return (atol or 0.0) + (rtol or 0.0) * reference


def _gate_observed_error(gate: dict[str, Any]) -> float | None:
    for key in ("abs_error", "observed"):
        value = _finite_float(gate.get(key))
        if value is not None:
            return abs(value)
    return None


def _gate_ratio(gate: dict[str, Any]) -> float | None:
    error = _gate_observed_error(gate)
    limit = _gate_limit(gate)
    if error is None or limit is None:
        return None
    if limit == 0.0:
        return 0.0 if error == 0.0 else math.inf
    return error / limit


def _format_gate_value(value: Any) -> str:
    finite = _finite_float(value)
    if finite is None:
        return "n/a"
    if abs(finite) >= 100 or (0 < abs(finite) < 1.0e-3):
        return f"{finite:.3e}"
    return f"{finite:.4g}"


def _failed_gate_details(gates: list[dict[str, Any]]) -> list[str]:
    details: list[str] = []
    for gate in gates:
        if bool(gate.get("passed", False)):
            continue
        metric = str(gate.get("metric", "unknown"))
        label = GATE_LABELS.get(metric, metric)
        observed = _format_gate_value(_gate_observed_error(gate))
        limit = _format_gate_value(_gate_limit(gate))
        details.append(f"{label}: {observed} > {limit}")
    return details


def _load_external_gate(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    gate_report = payload.get("gate_report", {})
    if not isinstance(gate_report, dict):
        gate_report = {}
    gates = gate_report.get("gates", [])
    if not isinstance(gates, list):
        gates = []
    promotion_gate = payload.get("promotion_gate", {})
    if not isinstance(promotion_gate, dict):
        promotion_gate = {}
    admission = external_vmec_holdout_admission_status(payload)
    passed = bool(admission["admissible_for_calibration"])
    ratios = [_gate_ratio(gate) for gate in gates]
    finite_ratios = [ratio for ratio in ratios if ratio is not None and math.isfinite(ratio)]
    failed_ratios = [
        ratio
        for ratio, gate in zip(ratios, gates, strict=False)
        if ratio is not None and not bool(gate.get("passed", False))
    ]
    finite_failed = [ratio for ratio in failed_ratios if math.isfinite(ratio)]
    failed_details = _failed_gate_details([gate for gate in gates if isinstance(gate, dict)])
    case = str(payload.get("case", path.stem))
    return {
        "artifact": _repo_relative_path(path),
        "case": case,
        "claim_level": str(payload.get("claim_level", "")),
        "family": _external_family(case, path.name),
        "passed": passed,
        "raw_gate_passed": bool(admission["raw_gate_passed"]),
        "promotion_gate_passed": bool(admission["promotion_gate_passed"]),
        "claim_level_acceptable": bool(admission["claim_level_acceptable"]),
        "admitted_for_calibration": passed,
        "negative_evidence": bool(admission["negative_evidence"]),
        "admission_blockers": list(admission["admission_blockers"]),
        "reason": str(promotion_gate.get("reason", "")),
        "failed_gates": [str(gate.get("metric", "unknown")) for gate in gates if not bool(gate.get("passed", False))],
        "failed_gate_details": failed_details,
        "max_gate_ratio": max(finite_ratios) if finite_ratios else 0.0,
        "max_failed_gate_ratio": max(finite_failed) if finite_failed else (math.inf if failed_ratios else 0.0),
        "common_pairwise_heat_flux_difference": _finite_float(
            (payload.get("common_window") or {}).get("max_pairwise_heat_flux_symmetric_relative_difference")
        ),
        "least_pairwise_heat_flux_difference": _finite_float(
            (payload.get("least_windows") or {}).get("max_pairwise_heat_flux_symmetric_relative_difference")
        ),
        "gate_report_passed": bool(gate_report.get("passed", False)),
    }


def _dataset_case_maps(dataset: dict[str, Any] | None) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    if not dataset:
        return {}, {}
    cases = dataset.get("cases", [])
    input_cases = (dataset.get("input_validation") or {}).get("cases", [])
    by_case = {str(item.get("case", "")): item for item in cases if isinstance(item, dict)}
    input_by_case = {
        str(item.get("case", "")): item
        for item in input_cases
        if isinstance(item, dict) and item.get("case")
    }
    return by_case, input_by_case


def _window_case_passes(window_stats: dict[str, Any]) -> dict[str, bool]:
    passed = window_stats.get("case_gate_passed", {})
    if not isinstance(passed, dict):
        return {}
    return {str(key): bool(value) for key, value in passed.items()}


def _window_case_thresholds(window_stats: dict[str, Any]) -> dict[str, float | None]:
    thresholds = window_stats.get("case_gate_thresholds", {})
    if not isinstance(thresholds, dict):
        return {}
    return {str(key): _finite_float(value) for key, value in thresholds.items()}


def _window_case_max_mean(window_stats: dict[str, Any]) -> dict[str, float | None]:
    values = window_stats.get("max_mean_rel_abs_by_case", {})
    if not isinstance(values, dict):
        return {}
    return {str(key): _finite_float(value) for key, value in values.items()}


def _point_gate_status(
    point: dict[str, Any],
    *,
    dataset_case: dict[str, Any] | None,
    input_case: dict[str, Any] | None,
    window_passed: dict[str, bool],
    external_by_case: dict[str, dict[str, Any]],
) -> tuple[str, bool | None, str]:
    gate_case = ""
    gate_passed: bool | None = None
    reason = ""
    if dataset_case:
        gate_case = str(dataset_case.get("gate_case", ""))
    if input_case:
        gate_case = str(input_case.get("gate_case", gate_case))
        gate_passed = bool(input_case.get("passed", input_case.get("gate_passed", False)))
        reason = str(input_case.get("reason", ""))
    if gate_case in window_passed:
        gate_passed = bool(window_passed[gate_case])
        reason = reason or "matched nonlinear release-window gate"
    if gate_case in external_by_case:
        gate_passed = bool(external_by_case[gate_case]["passed"])
        reason = reason or str(external_by_case[gate_case].get("reason", ""))
    return gate_case, gate_passed, reason


def _build_point_rows(
    train_holdout: dict[str, Any],
    *,
    dataset: dict[str, Any] | None,
    window_stats: dict[str, Any],
    external_by_case: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dataset_cases, input_cases = _dataset_case_maps(dataset)
    window_passed = _window_case_passes(window_stats)
    observed_floor = float(train_holdout.get("observed_floor", 1.0e-12))
    holdout_summary = (train_holdout.get("by_split") or {}).get("holdout", {})
    holdout_mean_error = _finite_float(holdout_summary.get("mean_abs_relative_error"))
    holdout_gate = _finite_float(train_holdout.get("holdout_mean_rel_gate"))
    rows: list[dict[str, Any]] = []
    training: list[dict[str, Any]] = []
    points = train_holdout.get("points", [])
    if not isinstance(points, list):
        raise ArtifactError("train/holdout report points must be a list")
    for point in points:
        if not isinstance(point, dict):
            continue
        case = str(point.get("case", ""))
        dataset_case = dataset_cases.get(case)
        input_case = input_cases.get(case)
        gate_case, gate_passed, gate_reason = _point_gate_status(
            point,
            dataset_case=dataset_case,
            input_case=input_case,
            window_passed=window_passed,
            external_by_case=external_by_case,
        )
        rel_error = _relative_error(
            point.get("observed_heat_flux"),
            point.get("predicted_heat_flux"),
            floor=observed_floor,
        )
        split = str(point.get("split", ""))
        status = "admitted_holdout" if split == "holdout" else "training_reference"
        reason = "admitted to current train/holdout artifact"
        if split == "holdout":
            reason = (
                "admitted nonlinear holdout with passed input gate; current absolute-flux "
                "calibration remains blocked by the aggregate holdout error gate"
            )
        elif split == "train":
            reason = "training/reference point, not an independent holdout"
        if gate_passed is False:
            reason = "listed in train/holdout report but matched input gate is not passed"
        if gate_reason:
            reason = f"{reason}; {gate_reason}"
        row = {
            "section": "admitted_holdouts" if split == "holdout" else "training_references",
            "rank": None,
            "status": status,
            "case": case,
            "case_label": _case_label(case),
            "geometry": str(point.get("geometry", "")),
            "split": split,
            "gate_case": gate_case,
            "gate_passed": gate_passed,
            "source_artifact": _repo_relative_path(train_holdout.get("artifact", DEFAULT_TRAIN_HOLDOUT)),
            "nonlinear_artifact": _repo_relative_path(point.get("nonlinear_artifact")),
            "quasilinear_artifact": _repo_relative_path(point.get("quasilinear_artifact")),
            "observed_heat_flux": _finite_float(point.get("observed_heat_flux")),
            "predicted_heat_flux": _finite_float(point.get("predicted_heat_flux")),
            "absolute_relative_error": rel_error,
            "holdout_mean_abs_relative_error": holdout_mean_error if split == "holdout" else None,
            "holdout_mean_rel_gate": holdout_gate if split == "holdout" else None,
            "next_best_score": None,
            "failed_gates": [],
            "reason": reason,
            "absolute_flux_promoted": False,
        }
        if split == "holdout":
            rows.append(row)
        else:
            training.append(row)
    return rows, training


def _dataset_excluded_rows(
    dataset: dict[str, Any] | None,
    *,
    window_stats: dict[str, Any],
) -> list[dict[str, Any]]:
    if not dataset:
        return []
    rows: list[dict[str, Any]] = []
    window_passed = _window_case_passes(window_stats)
    thresholds = _window_case_thresholds(window_stats)
    max_mean = _window_case_max_mean(window_stats)
    excluded = dataset.get("excluded_validated_nonlinear_cases", [])
    if not isinstance(excluded, list):
        return rows
    for item in excluded:
        if not isinstance(item, dict):
            continue
        case = str(item.get("case", ""))
        rows.append(
            {
                "section": "excluded_candidates",
                "rank": None,
                "status": "excluded_scope",
                "case": case,
                "case_label": _case_label(case),
                "geometry": str(item.get("geometry", "")),
                "split": "excluded",
                "gate_case": case,
                "gate_passed": bool(item.get("gate_passed", window_passed.get(case, False))),
                "source_artifact": _repo_relative_path(DEFAULT_DATASET_SUFFICIENCY),
                "observed_heat_flux": None,
                "predicted_heat_flux": None,
                "absolute_relative_error": None,
                "holdout_mean_abs_relative_error": max_mean.get(case),
                "holdout_mean_rel_gate": thresholds.get(case),
                "next_best_score": None,
                "failed_gates": [],
                "reason": str(item.get("reason", "validated nonlinear case is outside this electrostatic QL lane")),
                "absolute_flux_promoted": False,
                "eligible_for_next_candidate": False,
            }
        )
    return rows


def _window_excluded_rows(
    window_stats: dict[str, Any],
    *,
    known_gate_cases: set[str],
    already_reported: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    window_passed = _window_case_passes(window_stats)
    thresholds = _window_case_thresholds(window_stats)
    max_mean = _window_case_max_mean(window_stats)
    cases = window_stats.get("cases", [])
    if not isinstance(cases, list):
        return rows
    for raw_case in cases:
        case = str(raw_case)
        if case in known_gate_cases or case in already_reported:
            continue
        gate_passed = window_passed.get(case)
        status = "excluded_not_in_quasilinear_dataset" if gate_passed else "excluded_failed_window_gate"
        if gate_passed:
            reason = "passed nonlinear window gate but is absent from the current quasilinear train/holdout dataset"
        else:
            reason = "nonlinear window gate is not passed"
        rows.append(
            {
                "section": "excluded_candidates",
                "rank": None,
                "status": status,
                "case": case,
                "case_label": _case_label(case),
                "geometry": "",
                "split": "excluded",
                "gate_case": case,
                "gate_passed": gate_passed,
                "source_artifact": _repo_relative_path(DEFAULT_WINDOW_STATS),
                "observed_heat_flux": None,
                "predicted_heat_flux": None,
                "absolute_relative_error": None,
                "holdout_mean_abs_relative_error": max_mean.get(case),
                "holdout_mean_rel_gate": thresholds.get(case),
                "next_best_score": None,
                "failed_gates": [] if gate_passed else ["nonlinear_window_case_gate"],
                "reason": reason,
                "absolute_flux_promoted": False,
                "eligible_for_next_candidate": bool(gate_passed),
            }
        )
    return rows


def _external_excluded_rows(
    external_gates: list[dict[str, Any]],
    *,
    known_gate_cases: set[str],
    holdout_families: set[str],
    training_families: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for gate in external_gates:
        if str(gate["case"]) in known_gate_cases:
            continue
        family = str(gate.get("family", "external_vmec"))
        failed_gates = list(gate.get("failed_gates", []))
        failed_details = list(gate.get("failed_gate_details", []))
        passed = bool(gate.get("passed", False))
        raw_gate_passed = bool(gate.get("raw_gate_passed", passed))
        promotion_gate_passed = bool(gate.get("promotion_gate_passed", passed))
        claim_level_acceptable = bool(gate.get("claim_level_acceptable", passed))
        admitted_for_calibration = bool(gate.get("admitted_for_calibration", passed))
        negative_evidence = bool(gate.get("negative_evidence", not passed))
        admission_blockers = [str(item) for item in gate.get("admission_blockers", [])]
        if passed and family in holdout_families:
            status = "excluded_superseded_by_current_holdout_family"
            reason = "passed external-VMEC gate, but this family is already represented by a current admitted holdout"
            eligible = False
        elif passed and family in training_families:
            status = "excluded_same_family_training_audit"
            reason = (
                "passed external-VMEC same-family audit, but this family is already consumed by a training reference; "
                "it is reproducibility evidence rather than an independent holdout"
            )
            eligible = False
        elif passed:
            status = "next_best_passed_gate_not_in_holdout_report"
            reason = "passed external-VMEC gate but is not yet represented as a current holdout"
            eligible = True
        elif negative_evidence and raw_gate_passed:
            status = "excluded_negative_external_evidence"
            detail = "; ".join(admission_blockers) or str(
                gate.get("reason", "failed external-VMEC holdout admission")
            )
            reason = (
                "not admitted because external-VMEC holdout admission failed closed: "
                f"{detail}"
            )
            eligible = False
        else:
            status = "excluded_failed_external_gate"
            detail = "; ".join(failed_details) or str(gate.get("reason", "failed external-VMEC gate"))
            reason = f"not admitted because external-VMEC convergence gate failed: {detail}"
            eligible = family not in holdout_families
        rows.append(
            {
                "section": "excluded_candidates",
                "rank": None,
                "status": status,
                "case": str(gate["case"]),
                "case_label": str(gate["case"]),
                "geometry": family,
                "split": "excluded",
                "gate_case": str(gate["case"]),
                "gate_passed": passed,
                "raw_gate_passed": raw_gate_passed,
                "promotion_gate_passed": promotion_gate_passed,
                "claim_level": str(gate.get("claim_level", "")),
                "claim_level_acceptable": claim_level_acceptable,
                "admitted_for_calibration": admitted_for_calibration,
                "negative_evidence": negative_evidence,
                "source_artifact": str(gate["artifact"]),
                "observed_heat_flux": None,
                "predicted_heat_flux": None,
                "absolute_relative_error": None,
                "holdout_mean_abs_relative_error": None,
                "holdout_mean_rel_gate": None,
                "next_best_score": float(gate.get("max_failed_gate_ratio", 0.0)),
                "failed_gates": failed_gates,
                "failed_gate_details": failed_details,
                "admission_blockers": admission_blockers,
                "reason": reason,
                "absolute_flux_promoted": False,
                "eligible_for_next_candidate": eligible,
            }
        )
    return rows


def _next_candidates(
    training_references: list[dict[str, Any]],
    excluded_rows: list[dict[str, Any]],
    *,
    max_candidates: int,
) -> list[dict[str, Any]]:
    candidates: list[tuple[tuple[float, float, str], dict[str, Any]]] = []
    for row in training_references:
        geometry = str(row.get("geometry", ""))
        if "external_vmec" not in geometry:
            continue
        if row.get("gate_passed") is not True:
            continue
        candidate = dict(row)
        candidate.update(
            {
                "section": "next_best_candidates",
                "status": "training_reference_not_independent_holdout",
                "next_best_score": 0.0,
                "reason": (
                    "passed nonlinear gate but is currently consumed as a training/reference point; "
                    "promotion readiness needs a separate holdout in this family or another electrostatic VMEC family"
                ),
            }
        )
        candidates.append(((0.0, 0.0, str(candidate.get("case", ""))), candidate))
    for row in excluded_rows:
        if not bool(row.get("eligible_for_next_candidate", False)):
            continue
        score = _finite_float(row.get("next_best_score"))
        if score is None:
            score = 0.0 if row.get("gate_passed") is True else math.inf
        priority = 1.0 if row.get("gate_passed") is True else 2.0
        candidate = dict(row)
        candidate["section"] = "next_best_candidates"
        if not str(candidate.get("status", "")).startswith("next_best"):
            candidate["status"] = "next_best_failed_gate_candidate"
        candidates.append(((priority, score, str(candidate.get("case", ""))), candidate))
    sorted_candidates = [candidate for _, candidate in sorted(candidates, key=lambda item: item[0])]
    selected = sorted_candidates[:max_candidates]
    for rank, row in enumerate(selected, start=1):
        row["rank"] = rank
    return selected


def _next_actual_need(next_candidates: list[dict[str, Any]]) -> dict[str, Any]:
    train_candidate = next(
        (
            row
            for row in next_candidates
            if row.get("status") == "training_reference_not_independent_holdout"
        ),
        None,
    )
    failed_candidate = next(
        (
            row
            for row in next_candidates
            if row.get("status") == "next_best_failed_gate_candidate"
        ),
        None,
    )
    if train_candidate is not None:
        family = str(train_candidate.get("geometry", "external_vmec"))
        detail = {
            "needed": (
                "new independent electrostatic-compatible nonlinear transport window with a passed "
                "grid/window convergence gate and split=holdout"
            ),
            "preferred_family": family,
            "why": (
                f"{train_candidate.get('case')} already has a passed gate but is assigned to the train split; "
                "it cannot by itself add independent holdout leverage"
            ),
            "claim_boundary": "adding this holdout would expand evidence only; no current absolute-flux predictor is promoted",
        }
        if failed_candidate is not None:
            detail["nearest_tracked_gap"] = {
                "case": failed_candidate.get("case"),
                "source_artifact": failed_candidate.get("source_artifact"),
                "failed_gates": failed_candidate.get("failed_gate_details", failed_candidate.get("failed_gates", [])),
                "next_best_score": failed_candidate.get("next_best_score"),
            }
        return detail
    if failed_candidate is not None:
        return {
            "needed": "extend or retune the nearest failed external-VMEC nonlinear candidate until its convergence gate passes",
            "preferred_family": failed_candidate.get("geometry"),
            "nearest_tracked_gap": {
                "case": failed_candidate.get("case"),
                "source_artifact": failed_candidate.get("source_artifact"),
                "failed_gates": failed_candidate.get("failed_gate_details", failed_candidate.get("failed_gates", [])),
                "next_best_score": failed_candidate.get("next_best_score"),
            },
            "claim_boundary": "this is a proposed next holdout only; no current absolute-flux predictor is promoted",
        }
    return {
        "needed": "new independent nonlinear holdout with passed gates",
        "preferred_family": None,
        "claim_boundary": "no current absolute-flux predictor is promoted",
    }


def _top_case_requirements(
    next_candidates: list[dict[str, Any]],
    excluded_rows: list[dict[str, Any]],
    *,
    max_rows: int = 6,
) -> list[dict[str, Any]]:
    """Return concrete nonlinear cases that would reduce the promotion gap."""

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source, role in (
        (next_candidates, "ranked_next_best"),
        (
            [
                row
                for row in excluded_rows
                if _is_nonaxisymmetric_external_vmec_family(str(row.get("geometry", "")))
                and bool(row.get("eligible_for_next_candidate", False))
            ],
            "missing_nonaxisymmetric_external_vmec_holdout",
        ),
    ):
        for row in source:
            if not isinstance(row, dict):
                continue
            case = str(row.get("case", ""))
            if not case or case in seen:
                continue
            seen.add(case)
            passed = row.get("gate_passed")
            score = _finite_float(row.get("next_best_score"))
            if passed is True:
                action = "add this passed-gate case as an independent holdout, not a training/reference point"
            else:
                action = "extend/refine this nonlinear run until its grid/window convergence gate passes"
            rows.append(
                {
                    "case": case,
                    "family": row.get("geometry"),
                    "role": role,
                    "gate_passed": passed,
                    "next_best_score": score,
                    "failed_gates": list(row.get("failed_gate_details", row.get("failed_gates", []))),
                    "required_action": action,
                    "claim_boundary": "candidate case only; adding it does not by itself promote absolute flux",
                    "source_artifact": row.get("source_artifact"),
                }
            )
            if len(rows) >= max_rows:
                return rows
    return rows


def _absolute_flux_promotion_requirements(
    *,
    admitted: list[dict[str, Any]],
    training: list[dict[str, Any]],
    excluded_rows: list[dict[str, Any]],
    next_candidates: list[dict[str, Any]],
    calibration: dict[str, Any],
    model: dict[str, Any],
) -> dict[str, Any]:
    """Quantify what is missing before absolute-flux promotion can reopen."""

    admitted_families = {
        str(row.get("geometry", ""))
        for row in admitted
        if str(row.get("geometry", ""))
    }
    training_families = {
        str(row.get("geometry", ""))
        for row in training
        if str(row.get("geometry", ""))
    }
    external_holdout_families = sorted(
        family for family in admitted_families if _is_external_vmec_family(family)
    )
    nonaxisym_external_holdout_families = sorted(
        family
        for family in admitted_families
        if _is_nonaxisymmetric_external_vmec_family(family)
    )

    holdout_error = _finite_float(calibration.get("holdout_mean_abs_relative_error"))
    holdout_gate = _finite_float(calibration.get("holdout_mean_rel_gate"))
    error_excess = None
    error_factor = None
    if holdout_error is not None and holdout_gate is not None:
        error_excess = max(0.0, holdout_error - holdout_gate)
        error_factor = math.inf if holdout_gate == 0.0 else holdout_error / holdout_gate

    worst = None
    finite_admitted = [
        row
        for row in admitted
        if _finite_float(row.get("absolute_relative_error")) is not None
    ]
    if finite_admitted:
        worst_row = max(
            finite_admitted,
            key=lambda row: float(_finite_float(row.get("absolute_relative_error")) or 0.0),
        )
        worst = {
            "case": worst_row.get("case"),
            "family": worst_row.get("geometry"),
            "absolute_relative_error": _finite_float(worst_row.get("absolute_relative_error")),
            "observed_heat_flux": _finite_float(worst_row.get("observed_heat_flux")),
            "predicted_heat_flux": _finite_float(worst_row.get("predicted_heat_flux")),
        }

    total_needed = max(
        0,
        DEFAULT_MIN_ABSOLUTE_PROMOTION_HOLDOUTS - len(admitted),
    )
    external_needed = max(
        0,
        DEFAULT_MIN_EXTERNAL_VMEC_HOLDOUT_FAMILIES - len(external_holdout_families),
    )
    nonaxisym_external_needed = max(
        0,
        DEFAULT_MIN_NONAXISYMMETRIC_EXTERNAL_HOLDOUT_FAMILIES
        - len(nonaxisym_external_holdout_families),
    )
    reconsideration_passed = bool(
        holdout_error is not None
        and holdout_gate is not None
        and holdout_error <= holdout_gate
        and total_needed == 0
        and external_needed == 0
        and nonaxisym_external_needed == 0
        and bool(model.get("passed", False))
        and bool(calibration.get("passed", False))
    )

    gates = [
        {
            "metric": "absolute_train_holdout_report_passed",
            "passed": bool(calibration.get("passed", False)),
            "current": bool(calibration.get("passed", False)),
            "required": True,
            "detail": "the current one-constant absolute calibration report is still failed",
        },
        {
            "metric": "holdout_mean_abs_relative_error",
            "passed": bool(holdout_error is not None and holdout_gate is not None and holdout_error <= holdout_gate),
            "current": holdout_error,
            "required": f"<= {holdout_gate}" if holdout_gate is not None else "finite gate",
            "detail": "absolute-flux promotion needs a calibrated report that passes the held-out transport gate",
        },
        {
            "metric": "minimum_total_independent_holdouts",
            "passed": total_needed == 0,
            "current": len(admitted),
            "required": DEFAULT_MIN_ABSOLUTE_PROMOTION_HOLDOUTS,
            "additional_needed": total_needed,
            "detail": "holdouts must be passed, post-transient nonlinear transport windows outside the training split",
        },
        {
            "metric": "minimum_external_vmec_holdout_families",
            "passed": external_needed == 0,
            "current": len(external_holdout_families),
            "required": DEFAULT_MIN_EXTERNAL_VMEC_HOLDOUT_FAMILIES,
            "additional_needed": external_needed,
            "detail": "external-VMEC holdout families test transfer beyond built-in local cases",
        },
        {
            "metric": "minimum_nonaxisymmetric_external_vmec_holdout_families",
            "passed": nonaxisym_external_needed == 0,
            "current": len(nonaxisym_external_holdout_families),
            "required": DEFAULT_MIN_NONAXISYMMETRIC_EXTERNAL_HOLDOUT_FAMILIES,
            "additional_needed": nonaxisym_external_needed,
            "detail": "at least one external stellarator-like VMEC holdout is required before broad stellarator absolute-flux claims",
        },
        {
            "metric": "scoped_model_selection_gate_passed",
            "passed": bool(model.get("passed", False)),
            "current": bool(model.get("passed", False)),
            "required": True,
            "detail": "model-selection skill can pass while absolute-flux promotion remains blocked",
        },
    ]
    blockers = [str(gate["metric"]) for gate in gates if not bool(gate["passed"])]
    return _json_clean(
        {
            "kind": "quasilinear_absolute_flux_promotion_requirements",
            "absolute_flux_promoted": False,
            "reconsideration_ready": reconsideration_passed,
            "claim_boundary": (
                "This section defines the evidence needed to reopen absolute-flux promotion; "
                "it is not a promoted runtime/TOML predictor."
            ),
            "numeric_gap": {
                "holdout_mean_abs_relative_error": holdout_error,
                "holdout_mean_rel_gate": holdout_gate,
                "error_excess_over_gate": error_excess,
                "error_factor_to_gate": error_factor,
                "worst_admitted_holdout": worst,
                "scoped_model_selection_mean_abs_relative_error": model.get(
                    "candidate_mean_abs_relative_error"
                ),
                "scoped_model_selection_interval_coverage": model.get(
                    "candidate_prediction_interval_coverage"
                ),
            },
            "coverage_gap": {
                "admitted_holdout_families": sorted(admitted_families),
                "training_families": sorted(training_families),
                "external_vmec_holdout_families": external_holdout_families,
                "nonaxisymmetric_external_vmec_holdout_families": nonaxisym_external_holdout_families,
                "additional_total_independent_holdouts_needed": total_needed,
                "additional_external_vmec_holdout_families_needed": external_needed,
                "additional_nonaxisymmetric_external_vmec_holdout_families_needed": nonaxisym_external_needed,
            },
            "gates": gates,
            "blockers": blockers,
            "required_nonlinear_cases": _top_case_requirements(
                next_candidates,
                excluded_rows,
            ),
        }
    )


def _model_selection_summary(model_selection: dict[str, Any]) -> dict[str, Any]:
    metrics = model_selection.get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    return {
        "artifact": _repo_relative_path(DEFAULT_MODEL_SELECTION),
        "kind": str(model_selection.get("kind", "")),
        "claim_level": str(model_selection.get("claim_level", "")),
        "passed": bool(model_selection.get("passed", False)),
        "required_candidate": str(model_selection.get("required_candidate", "")),
        "accepted_candidates": list(model_selection.get("accepted_candidates", [])),
        "candidate_mean_abs_relative_error": _finite_float(metrics.get("candidate_mean_abs_relative_error")),
        "candidate_prediction_interval_coverage": _finite_float(metrics.get("candidate_prediction_interval_coverage")),
        "transport_mean_relative_error_gate": _finite_float(metrics.get("transport_mean_relative_error_gate")),
        "absolute_flux_promoted": False,
        "reason": "scoped model-selection result only; not a runtime/TOML absolute-flux predictor",
    }


def _calibration_summary(train_holdout: dict[str, Any]) -> dict[str, Any]:
    by_split = train_holdout.get("by_split", {})
    if not isinstance(by_split, dict):
        by_split = {}
    holdout = by_split.get("holdout", {}) if isinstance(by_split.get("holdout", {}), dict) else {}
    train = by_split.get("train", {}) if isinstance(by_split.get("train", {}), dict) else {}
    return {
        "artifact": _repo_relative_path(DEFAULT_TRAIN_HOLDOUT),
        "kind": str(train_holdout.get("kind", "")),
        "claim_level": str(train_holdout.get("claim_level", "")),
        "passed": bool(train_holdout.get("passed", False)),
        "n_train": int(train.get("n", 0) or 0),
        "n_holdout": int(holdout.get("n", 0) or 0),
        "holdout_mean_abs_relative_error": _finite_float(holdout.get("mean_abs_relative_error")),
        "holdout_max_abs_relative_error": _finite_float(holdout.get("max_abs_relative_error")),
        "holdout_mean_rel_gate": _finite_float(train_holdout.get("holdout_mean_rel_gate")),
        "absolute_flux_promoted": False,
        "reason": "train/holdout calibration report remains a calibration dataset, not a promoted absolute-flux model",
    }


def _screening_skill_summary(screening_skill: dict[str, Any] | None) -> dict[str, Any] | None:
    """Summarize rank/correlation readiness without promoting absolute flux."""

    if not isinstance(screening_skill, dict):
        return None
    gates = screening_skill.get("gates", {})
    gates = gates if isinstance(gates, dict) else {}
    models = screening_skill.get("models", [])
    models = models if isinstance(models, list) else []
    by_model = {
        str(row.get("model")): row
        for row in models
        if isinstance(row, dict) and row.get("model")
    }
    best = str(gates.get("best_screening_model") or "")
    best_holdout = str(gates.get("best_holdout_screening_model") or best)
    best_row = by_model.get(best, {})
    best_holdout_row = by_model.get(best_holdout, {})
    return {
        "artifact": _repo_relative_path(DEFAULT_SCREENING_SKILL),
        "kind": str(screening_skill.get("kind", "")),
        "claim_level": str(screening_skill.get("claim_level", "")),
        "accepted_screening_models": list(gates.get("accepted_screening_models", []) or []),
        "accepted_holdout_screening_models": list(gates.get("accepted_holdout_screening_models", []) or []),
        "accepted_absolute_flux_models": list(gates.get("accepted_absolute_flux_models", []) or []),
        "screening_correlation_passed": bool(gates.get("screening_correlation_passed", False)),
        "holdout_screening_correlation_passed": bool(
            gates.get("holdout_screening_correlation_passed", False)
        ),
        "absolute_flux_promotion_passed": bool(gates.get("absolute_flux_promotion_passed", False)),
        "best_screening_model": best or None,
        "best_screening_spearman": _finite_float(best_row.get("spearman")),
        "best_screening_pairwise_order_accuracy": _finite_float(
            best_row.get("pairwise_order_accuracy")
        ),
        "best_screening_mean_abs_relative_error": _finite_float(
            best_row.get("mean_abs_relative_error")
        ),
        "best_holdout_screening_model": best_holdout or None,
        "best_holdout_spearman": _finite_float(best_holdout_row.get("holdout_spearman")),
        "best_holdout_pairwise_order_accuracy": _finite_float(
            best_holdout_row.get("holdout_pairwise_order_accuracy")
        ),
        "best_holdout_mean_abs_relative_error": _finite_float(
            best_holdout_row.get("holdout_mean_abs_relative_error")
        ),
        "spearman_gate": _finite_float(gates.get("spearman_gate")),
        "pairwise_order_gate": _finite_float(gates.get("pairwise_order_gate")),
        "reason": (
            "full-portfolio screening can pass while held-out-only screening and "
            "absolute-flux promotion remain blocked"
        ),
    }


def _screening_promotion_requirements(
    screening: dict[str, Any] | None,
    *,
    absolute_requirements: dict[str, Any],
) -> dict[str, Any] | None:
    """Return evidence requirements for correlation-screening promotion."""

    if not isinstance(screening, dict):
        return None
    coverage_gap = absolute_requirements.get("coverage_gap", {})
    if not isinstance(coverage_gap, dict):
        coverage_gap = {}
    additional_holdouts = coverage_gap.get("additional_total_independent_holdouts_needed")
    gates = [
        {
            "metric": "full_portfolio_screening_correlation_passed",
            "passed": bool(screening.get("screening_correlation_passed", False)),
            "current": screening.get("accepted_screening_models", []),
            "required": "at least one scoped screening model",
            "detail": "useful model-development signal on the admitted portfolio",
        },
        {
            "metric": "heldout_screening_correlation_passed",
            "passed": bool(screening.get("holdout_screening_correlation_passed", False)),
            "current": screening.get("accepted_holdout_screening_models", []),
            "required": "at least one held-out screening model",
            "detail": "held-out-only rank/correlation gate must pass before screening promotion",
        },
        {
            "metric": "absolute_flux_promotion_passed",
            "passed": bool(screening.get("absolute_flux_promotion_passed", False)),
            "current": screening.get("accepted_absolute_flux_models", []),
            "required": "not required for screening, required for absolute-flux claims",
            "detail": "absolute-flux promotion remains a separate stricter target",
        },
        {
            "metric": "additional_independent_holdouts_needed",
            "passed": additional_holdouts == 0,
            "current": additional_holdouts,
            "required": 0,
            "detail": "more independent nonlinear holdouts reduce rank/correlation fragility",
        },
    ]
    blockers = [
        str(gate["metric"])
        for gate in gates
        if gate["metric"] != "absolute_flux_promotion_passed" and not bool(gate["passed"])
    ]
    return {
        "kind": "quasilinear_screening_promotion_requirements",
        "claim_boundary": (
            "This section defines what is missing before correlation-screening "
            "promotion; it does not promote absolute nonlinear heat-flux prediction."
        ),
        "screening_promoted": False,
        "reconsideration_ready": not blockers,
        "blockers": blockers,
        "gates": gates,
        "current_best_model": screening.get("best_screening_model"),
        "current_best_heldout_model": screening.get("best_holdout_screening_model"),
        "current_best_holdout_metrics": {
            "spearman": screening.get("best_holdout_spearman"),
            "pairwise_order_accuracy": screening.get("best_holdout_pairwise_order_accuracy"),
            "mean_abs_relative_error": screening.get("best_holdout_mean_abs_relative_error"),
        },
    }


def build_holdout_gap_report(
    *,
    model_selection: dict[str, Any],
    train_holdout: dict[str, Any],
    window_stats: dict[str, Any],
    external_gates: list[dict[str, Any]],
    dataset: dict[str, Any] | None = None,
    screening_skill: dict[str, Any] | None = None,
    external_patterns: list[str] | None = None,
    max_next_candidates: int = DEFAULT_NEXT_CANDIDATES,
) -> dict[str, Any]:
    """Return the JSON-ready holdout gap report."""

    external_by_case = {str(gate["case"]): gate for gate in external_gates}
    dataset_cases, input_cases = _dataset_case_maps(dataset)
    known_gate_cases = {
        str(item.get("gate_case"))
        for item in [*dataset_cases.values(), *input_cases.values()]
        if item.get("gate_case")
    }
    train_holdout = dict(train_holdout)
    train_holdout.setdefault("artifact", DEFAULT_TRAIN_HOLDOUT)
    admitted, training = _build_point_rows(
        train_holdout,
        dataset=dataset,
        window_stats=window_stats,
        external_by_case=external_by_case,
    )
    for row in [*admitted, *training]:
        if row.get("gate_case"):
            known_gate_cases.add(str(row["gate_case"]))
    holdout_families = {
        str(row.get("geometry"))
        for row in admitted
        if "external_vmec" in str(row.get("geometry", ""))
    }
    training_families = {
        str(row.get("geometry"))
        for row in training
        if "external_vmec" in str(row.get("geometry", ""))
    }
    excluded = _dataset_excluded_rows(dataset, window_stats=window_stats)
    already_reported = {str(row.get("case", "")) for row in excluded}
    excluded.extend(
        _window_excluded_rows(
            window_stats,
            known_gate_cases=known_gate_cases,
            already_reported=already_reported,
        )
    )
    excluded.extend(
        _external_excluded_rows(
            external_gates,
            known_gate_cases=known_gate_cases,
            holdout_families=holdout_families,
            training_families=training_families,
        )
    )
    next_candidates = _next_candidates(
        training,
        excluded,
        max_candidates=int(max_next_candidates),
    )
    calibration = _calibration_summary(train_holdout)
    model = _model_selection_summary(model_selection)
    absolute_requirements = _absolute_flux_promotion_requirements(
        admitted=admitted,
        training=training,
        excluded_rows=excluded,
        next_candidates=next_candidates,
        calibration=calibration,
        model=model,
    )
    screening = _screening_skill_summary(screening_skill)
    screening_requirements = _screening_promotion_requirements(
        screening,
        absolute_requirements=absolute_requirements,
    )
    external_passed = sum(1 for gate in external_gates if bool(gate.get("passed", False)))
    external_failed = len(external_gates) - external_passed
    external_negative = sum(
        1 for gate in external_gates if bool(gate.get("negative_evidence", False))
    )
    window_passed = _window_case_passes(window_stats)
    blockers = ["absolute_flux_predictor_not_promoted"]
    blockers.extend(
        f"absolute_requirement:{item}"
        for item in absolute_requirements.get("blockers", [])
        if item not in {"scoped_model_selection_gate_passed"}
    )
    if not bool(calibration["passed"]):
        blockers.append("stellarator_train_holdout_report_failed")
    if calibration.get("holdout_mean_abs_relative_error") is not None and calibration.get("holdout_mean_rel_gate") is not None:
        if float(calibration["holdout_mean_abs_relative_error"]) > float(calibration["holdout_mean_rel_gate"]):
            blockers.append("holdout_mean_error_exceeds_gate")
    if screening_requirements is not None:
        blockers.extend(
            f"screening_requirement:{item}"
            for item in screening_requirements.get("blockers", [])
        )
    payload = {
        "kind": "quasilinear_holdout_gap_report",
        "claim_level": CLAIM_LEVEL,
        "absolute_flux_promoted": False,
        "promotion_gate": {
            "passed": False,
            "blockers": blockers,
            "reason": (
                "This artifact reports holdout readiness gaps only. The scoped model-selection gate may pass, "
                "but no runtime/TOML absolute-flux predictor is promoted."
            ),
        },
        "inputs": {
            "model_selection": model["artifact"],
            "train_holdout": calibration["artifact"],
            "nonlinear_window_statistics": _repo_relative_path(DEFAULT_WINDOW_STATS),
            "dataset_sufficiency": _repo_relative_path(DEFAULT_DATASET_SUFFICIENCY) if dataset else None,
            "screening_skill": _repo_relative_path(DEFAULT_SCREENING_SKILL) if screening else None,
            "external_gate_patterns": [_repo_relative_pattern(pattern) for pattern in external_patterns or []],
        },
        "model_selection_status": model,
        "calibration_status": calibration,
        "screening_skill_status": screening,
        "summary": {
            "n_admitted_holdouts": len(admitted),
            "n_training_references": len(training),
            "n_excluded_candidates": len(excluded),
            "n_next_best_candidates": len(next_candidates),
            "n_external_gates": len(external_gates),
            "n_external_gates_passed": external_passed,
            "n_external_gates_failed": external_failed,
            "n_external_negative_evidence": external_negative,
            "n_nonlinear_window_cases": len(window_passed),
            "n_nonlinear_window_cases_passed": sum(1 for passed in window_passed.values() if passed),
            "holdout_mean_abs_relative_error": calibration.get("holdout_mean_abs_relative_error"),
            "holdout_mean_rel_gate": calibration.get("holdout_mean_rel_gate"),
            "model_selection_candidate_mean_abs_relative_error": model.get("candidate_mean_abs_relative_error"),
            "model_selection_interval_coverage": model.get("candidate_prediction_interval_coverage"),
            "screening_correlation_passed": None if screening is None else screening.get("screening_correlation_passed"),
            "holdout_screening_correlation_passed": None
            if screening is None
            else screening.get("holdout_screening_correlation_passed"),
            "best_holdout_screening_spearman": None
            if screening is None
            else screening.get("best_holdout_spearman"),
            "best_holdout_screening_pairwise_order_accuracy": None
            if screening is None
            else screening.get("best_holdout_pairwise_order_accuracy"),
        },
        "admitted_holdouts": admitted,
        "training_references": training,
        "excluded_candidates": excluded,
        "next_best_candidates": next_candidates,
        "next_actual_nonlinear_holdout_needed": _next_actual_need(next_candidates),
        "absolute_flux_promotion_requirements": absolute_requirements,
        "screening_promotion_requirements": screening_requirements,
        "notes": (
            "Admitted means the nonlinear window is already present in the current train/holdout metadata with a "
            "passed input/convergence gate. Excluded means the tracked nonlinear artifact is outside the current "
            "quasilinear absolute-flux holdout set or fails a gate. Next-best candidates are metadata-ranked only; "
            "the tool does not rerun simulations."
        ),
    }
    return _json_clean(payload)


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "; ".join(str(item) for item in value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def write_gap_report_csv(report: dict[str, Any], path: Path) -> None:
    """Write a flat CSV view of the report."""

    rows: list[dict[str, Any]] = []
    for section in ("admitted_holdouts", "training_references", "excluded_candidates", "next_best_candidates"):
        for row in report.get(section, []):
            if not isinstance(row, dict):
                continue
            out = {field: row.get(field) for field in CSV_FIELDS}
            out["section"] = section
            rows.append(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in CSV_FIELDS})


def _bar_values(rows: list[dict[str, Any]], key: str) -> tuple[list[str], list[float]]:
    labels: list[str] = []
    values: list[float] = []
    for row in rows:
        value = _finite_float(row.get(key))
        if value is None:
            continue
        labels.append(_short_label(str(row.get("case_label") or row.get("case"))))
        values.append(value)
    return labels, values


def holdout_gap_figure(report: dict[str, Any], *, title: str) -> plt.Figure:
    """Create a readable summary panel for the holdout gap report."""

    set_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(15.6, 9.8), constrained_layout=True)
    ax_error, ax_counts, ax_external, ax_text = axes.reshape(-1)

    admitted = [row for row in report.get("admitted_holdouts", []) if isinstance(row, dict)]
    labels, values = _bar_values(admitted, "absolute_relative_error")
    y = np.arange(len(labels))
    if values:
        colors = ["#2a9d8f" if value <= 0.35 else "#c44e52" for value in values]
        ax_error.barh(y, values, color=colors)
        ax_error.set_yticks(y, labels)
        ax_error.invert_yaxis()
        ax_error.set_xscale("log")
        gate = _finite_float((report.get("calibration_status") or {}).get("holdout_mean_rel_gate"))
        if gate is not None:
            ax_error.axvline(gate, color="#1f2937", linestyle="--", linewidth=1.7, label=f"gate {gate:.2g}")
            ax_error.legend(loc="lower right")
        for ypos, value in zip(y, values, strict=True):
            ax_error.text(value * 1.05, ypos, f"{value:.2g}", va="center", fontsize=8)
    else:
        ax_error.text(0.5, 0.5, "No admitted holdouts", ha="center", va="center")
        ax_error.set_axis_off()
    ax_error.set_title("Current admitted holdout errors")
    ax_error.set_xlabel("absolute relative error")
    ax_error.grid(True, axis="x", alpha=0.25)

    summary = report.get("summary", {}) if isinstance(report.get("summary", {}), dict) else {}
    count_labels = ["holdouts", "train refs", "excluded", "next-best"]
    count_values = [
        int(summary.get("n_admitted_holdouts", 0) or 0),
        int(summary.get("n_training_references", 0) or 0),
        int(summary.get("n_excluded_candidates", 0) or 0),
        int(summary.get("n_next_best_candidates", 0) or 0),
    ]
    count_y = np.arange(len(count_labels))
    ax_counts.barh(
        count_y,
        count_values,
        color=["#0f4c81", "#8d99ae", "#b45309", "#2a9d8f"],
    )
    ax_counts.set_yticks(count_y, count_labels)
    ax_counts.invert_yaxis()
    for ypos, value in enumerate(count_values):
        ax_counts.text(
            value + 0.08,
            ypos,
            str(value),
            ha="left",
            va="center",
            fontweight="bold",
        )
    ax_counts.set_xlim(0, max(count_values + [1]) * 1.22)
    ax_counts.set_title("Tracked candidate classes")
    ax_counts.set_xlabel("count")
    ax_counts.grid(True, axis="x", alpha=0.2)

    external_rows = [
        row
        for row in report.get("excluded_candidates", [])
        if isinstance(row, dict) and str(row.get("geometry", "")).endswith("external_vmec")
    ]
    ranked_external = sorted(
        external_rows,
        key=lambda row: (
            row.get("gate_passed") is True,
            _finite_float(row.get("next_best_score")) or math.inf,
            str(row.get("case")),
        ),
    )[:8]
    ext_labels = [_short_label(str(row.get("case", "")), max_chars=52) for row in ranked_external]
    ext_scores = [
        0.0 if row.get("gate_passed") is True else float(_finite_float(row.get("next_best_score")) or 0.0)
        for row in ranked_external
    ]
    if ranked_external:
        ypos = np.arange(len(ranked_external))
        colors = ["#2a9d8f" if row.get("gate_passed") is True else "#d1495b" for row in ranked_external]
        ax_external.barh(ypos, ext_scores, color=colors)
        ax_external.set_yticks(ypos, ext_labels)
        ax_external.invert_yaxis()
        ax_external.axvline(1.0, color="#1f2937", linestyle="--", linewidth=1.5)
        ax_external.set_xlabel("max failed gate / limit; <=1 passes")
        for pos, row, score in zip(ypos, ranked_external, ext_scores, strict=True):
            label = "pass" if row.get("gate_passed") is True else f"{score:.2g}x"
            ax_external.text(max(score, 0.03) + 0.04, pos, label, va="center", fontsize=8)
    else:
        ax_external.text(0.5, 0.5, "No external candidates", ha="center", va="center")
        ax_external.set_axis_off()
    ax_external.set_title("External-VMEC excluded/near-miss gates")
    ax_external.grid(True, axis="x", alpha=0.25)

    ax_text.set_axis_off()
    model = report.get("model_selection_status", {}) if isinstance(report.get("model_selection_status", {}), dict) else {}
    screening = report.get("screening_skill_status", {})
    screening = screening if isinstance(screening, dict) else {}
    calibration = report.get("calibration_status", {}) if isinstance(report.get("calibration_status", {}), dict) else {}
    need = report.get("next_actual_nonlinear_holdout_needed", {})
    if not isinstance(need, dict):
        need = {}
    absolute_requirements = report.get("absolute_flux_promotion_requirements", {})
    if not isinstance(absolute_requirements, dict):
        absolute_requirements = {}
    numeric_gap = absolute_requirements.get("numeric_gap", {})
    if not isinstance(numeric_gap, dict):
        numeric_gap = {}
    coverage_gap = absolute_requirements.get("coverage_gap", {})
    if not isinstance(coverage_gap, dict):
        coverage_gap = {}
    text_lines = [
        "Claim boundary",
        "No runtime/TOML absolute-flux predictor is promoted.",
        "",
        f"Model-selection passed: {bool(model.get('passed', False))}",
        f"Accepted scoped candidate: {model.get('required_candidate', 'n/a')}",
        f"Calibration report passed: {bool(calibration.get('passed', False))}",
        "Holdout mean/gate: "
        f"{_format_gate_value(calibration.get('holdout_mean_abs_relative_error'))} / "
        f"{_format_gate_value(calibration.get('holdout_mean_rel_gate'))}",
        "Error factor to gate: "
        f"{_format_gate_value(numeric_gap.get('error_factor_to_gate'))}x",
        "Additional holdouts needed: "
        f"{coverage_gap.get('additional_total_independent_holdouts_needed', 'n/a')}; "
        "external VMEC families: "
        f"{coverage_gap.get('additional_external_vmec_holdout_families_needed', 'n/a')}; "
        "nonaxisym external: "
        f"{coverage_gap.get('additional_nonaxisymmetric_external_vmec_holdout_families_needed', 'n/a')}",
        "",
        "Screening/correlation status",
        f"Full-portfolio screening passed: {bool(screening.get('screening_correlation_passed', False))}",
        f"Held-out screening passed: {bool(screening.get('holdout_screening_correlation_passed', False))}",
        "Best held-out Spearman/pairwise: "
        f"{_format_gate_value(screening.get('best_holdout_spearman'))} / "
        f"{_format_gate_value(screening.get('best_holdout_pairwise_order_accuracy'))}",
        "",
        "Next actual nonlinear holdout needed",
        str(need.get("needed", "new independent passed-gate holdout")),
        f"Preferred family: {need.get('preferred_family', 'n/a')}",
    ]
    nearest = need.get("nearest_tracked_gap")
    if isinstance(nearest, dict):
        text_lines.extend(
            [
                "",
                "Nearest tracked gap",
                str(nearest.get("case", "n/a")),
            ]
        )
    ax_text.text(
        0.02,
        0.98,
        "\n".join(text_lines),
        va="top",
        ha="left",
        fontsize=10,
        linespacing=1.35,
        bbox={"boxstyle": "round,pad=0.55", "facecolor": "#f8fafc", "edgecolor": "#cbd5e1"},
    )
    fig.suptitle(title, fontsize=15, fontweight="bold")
    return fig


def write_holdout_gap_report_artifacts(
    report: dict[str, Any],
    *,
    out: str | Path,
    title: str,
    dpi: int = 220,
    write_pdf: bool = True,
) -> dict[str, str]:
    """Write PNG/PDF/JSON/CSV artifacts for a holdout gap report."""

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig = holdout_gap_figure(report, title=title)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    paths = {"png": str(out_path)}
    if write_pdf:
        pdf_path = out_path.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        paths["pdf"] = str(pdf_path)
    plt.close(fig)

    json_path = out_path.with_suffix(".json")
    json_path.write_text(
        json.dumps(_json_clean(report), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    paths["json"] = str(json_path)

    csv_path = out_path.with_suffix(".csv")
    write_gap_report_csv(report, csv_path)
    paths["csv"] = str(csv_path)
    return paths


def _external_gate_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(Path(item) for item in glob.glob(pattern, recursive=True))
    return sorted(set(paths))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-selection", type=Path, default=DEFAULT_MODEL_SELECTION)
    parser.add_argument("--train-holdout", type=Path, default=DEFAULT_TRAIN_HOLDOUT)
    parser.add_argument("--nonlinear-window-statistics", type=Path, default=DEFAULT_WINDOW_STATS)
    parser.add_argument("--screening-skill", type=Path, default=DEFAULT_SCREENING_SKILL)
    parser.add_argument(
        "--dataset-sufficiency",
        type=Path,
        default=DEFAULT_DATASET_SUFFICIENCY,
        help="Optional dataset-sufficiency JSON used for gate-case mappings and scoped exclusions.",
    )
    parser.add_argument(
        "--external-gate-glob",
        action="append",
        dest="external_gate_patterns",
        default=None,
        help="External-VMEC convergence gate glob. Defaults to tracked external_vmec gates.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--title", default="Quasilinear holdout gap report")
    parser.add_argument("--max-next-candidates", type=int, default=DEFAULT_NEXT_CANDIDATES)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--no-pdf", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    patterns = args.external_gate_patterns or [DEFAULT_EXTERNAL_GLOB]
    dataset = _load_json(args.dataset_sufficiency) if args.dataset_sufficiency.exists() else None
    external_paths = _external_gate_paths(patterns)
    report = build_holdout_gap_report(
        model_selection=_load_json(args.model_selection),
        train_holdout=_load_json(args.train_holdout),
        window_stats=_load_json(args.nonlinear_window_statistics),
        external_gates=[_load_external_gate(path) for path in external_paths],
        dataset=dataset,
        screening_skill=_load_json(args.screening_skill) if args.screening_skill.exists() else None,
        external_patterns=patterns,
        max_next_candidates=args.max_next_candidates,
    )
    paths = write_holdout_gap_report_artifacts(
        report,
        out=args.out,
        title=args.title,
        dpi=args.dpi,
        write_pdf=not args.no_pdf,
    )
    for key in ("png", "pdf", "json", "csv"):
        if key in paths:
            print(f"saved {paths[key]}")
    summary = report["summary"]
    print(
        "admitted_holdouts={admitted} excluded={excluded} next_best={next_best} "
        "holdout_mean_abs_relative_error={mean} gate={gate} absolute_flux_promoted=false".format(
            admitted=summary["n_admitted_holdouts"],
            excluded=summary["n_excluded_candidates"],
            next_best=summary["n_next_best_candidates"],
            mean=summary["holdout_mean_abs_relative_error"],
            gate=summary["holdout_mean_rel_gate"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
