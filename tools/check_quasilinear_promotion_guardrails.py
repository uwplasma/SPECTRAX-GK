#!/usr/bin/env python3
"""Audit quasilinear absolute-flux promotion metadata and docs scope."""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spectraxgk.quasilinear_window import nonlinear_window_stats_promotion_ready


DEFAULT_REPORT_PATTERNS = (
    str(ROOT / "docs/_static/quasilinear_*train_holdout_report.json"),
    str(ROOT / "docs/_static/quasilinear_saturation_rule_sweep.json"),
    str(ROOT / "docs/_static/quasilinear_shape_aware_saturation.json"),
    str(ROOT / "docs/_static/quasilinear_candidate_uncertainty.json"),
    str(ROOT / "docs/_static/quasilinear_dataset_sufficiency.json"),
    str(ROOT / "docs/_static/quasilinear_validated_calibration_inputs.json"),
)
DEFAULT_DOCS = (
    ROOT / "docs/quasilinear.rst",
    ROOT / "docs/manuscript_figures.rst",
    ROOT / "docs/testing.rst",
)
DEFAULT_OUT = ROOT / "docs/_static/quasilinear_promotion_guardrails.json"

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
SCOPED_NON_ABSOLUTE_MARKERS = (
    "model_development",
    "not_runtime",
    "not runtime",
    "not a runtime",
    "not a transport model",
    "not validated transport",
    "candidate",
    "sufficiency",
    "scoped",
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


def _finite_number(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _nonnegative_finite(value: object) -> bool:
    return _finite_number(value) and float(value) >= 0.0


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

    has_holdout_gate = (
        _finite_number(data.get("holdout_mean_rel_gate"))
        and float(data["holdout_mean_rel_gate"]) > 0.0
    )
    holdout_metrics = by_split.get("holdout", {}) if isinstance(by_split, dict) else {}
    holdout_mean = (
        holdout_metrics.get("mean_abs_relative_error")
        if isinstance(holdout_metrics, dict)
        else None
    )
    holdout_passes = (
        has_holdout_gate
        and _finite_number(holdout_mean)
        and float(holdout_mean) <= float(data["holdout_mean_rel_gate"])
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


def _audit_docs(paths: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    gates: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        lower = _normalized_text(text)
        scope_present = any(marker in lower for marker in DOC_SCOPE_MARKERS)
        overclaim_lines = [
            f"{lineno}: {line.strip()}"
            for lineno, line in enumerate(text.splitlines(), start=1)
            if _line_overclaims_absolute_flux(line)
        ]
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
    report_patterns: list[str], doc_paths: list[str | Path]
) -> dict[str, Any]:
    report_paths = _expand_patterns(report_patterns)
    gates: list[dict[str, Any]] = []
    report_rows: list[dict[str, Any]] = []
    input_rows: list[dict[str, Any]] = []
    promotion_rows: list[dict[str, Any]] = []

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

    doc_gates, doc_rows = _audit_docs([Path(path) for path in doc_paths])
    gates.extend(doc_gates)
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
            "n_doc_checks": len(doc_rows),
            "n_failed_gates": len(failed),
        },
        "calibration_reports": report_rows,
        "input_validation_reports": input_rows,
        "promotion_gate_reports": promotion_rows,
        "doc_checks": doc_rows,
        "notes": (
            "Fast metadata guardrail only: it verifies finite nonlinear window statistics, "
            "train/holdout provenance, absolute-flux promotion gates, and conservative docs wording. "
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
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_guardrail_audit(
        args.reports or list(DEFAULT_REPORT_PATTERNS),
        args.docs or [str(path) for path in DEFAULT_DOCS],
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
