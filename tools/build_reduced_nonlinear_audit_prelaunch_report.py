#!/usr/bin/env python3
"""Build a reduced-objective prelaunch gate for nonlinear audit campaigns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from spectraxgk.validation.stellarator.transport_admission import (  # noqa: E402
    VMECJAXReducedPrelaunchPolicy,
    build_reduced_nonlinear_audit_prelaunch_report,
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _row_identifier(row: dict[str, Any]) -> tuple[str, ...]:
    identifiers = [str(row.get("label", ""))]
    fraction = row.get("relative_fraction")
    if fraction is not None:
        identifiers.append(str(fraction))
        try:
            identifiers.append(f"{float(fraction):.12g}")
        except (TypeError, ValueError):
            pass
    return tuple(item for item in identifiers if item)


def _select_row(rows: list[Any], selector: str) -> dict[str, Any]:
    for row in rows:
        if not isinstance(row, dict):
            continue
        if selector in _row_identifier(row):
            return row
    available = [
        item
        for row in rows
        if isinstance(row, dict)
        for item in _row_identifier(row)
    ]
    raise ValueError(f"no landscape row matches {selector!r}; available selectors={available}")


def _metric(row: dict[str, Any], metric_key: str) -> float:
    metrics = row.get("reduced_metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"landscape row {row.get('label')} is missing reduced_metrics")
    try:
        return float(metrics[metric_key])
    except KeyError as exc:
        raise ValueError(f"landscape row {row.get('label')} missing metric {metric_key!r}") from exc


def _sample_statistics(row: dict[str, Any], metric_key: str) -> dict[str, Any] | None:
    reports = row.get("reduced_metric_reports")
    if not isinstance(reports, dict):
        return None
    report = reports.get(metric_key)
    if not isinstance(report, dict):
        return None
    payload = report.get("payload")
    if not isinstance(payload, dict):
        return None
    statistics = payload.get("sample_statistics")
    return dict(statistics) if isinstance(statistics, dict) else None


def build_report(
    *,
    landscape_json: Path,
    baseline_selector: str,
    candidate_selector: str,
    metric_key: str,
    failed_reference_relative_reduction: float | None,
    policy: VMECJAXReducedPrelaunchPolicy,
) -> dict[str, Any]:
    """Build a reduced prelaunch report from a landscape artifact."""

    landscape = _load_json(landscape_json)
    rows = landscape.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"{landscape_json} is missing rows")
    baseline_row = _select_row(rows, baseline_selector)
    candidate_row = _select_row(rows, candidate_selector)
    report = build_reduced_nonlinear_audit_prelaunch_report(
        baseline_metric=_metric(baseline_row, metric_key),
        candidate_metric=_metric(candidate_row, metric_key),
        objective_sample_set=landscape.get("sample_set"),
        baseline_sample_statistics=_sample_statistics(baseline_row, metric_key),
        candidate_sample_statistics=_sample_statistics(candidate_row, metric_key),
        failed_reference_relative_reduction=failed_reference_relative_reduction,
        policy=policy,
    )
    report["artifacts"] = {
        "landscape_json": _repo_relative(landscape_json),
    }
    report["selected_rows"] = {
        "baseline": {
            "label": baseline_row.get("label"),
            "relative_fraction": baseline_row.get("relative_fraction"),
            "coefficient_value": baseline_row.get("coefficient_value"),
        },
        "candidate": {
            "label": candidate_row.get("label"),
            "relative_fraction": candidate_row.get("relative_fraction"),
            "coefficient_value": candidate_row.get("coefficient_value"),
        },
    }
    return report


def build_metric_report(
    *,
    baseline_metric: float,
    candidate_metric: float,
    sample_set: dict[str, list[float]] | None,
    metric_key: str,
    failed_reference_relative_reduction: float | None,
    policy: VMECJAXReducedPrelaunchPolicy,
) -> dict[str, Any]:
    """Build a reduced prelaunch report from explicit metrics."""

    report = build_reduced_nonlinear_audit_prelaunch_report(
        baseline_metric=baseline_metric,
        candidate_metric=candidate_metric,
        objective_sample_set=sample_set,
        failed_reference_relative_reduction=failed_reference_relative_reduction,
        policy=policy,
    )
    report["artifacts"] = {
        "landscape_json": None,
    }
    report["selected_rows"] = {
        "baseline": None,
        "candidate": None,
    }
    return report


def _sample_set_from_args(args: argparse.Namespace) -> dict[str, list[float]] | None:
    if args.sample_set_json is not None:
        payload = _load_json(args.sample_set_json)
        return {
            "surfaces": [float(item) for item in payload.get("surfaces", ())],
            "alphas": [float(item) for item in payload.get("alphas", ())],
            "ky_values": [float(item) for item in payload.get("ky_values", ())],
        }
    if args.surface or args.alpha or args.ky:
        return {
            "surfaces": [float(item) for item in args.surface],
            "alphas": [float(item) for item in args.alpha],
            "ky_values": [float(item) for item in args.ky],
        }
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--landscape-json", type=Path)
    parser.add_argument("--baseline-row", default="0")
    parser.add_argument("--candidate-row")
    parser.add_argument("--baseline-metric", type=float)
    parser.add_argument("--candidate-metric", type=float)
    parser.add_argument("--sample-set-json", type=Path)
    parser.add_argument("--surface", type=float, action="append", default=[])
    parser.add_argument("--alpha", type=float, action="append", default=[])
    parser.add_argument("--ky", type=float, action="append", default=[])
    parser.add_argument("--metric-key", default="nonlinear_window_heat_flux")
    parser.add_argument("--failed-reference-relative-reduction", type=float)
    parser.add_argument("--min-relative-reduction", type=float, default=0.04)
    parser.add_argument("--failed-reference-safety-factor", type=float, default=1.5)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--fail-on-blocked", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    policy = VMECJAXReducedPrelaunchPolicy(
        metric_key=str(args.metric_key),
        minimum_relative_reduction=float(args.min_relative_reduction),
        failed_reference_safety_factor=float(args.failed_reference_safety_factor),
    )
    if args.landscape_json is not None:
        if args.candidate_row is None:
            raise SystemExit("--candidate-row is required when --landscape-json is used")
        report = build_report(
            landscape_json=args.landscape_json,
            baseline_selector=str(args.baseline_row),
            candidate_selector=str(args.candidate_row),
            metric_key=str(args.metric_key),
            failed_reference_relative_reduction=args.failed_reference_relative_reduction,
            policy=policy,
        )
    else:
        if args.baseline_metric is None or args.candidate_metric is None:
            raise SystemExit(
                "either --landscape-json or both --baseline-metric and --candidate-metric are required"
            )
        report = build_metric_report(
            baseline_metric=float(args.baseline_metric),
            candidate_metric=float(args.candidate_metric),
            sample_set=_sample_set_from_args(args),
            metric_key=str(args.metric_key),
            failed_reference_relative_reduction=args.failed_reference_relative_reduction,
            policy=policy,
        )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "relative_reduced_reduction": report["relative_reduced_reduction"],
                "required_relative_reduced_reduction": report[
                    "required_relative_reduced_reduction"
                ],
                "blockers": report["blockers"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.fail_on_blocked and not bool(report["passed"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
