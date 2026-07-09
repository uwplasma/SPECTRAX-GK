#!/usr/bin/env python3
"""Build nonlinear transport admission artifacts from compact JSON evidence.

The two subcommands keep the evidence flow explicit:

* ``landscape`` selects an uncertainty-resolved nonlinear landscape candidate
  from matched replicated transport-window ensembles.
* ``prelaunch`` builds the reduced-objective prelaunch screen from either a
  transport landscape row pair or explicit reduced metrics.
* ``campaign`` decides whether a reduced prelaunch screen plus a landscape
  admission artifact is strong enough to launch the next optimizer campaign.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from spectraxgk.diagnostics.stellarator_transport_reports import (  # noqa: E402
    build_nonlinear_audit_redesign_report,
    build_nonlinear_campaign_admission_report,
    build_reduced_nonlinear_audit_prelaunch_report,
)
from spectraxgk.diagnostics.stellarator_transport_reports import (  # noqa: E402
    build_nonlinear_landscape_admission_report,
)
from spectraxgk.objectives.vmec_transport_admission import (  # noqa: E402
    VMECJAXNonlinearAuditPolicy,
    VMECJAXNonlinearCampaignPolicy,
    VMECJAXReducedPrelaunchPolicy,
)


DEFAULT_PRELAUNCH = (
    ROOT / "docs/_static/vmec_boundary_transport_landscape_p0p03_prelaunch_gate.json"
)
DEFAULT_LANDSCAPE = (
    ROOT / "docs/_static/vmec_boundary_transport_landscape_admission.json"
)
DEFAULT_CAMPAIGN_OUT = ROOT / "docs/_static/nonlinear_campaign_admission_report.json"


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
        item for row in rows if isinstance(row, dict) for item in _row_identifier(row)
    ]
    raise ValueError(
        f"no landscape row matches {selector!r}; available selectors={available}"
    )


def _metric(row: dict[str, Any], metric_key: str) -> float:
    metrics = row.get("reduced_metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"landscape row {row.get('label')} is missing reduced_metrics")
    try:
        return float(metrics[metric_key])
    except KeyError as exc:
        raise ValueError(
            f"landscape row {row.get('label')} missing metric {metric_key!r}"
        ) from exc


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


def build_landscape_report(
    *,
    baseline_ensemble: Path,
    candidate_ensembles: list[tuple[str, Path]],
    policy: VMECJAXNonlinearAuditPolicy,
) -> dict[str, Any]:
    """Load nonlinear ensemble artifacts and score landscape candidates."""

    baseline_payload = _load_json(baseline_ensemble)
    candidate_payloads = [_load_json(path) for _, path in candidate_ensembles]
    labels = [label for label, _ in candidate_ensembles]
    report = build_nonlinear_landscape_admission_report(
        baseline_payload,
        candidate_payloads,
        candidate_labels=labels,
        policy=policy,
    )
    report["artifacts"] = {
        "baseline_ensemble": _repo_relative(baseline_ensemble),
        "candidate_ensembles": [
            {"label": label, "path": _repo_relative(path)}
            for label, path in candidate_ensembles
        ],
    }
    return report


def build_campaign_report(
    *,
    prelaunch_report: Path,
    landscape_admission: Path,
    policy: VMECJAXNonlinearCampaignPolicy,
) -> dict[str, Any]:
    """Load compact JSON artifacts and score campaign admission."""

    report = build_nonlinear_campaign_admission_report(
        reduced_prelaunch_report=_load_json(prelaunch_report),
        landscape_admission_report=_load_json(landscape_admission),
        policy=policy,
    )
    report["artifacts"] = {
        "reduced_prelaunch_report": _repo_relative(prelaunch_report),
        "landscape_admission_report": _repo_relative(landscape_admission),
    }
    return report


def build_prelaunch_report(
    *,
    landscape_json: Path,
    baseline_selector: str,
    candidate_selector: str,
    metric_key: str,
    failed_reference_relative_reduction: float | None,
    policy: VMECJAXReducedPrelaunchPolicy,
) -> dict[str, Any]:
    """Build a reduced-objective prelaunch report from landscape rows."""

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


def build_prelaunch_metric_report(
    *,
    baseline_metric: float,
    candidate_metric: float,
    sample_set: dict[str, list[float]] | None,
    metric_key: str,
    failed_reference_relative_reduction: float | None,
    policy: VMECJAXReducedPrelaunchPolicy,
) -> dict[str, Any]:
    """Build a reduced-objective prelaunch report from explicit metrics."""

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


def build_redesign_report(
    *,
    matched_comparison: Path,
    objective_sample_set: dict[str, list[float]] | None,
    policy: VMECJAXNonlinearAuditPolicy,
) -> dict[str, Any]:
    """Load a matched comparison and recommend the next objective sample set."""

    return build_nonlinear_audit_redesign_report(
        _load_json(matched_comparison),
        objective_sample_set=objective_sample_set,
        policy=policy,
    )


def _add_landscape_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "landscape", help="select an admitted nonlinear landscape candidate"
    )
    parser.add_argument("--baseline-ensemble", type=Path, required=True)
    parser.add_argument(
        "--candidate-ensemble",
        action="append",
        nargs=2,
        metavar=("LABEL", "PATH"),
        default=[],
        help="Candidate label and nonlinear ensemble JSON path. May be repeated.",
    )
    parser.add_argument("--min-relative-reduction", type=float, default=0.02)
    parser.add_argument("--min-uncertainty-z-score", type=float, default=1.0)
    parser.add_argument("--max-combined-sem-rel", type=float, default=0.25)
    parser.add_argument("--min-replicate-count", type=int, default=3)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--fail-on-no-admission", action="store_true")
    parser.set_defaults(func=_run_landscape)


def _add_campaign_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "campaign", help="gate the next nonlinear optimizer campaign"
    )
    parser.add_argument("--prelaunch-report", type=Path, default=DEFAULT_PRELAUNCH)
    parser.add_argument("--landscape-admission", type=Path, default=DEFAULT_LANDSCAPE)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_CAMPAIGN_OUT)
    parser.add_argument("--min-landscape-relative-reduction", type=float, default=0.10)
    parser.add_argument("--min-landscape-z-score", type=float, default=3.0)
    parser.add_argument("--max-landscape-sem-rel", type=float, default=0.05)
    parser.add_argument("--min-landscape-replicates", type=int, default=3)
    parser.add_argument(
        "--allow-missing-cross-sample-gate",
        action="store_true",
        help="Do not block when older prelaunch artifacts lack reduced cross-sample statistics.",
    )
    parser.add_argument("--fail-on-blocked", action="store_true")
    parser.set_defaults(func=_run_campaign)


def _add_prelaunch_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "prelaunch",
        help="build the reduced-objective nonlinear audit prelaunch gate",
    )
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
    parser.set_defaults(func=_run_prelaunch)


def _add_redesign_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    parser = subparsers.add_parser(
        "redesign",
        help="recommend a broader objective sample set after a matched audit",
    )
    parser.add_argument("--matched-comparison", type=Path, required=True)
    parser.add_argument("--sample-set-json", type=Path)
    parser.add_argument("--surface", type=float, action="append", default=[])
    parser.add_argument("--alpha", type=float, action="append", default=[])
    parser.add_argument("--ky", type=float, action="append", default=[])
    parser.add_argument("--minimum-relative-reduction", type=float, default=0.02)
    parser.add_argument("--minimum-uncertainty-z-score", type=float, default=1.0)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--fail-on-redesign", action="store_true")
    parser.set_defaults(func=_run_redesign)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_landscape_parser(subparsers)
    _add_prelaunch_parser(subparsers)
    _add_campaign_parser(subparsers)
    _add_redesign_parser(subparsers)
    return parser


def _run_landscape(args: argparse.Namespace) -> int:
    if not args.candidate_ensemble:
        raise SystemExit(
            "at least one --candidate-ensemble LABEL PATH pair is required"
        )
    policy = VMECJAXNonlinearAuditPolicy(
        minimum_relative_reduction=float(args.min_relative_reduction),
        minimum_uncertainty_z_score=float(args.min_uncertainty_z_score),
        maximum_combined_sem_rel=float(args.max_combined_sem_rel),
        minimum_replicate_count=int(args.min_replicate_count),
    )
    report = build_landscape_report(
        baseline_ensemble=args.baseline_ensemble,
        candidate_ensembles=[
            (str(label), Path(path)) for label, path in args.candidate_ensemble
        ],
        policy=policy,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    selected = report.get("selected_candidate") or {}
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "selected_label": selected.get("label"),
                "relative_reduction": selected.get("relative_reduction"),
                "uncertainty_z_score": selected.get("uncertainty_z_score"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.fail_on_no_admission and not bool(report["passed"]):
        return 1
    return 0


def _run_campaign(args: argparse.Namespace) -> int:
    policy = VMECJAXNonlinearCampaignPolicy(
        minimum_landscape_relative_reduction=float(
            args.min_landscape_relative_reduction
        ),
        minimum_landscape_uncertainty_z_score=float(args.min_landscape_z_score),
        maximum_landscape_sem_rel=float(args.max_landscape_sem_rel),
        minimum_landscape_replicate_count=int(args.min_landscape_replicates),
        require_reduced_cross_sample_gate=not bool(
            args.allow_missing_cross_sample_gate
        ),
    )
    report = build_campaign_report(
        prelaunch_report=args.prelaunch_report,
        landscape_admission=args.landscape_admission,
        policy=policy,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "campaign_admitted": report["campaign_admitted"],
                "selected_label": (
                    (report.get("selected_landscape_candidate") or {}).get("label")
                    if isinstance(report.get("selected_landscape_candidate"), dict)
                    else None
                ),
                "blockers": report["blockers"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.fail_on_blocked and not bool(report["campaign_admitted"]):
        return 1
    return 0


def _run_prelaunch(args: argparse.Namespace) -> int:
    policy = VMECJAXReducedPrelaunchPolicy(
        metric_key=str(args.metric_key),
        minimum_relative_reduction=float(args.min_relative_reduction),
        failed_reference_safety_factor=float(args.failed_reference_safety_factor),
    )
    if args.landscape_json is not None:
        if args.candidate_row is None:
            raise SystemExit(
                "--candidate-row is required when --landscape-json is used"
            )
        report = build_prelaunch_report(
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
        report = build_prelaunch_metric_report(
            baseline_metric=float(args.baseline_metric),
            candidate_metric=float(args.candidate_metric),
            sample_set=_sample_set_from_args(args),
            metric_key=str(args.metric_key),
            failed_reference_relative_reduction=args.failed_reference_relative_reduction,
            policy=policy,
        )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "relative_reduced_reduction": report[
                    "relative_reduced_reduction"
                ],
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


def _run_redesign(args: argparse.Namespace) -> int:
    policy = VMECJAXNonlinearAuditPolicy(
        minimum_relative_reduction=float(args.minimum_relative_reduction),
        minimum_uncertainty_z_score=float(args.minimum_uncertainty_z_score),
    )
    report = build_redesign_report(
        matched_comparison=args.matched_comparison,
        objective_sample_set=_sample_set_from_args(args),
        policy=policy,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "nonlinear_audit_promoted": report["nonlinear_audit_promoted"],
                "requires_objective_redesign": report["requires_objective_redesign"],
                "blockers": report["blockers"],
                "recommended_sample_set": report["recommended_sample_set"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.fail_on_redesign and bool(report["requires_objective_redesign"]):
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
