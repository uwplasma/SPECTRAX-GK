#!/usr/bin/env python3
"""Build nonlinear transport admission artifacts from compact JSON evidence.

The two subcommands keep the evidence flow explicit:

* ``landscape`` selects an uncertainty-resolved nonlinear landscape candidate
  from matched replicated transport-window ensembles.
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
)
from spectraxgk.diagnostics.stellarator_transport_reports import (  # noqa: E402
    build_nonlinear_landscape_admission_report,
)
from spectraxgk.objectives.vmec_transport_admission import (  # noqa: E402
    VMECJAXNonlinearAuditPolicy,
    VMECJAXNonlinearCampaignPolicy,
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
