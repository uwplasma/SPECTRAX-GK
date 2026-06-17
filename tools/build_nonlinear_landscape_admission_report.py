#!/usr/bin/env python3
"""Build a nonlinear landscape admission report from ensemble-gate JSON files."""

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
    VMECJAXNonlinearAuditPolicy,
    build_nonlinear_landscape_admission_report,
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


def build_report(
    *,
    baseline_ensemble: Path,
    candidate_ensembles: list[tuple[str, Path]],
    policy: VMECJAXNonlinearAuditPolicy,
) -> dict[str, Any]:
    """Load ensemble artifacts and return a JSON-safe admission report."""

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.candidate_ensemble:
        raise SystemExit("at least one --candidate-ensemble LABEL PATH pair is required")
    policy = VMECJAXNonlinearAuditPolicy(
        minimum_relative_reduction=float(args.min_relative_reduction),
        minimum_uncertainty_z_score=float(args.min_uncertainty_z_score),
        maximum_combined_sem_rel=float(args.max_combined_sem_rel),
        minimum_replicate_count=int(args.min_replicate_count),
    )
    report = build_report(
        baseline_ensemble=args.baseline_ensemble,
        candidate_ensembles=[
            (str(label), Path(path))
            for label, path in args.candidate_ensemble
        ],
        policy=policy,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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


if __name__ == "__main__":
    raise SystemExit(main())
