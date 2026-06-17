#!/usr/bin/env python3
"""Rank failed nonlinear turbulence-gradient candidates for the next campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from spectraxgk.validation.nonlinear_gradient.evidence import (  # noqa: E402
    NonlinearTurbulenceGradientCandidateRankingConfig,
    load_json_artifact,
    nonlinear_turbulence_gradient_candidate_ranking_report,
)


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _artifact_label(payload: dict[str, Any], path: Path) -> str:
    slug = path.stem.removesuffix("_central_fd_gradient_gate")
    parameter = payload.get("parameter_name")
    if isinstance(parameter, str) and parameter:
        return f"{parameter}:{slug}"
    return slug


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "artifact",
        nargs="+",
        type=Path,
        help="Central finite-difference candidate JSON artifacts to rank.",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--max-gradient-uncertainty-rel", type=float, default=0.50)
    parser.add_argument("--max-fd-asymmetry-rel", type=float, default=0.50)
    parser.add_argument("--max-fd-condition-number", type=float, default=1.0e8)
    parser.add_argument("--min-fd-response-fraction", type=float, default=0.03)
    parser.add_argument(
        "--campaign-context",
        choices=("single_control_screen", "overdetermined_followup"),
        default="single_control_screen",
        help=(
            "Recommendation context. Use overdetermined_followup when the input "
            "candidates are the result of a completed multi-control follow-up."
        ),
    )
    parser.add_argument(
        "--fail-on-no-promotable",
        action="store_true",
        help="Return nonzero unless at least one candidate already passes all production gates.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifacts = [load_json_artifact(path) for path in args.artifact]
    labels = [_artifact_label(payload, path) for payload, path in zip(artifacts, args.artifact)]
    report = nonlinear_turbulence_gradient_candidate_ranking_report(
        artifacts,
        paths=[_repo_relative(path) for path in args.artifact],
        labels=labels,
        config=NonlinearTurbulenceGradientCandidateRankingConfig(
            max_gradient_uncertainty_rel=args.max_gradient_uncertainty_rel,
            max_fd_asymmetry_rel=args.max_fd_asymmetry_rel,
            max_fd_condition_number=args.max_fd_condition_number,
            min_fd_response_fraction=args.min_fd_response_fraction,
            campaign_context=args.campaign_context,
        ),
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
        print(f"saved {args.json_out}")
    else:
        print(text)
    if args.fail_on_no_promotable and not bool(report.get("passed", False)):
        print(
            "no nonlinear turbulence-gradient candidate passes production gates",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
