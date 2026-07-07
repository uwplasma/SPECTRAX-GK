#!/usr/bin/env python3
"""Check claim boundaries for nonlinear turbulence-gradient evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from spectraxgk.validation.nonlinear_gradient.evidence import (  # noqa: E402
    NonlinearTurbulenceGradientEvidenceConfig,
    NonlinearTurbulenceGradientGapConfig,
    load_json_artifact,
    nonlinear_turbulence_gradient_evidence_report,
)


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gradient-artifact", type=Path, required=True)
    parser.add_argument(
        "--window-artifact",
        action="append",
        type=Path,
        default=[],
        help="Replicated ensemble JSON or individual nonlinear-window convergence summary.",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument(
        "--gap-json-out",
        type=Path,
        help="Optional output path for the extracted fail-closed missing-campaign report.",
    )
    parser.add_argument(
        "--gap-case-slug",
        default="optimized_equilibrium_turbulence_gradient",
        help="Slug used when naming the required baseline/plus/minus campaign.",
    )
    parser.add_argument(
        "--gradient-parameter-name",
        default="vmec_state_control_or_profile_gradient",
        help="Parameter to perturb in the required paired nonlinear campaign.",
    )
    parser.add_argument("--perturbation-fraction", type=float, default=0.05)
    parser.add_argument("--analysis-tmin", type=float, default=350.0)
    parser.add_argument("--analysis-tmax", type=float, default=700.0)
    parser.add_argument("--minimum-tmax", type=float, default=700.0)
    parser.add_argument("--minimum-grid", default="n64x64x64x40x40")
    parser.add_argument(
        "--replicate-label",
        action="append",
        default=[],
        help=(
            "Required replicate label. Defaults to seed31, seed32, and dt0p04 "
            "when omitted."
        ),
    )
    parser.add_argument("--min-window-reports", type=int, default=2)
    parser.add_argument("--max-window-mean-rel-spread", type=float, default=0.15)
    parser.add_argument("--max-window-combined-sem-rel", type=float, default=0.25)
    parser.add_argument("--max-gradient-uncertainty-rel", type=float, default=0.50)
    parser.add_argument("--max-fd-asymmetry-rel", type=float, default=0.50)
    parser.add_argument("--max-fd-condition-number", type=float, default=1.0e8)
    parser.add_argument("--min-fd-response-fraction", type=float, default=0.03)
    parser.add_argument("--fail-on-blocked", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    gradient = load_json_artifact(args.gradient_artifact)
    windows = [load_json_artifact(path) for path in args.window_artifact]
    report = nonlinear_turbulence_gradient_evidence_report(
        gradient,
        window_artifacts=windows,
        gradient_path=_repo_relative(args.gradient_artifact),
        window_paths=[_repo_relative(path) for path in args.window_artifact],
        config=NonlinearTurbulenceGradientEvidenceConfig(
            min_window_reports=args.min_window_reports,
            max_window_mean_rel_spread=args.max_window_mean_rel_spread,
            max_window_combined_sem_rel=args.max_window_combined_sem_rel,
            max_gradient_uncertainty_rel=args.max_gradient_uncertainty_rel,
            max_fd_asymmetry_rel=args.max_fd_asymmetry_rel,
            max_fd_condition_number=args.max_fd_condition_number,
            min_fd_response_fraction=args.min_fd_response_fraction,
        ),
        gap_config=NonlinearTurbulenceGradientGapConfig(
            case_slug=args.gap_case_slug,
            parameter_name=args.gradient_parameter_name,
            perturbation_fraction=args.perturbation_fraction,
            analysis_tmin=args.analysis_tmin,
            analysis_tmax=args.analysis_tmax,
            minimum_tmax=args.minimum_tmax,
            minimum_grid=args.minimum_grid,
            replicate_labels=tuple(
                args.replicate_label or ["seed31", "seed32", "dt0p04"]
            ),
        ),
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
        print(f"saved {args.json_out}")
    else:
        print(text)
    if args.gap_json_out is not None:
        args.gap_json_out.parent.mkdir(parents=True, exist_ok=True)
        args.gap_json_out.write_text(
            json.dumps(report["evidence_gap"], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"saved {args.gap_json_out}")
    if args.fail_on_blocked and not bool(report["passed"]):
        print(
            "nonlinear turbulence-gradient evidence blocked: "
            + ", ".join(report["blockers"]),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
