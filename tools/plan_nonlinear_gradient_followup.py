#!/usr/bin/env python3
"""Plan bounded follow-up runs from nonlinear turbulence-gradient FD artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from spectraxgk.nonlinear_gradient_evidence import load_json_artifact  # noqa: E402
from spectraxgk.nonlinear_gradient_followup import (  # noqa: E402
    NonlinearGradientFollowupConfig,
    nonlinear_gradient_followup_plan,
)


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _label(payload: dict[str, Any], path: Path) -> str:
    parameter = payload.get("parameter_name")
    if isinstance(parameter, str) and parameter:
        return f"{parameter}:{path.stem.removesuffix('_central_fd_gradient_gate')}"
    return path.stem


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "artifact",
        nargs="+",
        type=Path,
        help="Production central-FD gradient JSON artifacts to inspect.",
    )
    parser.add_argument("--case", default="nonlinear_turbulence_gradient_followup")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--max-gradient-uncertainty-rel", type=float, default=0.50)
    parser.add_argument("--max-fd-asymmetry-rel", type=float, default=0.50)
    parser.add_argument("--min-fd-response-fraction", type=float, default=0.03)
    parser.add_argument(
        "--sem-safety-factor",
        type=float,
        default=1.10,
        help="Safety factor applied to the ideal 1/sqrt(N) replica estimate.",
    )
    parser.add_argument("--max-extra-replicates-per-state", type=int, default=4)
    parser.add_argument("--default-nominal-timestep", type=float, default=0.05)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifacts = [load_json_artifact(path) for path in args.artifact]
    report = nonlinear_gradient_followup_plan(
        artifacts,
        paths=[_repo_relative(path) for path in args.artifact],
        labels=[_label(payload, path) for payload, path in zip(artifacts, args.artifact)],
        case=args.case,
        config=NonlinearGradientFollowupConfig(
            max_gradient_uncertainty_rel=args.max_gradient_uncertainty_rel,
            max_fd_asymmetry_rel=args.max_fd_asymmetry_rel,
            min_fd_response_fraction=args.min_fd_response_fraction,
            sem_safety_factor=args.sem_safety_factor,
            max_extra_replicates_per_state=args.max_extra_replicates_per_state,
            default_nominal_timestep=args.default_nominal_timestep,
        ),
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
        print(f"saved {args.json_out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
