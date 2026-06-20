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

from spectraxgk.validation.nonlinear_gradient.evidence import load_json_artifact  # noqa: E402
from spectraxgk.validation.nonlinear_gradient.followup_core import (  # noqa: E402
    NonlinearGradientFollowupConfig,
)
from spectraxgk.validation.nonlinear_gradient.followup_plan import (  # noqa: E402
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


def _resolve_artifact_path(raw: Any) -> Path | None:
    if not isinstance(raw, str) or not raw:
        return None
    path = Path(raw)
    if path.is_absolute():
        return path if path.exists() else None
    for candidate in (ROOT / path, path):
        if candidate.exists():
            return candidate
    return None


def _hydrate_source_ensembles(payload: dict[str, Any]) -> dict[str, Any]:
    """Load compact source-ensemble rows from tracked ensemble artifacts."""

    source_ensembles = payload.get("source_ensembles")
    if not isinstance(source_ensembles, dict):
        return payload
    hydrated: dict[str, Any] = {}
    changed = False
    for state, raw in source_ensembles.items():
        if not isinstance(raw, dict):
            hydrated[state] = raw
            continue
        row = dict(raw)
        if isinstance(row.get("rows"), list):
            hydrated[state] = row
            continue
        ensemble_path = _resolve_artifact_path(row.get("path"))
        if ensemble_path is not None:
            ensemble = load_json_artifact(ensemble_path)
            if isinstance(ensemble.get("rows"), list):
                row["rows"] = ensemble["rows"]
                changed = True
        hydrated[state] = row
    if not changed:
        return payload
    return {**payload, "source_ensembles": hydrated}


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
    artifacts = [_hydrate_source_ensembles(load_json_artifact(path)) for path in args.artifact]
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
