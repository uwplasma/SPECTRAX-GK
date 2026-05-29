#!/usr/bin/env python3
"""Postprocess a nonlinear-gradient independent control-mean campaign.

This wrapper discovers matched plus/minus nonlinear outputs from a campaign
folder, builds replicated nonlinear-window ensemble gates for each state, and
then evaluates the independent control-mean uncertainty gate. It is intended to
make long office campaigns fail closed and reproducible with one command after
all matched seeds finish.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CASE_PREFIX = "qa_ess_zbs10_rel7p5_control_mean"
DEFAULT_VARIANCE_REPORT = ROOT / "docs" / "_static" / "qa_ess_zbs10_rel7p5_variance_reduction_plan.json"
DEFAULT_OUT_ROOT = ROOT / "docs" / "_static"

SEED_RE = re.compile(r"(?:^|_)seed(?P<seed>[0-9]+)(?:_|\.|$)")


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _seed_from_path(path: Path) -> int | None:
    match = SEED_RE.search(path.name)
    return None if match is None else int(match.group("seed"))


def _output_reaches_tmax(path: Path, min_tmax: float | None) -> bool:
    if min_tmax is None:
        return True
    try:
        import netCDF4

        with netCDF4.Dataset(path) as root:
            time = root.groups["Grids"].variables["time"][:]
    except Exception:
        return False
    if len(time) == 0:
        return False
    return float(max(time)) >= float(min_tmax)


def _discover_state_outputs(
    campaign_dir: Path,
    state: str,
    *,
    min_tmax: float | None = None,
) -> dict[int, Path]:
    folder = campaign_dir / "nonlinear_campaign" / state
    outputs: dict[int, Path] = {}
    if not folder.exists():
        return outputs
    for path in sorted(folder.glob("*_seed*.out.nc")):
        seed = _seed_from_path(path)
        if (
            seed is not None
            and path.stat().st_size > 0
            and _output_reaches_tmax(path, min_tmax)
        ):
            outputs[seed] = path
    return outputs


def discover_matched_outputs(
    campaign_dir: Path, *, min_tmax: float | None = None
) -> dict[str, Any]:
    """Return completed plus/minus output files keyed by common seed."""

    plus = _discover_state_outputs(campaign_dir, "plus_delta", min_tmax=min_tmax)
    minus = _discover_state_outputs(campaign_dir, "minus_delta", min_tmax=min_tmax)
    common = sorted(set(plus).intersection(minus))
    return {
        "plus": [plus[seed] for seed in common],
        "minus": [minus[seed] for seed in common],
        "common_seeds": common,
        "plus_completed": sorted(plus),
        "minus_completed": sorted(minus),
    }


def _run(cmd: list[str]) -> int:
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(ROOT), check=False).returncode


def _build_state_ensemble(
    *,
    state: str,
    outputs: list[Path],
    out_root: Path,
    case_prefix: str,
    tmin: float,
    tmax: float,
    bootstrap_samples: int,
    min_samples: int,
    min_blocks: int,
) -> dict[str, Any]:
    state_prefix = f"{case_prefix}_{state}"
    out_dir = out_root / f"{state_prefix}_replicates"
    ensemble_name = f"{state_prefix}_t{int(round(tmax))}_ensemble_gate.json"
    cmd = [
        sys.executable,
        str(ROOT / "tools" / "build_external_vmec_replicate_ensemble.py"),
        *[str(path) for path in outputs],
        "--out-dir",
        str(out_dir),
        "--case",
        f"{state_prefix}_replicated_nonlinear_window",
        "--tmin",
        f"{tmin:g}",
        "--tmax",
        f"{tmax:g}",
        "--artifact-prefix",
        _repo_relative(out_dir),
        "--readiness-json",
        f"{state_prefix}_readiness.json",
        "--ensemble-json",
        ensemble_name,
        "--out-png",
        f"{state_prefix}_t{int(round(tmax))}_ensemble_gate.png",
        "--bootstrap-samples",
        str(int(bootstrap_samples)),
        "--min-samples",
        str(int(min_samples)),
        "--min-blocks",
        str(int(min_blocks)),
    ]
    tool_rc = _run(cmd)
    ensemble_path = out_dir / ensemble_name
    ensemble_passed = False
    if ensemble_path.exists():
        try:
            ensemble_passed = bool(json.loads(ensemble_path.read_text(encoding="utf-8")).get("passed", False))
        except json.JSONDecodeError:
            ensemble_passed = False
    return {
        "tool_rc": tool_rc,
        "ensemble_path": ensemble_path,
        "ensemble_passed": ensemble_passed,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", required=True, type=Path)
    parser.add_argument("--variance-report", type=Path, default=DEFAULT_VARIANCE_REPORT)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--case-prefix", default=DEFAULT_CASE_PREFIX)
    parser.add_argument("--tmin", type=float, default=450.0)
    parser.add_argument("--tmax", type=float, default=900.0)
    parser.add_argument("--min-common-pairs", type=int, default=21)
    parser.add_argument("--min-control-mean-pairs", type=int, default=21)
    parser.add_argument("--target-response-uncertainty-rel", type=float, default=0.50)
    parser.add_argument("--bootstrap-samples", type=int, default=256)
    parser.add_argument("--min-samples", type=int, default=24)
    parser.add_argument("--min-blocks", type=int, default=4)
    parser.add_argument("--allow-failed-state-ensembles", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    matched = discover_matched_outputs(args.campaign_dir, min_tmax=float(args.tmax))
    common_seeds = matched["common_seeds"]
    summary = {
        "campaign_dir": str(args.campaign_dir),
        "common_pair_count": len(common_seeds),
        "common_seeds": common_seeds,
        "required_tmax": float(args.tmax),
        "plus_completed": matched["plus_completed"],
        "minus_completed": matched["minus_completed"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if len(common_seeds) < int(args.min_common_pairs):
        print(
            f"Need at least {args.min_common_pairs} matched completed pairs; "
            f"found {len(common_seeds)}.",
            file=sys.stderr,
        )
        return 2

    plus_state = _build_state_ensemble(
        state="plus_delta",
        outputs=list(matched["plus"]),
        out_root=args.out_root,
        case_prefix=str(args.case_prefix),
        tmin=float(args.tmin),
        tmax=float(args.tmax),
        bootstrap_samples=int(args.bootstrap_samples),
        min_samples=int(args.min_samples),
        min_blocks=int(args.min_blocks),
    )
    minus_state = _build_state_ensemble(
        state="minus_delta",
        outputs=list(matched["minus"]),
        out_root=args.out_root,
        case_prefix=str(args.case_prefix),
        tmin=float(args.tmin),
        tmax=float(args.tmax),
        bootstrap_samples=int(args.bootstrap_samples),
        min_samples=int(args.min_samples),
        min_blocks=int(args.min_blocks),
    )

    gate_prefix = args.out_root / f"{args.case_prefix}_gate"
    gate_cmd = [
        sys.executable,
        str(ROOT / "tools" / "build_nonlinear_gradient_control_mean_gate.py"),
        "--variance-report",
        str(args.variance_report),
        "--plus-ensemble",
        str(plus_state["ensemble_path"]),
        "--minus-ensemble",
        str(minus_state["ensemble_path"]),
        "--case",
        f"{args.case_prefix}_gate",
        "--out-prefix",
        str(gate_prefix),
        "--target-response-uncertainty-rel",
        str(float(args.target_response_uncertainty_rel)),
        "--min-control-mean-pairs",
        str(int(args.min_control_mean_pairs)),
    ]
    if args.allow_failed_state_ensembles:
        gate_cmd.append("--allow-failed-state-ensembles")
    gate_rc = _run(gate_cmd)
    payload = {
        "plus_ensemble_tool_rc": plus_state["tool_rc"],
        "minus_ensemble_tool_rc": minus_state["tool_rc"],
        "plus_ensemble_passed": plus_state["ensemble_passed"],
        "minus_ensemble_passed": minus_state["ensemble_passed"],
        "gate_rc": gate_rc,
        "plus_ensemble": _repo_relative(plus_state["ensemble_path"]),
        "minus_ensemble": _repo_relative(minus_state["ensemble_path"]),
        "gate": _repo_relative(gate_prefix.with_suffix(".json")),
    }
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    state_gate_ok = (
        bool(args.allow_failed_state_ensembles)
        or (plus_state["ensemble_passed"] and minus_state["ensemble_passed"])
    )
    return 0 if state_gate_ok and gate_rc == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
