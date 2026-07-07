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
    final_time = _output_final_time(path)
    return final_time is not None and final_time >= float(min_tmax)


def _output_final_time(path: Path) -> float | None:
    try:
        import netCDF4

        with netCDF4.Dataset(path) as root:
            time = root.groups["Grids"].variables["time"][:]
    except Exception:
        return None
    if len(time) == 0:
        return None
    return float(max(time))


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


def _planned_state_seeds(campaign_dir: Path, state: str) -> list[int]:
    folder = campaign_dir / "nonlinear_campaign" / state
    if not folder.exists():
        return []
    seeds = [_seed_from_path(path) for path in sorted(folder.glob("*_seed*.toml"))]
    return sorted({seed for seed in seeds if seed is not None})


def _state_output_status(campaign_dir: Path, state: str, *, min_tmax: float) -> dict[str, Any]:
    folder = campaign_dir / "nonlinear_campaign" / state
    planned = _planned_state_seeds(campaign_dir, state)
    output_by_seed: dict[int, Path] = {}
    if folder.exists():
        for path in sorted(folder.glob("*_seed*.out.nc")):
            seed = _seed_from_path(path)
            if seed is not None:
                output_by_seed[seed] = path
    completed: list[int] = []
    partial: list[dict[str, Any]] = []
    missing: list[int] = []
    for seed in sorted(set(planned).union(output_by_seed)):
        path = output_by_seed.get(seed)
        if path is None:
            missing.append(seed)
            continue
        final_time = _output_final_time(path)
        row = {
            "seed": seed,
            "path": str(path),
            "final_time": final_time,
            "size_bytes": path.stat().st_size,
        }
        if final_time is not None and final_time >= min_tmax:
            completed.append(seed)
        else:
            partial.append(row)
    return {
        "planned_count": len(planned),
        "planned_seeds": planned,
        "completed_count": len(completed),
        "completed_seeds": completed,
        "partial_count": len(partial),
        "partial_outputs": partial,
        "missing_count": len(missing),
        "missing_seeds": missing,
    }


def discover_campaign_status(
    campaign_dir: Path,
    *,
    min_tmax: float,
    min_common_pairs: int,
) -> dict[str, Any]:
    """Summarize output readiness without building ensemble artifacts."""

    plus = _state_output_status(campaign_dir, "plus_delta", min_tmax=float(min_tmax))
    minus = _state_output_status(campaign_dir, "minus_delta", min_tmax=float(min_tmax))
    common = sorted(set(plus["completed_seeds"]).intersection(minus["completed_seeds"]))
    return {
        "kind": "nonlinear_gradient_control_mean_campaign_status",
        "campaign_dir": str(campaign_dir),
        "min_output_tmax": float(min_tmax),
        "min_common_pairs": int(min_common_pairs),
        "common_pair_count": len(common),
        "common_seeds": common,
        "ready_for_strict_postprocess": len(common) >= int(min_common_pairs),
        "states": {
            "plus_delta": plus,
            "minus_delta": minus,
        },
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
        str(ROOT / "tools" / "artifacts" / "build_external_vmec_replicate_ensemble.py"),
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
    parser.add_argument(
        "--min-output-tmax",
        type=float,
        default=None,
        help="Minimum final time required in each output. Defaults to 0.99 * --tmax to allow fixed-step/sample-stride roundoff.",
    )
    parser.add_argument("--min-control-mean-pairs", type=int, default=21)
    parser.add_argument("--target-response-uncertainty-rel", type=float, default=0.50)
    parser.add_argument("--bootstrap-samples", type=int, default=256)
    parser.add_argument("--min-samples", type=int, default=24)
    parser.add_argument("--min-blocks", type=int, default=4)
    parser.add_argument("--allow-failed-state-ensembles", action="store_true")
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Only summarize completed/partial/missing outputs. Return 0 when ready, 2 otherwise.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    min_output_tmax = (
        float(args.min_output_tmax)
        if args.min_output_tmax is not None
        else 0.99 * float(args.tmax)
    )
    if args.status_only:
        status = discover_campaign_status(
            args.campaign_dir,
            min_tmax=min_output_tmax,
            min_common_pairs=int(args.min_common_pairs),
        )
        print(json.dumps(status, indent=2, sort_keys=True), flush=True)
        return 0 if bool(status["ready_for_strict_postprocess"]) else 2
    matched = discover_matched_outputs(args.campaign_dir, min_tmax=min_output_tmax)
    common_seeds = matched["common_seeds"]
    summary = {
        "campaign_dir": str(args.campaign_dir),
        "common_pair_count": len(common_seeds),
        "common_seeds": common_seeds,
        "requested_tmax": float(args.tmax),
        "min_output_tmax": min_output_tmax,
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
        str(ROOT / "tools" / "artifacts" / "build_nonlinear_gradient_control_mean_gate.py"),
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
