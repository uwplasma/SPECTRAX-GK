#!/usr/bin/env python3
"""Run a fail-closed VMEC-JAX/SPECTRAX-GK transport-weight ladder.

The ladder starts from a solved QA candidate directory containing ``input.final``.
Each transport-weight refinement is run in its own output directory with
``--allow-failed-solved-wout-gate`` so failed branches remain inspectable. The
promotion rule is intentionally conservative: only candidates whose
``solved_wout_gate.json`` passes may be selected, and the selected candidate is
the largest passing transport weight. A separate long-window nonlinear
SPECTRAX-GK audit is still required before making turbulent-flux claims.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DRIVER = ROOT / "examples" / "optimization" / "vmec_jax_qa_low_turbulence_optimization.py"
DEFAULT_OUTDIR = ROOT / "tools_out" / "vmec_jax_guarded_transport_ladder"
DEFAULT_WEIGHTS = (5.0e-4, 1.0e-3, 2.5e-3, 5.0e-3, 1.0e-2)


def _read_json_object(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _weight_token(weight: float) -> str:
    token = f"{float(weight):.8g}".replace("-", "m").replace(".", "p")
    return token.replace("+", "")


def _parse_weights(raw: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated weight")
    if any(weight <= 0.0 for weight in values):
        raise argparse.ArgumentTypeError("transport weights must be positive")
    return values


def candidate_summary(root: Path, *, label: str, weight: float | None = None, baseline: bool = False) -> dict[str, Any]:
    """Return a compact promotion summary for a solved candidate directory."""

    history_path = root / "history.json"
    gate_path = root / "solved_wout_gate.json"
    history = _read_json_object(history_path) if history_path.exists() else {}
    gate = _read_json_object(gate_path) if gate_path.exists() else {}
    passed = bool(gate.get("passed", False))
    return {
        "label": label,
        "root": str(root),
        "baseline": bool(baseline),
        "transport_weight": None if weight is None else float(weight),
        "history_path": str(history_path) if history_path.exists() else None,
        "gate_path": str(gate_path) if gate_path.exists() else None,
        "passed": passed,
        "objective_final": history.get("objective_final"),
        "aspect_final": history.get("aspect_final"),
        "iota_final": history.get("iota_final"),
        "qs_final": history.get("qs_final"),
        "gate_checks": {name: check.get("passed") for name, check in gate.get("checks", {}).items()},
        "next_action": gate.get("next_action"),
    }


def select_promoted_candidate(summaries: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Select the largest passing transport-weight candidate, if any."""

    passing_transport = [
        item
        for item in summaries
        if bool(item.get("passed", False)) and item.get("transport_weight") is not None
    ]
    if passing_transport:
        return max(
            passing_transport,
            key=lambda item: (float(item["transport_weight"]), -float(item.get("objective_final") or 0.0)),
        )
    passing_baselines = [item for item in summaries if bool(item.get("passed", False)) and bool(item.get("baseline"))]
    return passing_baselines[0] if passing_baselines else None


def build_driver_command(
    *,
    python_executable: str,
    driver: Path,
    input_file: Path,
    outdir: Path,
    weight: float,
    driver_args: tuple[str, ...],
) -> list[str]:
    """Build one fail-closed transport-refinement command."""

    return [
        python_executable,
        str(driver),
        "--input",
        str(input_file),
        "--outdir",
        str(outdir),
        "--disable-mode-continuation",
        "--spectrax-weight",
        f"{float(weight):.16g}",
        "--allow-failed-solved-wout-gate",
        *driver_args,
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--constraints-dir", type=Path, required=True, help="Passing QA-only candidate directory")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR, help="Transport ladder output root")
    parser.add_argument("--weights", type=_parse_weights, default=DEFAULT_WEIGHTS)
    parser.add_argument("--driver", type=Path, default=DEFAULT_DRIVER)
    parser.add_argument("--python", default=sys.executable, help="Python executable used for candidate runs")
    parser.add_argument(
        "--driver-args",
        default="",
        help="Additional arguments passed to the VMEC-JAX QA driver, parsed with shlex.split",
    )
    parser.add_argument("--timeout-s", type=float, default=0.0, help="Per-candidate subprocess timeout; 0 disables")
    parser.add_argument("--dry-run", action="store_true", help="Write the launch plan without running candidates")
    parser.add_argument("--out-json", type=Path, default=None, help="Summary JSON path; defaults inside --outdir")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    constraints_dir = Path(args.constraints_dir)
    input_file = constraints_dir / "input.final"
    if not input_file.exists():
        raise FileNotFoundError(f"missing constraints restart input: {input_file}")
    driver_args = tuple(shlex.split(str(args.driver_args)))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    commands: list[dict[str, Any]] = []
    summaries = [candidate_summary(constraints_dir, label="QA constraints restart", baseline=True)]
    run_failures: list[dict[str, Any]] = []
    for weight in tuple(float(w) for w in args.weights):
        candidate_dir = outdir / f"transport_weight_{_weight_token(weight)}"
        command = build_driver_command(
            python_executable=str(args.python),
            driver=Path(args.driver),
            input_file=input_file,
            outdir=candidate_dir,
            weight=weight,
            driver_args=driver_args,
        )
        commands.append({"transport_weight": weight, "outdir": str(candidate_dir), "command": command})
        if not bool(args.dry_run):
            try:
                subprocess.run(
                    command,
                    cwd=ROOT,
                    check=True,
                    timeout=None if float(args.timeout_s) <= 0.0 else float(args.timeout_s),
                )
            except subprocess.CalledProcessError as exc:
                run_failures.append({"transport_weight": weight, "outdir": str(candidate_dir), "returncode": exc.returncode})
            except subprocess.TimeoutExpired:
                run_failures.append({"transport_weight": weight, "outdir": str(candidate_dir), "timeout_s": float(args.timeout_s)})
            summaries.append(candidate_summary(candidate_dir, label=f"transport weight {weight:.3g}", weight=weight))
    promoted = select_promoted_candidate(summaries)
    payload = {
        "kind": "vmec_jax_guarded_transport_ladder",
        "claim_scope": (
            "solved-candidate transport-weight admission only; long-window nonlinear SPECTRAX-GK "
            "audits are required before turbulent-flux optimization claims"
        ),
        "constraints_dir": str(constraints_dir),
        "restart_input": str(input_file),
        "dry_run": bool(args.dry_run),
        "commands": commands,
        "candidates": summaries,
        "run_failures": run_failures,
        "promoted_candidate": promoted,
        "passed": promoted is not None,
        "next_action": (
            "launch matched long-window nonlinear SPECTRAX-GK audits for the promoted candidate"
            if promoted is not None and promoted.get("transport_weight") is not None
            else "no transport-weight refinement passed; keep QA-only candidate or run a more conservative ladder"
        ),
    }
    out_json = Path(args.out_json) if args.out_json is not None else outdir / "guarded_transport_ladder.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": payload["passed"], "out_json": str(out_json), "promoted_candidate": promoted}, indent=2))
    return 0 if bool(payload["passed"]) or bool(args.dry_run) else 1


if __name__ == "__main__":
    raise SystemExit(main())
