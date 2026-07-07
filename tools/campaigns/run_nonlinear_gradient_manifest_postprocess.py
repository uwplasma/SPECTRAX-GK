#!/usr/bin/env python3
"""Run post-processing commands from a nonlinear-gradient campaign manifest."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

STATE_ORDER = ("baseline", "plus_delta", "minus_delta")
STEP_CHOICES = ("output-gates", "ensembles", "central-fd", "evidence")


@dataclass(frozen=True)
class ManifestCommand:
    """One command extracted from a nonlinear-gradient manifest."""

    step: str
    label: str
    command: str


def _ordered_states(state_commands: dict[str, Any]) -> list[str]:
    ordered = [state for state in STATE_ORDER if state in state_commands]
    ordered.extend(sorted(state for state in state_commands if state not in ordered))
    return ordered


def load_manifest(path: Path) -> dict[str, Any]:
    """Load and minimally validate a nonlinear-gradient campaign manifest."""

    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("kind") != "nonlinear_turbulence_gradient_campaign_manifest":
        raise ValueError(
            "expected kind='nonlinear_turbulence_gradient_campaign_manifest', "
            f"got {manifest.get('kind')!r}"
        )
    if not isinstance(manifest.get("state_ensemble_commands"), dict):
        raise ValueError("manifest is missing state_ensemble_commands")
    if not isinstance(manifest.get("promotion_contract"), dict):
        raise ValueError("manifest is missing promotion_contract")
    return manifest


def missing_expected_outputs(
    manifest: dict[str, Any], *, root: Path = ROOT
) -> list[Path]:
    """Return expected runtime outputs that are not present yet."""

    missing: list[Path] = []
    state_commands = manifest["state_ensemble_commands"]
    for state in _ordered_states(state_commands):
        row = state_commands[state]
        for raw_path in row.get("expected_outputs", []):
            path = root / str(raw_path)
            if not path.exists():
                missing.append(path)
    return missing


def collect_postprocess_commands(
    manifest: dict[str, Any],
    *,
    steps: set[str] | None = None,
) -> list[ManifestCommand]:
    """Collect post-processing commands in dependency order."""

    selected_steps = set(STEP_CHOICES) if steps is None else set(steps)
    unknown = selected_steps.difference(STEP_CHOICES)
    if unknown:
        raise ValueError(f"unknown post-processing step(s): {sorted(unknown)}")

    commands: list[ManifestCommand] = []
    state_commands = manifest["state_ensemble_commands"]
    for state in _ordered_states(state_commands):
        row = state_commands[state]
        if "output-gates" in selected_steps:
            command = row.get("output_gate_command")
            if not command:
                raise ValueError(f"state {state!r} is missing output_gate_command")
            commands.append(ManifestCommand("output-gates", state, command))
    for state in _ordered_states(state_commands):
        row = state_commands[state]
        if "ensembles" in selected_steps:
            command = row.get("build_ensemble_command")
            if not command:
                raise ValueError(f"state {state!r} is missing build_ensemble_command")
            commands.append(ManifestCommand("ensembles", state, command))

    contract = manifest["promotion_contract"]
    if "central-fd" in selected_steps:
        command = contract.get("central_fd_command")
        if not command:
            raise ValueError("promotion_contract is missing central_fd_command")
        commands.append(ManifestCommand("central-fd", "promotion", command))
    if "evidence" in selected_steps:
        command = contract.get("evidence_check_command")
        if not command:
            raise ValueError("promotion_contract is missing evidence_check_command")
        commands.append(ManifestCommand("evidence", "promotion", command))
    return commands


def _run_command(command: ManifestCommand, *, cwd: Path) -> int:
    print(f"[{command.step}:{command.label}] {command.command}", flush=True)
    completed = subprocess.run(shlex.split(command.command), cwd=cwd, check=False)
    return int(completed.returncode)


def _write_summary(
    *,
    path: Path,
    manifest_path: Path,
    commands: list[ManifestCommand],
    results: list[dict[str, Any]],
    dry_run: bool,
    missing_outputs: list[Path],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "nonlinear_gradient_manifest_postprocess_summary",
        "manifest": str(manifest_path),
        "dry_run": dry_run,
        "passed": all(row["returncode"] == 0 for row in results)
        and not missing_outputs,
        "missing_expected_outputs": [str(path) for path in missing_outputs],
        "commands": [
            {"step": item.step, "label": item.label, "command": item.command}
            for item in commands
        ],
        "results": results,
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--step",
        action="append",
        choices=STEP_CHOICES,
        help="Step to run. Repeat to select multiple steps; default runs all.",
    )
    parser.add_argument(
        "--require-outputs",
        action="store_true",
        help="Fail before running commands if any expected runtime output is missing.",
    )
    parser.add_argument(
        "--allow-blocked",
        action="store_true",
        help="Continue through non-zero post-processing exits and return success.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print/record commands without executing them.",
    )
    parser.add_argument("--summary-json", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_path = args.manifest.expanduser().resolve()
    manifest = load_manifest(manifest_path)
    steps = set(args.step) if args.step else None
    commands = collect_postprocess_commands(manifest, steps=steps)
    missing = missing_expected_outputs(manifest)
    if args.require_outputs and missing:
        for path in missing:
            print(f"missing expected output: {path}", file=sys.stderr)
        if args.summary_json:
            _write_summary(
                path=args.summary_json,
                manifest_path=manifest_path,
                commands=commands,
                results=[],
                dry_run=args.dry_run,
                missing_outputs=missing,
            )
        return 2

    results: list[dict[str, Any]] = []
    for command in commands:
        if args.dry_run:
            print(f"[dry-run:{command.step}:{command.label}] {command.command}")
            returncode = 0
        else:
            returncode = _run_command(command, cwd=ROOT)
        results.append(
            {
                "step": command.step,
                "label": command.label,
                "command": command.command,
                "returncode": returncode,
            }
        )
        if returncode != 0 and not args.allow_blocked:
            break

    if args.summary_json:
        _write_summary(
            path=args.summary_json,
            manifest_path=manifest_path,
            commands=commands,
            results=results,
            dry_run=args.dry_run,
            missing_outputs=[] if not args.require_outputs else missing,
        )
    failed = any(row["returncode"] != 0 for row in results)
    if failed and not args.allow_blocked:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
