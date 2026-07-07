#!/usr/bin/env python3
"""Post-process and promote an overdetermined nonlinear-gradient campaign."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shlex
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PlannedCommand:
    """One command in the overdetermined post-processing sequence."""

    step: str
    label: str
    command: str


def _repo_path(path: Path | str) -> str:
    raw = Path(path)
    try:
        return raw.resolve().relative_to(ROOT.resolve()).as_posix()
    except (OSError, ValueError):
        return str(path)


def _resolve_repo_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    return ROOT / path


def _default_status_json(manifest_path: Path) -> Path:
    stem = manifest_path.stem
    if stem.endswith("_plan"):
        return manifest_path.with_name(f"{stem[:-5]}_status.json")
    return manifest_path.with_suffix(".status.json")


def load_manifest(path: Path) -> dict[str, Any]:
    """Load and validate an overdetermined nonlinear-gradient manifest."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    if (
        payload.get("kind")
        != "overdetermined_nonlinear_turbulence_gradient_campaign_manifest"
    ):
        raise ValueError(
            "expected kind='overdetermined_nonlinear_turbulence_gradient_campaign_manifest', "
            f"got {payload.get('kind')!r}"
        )
    controls = payload.get("controls")
    if not isinstance(controls, list) or not controls:
        raise ValueError("manifest is missing a non-empty controls list")
    if not isinstance(payload.get("promotion_contract"), dict):
        raise ValueError("manifest is missing promotion_contract")
    return payload


def build_postprocess_commands(
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
    status_json: Path | None = None,
) -> list[PlannedCommand]:
    """Return the fail-closed post-processing command sequence."""

    commands: list[PlannedCommand] = []
    for control in manifest["controls"]:
        if not isinstance(control, dict):
            raise ValueError("all controls must be JSON objects")
        slug = str(control.get("coefficient_slug", "unknown"))
        nested_raw = control.get("expected_nonlinear_campaign_manifest")
        if not nested_raw:
            raise ValueError(
                f"control {slug!r} is missing expected_nonlinear_campaign_manifest"
            )
        nested_manifest = _resolve_repo_path(str(nested_raw))
        summary_json = nested_manifest.with_name("postprocess_summary.json")
        commands.append(
            PlannedCommand(
                "per-control-postprocess",
                slug,
                " ".join(
                    [
                        "python3",
                        "tools/campaigns/run_nonlinear_gradient_manifest_postprocess.py",
                        shlex.quote(_repo_path(nested_manifest)),
                        "--require-outputs",
                        "--summary-json",
                        shlex.quote(_repo_path(summary_json)),
                    ]
                ),
            )
        )

    contract = manifest["promotion_contract"]
    ranking_command = contract.get("candidate_ranking_command")
    if not ranking_command:
        raise ValueError("promotion_contract is missing candidate_ranking_command")
    commands.append(
        PlannedCommand("candidate-ranking", "promotion", str(ranking_command))
    )

    final_status = status_json or _default_status_json(manifest_path)
    commands.append(
        PlannedCommand(
            "final-status",
            "promotion",
            " ".join(
                [
                    "python3",
                    "tools/release/check_overdetermined_nonlinear_gradient_campaign.py",
                    shlex.quote(_repo_path(manifest_path)),
                    "--out-json",
                    shlex.quote(_repo_path(final_status)),
                    "--fail-on-blocked",
                ]
            ),
        )
    )
    return commands


def _run_command(command: PlannedCommand) -> int:
    print(f"[{command.step}:{command.label}] {command.command}", flush=True)
    completed = subprocess.run(shlex.split(command.command), cwd=ROOT, check=False)
    return int(completed.returncode)


def _write_summary(
    *,
    path: Path,
    manifest_path: Path,
    commands: list[PlannedCommand],
    results: list[dict[str, Any]],
    dry_run: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "overdetermined_nonlinear_gradient_postprocess_summary",
        "manifest": _repo_path(manifest_path),
        "dry_run": dry_run,
        "passed": all(int(row["returncode"]) == 0 for row in results),
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
    parser.add_argument("--status-json", type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--allow-blocked", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_path = args.manifest.expanduser().resolve()
    manifest = load_manifest(manifest_path)
    status_json = args.status_json.expanduser().resolve() if args.status_json else None
    commands = build_postprocess_commands(
        manifest,
        manifest_path=manifest_path,
        status_json=status_json,
    )
    results: list[dict[str, Any]] = []
    for command in commands:
        if args.dry_run:
            print(f"[dry-run:{command.step}:{command.label}] {command.command}")
            returncode = 0
        else:
            returncode = _run_command(command)
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

    summary_json = args.summary_json or (
        (status_json or _default_status_json(manifest_path)).with_name(
            "overdetermined_postprocess_summary.json"
        )
    )
    _write_summary(
        path=summary_json,
        manifest_path=manifest_path,
        commands=commands,
        results=results,
        dry_run=bool(args.dry_run),
    )
    failed = any(int(row["returncode"]) != 0 for row in results)
    if failed and not args.allow_blocked:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
