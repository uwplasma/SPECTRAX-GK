#!/usr/bin/env python3
"""Run all nested nonlinear tasks from an overdetermined campaign manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.campaigns.run_nonlinear_gradient_direct_campaign import (  # noqa: E402
    DirectTask,
    collect_direct_tasks,
    load_manifest as load_nested_manifest,
    run_tasks,
)


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


def load_overdetermined_manifest(path: Path) -> dict[str, Any]:
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
    if not isinstance(payload.get("controls"), list):
        raise ValueError("manifest is missing controls")
    return payload


def nested_manifest_paths(manifest: dict[str, Any]) -> list[tuple[str, Path]]:
    """Return ``(control_slug, nested_manifest_path)`` entries."""

    paths: list[tuple[str, Path]] = []
    for control in manifest.get("controls", []):
        if not isinstance(control, dict):
            raise ValueError("all controls must be JSON objects")
        slug = str(control.get("coefficient_slug", "unknown"))
        raw_path = control.get("expected_nonlinear_campaign_manifest")
        if not raw_path:
            raise ValueError(
                f"control {slug!r} is missing expected_nonlinear_campaign_manifest"
            )
        paths.append((slug, _resolve_repo_path(str(raw_path))))
    return paths


def collect_overdetermined_direct_tasks(
    manifest: dict[str, Any],
    *,
    controls: set[str] | None = None,
    states: set[str] | None = None,
    labels: set[str] | None = None,
) -> list[DirectTask]:
    """Collect nested direct tasks in control/state order."""

    tasks: list[DirectTask] = []
    for slug, path in nested_manifest_paths(manifest):
        if controls is not None and slug not in controls:
            continue
        nested = load_nested_manifest(path)
        for task in collect_direct_tasks(nested, states=states, labels=labels):
            tasks.append(
                DirectTask(
                    state=f"{slug}:{task.state}",
                    label=task.label,
                    command=task.command,
                    config=task.config,
                    output=task.output,
                )
            )
    return tasks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--gpu",
        action="append",
        default=None,
        help="GPU id to use; repeat for multiple GPUs.",
    )
    parser.add_argument(
        "--control",
        action="append",
        default=None,
        help="Control slug to run; repeat to filter.",
    )
    parser.add_argument(
        "--state", action="append", default=None, help="State to run; repeat to filter."
    )
    parser.add_argument(
        "--label",
        action="append",
        default=None,
        help="Output label/stem to run; repeat to filter.",
    )
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--status-json", type=Path)
    parser.add_argument("--timeout-s", type=float, default=10800.0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_path = args.manifest.expanduser().resolve()
    manifest = load_overdetermined_manifest(manifest_path)
    tasks = collect_overdetermined_direct_tasks(
        manifest,
        controls=set(args.control) if args.control else None,
        states=set(args.state) if args.state else None,
        labels=set(args.label) if args.label else None,
    )
    if not tasks:
        print("No direct full-horizon tasks selected.", file=sys.stderr)
        return 2

    log_dir = args.log_dir or manifest_path.parent / "overdetermined_full_direct_logs"
    status_json = args.status_json or log_dir / "status.json"
    gpus = tuple(args.gpu or ["0"])
    if args.dry_run:
        for index, task in enumerate(tasks):
            gpu = gpus[index % len(gpus)]
            print(f"[dry-run gpu={gpu} state={task.state}] {task.command}")
        return 0

    results = run_tasks(
        tasks,
        gpus=gpus,
        log_dir=log_dir,
        status_json=status_json,
        timeout_s=args.timeout_s,
        skip_existing=bool(args.skip_existing),
        stop_on_failure=bool(args.stop_on_failure),
    )
    status_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "overdetermined_nonlinear_gradient_direct_campaign_status",
        "manifest": _repo_path(manifest_path),
        "task_count": len(tasks),
        "finished_count": sum(
            row["status"] in {"finished", "skipped"} for row in results
        ),
        "failed_count": sum(row["status"] == "failed" for row in results),
        "results": results,
    }
    status_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 1 if payload["failed_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
