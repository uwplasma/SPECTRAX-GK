#!/usr/bin/env python3
"""Run direct full-horizon nonlinear-gradient commands from a manifest."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import queue
import shlex
import subprocess
import sys
import threading
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STATE_ORDER = ("baseline", "plus_delta", "minus_delta")


@dataclass(frozen=True)
class DirectTask:
    """One direct full-horizon runtime task from the campaign manifest."""

    state: str
    label: str
    command: str
    config: Path
    output: Path


def _repo_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except (OSError, ValueError):
        return str(path)


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
    return manifest


def _config_from_command(command: str) -> Path:
    parts = shlex.split(command)
    try:
        raw = parts[parts.index("--config") + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"direct command is missing '--config': {command}") from exc
    return ROOT / raw


def collect_direct_tasks(
    manifest: dict[str, Any],
    *,
    states: set[str] | None = None,
    labels: set[str] | None = None,
) -> list[DirectTask]:
    """Collect direct full-horizon tasks in deterministic state order."""

    state_commands = manifest["state_ensemble_commands"]
    tasks: list[DirectTask] = []
    for state in _ordered_states(state_commands):
        if states is not None and state not in states:
            continue
        row = state_commands[state]
        for command in row.get("direct_full_horizon_launch_commands", []):
            config = _config_from_command(command)
            output = config.with_suffix(".out.nc")
            label = output.name
            variant_label = label.removesuffix(".out.nc")
            if labels is not None and label not in labels and variant_label not in labels:
                continue
            tasks.append(
                DirectTask(
                    state=state,
                    label=label,
                    command=command,
                    config=config,
                    output=output,
                )
            )
    return tasks


def _task_env(gpu: str, *, extra_env: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env.update(extra_env or {})
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["PYTHONPATH"] = "src" if not env.get("PYTHONPATH") else f"src:{env['PYTHONPATH']}"
    return env


def _write_status(
    path: Path,
    results: list[dict[str, Any]],
    *,
    task_count: int | None = None,
    campaign_status: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = len(results) if task_count is None else task_count
    finished_count = sum(row["status"] in {"finished", "skipped"} for row in results)
    failed_count = sum(row["status"] == "failed" for row in results)
    payload = {
        "kind": "nonlinear_gradient_direct_campaign_status",
        "updated": time.time(),
        "status": campaign_status or ("failed" if failed_count else "finished"),
        "task_count": expected,
        "results": results,
        "finished_count": finished_count,
        "failed_count": failed_count,
        "pending_count": max(expected - len(results), 0),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_one(
    task: DirectTask,
    *,
    gpu: str,
    log_dir: Path,
    timeout_s: float,
    skip_existing: bool,
) -> dict[str, Any]:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{task.output.stem}.gpu{gpu}.log"
    row = {
        "state": task.state,
        "label": task.label,
        "command": task.command,
        "config": _repo_path(task.config),
        "output": _repo_path(task.output),
        "gpu": gpu,
        "log": _repo_path(log_path),
    }
    if skip_existing and task.output.exists() and task.output.stat().st_size > 0:
        return {**row, "status": "skipped", "returncode": 0, "elapsed_s": 0.0}

    start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"task={task.label} gpu={gpu}\ncommand={task.command}\n\n")
        log.flush()
        completed = subprocess.run(
            shlex.split(task.command),
            cwd=ROOT,
            env=_task_env(gpu),
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
            check=False,
        )
    elapsed = time.monotonic() - start
    status = "finished" if completed.returncode == 0 else "failed"
    return {**row, "status": status, "returncode": int(completed.returncode), "elapsed_s": elapsed}


def run_tasks(
    tasks: list[DirectTask],
    *,
    gpus: tuple[str, ...],
    log_dir: Path,
    status_json: Path,
    timeout_s: float,
    skip_existing: bool = False,
    stop_on_failure: bool = False,
) -> list[dict[str, Any]]:
    """Run direct tasks with one worker per listed GPU."""

    if not gpus:
        raise ValueError("at least one GPU id must be supplied")
    work_queue: queue.Queue[DirectTask] = queue.Queue()
    for task in tasks:
        work_queue.put(task)

    results: list[dict[str, Any]] = []
    lock = threading.Lock()
    stop_event = threading.Event()
    _write_status(status_json, results, task_count=len(tasks), campaign_status="running")

    def worker(gpu: str) -> None:
        while not stop_event.is_set():
            try:
                task = work_queue.get_nowait()
            except queue.Empty:
                return
            try:
                result = _run_one(
                    task,
                    gpu=gpu,
                    log_dir=log_dir,
                    timeout_s=timeout_s,
                    skip_existing=skip_existing,
                )
            except subprocess.TimeoutExpired as exc:
                result = {
                    "state": task.state,
                    "label": task.label,
                    "command": task.command,
                    "config": _repo_path(task.config),
                    "output": _repo_path(task.output),
                    "gpu": gpu,
                    "log": _repo_path(log_dir / f"{task.output.stem}.gpu{gpu}.log"),
                    "status": "failed",
                    "returncode": None,
                    "elapsed_s": timeout_s,
                    "error": f"timeout after {exc.timeout} s",
                }
            with lock:
                results.append(result)
                _write_status(status_json, results, task_count=len(tasks), campaign_status="running")
                if stop_on_failure and result["status"] == "failed":
                    stop_event.set()
            work_queue.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,), daemon=True) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    final_status = "failed" if any(row["status"] == "failed" for row in results) else "finished"
    _write_status(status_json, results, task_count=len(tasks), campaign_status=final_status)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--gpu", action="append", default=None, help="GPU id to use; repeat for multiple GPUs.")
    parser.add_argument("--state", action="append", default=None, help="State to run; repeat to filter.")
    parser.add_argument("--label", action="append", default=None, help="Output label/stem to run; repeat to filter.")
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
    manifest = load_manifest(manifest_path)
    tasks = collect_direct_tasks(
        manifest,
        states=set(args.state) if args.state else None,
        labels=set(args.label) if args.label else None,
    )
    if not tasks:
        print("No direct full-horizon tasks selected.", file=sys.stderr)
        return 2

    log_dir = args.log_dir or manifest_path.parent.parent / "full_direct_logs"
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
        skip_existing=args.skip_existing,
        stop_on_failure=args.stop_on_failure,
    )
    return 1 if any(row["status"] == "failed" for row in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
