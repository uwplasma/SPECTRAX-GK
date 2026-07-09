#!/usr/bin/env python3
"""Run direct full-horizon nonlinear commands from a launch manifest."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import queue
import re
import shlex
import subprocess
import sys
import threading
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
STATE_ORDER = ("baseline", "plus_delta", "minus_delta")
SUPPORTED_MANIFEST_KINDS = {
    "nonlinear_turbulence_gradient_campaign_manifest",
    "external_vmec_holdout_config_manifest",
}
POSTPROCESS_STEP_CHOICES = ("output-gates", "ensembles", "central-fd", "evidence")
DEFAULT_CONTROL_MEAN_CASE_PREFIX = "qa_ess_zbs10_rel7p5_control_mean"
DEFAULT_CONTROL_MEAN_VARIANCE_REPORT = (
    ROOT / "docs" / "_static" / "qa_ess_zbs10_rel7p5_variance_reduction_plan.json"
)
DEFAULT_CONTROL_MEAN_OUT_ROOT = ROOT / "docs" / "_static"
SEED_RE = re.compile(r"(?:^|_)seed(?P<seed>[0-9]+)(?:_|\.|$)")


@dataclass(frozen=True)
class DirectTask:
    """One direct full-horizon runtime task from the campaign manifest."""

    state: str
    label: str
    command: str
    config: Path
    output: Path


@dataclass(frozen=True)
class ManifestCommand:
    """One post-processing command extracted from a nonlinear-gradient manifest."""

    step: str
    label: str
    command: str


@dataclass(frozen=True)
class PlannedCommand:
    """One command in the overdetermined post-processing sequence."""

    step: str
    label: str
    command: str


def _repo_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except (OSError, ValueError):
        return str(path)


def _resolve_repo_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    return ROOT / path


def _ordered_states(state_commands: dict[str, Any]) -> list[str]:
    ordered = [state for state in STATE_ORDER if state in state_commands]
    ordered.extend(sorted(state for state in state_commands if state not in ordered))
    return ordered


def load_manifest(path: Path) -> dict[str, Any]:
    """Load and minimally validate a direct nonlinear campaign manifest."""

    manifest = json.loads(path.read_text(encoding="utf-8"))
    kind = manifest.get("kind")
    if kind not in SUPPORTED_MANIFEST_KINDS:
        supported = ", ".join(sorted(SUPPORTED_MANIFEST_KINDS))
        raise ValueError(f"expected manifest kind in {{{supported}}}, got {kind!r}")
    if kind == "nonlinear_turbulence_gradient_campaign_manifest" and not isinstance(
        manifest.get("state_ensemble_commands"),
        dict,
    ):
        raise ValueError("manifest is missing state_ensemble_commands")
    if kind == "external_vmec_holdout_config_manifest" and not isinstance(
        manifest.get("direct_full_horizon_launch_commands"),
        list,
    ):
        raise ValueError("manifest is missing direct_full_horizon_launch_commands")
    return manifest


def _split_command_env(command: str) -> tuple[list[str], dict[str, str]]:
    """Split a manifest command into argv and leading environment overrides."""

    parts = shlex.split(command)
    env: dict[str, str] = {}
    index = 0
    while index < len(parts):
        part = parts[index]
        if "=" not in part or part.startswith("-"):
            break
        key, value = part.split("=", maxsplit=1)
        if not key.isidentifier():
            break
        env[key] = value
        index += 1
    argv = parts[index:]
    if not argv:
        raise ValueError(f"direct command does not contain an executable: {command}")
    return argv, env


def _config_from_command(command: str) -> Path:
    parts, _ = _split_command_env(command)
    try:
        raw = parts[parts.index("--config") + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"direct command is missing '--config': {command}") from exc
    return ROOT / raw


def _task_from_command(
    *,
    state: str,
    command: str,
    labels: set[str] | None = None,
    output_override: Path | None = None,
) -> DirectTask | None:
    config = _config_from_command(command)
    output = (
        config.with_suffix(".out.nc") if output_override is None else output_override
    )
    label = output.name
    variant_label = label.removesuffix(".out.nc")
    config_label = config.name
    config_stem = config.stem
    if labels is not None and labels.isdisjoint(
        {label, variant_label, config_label, config_stem}
    ):
        return None
    return DirectTask(
        state=state,
        label=label,
        command=command,
        config=config,
        output=output,
    )


def collect_direct_tasks(
    manifest: dict[str, Any],
    *,
    states: set[str] | None = None,
    labels: set[str] | None = None,
) -> list[DirectTask]:
    """Collect direct full-horizon tasks in deterministic state order."""

    if manifest.get("kind") == "external_vmec_holdout_config_manifest":
        state = "external_vmec"
        if states is not None and state not in states:
            return []
        output_by_config = {
            str(row.get("path")): ROOT / str(row["output_path"])
            for row in manifest.get("configs", [])
            if isinstance(row, dict) and row.get("path") and row.get("output_path")
        }
        tasks = []
        for command in manifest.get("direct_full_horizon_launch_commands", []):
            command_str = str(command)
            config = _config_from_command(command_str)
            output_override = output_by_config.get(_repo_path(config))
            task = _task_from_command(
                state=state,
                command=command_str,
                labels=labels,
                output_override=output_override,
            )
            if task is not None:
                tasks.append(task)
        return tasks

    state_commands = manifest["state_ensemble_commands"]
    tasks: list[DirectTask] = []
    for state in _ordered_states(state_commands):
        if states is not None and state not in states:
            continue
        row = state_commands[state]
        for command in row.get("direct_full_horizon_launch_commands", []):
            task = _task_from_command(state=state, command=str(command), labels=labels)
            if task is not None:
                tasks.append(task)
    return tasks


def _task_env(gpu: str, *, extra_env: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env.update(extra_env or {})
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["PYTHONPATH"] = (
        "src" if not env.get("PYTHONPATH") else f"src:{env['PYTHONPATH']}"
    )
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
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _output_bundle_paths(output: Path) -> tuple[Path, ...]:
    """Return the files that make a nonlinear runtime output restart-complete."""

    name = output.name
    if name.endswith(".out.nc"):
        base = output.with_name(name[: -len(".out.nc")])
        return tuple(
            base.with_suffix(f".{suffix}")
            for suffix in ("out.nc", "restart.nc", "big.nc")
        )
    return (output,)


def _output_bundle_complete(output: Path) -> bool:
    return all(
        path.exists() and path.stat().st_size > 0
        for path in _output_bundle_paths(output)
    )


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
    bundle_paths = _output_bundle_paths(task.output)
    row["required_output_bundle"] = [_repo_path(path) for path in bundle_paths]
    if skip_existing and _output_bundle_complete(task.output):
        return {**row, "status": "skipped", "returncode": 0, "elapsed_s": 0.0}

    start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"task={task.label} gpu={gpu}\ncommand={task.command}\n\n")
        log.flush()
        argv, command_env = _split_command_env(task.command)
        completed = subprocess.run(
            argv,
            cwd=ROOT,
            env=_task_env(gpu, extra_env=command_env),
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
            check=False,
        )
    elapsed = time.monotonic() - start
    status = "finished" if completed.returncode == 0 else "failed"
    return {
        **row,
        "status": status,
        "returncode": int(completed.returncode),
        "elapsed_s": elapsed,
    }


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
    _write_status(
        status_json, results, task_count=len(tasks), campaign_status="running"
    )

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
                bundle_paths = _output_bundle_paths(task.output)
                result = {
                    "state": task.state,
                    "label": task.label,
                    "command": task.command,
                    "config": _repo_path(task.config),
                    "output": _repo_path(task.output),
                    "required_output_bundle": [
                        _repo_path(path) for path in bundle_paths
                    ],
                    "gpu": gpu,
                    "log": _repo_path(log_dir / f"{task.output.stem}.gpu{gpu}.log"),
                    "status": "failed",
                    "returncode": None,
                    "elapsed_s": timeout_s,
                    "error": f"timeout after {exc.timeout} s",
                }
            with lock:
                results.append(result)
                _write_status(
                    status_json,
                    results,
                    task_count=len(tasks),
                    campaign_status="running",
                )
                if stop_on_failure and result["status"] == "failed":
                    stop_event.set()
            work_queue.task_done()

    threads = [
        threading.Thread(target=worker, args=(gpu,), daemon=True) for gpu in gpus
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    final_status = (
        "failed" if any(row["status"] == "failed" for row in results) else "finished"
    )
    _write_status(
        status_json, results, task_count=len(tasks), campaign_status=final_status
    )
    return results


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
        nested = load_manifest(path)
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


def load_postprocess_manifest(path: Path) -> dict[str, Any]:
    """Load and validate a nonlinear-gradient post-processing manifest."""

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
    for state in _ordered_states(manifest["state_ensemble_commands"]):
        row = manifest["state_ensemble_commands"][state]
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

    selected_steps = set(POSTPROCESS_STEP_CHOICES) if steps is None else set(steps)
    unknown = selected_steps.difference(POSTPROCESS_STEP_CHOICES)
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


def _run_manifest_postprocess_command(command: ManifestCommand, *, cwd: Path) -> int:
    print(f"[{command.step}:{command.label}] {command.command}", flush=True)
    completed = subprocess.run(shlex.split(command.command), cwd=cwd, check=False)
    return int(completed.returncode)


def _write_manifest_postprocess_summary(
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


def _default_overdetermined_status_json(manifest_path: Path) -> Path:
    stem = manifest_path.stem
    if stem.endswith("_plan"):
        return manifest_path.with_name(f"{stem[:-5]}_status.json")
    return manifest_path.with_suffix(".status.json")


def build_postprocess_commands(
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
    status_json: Path | None = None,
) -> list[PlannedCommand]:
    """Return the fail-closed overdetermined post-processing sequence."""

    if not isinstance(manifest.get("controls"), list) or not manifest["controls"]:
        raise ValueError("manifest is missing a non-empty controls list")
    if not isinstance(manifest.get("promotion_contract"), dict):
        raise ValueError("manifest is missing promotion_contract")

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
                        "tools/campaigns/run_nonlinear_gradient_direct_campaign.py",
                        "postprocess",
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

    final_status = status_json or _default_overdetermined_status_json(manifest_path)
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


def _run_overdetermined_postprocess_command(command: PlannedCommand) -> int:
    print(f"[{command.step}:{command.label}] {command.command}", flush=True)
    completed = subprocess.run(shlex.split(command.command), cwd=ROOT, check=False)
    return int(completed.returncode)


def _write_overdetermined_postprocess_summary(
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


def _seed_from_path(path: Path) -> int | None:
    match = SEED_RE.search(path.name)
    return None if match is None else int(match.group("seed"))


def _output_final_time(path: Path) -> float | None:
    try:
        import netCDF4

        with netCDF4.Dataset(path) as root:
            time_values = root.groups["Grids"].variables["time"][:]
    except Exception:
        return None
    if len(time_values) == 0:
        return None
    return float(max(time_values))


def _output_reaches_tmax(path: Path, min_tmax: float | None) -> bool:
    if min_tmax is None:
        return True
    final_time = _output_final_time(path)
    return final_time is not None and final_time >= float(min_tmax)


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


def _state_output_status(
    campaign_dir: Path, state: str, *, min_tmax: float
) -> dict[str, Any]:
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
    """Summarize control-mean output readiness without building ensembles."""

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


def _run_control_mean_tool(cmd: list[str]) -> int:
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(ROOT), check=False).returncode


def _build_control_mean_state_ensemble(
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
        _repo_path(out_dir),
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
    tool_rc = _run_control_mean_tool(cmd)
    ensemble_path = out_dir / ensemble_name
    ensemble_passed = False
    if ensemble_path.exists():
        try:
            ensemble_passed = bool(
                json.loads(ensemble_path.read_text(encoding="utf-8")).get(
                    "passed", False
                )
            )
        except json.JSONDecodeError:
            ensemble_passed = False
    return {
        "tool_rc": tool_rc,
        "ensemble_path": ensemble_path,
        "ensemble_passed": ensemble_passed,
    }


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


def build_overdetermined_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run all nested nonlinear tasks from an overdetermined campaign manifest."
    )
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


def build_postprocess_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run post-processing commands from a nonlinear-gradient campaign manifest."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--step",
        action="append",
        choices=POSTPROCESS_STEP_CHOICES,
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


def build_overdetermined_postprocess_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Post-process and promote an overdetermined nonlinear-gradient campaign."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--status-json", type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--allow-blocked", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def build_control_mean_postprocess_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Postprocess a nonlinear-gradient independent control-mean campaign."
    )
    parser.add_argument("--campaign-dir", required=True, type=Path)
    parser.add_argument(
        "--variance-report", type=Path, default=DEFAULT_CONTROL_MEAN_VARIANCE_REPORT
    )
    parser.add_argument("--out-root", type=Path, default=DEFAULT_CONTROL_MEAN_OUT_ROOT)
    parser.add_argument("--case-prefix", default=DEFAULT_CONTROL_MEAN_CASE_PREFIX)
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


def main_direct(argv: list[str] | None = None) -> int:
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


def main_overdetermined(argv: list[str] | None = None) -> int:
    args = build_overdetermined_parser().parse_args(argv)
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


def main_postprocess(argv: list[str] | None = None) -> int:
    args = build_postprocess_parser().parse_args(argv)
    manifest_path = args.manifest.expanduser().resolve()
    manifest = load_postprocess_manifest(manifest_path)
    steps = set(args.step) if args.step else None
    commands = collect_postprocess_commands(manifest, steps=steps)
    missing = missing_expected_outputs(manifest)
    if args.require_outputs and missing:
        for path in missing:
            print(f"missing expected output: {path}", file=sys.stderr)
        if args.summary_json:
            _write_manifest_postprocess_summary(
                path=args.summary_json,
                manifest_path=manifest_path,
                commands=commands,
                results=[],
                dry_run=bool(args.dry_run),
                missing_outputs=missing,
            )
        return 2

    results: list[dict[str, Any]] = []
    for command in commands:
        if args.dry_run:
            print(f"[dry-run:{command.step}:{command.label}] {command.command}")
            returncode = 0
        else:
            returncode = _run_manifest_postprocess_command(command, cwd=ROOT)
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
        _write_manifest_postprocess_summary(
            path=args.summary_json,
            manifest_path=manifest_path,
            commands=commands,
            results=results,
            dry_run=bool(args.dry_run),
            missing_outputs=[] if not args.require_outputs else missing,
        )
    failed = any(row["returncode"] != 0 for row in results)
    if failed and not args.allow_blocked:
        return 1
    return 0


def main_overdetermined_postprocess(argv: list[str] | None = None) -> int:
    args = build_overdetermined_postprocess_parser().parse_args(argv)
    manifest_path = args.manifest.expanduser().resolve()
    manifest = load_overdetermined_manifest(manifest_path)
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
            returncode = _run_overdetermined_postprocess_command(command)
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
        (status_json or _default_overdetermined_status_json(manifest_path)).with_name(
            "overdetermined_postprocess_summary.json"
        )
    )
    _write_overdetermined_postprocess_summary(
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


def main_control_mean_postprocess(argv: list[str] | None = None) -> int:
    args = build_control_mean_postprocess_parser().parse_args(argv)
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

    plus_state = _build_control_mean_state_ensemble(
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
    minus_state = _build_control_mean_state_ensemble(
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
        str(ROOT / "tools" / "artifacts" / "build_nonlinear_gradient_evidence.py"),
        "control-mean",
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
    gate_rc = _run_control_mean_tool(gate_cmd)
    payload = {
        "plus_ensemble_tool_rc": plus_state["tool_rc"],
        "minus_ensemble_tool_rc": minus_state["tool_rc"],
        "plus_ensemble_passed": plus_state["ensemble_passed"],
        "minus_ensemble_passed": minus_state["ensemble_passed"],
        "gate_rc": gate_rc,
        "plus_ensemble": _repo_path(plus_state["ensemble_path"]),
        "minus_ensemble": _repo_path(minus_state["ensemble_path"]),
        "gate": _repo_path(gate_prefix.with_suffix(".json")),
    }
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    state_gate_ok = bool(args.allow_failed_state_ensembles) or (
        plus_state["ensemble_passed"] and minus_state["ensemble_passed"]
    )
    return 0 if state_gate_ok and gate_rc == 0 else 1


def main(argv: list[str] | None = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    if tokens and tokens[0] == "control-mean-postprocess":
        return main_control_mean_postprocess(tokens[1:])
    if tokens and tokens[0] == "postprocess-overdetermined":
        return main_overdetermined_postprocess(tokens[1:])
    if tokens and tokens[0] == "postprocess":
        return main_postprocess(tokens[1:])
    if tokens and tokens[0] == "overdetermined":
        return main_overdetermined(tokens[1:])
    if tokens and tokens[0] == "direct":
        return main_direct(tokens[1:])
    return main_direct(tokens)


if __name__ == "__main__":
    raise SystemExit(main())
