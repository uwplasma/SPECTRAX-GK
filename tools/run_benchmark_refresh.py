#!/usr/bin/env python3
"""Run the benchmark refresh matrix from a TOML manifest."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RefreshJob:
    name: str
    description: str
    cwd: str
    command: str
    outputs: tuple[str, ...]
    requires_env: tuple[str, ...]
    enabled: bool = True


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _render(text: str) -> str:
    return os.path.expandvars(text.replace("{root}", str(ROOT)))


def _load_manifest(path: Path) -> list[RefreshJob]:
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    jobs_raw = data.get("job", [])
    jobs: list[RefreshJob] = []
    for item in jobs_raw:
        jobs.append(
            RefreshJob(
                name=str(item["name"]),
                description=str(item.get("description", "")),
                cwd=str(item.get("cwd", "{root}")),
                command=str(item["command"]),
                outputs=tuple(str(x) for x in item.get("outputs", [])),
                requires_env=tuple(str(x) for x in item.get("requires_env", [])),
                enabled=bool(item.get("enabled", True)),
            )
        )
    return jobs


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "tools" / "benchmark_refresh_manifest.toml",
        help="Benchmark refresh manifest.",
    )
    p.add_argument("--job", action="append", default=None, help="Run only the named job (repeatable).")
    p.add_argument("--list", action="store_true", help="List jobs and exit.")
    p.add_argument("--dry-run", action="store_true", help="Print the selected commands without executing them.")
    p.add_argument(
        "--skip-missing-env",
        action="store_true",
        help="Skip jobs whose required environment variables are not set.",
    )
    p.add_argument(
        "--summary-out",
        type=Path,
        default=ROOT / "tools_out" / "benchmark_refresh_summary.json",
        help="Write a JSON summary for the attempted refresh.",
    )
    return p


def _select_jobs(jobs: list[RefreshJob], selected_names: set[str] | None) -> list[RefreshJob]:
    out = [job for job in jobs if job.enabled]
    if selected_names:
        out = [job for job in out if job.name in selected_names]
    return out


def _missing_env(job: RefreshJob) -> list[str]:
    return [name for name in job.requires_env if not os.environ.get(name)]


def _check_outputs(job: RefreshJob) -> list[str]:
    missing: list[str] = []
    for output in job.outputs:
        if not _resolve(_render(output)).exists():
            missing.append(str(_resolve(_render(output))))
    return missing


def _write_summary(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = _build_parser().parse_args()
    manifest = _resolve(args.manifest)
    jobs = _load_manifest(manifest)
    selected = set(args.job or [])
    run_jobs = _select_jobs(jobs, selected if selected else None)

    if args.list:
        for job in run_jobs:
            req = f" requires_env={list(job.requires_env)}" if job.requires_env else ""
            print(f"{job.name}: {job.description}{req}")
        return 0

    summary: dict[str, object] = {
        "manifest": str(manifest),
        "jobs": [],
    }
    env = os.environ.copy()
    rc = 0

    for job in run_jobs:
        missing_env = _missing_env(job)
        rendered_command = _render(job.command)
        rendered_cwd = _resolve(_render(job.cwd))
        entry = {
            "name": job.name,
            "description": job.description,
            "cwd": str(rendered_cwd),
            "command": rendered_command,
            "requires_env": list(job.requires_env),
            "missing_env": missing_env,
            "status": "pending",
        }
        if missing_env:
            if args.skip_missing_env:
                entry["status"] = "skipped_missing_env"
                summary["jobs"].append(entry)
                continue
            entry["status"] = "failed_missing_env"
            summary["jobs"].append(entry)
            _write_summary(_resolve(args.summary_out), summary)
            return 2

        if args.dry_run:
            entry["status"] = "dry_run"
            summary["jobs"].append(entry)
            print(f"[dry-run] {job.name}: cd {rendered_cwd} && {rendered_command}")
            continue

        proc = subprocess.run(rendered_command, shell=True, cwd=rendered_cwd, env=env)
        if proc.returncode != 0:
            entry["status"] = "failed"
            entry["returncode"] = proc.returncode
            summary["jobs"].append(entry)
            rc = proc.returncode
            break

        missing_outputs = _check_outputs(job)
        if missing_outputs:
            entry["status"] = "failed_missing_output"
            entry["missing_outputs"] = missing_outputs
            summary["jobs"].append(entry)
            rc = 3
            break

        entry["status"] = "success"
        entry["outputs"] = [str(_resolve(_render(path))) for path in job.outputs]
        summary["jobs"].append(entry)

    _write_summary(_resolve(args.summary_out), summary)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
