#!/usr/bin/env python3
"""Orchestrate exact-state GX parity audits for multiple lanes.

This runner is intentionally file-path driven (manifest-based) so it can be
used on local laptops and remote machines without hardcoding office-only paths
into the repo.

It wraps:
- tools/compare_gx_runtime_startup.py
- tools/compare_gx_runtime_diag_state.py
- tools/compare_gx_runtime_window.py
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from spectraxgk.io import load_toml


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True, help="TOML manifest describing each exact-state lane.")
    p.add_argument("--lane", type=str, default=None, help="Optional single lane key to run.")
    p.add_argument("--outdir", type=Path, default=Path("tools_out") / "exact_state_audit", help="Output directory root.")
    p.add_argument("--python", type=str, default=sys.executable, help="Python interpreter to use for subprocess tool calls.")
    return p


def _tool_env(repo_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    entries = [str((repo_root / "src").resolve()), str(repo_root.resolve())]
    if existing:
        entries.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(entries)
    return env


def _run_tool(cmd: list[str], *, cwd: Path | None, log_path: Path, env: dict[str, str] | None = None) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    env_lines: list[str] = []
    if env is not None:
        for key in ("PYTHONPATH", "W7X_VMEC_FILE", "HSX_VMEC_FILE"):
            if key in env:
                env_lines.append(f"{key}={env[key]}")
    log_path.write_text(
        f"$ {' '.join(cmd)}\n"
        + (f"env:\n" + "\n".join(env_lines) + "\n\n" if env_lines else "\n")
        + f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}\n",
        encoding="utf-8",
    )
    return {"returncode": int(proc.returncode), "log": str(log_path)}


@contextmanager
def _temporary_env(env_updates: dict[str, str] | None):
    if not env_updates:
        yield
        return
    previous: dict[str, str | None] = {key: os.environ.get(key) for key in env_updates}
    try:
        for key, value in env_updates.items():
            os.environ[str(key)] = str(value)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _lane_section(manifest: dict[str, Any]) -> dict[str, Any]:
    lanes = manifest.get("lane")
    if not isinstance(lanes, dict):
        raise ValueError("Manifest must contain a [lane.<name>] table per lane.")
    return lanes


def _resolve_manifest_path(value: str | Path, *, manifest_dir: Path) -> Path:
    raw = os.path.expandvars(str(value))
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = manifest_dir / path
    return path.resolve()


def _expand_env_value(value: object) -> str:
    text = str(value)
    for _ in range(4):
        expanded = os.path.expanduser(os.path.expandvars(text))
        if expanded == text:
            return expanded
        text = expanded
    return text


def main() -> None:
    args = build_parser().parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    manifest_dir = manifest_path.parent
    manifest = load_toml(manifest_path)
    lanes = _lane_section(manifest)

    selected = [args.lane] if args.lane is not None else list(lanes.keys())
    out_root = args.outdir.expanduser().resolve()
    py = str(args.python)
    here = Path(__file__).resolve().parent
    repo_root = here.parent

    summary: dict[str, Any] = {"manifest": str(manifest_path), "lanes": {}}

    for lane_key in selected:
        if lane_key not in lanes:
            raise SystemExit(f"Lane not found in manifest: {lane_key}")
        cfg = lanes[lane_key]
        if not isinstance(cfg, dict):
            raise SystemExit(f"Lane config must be a table: lane.{lane_key}")

        gx_out = _resolve_manifest_path(cfg["gx_out"], manifest_dir=manifest_dir)
        config = _resolve_manifest_path(cfg["config"], manifest_dir=manifest_dir)
        out_dir = out_root / lane_key
        out_dir.mkdir(parents=True, exist_ok=True)

        lane_summary: dict[str, Any] = {"gx_out": str(gx_out), "config": str(config)}
        env_cfg = cfg.get("env")
        env_updates = (
            {str(key): _expand_env_value(value) for key, value in env_cfg.items()}
            if isinstance(env_cfg, dict)
            else None
        )
        if env_updates:
            lane_summary["env"] = env_updates

        with _temporary_env(env_updates):
            tool_env = _tool_env(repo_root)
            if "startup" in cfg:
                st = cfg["startup"]
                cmd = [
                    py,
                    str(here / "compare_gx_runtime_startup.py"),
                    "--gx-dir",
                    str(_resolve_manifest_path(st["gx_dir"], manifest_dir=manifest_dir)),
                    "--gx-out",
                    str(gx_out),
                    "--config",
                    str(config),
                    "--ky",
                    str(st["ky"]),
                ]
                if "kx_target" in st:
                    cmd += ["--kx-target", str(st["kx_target"])]
                if "y0" in st:
                    cmd += ["--y0", str(st["y0"])]
                lane_summary["startup"] = _run_tool(cmd, cwd=here, log_path=out_dir / "startup.log", env=tool_env)

            if "diag_state" in cfg:
                ds = cfg["diag_state"]
                cmd = [
                    py,
                    str(here / "compare_gx_runtime_diag_state.py"),
                    "--gx-dir",
                    str(_resolve_manifest_path(ds["gx_dir"], manifest_dir=manifest_dir)),
                    "--gx-out",
                    str(gx_out),
                    "--config",
                    str(config),
                    "--time-index",
                    str(ds["time_index"]),
                    "--out",
                    str(out_dir / "diag_state.csv"),
                ]
                if "y0" in ds:
                    cmd += ["--y0", str(ds["y0"])]
                lane_summary["diag_state"] = _run_tool(cmd, cwd=here, log_path=out_dir / "diag_state.log", env=tool_env)

            if "window" in cfg:
                w = cfg["window"]
                cmd = [
                    py,
                    str(here / "compare_gx_runtime_window.py"),
                    "--gx-dir",
                    str(_resolve_manifest_path(w["gx_dir"], manifest_dir=manifest_dir)),
                    "--gx-out",
                    str(gx_out),
                    "--config",
                    str(config),
                    "--time-index-start",
                    str(w["time_index_start"]),
                    "--time-index-stop",
                    str(w["time_index_stop"]),
                    "--out",
                    str(out_dir / "window.csv"),
                ]
                if "steps" in w:
                    cmd += ["--steps", str(w["steps"])]
                if "ky" in w:
                    cmd += ["--ky", str(w["ky"])]
                if "y0" in w:
                    cmd += ["--y0", str(w["y0"])]
                lane_summary["window"] = _run_tool(cmd, cwd=here, log_path=out_dir / "window.log", env=tool_env)

        summary["lanes"][lane_key] = lane_summary

    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"saved {summary_path}")


if __name__ == "__main__":
    main()
