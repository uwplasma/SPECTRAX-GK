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
import json
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


def _run_tool(cmd: list[str], *, cwd: Path | None, log_path: Path) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd is not None else None, capture_output=True, text=True, check=False)
    log_path.write_text(
        f"$ {' '.join(cmd)}\n\nstdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}\n",
        encoding="utf-8",
    )
    return {"returncode": int(proc.returncode), "log": str(log_path)}


def _lane_section(manifest: dict[str, Any]) -> dict[str, Any]:
    lanes = manifest.get("lane")
    if not isinstance(lanes, dict):
        raise ValueError("Manifest must contain a [lane.<name>] table per lane.")
    return lanes


def main() -> None:
    args = build_parser().parse_args()
    manifest = load_toml(args.manifest)
    lanes = _lane_section(manifest)

    selected = [args.lane] if args.lane is not None else list(lanes.keys())
    out_root = args.outdir.expanduser().resolve()
    py = str(args.python)
    here = Path(__file__).resolve().parent

    summary: dict[str, Any] = {"manifest": str(Path(args.manifest).resolve()), "lanes": {}}

    for lane_key in selected:
        if lane_key not in lanes:
            raise SystemExit(f"Lane not found in manifest: {lane_key}")
        cfg = lanes[lane_key]
        if not isinstance(cfg, dict):
            raise SystemExit(f"Lane config must be a table: lane.{lane_key}")

        gx_out = Path(cfg["gx_out"]).expanduser()
        config = Path(cfg["config"]).expanduser()
        out_dir = out_root / lane_key
        out_dir.mkdir(parents=True, exist_ok=True)

        lane_summary: dict[str, Any] = {"gx_out": str(gx_out), "config": str(config)}

        if "startup" in cfg:
            st = cfg["startup"]
            cmd = [
                py,
                str(here / "compare_gx_runtime_startup.py"),
                "--gx-dir",
                str(Path(st["gx_dir"]).expanduser()),
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
            lane_summary["startup"] = _run_tool(cmd, cwd=here, log_path=out_dir / "startup.log")

        if "diag_state" in cfg:
            ds = cfg["diag_state"]
            cmd = [
                py,
                str(here / "compare_gx_runtime_diag_state.py"),
                "--gx-dir",
                str(Path(ds["gx_dir"]).expanduser()),
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
            lane_summary["diag_state"] = _run_tool(cmd, cwd=here, log_path=out_dir / "diag_state.log")

        if "window" in cfg:
            w = cfg["window"]
            cmd = [
                py,
                str(here / "compare_gx_runtime_window.py"),
                "--gx-dir",
                str(Path(w["gx_dir"]).expanduser()),
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
            lane_summary["window"] = _run_tool(cmd, cwd=here, log_path=out_dir / "window.log")

        summary["lanes"][lane_key] = lane_summary

    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"saved {summary_path}")


if __name__ == "__main__":
    main()

