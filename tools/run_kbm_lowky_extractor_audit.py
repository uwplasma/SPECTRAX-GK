#!/usr/bin/env python3
"""Run the direct low-ky KBM extractor audit using cached gx_time trajectories."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess

from spectraxgk.io import load_toml


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gx", type=Path, required=True, help="Path to GX KBM .out.nc file.")
    p.add_argument("--gx-input", type=Path, default=None, help="Optional GX input file for exact benchmark contract overrides.")
    p.add_argument(
        "--geometry-file",
        type=Path,
        default=None,
        help="Optional geometry source for imported-geometry KBM audits (defaults to --gx).",
    )
    p.add_argument(
        "--gx-big",
        type=Path,
        default=None,
        help="Optional GX big/eigenfunction file. If omitted and unavailable, eigenfunction scoring is skipped.",
    )
    p.add_argument("--ky", type=str, default="0.3,0.4", help="Comma-separated low-ky values to score.")
    p.add_argument(
        "--trajectory-dir",
        type=Path,
        default=Path("tools_out") / "kbm_lowky_traj",
        help="Trajectory cache directory.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("tools_out") / "kbm_lowky_branch.csv",
        help="Main branch-score CSV output.",
    )
    p.add_argument(
        "--candidate-out",
        type=Path,
        default=Path("tools_out") / "kbm_lowky_candidates.csv",
        help="Per-candidate CSV output.",
    )
    p.add_argument(
        "--branch-solvers",
        type=str,
        default="gx_time@project,gx_time@project_late,gx_time@svd,gx_time@svd_late,gx_time@max,gx_time@z_index",
        help="Candidate extractor list passed through to compare_gx_kbm.py.",
    )
    p.add_argument(
        "--sample-step-stride",
        type=int,
        default=1,
        help="Subsample saved GX diagnostic samples for imported-geometry audits.",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on selected GX diagnostic samples for imported-geometry audits.",
    )
    return p


def _geo_option(path: Path) -> str:
    data = load_toml(path)
    return str(data.get("Geometry", {}).get("geo_option", "s-alpha")).strip().lower()


def _build_command(args, *, here: Path, gx: Path, gx_input: Path, gx_big: Path) -> list[str]:
    geo_option = _geo_option(gx_input)
    if geo_option == "s-alpha":
        return [
            "python",
            str(here / "compare_gx_kbm.py"),
            "--gx",
            str(gx),
            "--gx-input",
            str(gx_input),
            "--gx-big",
            str(gx_big),
            "--ky",
            str(args.ky),
            "--trajectory-dir",
            str(args.trajectory_dir.expanduser()),
            "--reuse-trajectory",
            "--branch-solvers",
            str(args.branch_solvers),
            "--out",
            str(args.out.expanduser()),
            "--candidate-out",
            str(args.candidate_out.expanduser()),
        ]

    geometry_file = args.geometry_file.expanduser().resolve() if args.geometry_file is not None else gx
    cmd = [
        "python",
        str(here / "run_imported_linear_targeted_audit.py"),
        "--gx",
        str(gx),
        "--geometry-file",
        str(geometry_file),
        "--gx-input",
        str(gx_input),
        "--out",
        str(args.out.expanduser()),
        "--cache-dir",
        str(args.trajectory_dir.expanduser()),
        "--sample-step-stride",
        str(int(args.sample_step_stride)),
    ]
    if args.max_samples is not None:
        cmd += ["--max-samples", str(int(args.max_samples))]
    ky_values = [k.strip() for k in str(args.ky).split(",") if k.strip()]
    if ky_values:
        cmd += ["--ky", *ky_values]
    return cmd


def main() -> None:
    args = build_parser().parse_args()
    here = Path(__file__).resolve().parent
    gx = args.gx.expanduser().resolve()
    gx_input = args.gx_input.expanduser().resolve() if args.gx_input is not None else gx.with_suffix(".in")
    gx_big = args.gx_big.expanduser().resolve() if args.gx_big is not None else gx.with_suffix(".big.nc")
    if not gx_big.exists():
        gx_big = (args.out.parent if args.out.parent != Path("") else Path.cwd()) / "missing_gx_big.nc"

    cmd = _build_command(args, here=here, gx=gx, gx_input=gx_input, gx_big=gx_big)
    env = os.environ.copy()
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
