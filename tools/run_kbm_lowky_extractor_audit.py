#!/usr/bin/env python3
"""Run the direct low-ky KBM extractor audit using cached gx_time trajectories."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gx", type=Path, required=True, help="Path to GX KBM .out.nc file.")
    p.add_argument("--gx-input", type=Path, default=None, help="Optional GX input file for exact benchmark contract overrides.")
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
    return p


def main() -> None:
    args = build_parser().parse_args()
    here = Path(__file__).resolve().parent
    gx = args.gx.expanduser().resolve()
    gx_input = args.gx_input.expanduser().resolve() if args.gx_input is not None else gx.with_suffix(".in")
    gx_big = args.gx_big.expanduser().resolve() if args.gx_big is not None else gx.with_suffix(".big.nc")
    if not gx_big.exists():
        gx_big = (args.out.parent if args.out.parent != Path("") else Path.cwd()) / "missing_gx_big.nc"

    cmd = [
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
    env = os.environ.copy()
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
