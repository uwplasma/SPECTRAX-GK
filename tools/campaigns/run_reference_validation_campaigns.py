#!/usr/bin/env python3
"""Run reference-validation and benchmark-refresh campaign helpers.

This module groups small campaign wrappers that used to live as separate
entry points. Keeping them together makes the tool surface easier to navigate
while preserving the existing benchmark and reference-comparison workflows.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib

import pandas as pd
from netCDF4 import Dataset

from spectraxgk.workflows.runtime.toml import load_toml

ROOT = Path(__file__).resolve().parents[2]


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


def _select_jobs(
    jobs: list[RefreshJob], selected_names: set[str] | None
) -> list[RefreshJob]:
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


def build_benchmark_refresh_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the benchmark refresh matrix.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "tools" / "benchmark_refresh_manifest.toml",
        help="Benchmark refresh manifest.",
    )
    parser.add_argument(
        "--job",
        action="append",
        default=None,
        help="Run only the named job (repeatable).",
    )
    parser.add_argument("--list", action="store_true", help="List jobs and exit.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected commands without executing them.",
    )
    parser.add_argument(
        "--skip-missing-env",
        action="store_true",
        help="Skip jobs whose required environment variables are not set.",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=ROOT / "tools_out" / "benchmark_refresh_summary.json",
        help="Write a JSON summary for the attempted refresh.",
    )
    return parser


def main_benchmark_refresh(argv: list[str] | None = None) -> int:
    args = build_benchmark_refresh_parser().parse_args(argv)
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
                summary["jobs"].append(entry)  # type: ignore[union-attr]
                continue
            entry["status"] = "failed_missing_env"
            summary["jobs"].append(entry)  # type: ignore[union-attr]
            _write_summary(_resolve(args.summary_out), summary)
            return 2

        if args.dry_run:
            entry["status"] = "dry_run"
            summary["jobs"].append(entry)  # type: ignore[union-attr]
            print(f"[dry-run] {job.name}: cd {rendered_cwd} && {rendered_command}")
            continue

        proc = subprocess.run(rendered_command, shell=True, cwd=rendered_cwd, env=env)
        if proc.returncode != 0:
            entry["status"] = "failed"
            entry["returncode"] = proc.returncode
            summary["jobs"].append(entry)  # type: ignore[union-attr]
            rc = proc.returncode
            break

        missing_outputs = _check_outputs(job)
        if missing_outputs:
            entry["status"] = "failed_missing_output"
            entry["missing_outputs"] = missing_outputs
            summary["jobs"].append(entry)  # type: ignore[union-attr]
            rc = 3
            break

        entry["status"] = "success"
        entry["outputs"] = [str(_resolve(_render(path))) for path in job.outputs]
        summary["jobs"].append(entry)  # type: ignore[union-attr]

    _write_summary(_resolve(args.summary_out), summary)
    return rc


def build_imported_linear_targeted_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run targeted imported-linear audits one ky at a time."
    )
    parser.add_argument(
        "--gx", type=Path, required=True, help="Path to reference .out.nc file."
    )
    parser.add_argument(
        "--geometry-file",
        type=Path,
        required=True,
        help="Geometry file passed to the imported-linear comparison tool.",
    )
    parser.add_argument(
        "--gx-input", type=Path, default=None, help="Optional reference input file."
    )
    parser.add_argument("--out", type=Path, required=True, help="Combined CSV output.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("tools_out") / "imported_linear_cache",
        help="Directory for per-ky CSV cache artifacts.",
    )
    parser.add_argument(
        "--ky", type=float, nargs="*", default=None, help="Explicit ky values to run."
    )
    parser.add_argument(
        "--max-kys",
        type=int,
        default=None,
        help="If set and --ky is omitted, only run the first N positive ky values.",
    )
    parser.add_argument("--Nl", type=int, default=None)
    parser.add_argument("--Nm", type=int, default=None)
    parser.add_argument("--tprim", type=float, default=3.0)
    parser.add_argument("--fprim", type=float, default=1.0)
    parser.add_argument("--tau-e", type=float, default=1.0, dest="tau_e")
    parser.add_argument("--damp-ends-amp", type=float, default=0.1)
    parser.add_argument("--damp-ends-widthfrac", type=float, default=1.0 / 8.0)
    parser.add_argument(
        "--mode-method", choices=("z_index", "max", "project", "svd"), default="z_index"
    )
    parser.add_argument("--rel-floor-fraction", type=float, default=1.0e-2)
    parser.add_argument(
        "--sample-step-stride",
        type=int,
        default=1,
        help="Subsample saved diagnostic samples by this stride.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="If set, only score the first N selected samples per ky.",
    )
    parser.add_argument(
        "--sample-window",
        choices=("head", "tail"),
        default="head",
        help="When --max-samples is set, select first or last N stride-filtered samples.",
    )
    parser.add_argument(
        "--reuse-cache",
        action="store_true",
        help="Reuse existing per-ky CSV rows if present.",
    )
    return parser


def _positive_ky(path: Path) -> list[float]:
    root = Dataset(path, "r")
    try:
        ky = [
            float(v) for v in root.groups["Grids"].variables["ky"][:] if float(v) > 0.0
        ]
    finally:
        root.close()
    return ky


def _ky_tag(ky: float) -> str:
    return f"{float(ky):0.4f}".replace(".", "p")


def _combine_csvs(cache_dir: Path, ky_values: list[float], out: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for ky in ky_values:
        csv_path = cache_dir / f"ky_{_ky_tag(ky)}.csv"
        if not csv_path.exists():
            continue
        rows.append(pd.read_csv(csv_path))
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not df.empty and "ky" in df.columns:
        df = df.sort_values("ky").reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def main_imported_linear_targeted(argv: list[str] | None = None) -> int:
    args = build_imported_linear_targeted_parser().parse_args(argv)
    here = Path(__file__).resolve().parent

    gx = args.gx.expanduser().resolve()
    geometry_file = args.geometry_file.expanduser().resolve()
    gx_input = None if args.gx_input is None else args.gx_input.expanduser().resolve()
    out = args.out.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.ky:
        ky_values = [float(v) for v in args.ky]
    else:
        ky_values = _positive_ky(gx)
        if args.max_kys is not None:
            ky_values = ky_values[: max(0, int(args.max_kys))]

    for ky in ky_values:
        cache_csv = cache_dir / f"ky_{_ky_tag(ky)}.csv"
        if args.reuse_cache and cache_csv.exists():
            _combine_csvs(cache_dir, ky_values, out)
            continue

        cmd = [
            sys.executable,
            str(here.parent / "comparison" / "compare_gx_imported_linear.py"),
            "--gx",
            str(gx),
            "--geometry-file",
            str(geometry_file),
            "--out",
            str(cache_csv),
            "--cache-dir",
            str(cache_dir),
            "--ky",
            str(float(ky)),
            "--sample-step-stride",
            str(int(args.sample_step_stride)),
            "--sample-window",
            str(args.sample_window),
            "--tprim",
            str(float(args.tprim)),
            "--fprim",
            str(float(args.fprim)),
            "--tau-e",
            str(float(args.tau_e)),
            "--damp-ends-amp",
            str(float(args.damp_ends_amp)),
            "--damp-ends-widthfrac",
            str(float(args.damp_ends_widthfrac)),
            "--mode-method",
            str(args.mode_method),
            "--rel-floor-fraction",
            str(float(args.rel_floor_fraction)),
        ]
        if args.Nl is not None:
            cmd += ["--Nl", str(int(args.Nl))]
        if args.Nm is not None:
            cmd += ["--Nm", str(int(args.Nm))]
        if args.max_samples is not None:
            cmd += ["--max-samples", str(int(args.max_samples))]
        if args.reuse_cache:
            cmd += ["--reuse-cache"]
        if gx_input is not None:
            cmd += ["--gx-input", str(gx_input)]
        subprocess.run(cmd, check=True)
        _combine_csvs(cache_dir, ky_values, out)

    df = _combine_csvs(cache_dir, ky_values, out)
    print(df.to_string(index=False))
    print(f"saved {out}")
    return 0


def build_kbm_lowky_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the direct low-ky KBM extractor audit using cached trajectories."
    )
    parser.add_argument(
        "--gx", type=Path, required=True, help="Path to reference KBM .out.nc file."
    )
    parser.add_argument(
        "--gx-input",
        type=Path,
        default=None,
        help="Optional input file for exact benchmark contract overrides.",
    )
    parser.add_argument(
        "--geometry-file",
        type=Path,
        default=None,
        help="Optional geometry source for imported-geometry KBM audits (defaults to --gx).",
    )
    parser.add_argument(
        "--gx-big",
        type=Path,
        default=None,
        help="Optional big/eigenfunction file. If omitted and unavailable, eigenfunction scoring is skipped.",
    )
    parser.add_argument(
        "--ky",
        type=str,
        default="0.3,0.4",
        help="Comma-separated low-ky values to score.",
    )
    parser.add_argument(
        "--trajectory-dir",
        type=Path,
        default=Path("tools_out") / "kbm_lowky_traj",
        help="Trajectory cache directory.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("tools_out") / "kbm_lowky_branch.csv",
        help="Main branch-score CSV output.",
    )
    parser.add_argument(
        "--candidate-out",
        type=Path,
        default=Path("tools_out") / "kbm_lowky_candidates.csv",
        help="Per-candidate CSV output.",
    )
    parser.add_argument(
        "--branch-solvers",
        type=str,
        default="gx_time@project,gx_time@project_late,gx_time@svd,gx_time@svd_late,gx_time@max,gx_time@z_index",
        help="Candidate extractor list passed through to compare_gx_kbm.py.",
    )
    parser.add_argument(
        "--sample-step-stride",
        type=int,
        default=1,
        help="Subsample saved diagnostic samples for imported-geometry audits.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on selected diagnostic samples for imported-geometry audits.",
    )
    return parser


def _geo_option(path: Path) -> str:
    data = load_toml(path)
    return str(data.get("Geometry", {}).get("geo_option", "s-alpha")).strip().lower()


def _build_kbm_lowky_command(
    args, *, here: Path, gx: Path, gx_input: Path, gx_big: Path
) -> list[str]:
    geo_option = _geo_option(gx_input)
    comparison_dir = here.parent / "comparison"
    if geo_option == "s-alpha":
        return [
            "python",
            str(comparison_dir / "compare_gx_kbm.py"),
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

    geometry_file = (
        args.geometry_file.expanduser().resolve()
        if args.geometry_file is not None
        else gx
    )
    cmd = [
        "python",
        str(here / "run_reference_validation_campaigns.py"),
        "imported-linear-targeted",
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


# Compatibility alias for existing tests and downstream scripts that import helpers.
_build_command = _build_kbm_lowky_command


def main_kbm_lowky_extractor(argv: list[str] | None = None) -> int:
    args = build_kbm_lowky_parser().parse_args(argv)
    here = Path(__file__).resolve().parent
    gx = args.gx.expanduser().resolve()
    gx_input = (
        args.gx_input.expanduser().resolve()
        if args.gx_input is not None
        else gx.with_suffix(".in")
    )
    gx_big = (
        args.gx_big.expanduser().resolve()
        if args.gx_big is not None
        else gx.with_suffix(".big.nc")
    )
    if not gx_big.exists():
        gx_big = (
            args.out.parent if args.out.parent != Path("") else Path.cwd()
        ) / "missing_gx_big.nc"

    cmd = _build_kbm_lowky_command(
        args, here=here, gx=gx, gx_input=gx_input, gx_big=gx_big
    )
    subprocess.run(cmd, check=True, env=os.environ.copy())
    return 0


SUBCOMMANDS = {
    "benchmark-refresh": main_benchmark_refresh,
    "imported-linear-targeted": main_imported_linear_targeted,
    "kbm-lowky-extractor": main_kbm_lowky_extractor,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=sorted(SUBCOMMANDS))
    return parser


def main(argv: list[str] | None = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    if not tokens or tokens[0] in {"-h", "--help"}:
        build_parser().parse_args(tokens)
        return 0
    command, rest = tokens[0], tokens[1:]
    try:
        handler = SUBCOMMANDS[command]
    except KeyError:
        build_parser().parse_args([command])
        return 2
    return handler(rest)


if __name__ == "__main__":
    raise SystemExit(main())
