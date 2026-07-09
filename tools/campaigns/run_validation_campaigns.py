#!/usr/bin/env python3
"""Run validation, benchmark-refresh, and runtime-gate campaign helpers.

This module groups command-style campaign wrappers under one entry point so the
tool surface stays navigable while preserving benchmark, comparison, and runtime
validation workflows.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib
from typing import Any

import jax
import numpy as np
import pandas as pd
from netCDF4 import Dataset

from spectraxgk.artifacts.restart import write_netcdf_restart_state
from spectraxgk.geometry import load_imported_geometry_netcdf
from spectraxgk.geometry.vmec_eik import generate_runtime_vmec_eik
from spectraxgk.runtime import RuntimeNonlinearResult, run_runtime_nonlinear
from spectraxgk.workflows.runtime.toml import load_runtime_from_toml, load_toml

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
        str(here / "run_validation_campaigns.py"),
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



ROUNDTRIP_FIELDS = (
    "theta",
    "bmag_profile",
    "gds2_profile",
    "gds21_profile",
    "gds22_profile",
    "cv_profile",
    "gb_profile",
    "jacobian_profile",
    "grho_profile",
)


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


def _lane_env(lane_cfg: dict[str, Any]) -> dict[str, str] | None:
    env_cfg = lane_cfg.get("env")
    return (
        {str(key): _expand_env_value(value) for key, value in env_cfg.items()}
        if isinstance(env_cfg, dict)
        else None
    )


def _raw_run_value(raw: dict[str, Any], key: str, *, default: Any = None) -> Any:
    run_cfg = raw.get("run")
    if not isinstance(run_cfg, dict):
        return default
    return run_cfg.get(key, default)


def _scalar_rel(ref: float, test: float) -> dict[str, float]:
    diff = abs(test - ref)
    denom = max(abs(ref), 1.0e-30)
    return {"abs": float(diff), "rel": float(diff / denom)}


def _complex_rel_metrics(ref: np.ndarray, test: np.ndarray) -> dict[str, float]:
    ref_arr = np.asarray(ref, dtype=np.complex128)
    test_arr = np.asarray(test, dtype=np.complex128)
    diff = np.abs(test_arr - ref_arr)
    denom = np.maximum(np.abs(ref_arr), 1.0e-30)
    rel = diff / denom
    return {
        "max_abs": float(np.max(diff)),
        "max_rel": float(np.max(rel)),
        "rms_rel": float(np.sqrt(np.mean(rel**2))),
        "norm_ref": float(np.linalg.norm(ref_arr.ravel())),
        "norm_test": float(np.linalg.norm(test_arr.ravel())),
    }


def _diag_scalar_map(result: RuntimeNonlinearResult) -> dict[str, float]:
    if result.diagnostics is None:
        return {}
    diag = result.diagnostics
    out: dict[str, float] = {}
    for name, arr in (
        ("Wg", diag.Wg_t),
        ("Wphi", diag.Wphi_t),
        ("Wapar", diag.Wapar_t),
        ("heat", diag.heat_flux_t),
        ("pflux", diag.particle_flux_t),
    ):
        arr_np = np.asarray(arr, dtype=float)
        if arr_np.size:
            out[name] = float(arr_np[-1])
    return out


def _device_result_summary(result: RuntimeNonlinearResult) -> dict[str, float]:
    if result.state is None:
        raise RuntimeError("Device parity gate requires return_state=True results.")
    summary = {
        "state_norm": float(np.linalg.norm(np.asarray(result.state).ravel())),
    }
    summary.update(_diag_scalar_map(result))
    return summary


def build_device_parity_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run CPU/GPU short-window parity gates."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--lane", type=str, default=None)
    parser.add_argument(
        "--outdir", type=Path, default=Path("tools_out") / "device_parity_gate"
    )
    return parser


def _run_device_lane(cfg_path: Path, lane_cfg: dict[str, Any]) -> dict[str, Any]:
    env_updates = _lane_env(lane_cfg)
    with _temporary_env(env_updates):
        cfg, raw = load_runtime_from_toml(cfg_path)
        ky = float(lane_cfg.get("ky", _raw_run_value(raw, "ky", default=0.3)))
        kx_target = lane_cfg.get("kx_target", _raw_run_value(raw, "kx", default=0.0))
        if kx_target is not None:
            kx_target = float(kx_target)
        nl = int(lane_cfg.get("Nl", _raw_run_value(raw, "Nl", default=4)))
        nm = int(lane_cfg.get("Nm", _raw_run_value(raw, "Nm", default=8)))
        dt = float(lane_cfg.get("dt", cfg.time.dt))
        steps = int(lane_cfg["steps"])
        sample_stride = int(lane_cfg.get("sample_stride", 1))
        diagnostics_stride = int(lane_cfg.get("diagnostics_stride", 1))
        fixed_dt = bool(lane_cfg.get("fixed_dt", cfg.time.fixed_dt))
        rtol = float(lane_cfg.get("rtol", 2.0e-4))
        atol = float(lane_cfg.get("atol", 1.0e-7))
        required_nonzero = tuple(
            str(v) for v in lane_cfg.get("required_nonzero", ("state_norm", "Wphi"))
        )

        cpu_devices = jax.devices("cpu")
        gpu_devices = jax.devices("gpu")
        if not cpu_devices or not gpu_devices:
            raise RuntimeError(
                "CPU and GPU backends are both required for the device parity gate."
            )

        time_cfg = replace(
            cfg.time,
            dt=dt,
            fixed_dt=fixed_dt,
            diagnostics=True,
            sample_stride=sample_stride,
            diagnostics_stride=diagnostics_stride,
            t_max=dt * float(steps),
        )
        cfg_gate = replace(cfg, time=time_cfg)
        common = dict(
            ky_target=ky,
            kx_target=kx_target,
            Nl=nl,
            Nm=nm,
            dt=dt,
            steps=steps,
            sample_stride=sample_stride,
            diagnostics_stride=diagnostics_stride,
            return_state=True,
        )

        def _run_on(device) -> dict[str, float]:
            with jax.default_device(device):
                result = run_runtime_nonlinear(cfg_gate, **common)
            return _device_result_summary(result)

        cpu_summary = _run_on(cpu_devices[0])
        gpu_summary = _run_on(gpu_devices[0])
        metric_names = sorted(cpu_summary.keys() & gpu_summary.keys())
        metrics = {
            name: _scalar_rel(cpu_summary[name], gpu_summary[name])
            for name in metric_names
        }
        missing_nonzero = [
            name
            for name in required_nonzero
            if abs(cpu_summary.get(name, 0.0)) <= 1.0e-30
        ]
        ok = not missing_nonzero and all(
            values["abs"] <= atol or values["rel"] <= rtol
            for values in metrics.values()
        )
        return {
            "config": str(cfg_path),
            "env": env_updates,
            "ky": ky,
            "kx_target": kx_target,
            "Nl": nl,
            "Nm": nm,
            "dt": dt,
            "steps": steps,
            "fixed_dt": fixed_dt,
            "required_nonzero": list(required_nonzero),
            "missing_nonzero": missing_nonzero,
            "cpu": cpu_summary,
            "gpu": gpu_summary,
            "metrics": metrics,
            "tolerances": {"rtol": rtol, "atol": atol},
            "ok": bool(ok),
        }


def main_device_parity(argv: list[str] | None = None) -> int:
    args = build_device_parity_parser().parse_args(argv)
    manifest_path = args.manifest.expanduser().resolve()
    manifest = load_toml(manifest_path)
    lanes = _lane_section(manifest)
    selected = [args.lane] if args.lane is not None else list(lanes.keys())
    out_root = args.outdir.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {"manifest": str(manifest_path), "lanes": {}}
    failed: list[str] = []
    for lane_key in selected:
        lane_cfg = lanes.get(lane_key)
        if not isinstance(lane_cfg, dict):
            raise SystemExit(f"Lane config must be a table: lane.{lane_key}")
        config_path = _resolve_manifest_path(
            lane_cfg["config"], manifest_dir=manifest_path.parent
        )
        lane_summary = _run_device_lane(config_path, lane_cfg)
        summary["lanes"][lane_key] = lane_summary
        lane_out = out_root / lane_key
        lane_out.mkdir(parents=True, exist_ok=True)
        (lane_out / "summary.json").write_text(
            json.dumps(lane_summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        if not bool(lane_summary["ok"]):
            failed.append(lane_key)

    summary_path = out_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"saved {summary_path}")
    if failed:
        raise SystemExit(f"Device parity gate failed for lanes: {', '.join(failed)}")
    return 0


def build_vmec_roundtrip_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run VMEC roundtrip determinism gates."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--lane", type=str, default=None)
    parser.add_argument(
        "--outdir", type=Path, default=Path("tools_out") / "vmec_roundtrip_gate"
    )
    return parser


def _array_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    diff = np.asarray(a) - np.asarray(b)
    return {
        "max_abs": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "l2_abs": float(np.linalg.norm(diff.ravel())) if diff.size else 0.0,
    }


def _run_vmec_roundtrip_lane(
    cfg_path: Path, lane_cfg: dict[str, Any], *, out_dir: Path
) -> dict[str, Any]:
    env_updates = _lane_env(lane_cfg)
    with _temporary_env(env_updates):
        cfg, _raw = load_runtime_from_toml(cfg_path)
        out1 = out_dir / "geom1.eik.nc"
        out2 = out_dir / "geom2.eik.nc"
        generate_runtime_vmec_eik(cfg, output_path=out1, force=True)
        generate_runtime_vmec_eik(cfg, output_path=out2, force=True)
        g1 = load_imported_geometry_netcdf(out1)
        g2 = load_imported_geometry_netcdf(out2)
        field_metrics = {
            name: _array_metrics(
                np.asarray(getattr(g1, name)), np.asarray(getattr(g2, name))
            )
            for name in ROUNDTRIP_FIELDS
        }
        ok = all(values["max_abs"] == 0.0 for values in field_metrics.values())
        return {
            "config": str(cfg_path),
            "env": env_updates,
            "out1": str(out1),
            "out2": str(out2),
            "fields": field_metrics,
            "ok": bool(ok),
        }


def main_vmec_roundtrip(argv: list[str] | None = None) -> int:
    args = build_vmec_roundtrip_parser().parse_args(argv)
    manifest_path = args.manifest.expanduser().resolve()
    manifest = load_toml(manifest_path)
    lanes = _lane_section(manifest)
    selected = [args.lane] if args.lane is not None else list(lanes.keys())
    out_root = args.outdir.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {"manifest": str(manifest_path), "lanes": {}}
    failed: list[str] = []
    for lane_key in selected:
        lane_cfg = lanes.get(lane_key)
        if not isinstance(lane_cfg, dict):
            raise SystemExit(f"Lane config must be a table: lane.{lane_key}")
        config_path = _resolve_manifest_path(
            lane_cfg["config"], manifest_dir=manifest_path.parent
        )
        lane_out = out_root / lane_key
        lane_out.mkdir(parents=True, exist_ok=True)
        lane_summary = _run_vmec_roundtrip_lane(config_path, lane_cfg, out_dir=lane_out)
        summary["lanes"][lane_key] = lane_summary
        (lane_out / "summary.json").write_text(
            json.dumps(lane_summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        if not bool(lane_summary["ok"]):
            failed.append(lane_key)

    summary_path = out_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"saved {summary_path}")
    if failed:
        raise SystemExit(f"VMEC roundtrip gate failed for lanes: {', '.join(failed)}")
    return 0


def build_restart_parity_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run nonlinear restart/continuation parity gates."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--lane", type=str, default=None)
    parser.add_argument(
        "--outdir", type=Path, default=Path("tools_out") / "restart_parity_gate"
    )
    return parser


def _restart_lane_summary(
    cfg_path: Path,
    lane_cfg: dict[str, Any],
    *,
    out_dir: Path,
) -> dict[str, Any]:
    env_updates = _lane_env(lane_cfg)
    with _temporary_env(env_updates):
        cfg, raw = load_runtime_from_toml(cfg_path)
        ky = float(lane_cfg.get("ky", _raw_run_value(raw, "ky", default=0.3)))
        kx_target = lane_cfg.get("kx_target", _raw_run_value(raw, "kx", default=0.0))
        if kx_target is not None:
            kx_target = float(kx_target)
        nl = int(lane_cfg.get("Nl", _raw_run_value(raw, "Nl", default=4)))
        nm = int(lane_cfg.get("Nm", _raw_run_value(raw, "Nm", default=8)))
        dt = float(lane_cfg.get("dt", cfg.time.dt))
        steps_first = int(lane_cfg["steps_first"])
        steps_second = int(lane_cfg["steps_second"])
        fixed_dt = bool(lane_cfg.get("fixed_dt", cfg.time.fixed_dt))
        sample_stride = int(lane_cfg.get("sample_stride", 1))
        diagnostics_stride = int(lane_cfg.get("diagnostics_stride", 1))
        state_rtol = float(lane_cfg.get("state_rtol", 1.0e-6))
        state_atol = float(lane_cfg.get("state_atol", 1.0e-9))
        diag_rtol = float(lane_cfg.get("diag_rtol", 1.0e-6))
        diag_atol = float(lane_cfg.get("diag_atol", 1.0e-9))

        time_cfg = replace(
            cfg.time,
            dt=dt,
            fixed_dt=fixed_dt,
            diagnostics=True,
            sample_stride=sample_stride,
            diagnostics_stride=diagnostics_stride,
            t_max=dt * float(steps_first + steps_second),
        )
        cfg_gate = replace(cfg, time=time_cfg)
        common = dict(
            ky_target=ky,
            kx_target=kx_target,
            Nl=nl,
            Nm=nm,
            dt=dt,
            sample_stride=sample_stride,
            diagnostics_stride=diagnostics_stride,
            return_state=True,
        )
        full = run_runtime_nonlinear(
            cfg_gate, steps=steps_first + steps_second, **common
        )
        part1 = run_runtime_nonlinear(cfg_gate, steps=steps_first, **common)
        if full.state is None or part1.state is None:
            raise RuntimeError(
                "Restart parity gate requires return_state=True results."
            )

        restart_path = out_dir / "restart.bin"
        write_netcdf_restart_state(
            restart_path, np.asarray(part1.state, dtype=np.complex64)
        )
        cfg_restart = replace(
            cfg_gate,
            init=replace(
                cfg_gate.init,
                init_file=str(restart_path),
                init_file_scale=1.0,
                init_file_mode="replace",
            ),
        )
        cont = run_runtime_nonlinear(cfg_restart, steps=steps_second, **common)
        if cont.state is None:
            raise RuntimeError("Restart continuation did not return a final state.")

        state_metrics = _complex_rel_metrics(
            np.asarray(full.state), np.asarray(cont.state)
        )
        diag_full = _diag_scalar_map(full)
        diag_cont = _diag_scalar_map(cont)
        diag_metrics = {
            name: _scalar_rel(diag_full[name], diag_cont[name])
            for name in diag_full.keys() & diag_cont.keys()
        }
        state_ok = bool(
            state_metrics["max_abs"] <= state_atol
            or state_metrics["max_rel"] <= state_rtol
        )
        diag_ok = all(
            values["abs"] <= diag_atol or values["rel"] <= diag_rtol
            for values in diag_metrics.values()
        )
        ok = bool(state_ok and diag_ok)
        summary = {
            "config": str(cfg_path),
            "env": env_updates,
            "ky": ky,
            "kx_target": kx_target,
            "Nl": nl,
            "Nm": nm,
            "dt": dt,
            "fixed_dt": fixed_dt,
            "steps_first": steps_first,
            "steps_second": steps_second,
            "state_metrics": state_metrics,
            "diag_metrics": diag_metrics,
            "state_tolerances": {"rtol": state_rtol, "atol": state_atol},
            "diag_tolerances": {"rtol": diag_rtol, "atol": diag_atol},
            "ok": ok,
        }
        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        return summary


def main_restart_parity(argv: list[str] | None = None) -> int:
    args = build_restart_parity_parser().parse_args(argv)
    manifest_path = args.manifest.expanduser().resolve()
    manifest = load_toml(manifest_path)
    lanes = _lane_section(manifest)
    selected = [args.lane] if args.lane is not None else list(lanes.keys())
    out_root = args.outdir.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {"manifest": str(manifest_path), "lanes": {}}
    failed: list[str] = []
    for lane_key in selected:
        lane_cfg = lanes.get(lane_key)
        if not isinstance(lane_cfg, dict):
            raise SystemExit(f"Lane config must be a table: lane.{lane_key}")
        config_path = _resolve_manifest_path(
            lane_cfg["config"], manifest_dir=manifest_path.parent
        )
        lane_out = out_root / lane_key
        lane_out.mkdir(parents=True, exist_ok=True)
        lane_summary = _restart_lane_summary(config_path, lane_cfg, out_dir=lane_out)
        summary["lanes"][lane_key] = lane_summary
        if not bool(lane_summary["ok"]):
            failed.append(lane_key)

    summary_path = out_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"saved {summary_path}")
    if failed:
        raise SystemExit(f"Restart parity gate failed for lanes: {', '.join(failed)}")
    return 0


SUBCOMMANDS = {
    "benchmark-refresh": main_benchmark_refresh,
    "device-parity": main_device_parity,
    "imported-linear-targeted": main_imported_linear_targeted,
    "kbm-lowky-extractor": main_kbm_lowky_extractor,
    "restart-parity": main_restart_parity,
    "vmec-roundtrip": main_vmec_roundtrip,
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
