#!/usr/bin/env python3
"""Profile solver-backed independent-ky strong scaling with identity gates."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREFIX = REPO_ROOT / "docs" / "_static" / "independent_ky_scan_scaling"


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_clean(value.tolist())
    if isinstance(value, np.generic):
        return _json_clean(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _parse_float_list(text: str) -> list[float]:
    values = [float(part.strip()) for part in str(text).split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated list of floats")
    return values


def _parse_int_list(text: str) -> list[int]:
    values = [int(part.strip()) for part in str(text).split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError(
            "expected a comma-separated list of positive integers"
        )
    if any(value < 1 for value in values):
        raise argparse.ArgumentTypeError("all device counts must be positive")
    return values


def _split_ky(ky_values: np.ndarray, n_parts: int) -> list[np.ndarray]:
    ky = np.asarray(ky_values, dtype=float)
    if ky.ndim != 1 or ky.size == 0:
        raise ValueError("ky_values must be a nonempty one-dimensional array")
    parts = min(int(n_parts), int(ky.size))
    return [chunk for chunk in np.array_split(ky, parts) if chunk.size > 0]


def _worker_env(
    base_env: dict[str, str], *, backend: str, worker_index: int
) -> dict[str, str]:
    env = dict(base_env)
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("PYTHONPATH", str(REPO_ROOT / "src"))
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    backend_key = str(backend).strip().lower()
    if backend_key == "cpu":
        env["JAX_PLATFORMS"] = "cpu"
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
    elif backend_key in {"gpu", "cuda"}:
        env["JAX_PLATFORMS"] = "cuda"
        env["CUDA_VISIBLE_DEVICES"] = str(int(worker_index))
    else:
        raise ValueError("backend must be 'cpu' or 'gpu'")
    return env


def _time_stats(samples: list[float]) -> dict[str, float]:
    if not samples:
        raise ValueError("at least one timing sample is required")
    return {
        "min": float(min(samples)),
        "median": float(statistics.median(samples)),
        "mean": float(statistics.fmean(samples)),
        "max": float(max(samples)),
        "std": float(statistics.pstdev(samples)) if len(samples) > 1 else 0.0,
    }


def _run_solver_chunk(args: argparse.Namespace) -> dict[str, Any]:
    from spectraxgk.benchmarks import run_cyclone_scan
    from spectraxgk.config import CycloneBaseCase, GridConfig

    ky = np.asarray(args.worker_ky, dtype=float)
    cfg = CycloneBaseCase(
        grid=GridConfig(
            Nx=int(args.nx),
            Ny=int(args.ny),
            Nz=int(args.nz),
            ntheta=int(args.nz),
            nperiod=1,
            y0=10.0,
        )
    )
    tmin = float(args.fit_start_fraction) * float(args.dt) * int(args.steps)
    tmax = float(args.fit_end_fraction) * float(args.dt) * int(args.steps)
    samples: list[float] = []
    last = None
    for repeat_index in range(int(args.warmups) + int(args.repeats)):
        start = time.perf_counter()
        result = run_cyclone_scan(
            ky,
            cfg=cfg,
            Nl=int(args.nl),
            Nm=int(args.nm),
            dt=float(args.dt),
            steps=int(args.steps),
            solver="time",
            method=str(args.method),
            sample_stride=int(args.sample_stride),
            fit_signal="phi",
            auto_window=False,
            tmin=tmin,
            tmax=tmax,
            min_points=int(args.min_points),
            ky_batch=1,
            fixed_batch_shape=False,
            use_jit=True,
            mode_only=True,
            require_positive=False,
        )
        elapsed = time.perf_counter() - start
        last = result
        if repeat_index >= int(args.warmups):
            samples.append(float(elapsed))
    assert last is not None
    payload = {
        "ky": np.asarray(last.ky, dtype=float).tolist(),
        "gamma": np.asarray(last.gamma, dtype=float).tolist(),
        "omega": np.asarray(last.omega, dtype=float).tolist(),
        "samples_s": samples,
        "stats_s": _time_stats(samples),
    }
    Path(args.worker_out).write_text(
        json.dumps(_json_clean(payload), indent=2) + "\n", encoding="utf-8"
    )
    return payload


def _worker_command(
    args: argparse.Namespace, *, ky: np.ndarray, out_path: Path
) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--worker-out",
        str(out_path),
        "--worker-ky",
        ",".join(f"{float(value):.16g}" for value in ky),
        "--nx",
        str(int(args.nx)),
        "--ny",
        str(int(args.ny)),
        "--nz",
        str(int(args.nz)),
        "--nl",
        str(int(args.nl)),
        "--nm",
        str(int(args.nm)),
        "--dt",
        str(float(args.dt)),
        "--steps",
        str(int(args.steps)),
        "--method",
        str(args.method),
        "--sample-stride",
        str(int(args.sample_stride)),
        "--fit-start-fraction",
        str(float(args.fit_start_fraction)),
        "--fit-end-fraction",
        str(float(args.fit_end_fraction)),
        "--min-points",
        str(int(args.min_points)),
        "--warmups",
        str(int(args.warmups)),
        "--repeats",
        str(int(args.repeats)),
    ]
    return cmd


def _run_device_count(
    args: argparse.Namespace,
    *,
    ky_values: np.ndarray,
    device_count: int,
    tmp: Path,
) -> dict[str, Any]:
    chunks = _split_ky(ky_values, int(device_count))
    processes: list[tuple[int, Path, subprocess.Popen[str]]] = []
    start_wall = time.perf_counter()
    for worker_index, chunk in enumerate(chunks):
        out_path = tmp / f"{args.backend}_{device_count}_worker_{worker_index}.json"
        env = _worker_env(
            os.environ, backend=str(args.backend), worker_index=worker_index
        )
        proc = subprocess.Popen(
            _worker_command(args, ky=chunk, out_path=out_path),
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        processes.append((worker_index, out_path, proc))

    worker_payloads: list[dict[str, Any]] = []
    errors: list[str] = []
    for worker_index, out_path, proc in processes:
        try:
            stdout, stderr = proc.communicate(timeout=float(args.timeout_s))
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            errors.append(
                f"worker {worker_index} timed out after {args.timeout_s}s\n{stderr[-2000:]}"
            )
            continue
        if proc.returncode != 0:
            errors.append(
                f"worker {worker_index} failed with code {proc.returncode}\n{stderr[-4000:]}\n{stdout[-1000:]}"
            )
            continue
        worker_payloads.append(json.loads(out_path.read_text(encoding="utf-8")))
    wall_s = time.perf_counter() - start_wall

    if errors:
        return {
            "requested_devices": int(device_count),
            "actual_workers": len(chunks),
            "wall_s": wall_s,
            "timed_wall_s": math.nan,
            "ky": [],
            "gamma": [],
            "omega": [],
            "worker_stats": worker_payloads,
            "error": "\n".join(errors),
        }

    ky_all = np.concatenate(
        [np.asarray(payload["ky"], dtype=float) for payload in worker_payloads]
    )
    gamma_all = np.concatenate(
        [np.asarray(payload["gamma"], dtype=float) for payload in worker_payloads]
    )
    omega_all = np.concatenate(
        [np.asarray(payload["omega"], dtype=float) for payload in worker_payloads]
    )
    order = np.argsort(ky_all)
    timed_wall_s = max(
        float(payload["stats_s"]["median"]) for payload in worker_payloads
    )
    return _json_clean(
        {
            "requested_devices": int(device_count),
            "actual_workers": len(chunks),
            "wall_s": float(wall_s),
            "timed_wall_s": float(timed_wall_s),
            "ky": ky_all[order].tolist(),
            "gamma": gamma_all[order].tolist(),
            "omega": omega_all[order].tolist(),
            "worker_stats": worker_payloads,
            "error": None,
        }
    )


def _identity_metrics(
    reference: dict[str, Any], row: dict[str, Any]
) -> dict[str, float | bool]:
    if row.get("error") is not None:
        return {
            "max_gamma_abs_error": math.nan,
            "max_gamma_rel_error": math.nan,
            "max_omega_abs_error": math.nan,
            "identity_gate_pass": False,
        }
    ref_ky = np.asarray(reference["ky"], dtype=float)
    ky = np.asarray(row["ky"], dtype=float)
    if ref_ky.shape != ky.shape or not np.allclose(ref_ky, ky, rtol=0.0, atol=1.0e-14):
        return {
            "max_gamma_abs_error": math.nan,
            "max_gamma_rel_error": math.nan,
            "max_omega_abs_error": math.nan,
            "identity_gate_pass": False,
        }
    ref_gamma = np.asarray(reference["gamma"], dtype=float)
    gamma = np.asarray(row["gamma"], dtype=float)
    ref_omega = np.asarray(reference["omega"], dtype=float)
    omega = np.asarray(row["omega"], dtype=float)
    gamma_abs = np.abs(gamma - ref_gamma)
    gamma_rel = gamma_abs / np.maximum(np.abs(ref_gamma), 1.0e-12)
    omega_abs = np.abs(omega - ref_omega)
    return {
        "max_gamma_abs_error": float(np.max(gamma_abs)),
        "max_gamma_rel_error": float(np.max(gamma_rel)),
        "max_omega_abs_error": float(np.max(omega_abs)),
        "identity_gate_pass": bool(
            float(np.max(gamma_rel)) <= float(args_global_gamma_rtol())
            and float(np.max(omega_abs)) <= float(args_global_omega_atol())
        ),
    }


_GAMMA_RTOL = 1.0e-7
_OMEGA_ATOL = 1.0e-7


def args_global_gamma_rtol() -> float:
    return _GAMMA_RTOL


def args_global_omega_atol() -> float:
    return _OMEGA_ATOL


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    global _GAMMA_RTOL, _OMEGA_ATOL
    _GAMMA_RTOL = float(args.gamma_rtol)
    _OMEGA_ATOL = float(args.omega_atol)
    ky_values = np.asarray(args.ky, dtype=float)
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(
        prefix="spectraxgk-independent-ky-scaling-"
    ) as tmp_name:
        tmp = Path(tmp_name)
        for device_count in list(args.devices):
            rows.append(
                _run_device_count(
                    args, ky_values=ky_values, device_count=int(device_count), tmp=tmp
                )
            )

    reference = next(
        (
            row
            for row in rows
            if int(row["requested_devices"]) == 1 and row.get("error") is None
        ),
        None,
    )
    baseline = float(reference["timed_wall_s"]) if reference is not None else math.nan
    for row in rows:
        metrics = (
            _identity_metrics(reference, row)
            if reference is not None
            else {
                "max_gamma_abs_error": math.nan,
                "max_gamma_rel_error": math.nan,
                "max_omega_abs_error": math.nan,
                "identity_gate_pass": False,
            }
        )
        row.update(metrics)
        current = (
            float(row["timed_wall_s"])
            if row.get("timed_wall_s") is not None
            else math.nan
        )
        row["strong_speedup_vs_1_device"] = (
            baseline / current if baseline > 0.0 and current > 0.0 else math.nan
        )
        row["parallel_efficiency"] = (
            row["strong_speedup_vs_1_device"] / float(row["requested_devices"])
            if math.isfinite(float(row["strong_speedup_vs_1_device"]))
            else math.nan
        )
    return _json_clean(
        {
            "kind": "independent_ky_scan_strong_scaling",
            "backend": str(args.backend),
            "devices": [int(value) for value in args.devices],
            "ky": ky_values.tolist(),
            "grid": {
                "Nx": int(args.nx),
                "Ny": int(args.ny),
                "Nz": int(args.nz),
                "Nl": int(args.nl),
                "Nm": int(args.nm),
            },
            "time": {
                "dt": float(args.dt),
                "steps": int(args.steps),
                "method": str(args.method),
                "sample_stride": int(args.sample_stride),
                "fit_start_fraction": float(args.fit_start_fraction),
                "fit_end_fraction": float(args.fit_end_fraction),
            },
            "warmups": int(args.warmups),
            "repeats": int(args.repeats),
            "gamma_rtol": float(args.gamma_rtol),
            "omega_atol": float(args.omega_atol),
            "identity_passed": all(bool(row.get("identity_gate_pass")) for row in rows),
            "claim_scope": (
                "solver-backed independent ky scan strong-scaling artifact. This validates the "
                "preferred production parallelization path for independent scans/UQ ensembles; "
                "it is not a nonlinear domain-decomposition speedup claim."
            ),
            "rows": rows,
        }
    )


def write_artifacts(summary: dict[str, Any], out_prefix: Path) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from spectraxgk.artifacts.plotting import set_plot_style

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_prefix.with_suffix(".json")
    csv_path = out_prefix.with_suffix(".csv")
    png_path = out_prefix.with_suffix(".png")
    pdf_path = out_prefix.with_suffix(".pdf")
    json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows = list(summary["rows"])
    fieldnames = [
        "requested_devices",
        "actual_workers",
        "timed_wall_s",
        "wall_s",
        "strong_speedup_vs_1_device",
        "parallel_efficiency",
        "max_gamma_abs_error",
        "max_gamma_rel_error",
        "max_omega_abs_error",
        "identity_gate_pass",
        "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    set_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9), constrained_layout=True)
    x = np.asarray([int(row["requested_devices"]) for row in rows], dtype=float)
    speedup = np.asarray(
        [float(row["strong_speedup_vs_1_device"]) for row in rows], dtype=float
    )
    elapsed = np.asarray([float(row["timed_wall_s"]) for row in rows], dtype=float)
    gamma_rel = np.asarray(
        [float(row["max_gamma_rel_error"]) for row in rows], dtype=float
    )
    omega_abs = np.asarray(
        [float(row["max_omega_abs_error"]) for row in rows], dtype=float
    )

    axes[0].plot(x, speedup, "o-", lw=2.2, color="#276b8e", label="measured")
    axes[0].plot(x, x, ":", lw=1.3, color="0.35", label="ideal")
    axes[0].set_xlabel("workers/devices")
    axes[0].set_ylabel("speedup vs one worker")
    axes[0].set_title(f"{str(summary['backend']).upper()} independent ky scaling")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].semilogy(x, np.maximum(elapsed, 1.0e-16), "s-", lw=2.0, color="#b45f06")
    axes[1].set_xlabel("workers/devices")
    axes[1].set_ylabel("median worker wall time [s]")
    axes[1].set_title("Timed scan throughput")

    axes[2].semilogy(
        x, np.maximum(gamma_rel, 1.0e-16), "o-", lw=2.0, label=r"$\gamma$ rel."
    )
    axes[2].semilogy(
        x + 0.03, np.maximum(omega_abs, 1.0e-16), "s-", lw=2.0, label=r"$\omega$ abs."
    )
    axes[2].axhline(float(summary["gamma_rtol"]), color="#276b8e", ls=":", lw=1.2)
    axes[2].axhline(float(summary["omega_atol"]), color="#b45f06", ls=":", lw=1.2)
    axes[2].set_xlabel("workers/devices")
    axes[2].set_ylabel("identity error")
    axes[2].set_title("Serial identity gate")
    axes[2].legend(frameon=False, fontsize=8)

    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "png": str(png_path),
        "pdf": str(pdf_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-out", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-ky", type=_parse_float_list, default=None, help=argparse.SUPPRESS
    )
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_PREFIX)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--devices", type=_parse_int_list, default=[1, 2, 4])
    parser.add_argument(
        "--ky",
        type=_parse_float_list,
        default=[
            0.08,
            0.12,
            0.16,
            0.20,
            0.24,
            0.28,
            0.32,
            0.36,
            0.40,
            0.44,
            0.48,
            0.52,
        ],
    )
    parser.add_argument("--nx", type=int, default=1)
    parser.add_argument("--ny", type=int, default=128)
    parser.add_argument("--nz", type=int, default=96)
    parser.add_argument("--nl", type=int, default=4)
    parser.add_argument("--nm", type=int, default=8)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--method", default="rk2")
    parser.add_argument("--sample-stride", type=int, default=5)
    parser.add_argument("--fit-start-fraction", type=float, default=0.35)
    parser.add_argument("--fit-end-fraction", type=float, default=0.9)
    parser.add_argument("--min-points", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--gamma-rtol", type=float, default=1.0e-7)
    parser.add_argument("--omega-atol", type=float, default=1.0e-7)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if bool(args.worker):
        if args.worker_out is None or args.worker_ky is None:
            raise ValueError("--worker requires --worker-out and --worker-ky")
        _run_solver_chunk(args)
        return 0
    summary = run_sweep(args)
    paths = write_artifacts(summary, Path(args.out_prefix))
    print(
        json.dumps(
            {"identity_passed": summary["identity_passed"], "paths": paths}, indent=2
        )
    )
    return 0 if bool(summary["identity_passed"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
