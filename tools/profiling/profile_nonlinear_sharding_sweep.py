#!/usr/bin/env python3
"""Run large fixed-step nonlinear strong-scaling sweeps in isolated processes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREFIX = REPO_ROOT / "docs" / "_static" / "nonlinear_sharding_strong_scaling"
PROFILE_TOOL = REPO_ROOT / "tools" / "profiling" / "profile_nonlinear_sharding.py"
OFFICE_GPU_XLARGE_PREFIX = (
    REPO_ROOT / "docs" / "_static" / "nonlinear_sharding_strong_scaling_gpu_xlarge"
)


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


def _parse_int_list(text: str) -> list[int]:
    values = [int(part.strip()) for part in str(text).split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError(
            "expected a comma-separated list of positive integers"
        )
    if any(value < 1 for value in values):
        raise argparse.ArgumentTypeError("all values must be positive")
    return values


def _append_xla_flag(existing: str, flag: str) -> str:
    key = flag.split("=")[0]
    if key == "--xla_force_host_platform_device_count":
        cleaned = re.sub(
            r"--xla_force_host_platform_device_count=\S+", "", existing
        ).strip()
        return f"{cleaned} {flag}".strip()
    if key in existing:
        return existing
    return f"{existing} {flag}".strip()


def _tail_text(value: Any, *, limit: int = 4000) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = str(value)
    return text[-int(limit) :]


def _device_env(
    base_env: dict[str, str], *, backend: str, devices: int
) -> dict[str, str]:
    env = dict(base_env)
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    backend_key = str(backend).strip().lower()
    if backend_key == "cpu":
        env["JAX_PLATFORMS"] = "cpu"
        env["XLA_FLAGS"] = _append_xla_flag(
            env.get("XLA_FLAGS", ""),
            f"--xla_force_host_platform_device_count={int(devices)}",
        )
    elif backend_key in {"gpu", "cuda"}:
        env["JAX_PLATFORMS"] = "cuda"
        env["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(index) for index in range(int(devices))
        )
    else:
        raise ValueError("backend must be 'cpu' or 'gpu'")
    return env


def _profile_command(
    *,
    out_json: Path,
    trace_dir: Path | None,
    nx: int,
    ny: int,
    nz: int,
    nl: int,
    nm: int,
    dt: float,
    steps: int,
    method: str,
    sharding: str,
    sharding_options: str,
    laguerre_mode: str,
    warmups: int,
    repeats: int,
) -> list[str]:
    cmd = [
        sys.executable,
        str(PROFILE_TOOL),
        "--out-json",
        str(out_json),
        "--nx",
        str(int(nx)),
        "--ny",
        str(int(ny)),
        "--nz",
        str(int(nz)),
        "--nl",
        str(int(nl)),
        "--nm",
        str(int(nm)),
        "--dt",
        str(float(dt)),
        "--steps",
        str(int(steps)),
        "--method",
        str(method),
        "--sharding",
        str(sharding),
        "--sharding-options",
        str(sharding_options),
        "--laguerre-mode",
        str(laguerre_mode),
        "--warmups",
        str(int(warmups)),
        "--repeats",
        str(int(repeats)),
    ]
    if trace_dir is not None:
        cmd.extend(["--trace-dir", str(trace_dir)])
    return cmd


def _select_parallel_row(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    best = dict(payload.get("best_identity_preserving_candidate") or {})
    best_spec = best.get("spec")
    if best_spec is not None:
        result = dict(payload["sharded_results"][str(best_spec)])
        if (
            bool(result.get("identity_gate_pass", False))
            and result.get("stats_s") is not None
        ):
            return str(best_spec), result
    requested = str(payload.get("state_sharding_requested", "auto"))
    return requested, dict(payload["sharded_results"][requested])


def _row_from_payload(
    payload: dict[str, Any], *, requested_devices: int
) -> dict[str, Any]:
    spec, result = _select_parallel_row(payload)
    stats = result.get("stats_s")
    parallel_median = float(stats["median"]) if stats else math.nan
    serial_median = float(payload["serial_stats_s"]["median"])
    return {
        "requested_devices": int(requested_devices),
        "actual_devices": int(payload["device_count"]),
        "backend": str(payload["default_backend"]),
        "state_shape": tuple(int(x) for x in payload["state_shape"]),
        "best_spec": spec,
        "state_sharding_active": bool(result.get("state_sharding_active", False)),
        "identity_gate_pass": bool(result.get("identity_gate_pass", False)),
        "serial_median_s": serial_median,
        "parallel_median_s": parallel_median,
        "same_process_speedup": serial_median / parallel_median
        if parallel_median > 0.0
        else math.nan,
        "strong_speedup_vs_1_device": math.nan,
        "max_abs_state_error": result.get("max_abs_state_error"),
        "max_rel_state_error": result.get("max_rel_state_error"),
        "profile_json": str(payload.get("_profile_json", "")),
        "source_contract_version": payload.get("source_contract_version"),
        "profile_command": payload.get("profile_command"),
        "profile_command_argv": payload.get("profile_command_argv"),
        "source_artifact": payload.get("source_artifact"),
        "software_versions": payload.get("software_versions"),
        "timing_warmup_repeat": payload.get("timing_warmup_repeat"),
        "profile_backend": payload.get("backend", payload.get("default_backend")),
        "profile_device_count": payload.get("device_count"),
        "profile_sharding_axis": payload.get(
            "sharding_axis", payload.get("state_sharding_requested")
        ),
        "error": result.get("error"),
    }


def _failure_row(
    *,
    requested_devices: int,
    backend: str,
    profile_json: Path,
    error: str,
) -> dict[str, Any]:
    return {
        "requested_devices": int(requested_devices),
        "actual_devices": None,
        "backend": str(backend),
        "state_shape": None,
        "best_spec": None,
        "state_sharding_active": False,
        "identity_gate_pass": False,
        "serial_median_s": math.nan,
        "parallel_median_s": math.nan,
        "same_process_speedup": math.nan,
        "strong_speedup_vs_1_device": math.nan,
        "max_abs_state_error": None,
        "max_rel_state_error": None,
        "profile_json": str(profile_json),
        "error": str(error),
    }


def _speedup_status(rows: list[dict[str, Any]], *, backend: str) -> dict[str, Any]:
    parallel_rows = [row for row in rows if int(row.get("requested_devices") or 0) > 1]
    speedup_threshold = 1.0
    speedup_blockers: list[str] = []
    for row in parallel_rows:
        requested_devices = int(row.get("requested_devices") or 0)
        if not bool(row.get("identity_gate_pass", False)):
            speedup_blockers.append(
                f"{backend}_{requested_devices}devices_identity_failed"
            )
            continue
        speedup = row.get("strong_speedup_vs_1_device")
        speedup_value = float(speedup) if speedup is not None else math.nan
        if not math.isfinite(speedup_value):
            speedup_blockers.append(
                f"{backend}_{requested_devices}devices_speedup_missing"
            )
        elif speedup_value < speedup_threshold:
            speedup_blockers.append(
                f"{backend}_{requested_devices}devices_speedup_{speedup_value:.3g}_below_{speedup_threshold:.3g}"
            )
    speedup_passed = bool(parallel_rows) and not speedup_blockers
    return {
        "speedup_passed": speedup_passed,
        "speedup_threshold_vs_1_device": speedup_threshold,
        "speedup_blockers": speedup_blockers,
        "status": "identity_and_speedup"
        if speedup_passed
        else "diagnostic_identity_only",
    }


def run_sweep(
    *,
    backend: str,
    devices: list[int],
    nx: int,
    ny: int,
    nz: int,
    nl: int,
    nm: int,
    dt: float,
    steps: int,
    method: str,
    sharding: str,
    sharding_options: str,
    laguerre_mode: str,
    warmups: int,
    repeats: int,
    timeout_s: float,
    trace: bool,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    profiles: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(
        prefix="spectraxgk-nonlinear-scaling-"
    ) as tmp_name:
        tmp = Path(tmp_name)
        for requested_devices in devices:
            profile_json = tmp / f"{backend}_{requested_devices}devices.json"
            trace_dir = (
                tmp / f"trace_{backend}_{requested_devices}devices" if trace else None
            )
            env = _device_env(
                os.environ, backend=backend, devices=int(requested_devices)
            )
            cmd = _profile_command(
                out_json=profile_json,
                trace_dir=trace_dir,
                nx=nx,
                ny=ny,
                nz=nz,
                nl=nl,
                nm=nm,
                dt=dt,
                steps=steps,
                method=method,
                sharding=sharding,
                sharding_options=sharding_options,
                laguerre_mode=laguerre_mode,
                warmups=warmups,
                repeats=repeats,
            )
            try:
                completed = subprocess.run(
                    cmd,
                    cwd=REPO_ROOT,
                    env=env,
                    text=True,
                    capture_output=True,
                    timeout=float(timeout_s),
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                output_tail = _tail_text(exc.stderr) or _tail_text(exc.stdout)
                rows.append(
                    _failure_row(
                        requested_devices=int(requested_devices),
                        backend=backend,
                        profile_json=profile_json,
                        error=(
                            f"profile timed out after {float(timeout_s):.3g} s"
                            + (f"\n{output_tail}" if output_tail else "")
                        ),
                    )
                )
                continue
            if profile_json.exists():
                try:
                    payload = json.loads(profile_json.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    payload = None
                if isinstance(payload, dict):
                    payload["_profile_json"] = str(profile_json)
                    payload["profile_returncode"] = int(completed.returncode)
                    profiles[str(requested_devices)] = payload
                    row = _row_from_payload(
                        payload, requested_devices=int(requested_devices)
                    )
                    row["profile_returncode"] = int(completed.returncode)
                    rows.append(row)
                    continue
            if completed.returncode != 0:
                rows.append(
                    _failure_row(
                        requested_devices=int(requested_devices),
                        backend=backend,
                        profile_json=profile_json,
                        error=_tail_text(completed.stderr)
                        or _tail_text(completed.stdout),
                    )
                )
                continue
            rows.append(
                _failure_row(
                    requested_devices=int(requested_devices),
                    backend=backend,
                    profile_json=profile_json,
                    error="profile returned success but did not write a valid JSON artifact",
                )
            )

    valid_rows = [
        row
        for row in rows
        if bool(row.get("identity_gate_pass")) and float(row["parallel_median_s"]) > 0.0
    ]
    baseline_row = next(
        (row for row in valid_rows if int(row["requested_devices"]) == 1), None
    )
    baseline = float(baseline_row["parallel_median_s"]) if baseline_row else math.nan
    for row in rows:
        current = (
            float(row["parallel_median_s"])
            if row["parallel_median_s"] is not None
            else math.nan
        )
        row["strong_speedup_vs_1_device"] = (
            baseline / current if baseline > 0.0 and current > 0.0 else math.nan
        )

    passed = all(bool(row.get("identity_gate_pass")) for row in rows)
    speedup_status = _speedup_status(rows, backend=backend)
    return _json_clean(
        {
            "kind": "nonlinear_sharding_strong_scaling_sweep",
            "backend": str(backend),
            "devices": [int(device) for device in devices],
            "grid": {
                "Nx": int(nx),
                "Ny_requested": int(ny),
                "Nz": int(nz),
                "Nl": int(nl),
                "Nm": int(nm),
            },
            "dt": float(dt),
            "steps": int(steps),
            "method": str(method),
            "sharding": str(sharding),
            "sharding_options": str(sharding_options),
            "laguerre_mode": str(laguerre_mode),
            "warmups": int(warmups),
            "repeats": int(repeats),
            "timeout_s": float(timeout_s),
            "identity_passed": passed,
            **speedup_status,
            "claim_scope": (
                "large fixed-step nonlinear state-sharding strong-scaling artifact with numerical identity gates; "
                "speedup is a separate fail-closed field, and this artifact is profiler evidence unless "
                "speedup_passed is true, not as a broad production speedup claim"
            ),
            "rows": rows,
            "profiles": profiles,
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
        "actual_devices",
        "backend",
        "state_shape",
        "best_spec",
        "state_sharding_active",
        "identity_gate_pass",
        "serial_median_s",
        "parallel_median_s",
        "same_process_speedup",
        "strong_speedup_vs_1_device",
        "max_abs_state_error",
        "max_rel_state_error",
        "profile_json",
        "source_contract_version",
        "profile_command",
        "profile_command_argv",
        "source_artifact",
        "software_versions",
        "timing_warmup_repeat",
        "profile_backend",
        "profile_device_count",
        "profile_sharding_axis",
        "profile_returncode",
        "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.9), constrained_layout=True)
    valid = [
        row
        for row in rows
        if row["parallel_median_s"] is not None
        and math.isfinite(float(row["parallel_median_s"]))
    ]
    x = np.asarray([int(row["requested_devices"]) for row in valid], dtype=float)
    y = np.asarray(
        [float(row["strong_speedup_vs_1_device"]) for row in valid], dtype=float
    )
    times = np.asarray([float(row["parallel_median_s"]) for row in valid], dtype=float)
    errors = np.asarray(
        [
            float(row["max_rel_state_error"])
            if row["max_rel_state_error"] is not None
            else np.nan
            for row in valid
        ],
        dtype=float,
    )

    axes[0].plot(x, y, "o-", lw=2.2, color="#276b8e", label="measured")
    if x.size:
        axes[0].plot(x, x, ":", lw=1.5, color="0.35", label="ideal")
    axes[0].set_xlabel("devices")
    axes[0].set_ylabel("speedup vs 1 device")
    axes[0].set_title(f"{str(summary['backend']).upper()} nonlinear strong scaling")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].semilogy(
        x,
        np.maximum(times, 1.0e-16),
        "s-",
        lw=2.0,
        color="#b45f06",
        label="median time",
    )
    ax_err = axes[1].twinx()
    ax_err.semilogy(
        x + 0.03,
        np.maximum(errors, 1.0e-16),
        "^-",
        lw=1.7,
        color="#4f7f2d",
        label="rel. error",
    )
    axes[1].set_xlabel("devices")
    axes[1].set_ylabel("median time [s]")
    ax_err.set_ylabel("max relative state error")
    axes[1].set_title("Timing and identity")
    handles, labels = axes[1].get_legend_handles_labels()
    handles2, labels2 = ax_err.get_legend_handles_labels()
    axes[1].legend(handles + handles2, labels + labels2, frameon=False, fontsize=8)

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
    parser.add_argument(
        "--office-gpu-xlarge",
        action="store_true",
        help=(
            "Use the canonical office two-GPU nonlinear sharding profile: "
            "gpu backend, devices 1,2, Nx=48, Ny=96, Nz=128, Nl=4, Nm=8, steps=12, trace enabled."
        ),
    )
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_PREFIX)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--devices", type=_parse_int_list, default=[1, 2])
    parser.add_argument("--nx", type=int, default=16)
    parser.add_argument("--ny", type=int, default=16)
    parser.add_argument("--nz", type=int, default=32)
    parser.add_argument("--nl", type=int, default=3)
    parser.add_argument("--nm", type=int, default=4)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--method", default="rk2")
    parser.add_argument("--sharding", default="auto")
    parser.add_argument("--sharding-options", default="auto,kx")
    parser.add_argument("--laguerre-mode", default="grid")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--trace", action="store_true")
    return parser


def apply_profile_preset(args: argparse.Namespace) -> argparse.Namespace:
    """Apply named profiling presets after argparse has filled defaults."""

    if not bool(getattr(args, "office_gpu_xlarge", False)):
        return args
    args.backend = "gpu"
    args.devices = [1, 2]
    args.nx = 48
    args.ny = 96
    args.nz = 128
    args.nl = 4
    args.nm = 8
    args.steps = 12
    args.sharding = "auto"
    args.sharding_options = "auto,kx"
    args.out_prefix = OFFICE_GPU_XLARGE_PREFIX
    args.trace = True
    return args


def main() -> int:
    args = apply_profile_preset(build_parser().parse_args())
    summary = run_sweep(
        backend=str(args.backend),
        devices=list(args.devices),
        nx=int(args.nx),
        ny=int(args.ny),
        nz=int(args.nz),
        nl=int(args.nl),
        nm=int(args.nm),
        dt=float(args.dt),
        steps=int(args.steps),
        method=str(args.method),
        sharding=str(args.sharding),
        sharding_options=str(args.sharding_options),
        laguerre_mode=str(args.laguerre_mode),
        warmups=int(args.warmups),
        repeats=int(args.repeats),
        timeout_s=float(args.timeout_s),
        trace=bool(args.trace),
    )
    paths = write_artifacts(summary, Path(args.out_prefix))
    print(
        json.dumps(
            {
                "identity_passed": summary["identity_passed"],
                "speedup_passed": summary["speedup_passed"],
                "status": summary["status"],
                "paths": paths,
            },
            indent=2,
        )
    )
    return 0 if bool(summary["identity_passed"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
