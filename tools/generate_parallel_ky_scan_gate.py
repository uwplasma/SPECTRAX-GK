#!/usr/bin/env python3
"""Generate a solver-backed ky-batch parallelization identity gate."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

from spectraxgk.benchmarks import CycloneScanResult, run_cyclone_scan
from spectraxgk.config import CycloneBaseCase, GridConfig
from spectraxgk.plotting import set_plot_style


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREFIX = REPO_ROOT / "docs" / "_static" / "parallel_ky_scan_gate"


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


def _timed_cyclone_scan(
    ky_values: np.ndarray,
    *,
    ky_batch: int,
    cfg: CycloneBaseCase,
    steps: int,
    dt: float,
    nlaguerre: int,
    nhermite: int,
) -> tuple[CycloneScanResult, float]:
    start = time.perf_counter()
    result = run_cyclone_scan(
        ky_values,
        cfg=cfg,
        Nl=nlaguerre,
        Nm=nhermite,
        dt=dt,
        steps=steps,
        solver="time",
        method="rk2",
        sample_stride=2,
        fit_signal="phi",
        auto_window=False,
        tmin=0.2 * dt * steps,
        tmax=0.9 * dt * steps,
        min_points=4,
        ky_batch=ky_batch,
        fixed_batch_shape=True,
        use_jit=True,
    )
    return result, time.perf_counter() - start


def build_parallel_ky_scan_gate(
    *,
    ky_values: np.ndarray,
    serial_batch: int,
    parallel_batch: int,
    gamma_rtol: float,
    omega_atol: float,
    steps: int,
    dt: float,
    nx: int,
    ny: int,
    nz: int,
    nlaguerre: int,
    nhermite: int,
) -> dict[str, object]:
    """Run serial and batched Cyclone scans and build identity-gate metadata."""

    cfg = CycloneBaseCase(grid=GridConfig(Nx=nx, Ny=ny, Nz=nz, ntheta=nz, nperiod=1, y0=10.0))
    serial, serial_elapsed = _timed_cyclone_scan(
        ky_values,
        ky_batch=serial_batch,
        cfg=cfg,
        steps=steps,
        dt=dt,
        nlaguerre=nlaguerre,
        nhermite=nhermite,
    )
    batched, batch_elapsed = _timed_cyclone_scan(
        ky_values,
        ky_batch=parallel_batch,
        cfg=cfg,
        steps=steps,
        dt=dt,
        nlaguerre=nlaguerre,
        nhermite=nhermite,
    )
    if not np.allclose(serial.ky, batched.ky, rtol=0.0, atol=0.0):
        raise ValueError("serial and batched scans returned different ky ordering")

    gamma_denom = np.maximum(np.abs(serial.gamma), 1.0e-12)
    gamma_rel = np.abs(batched.gamma - serial.gamma) / gamma_denom
    omega_abs = np.abs(batched.omega - serial.omega)
    max_gamma_rel = float(np.max(gamma_rel))
    max_omega_abs = float(np.max(omega_abs))
    identity_passed = bool(max_gamma_rel <= gamma_rtol and max_omega_abs <= omega_atol)
    speedup = float(serial_elapsed / batch_elapsed) if batch_elapsed > 0.0 else math.inf

    rows = []
    for idx, ky in enumerate(serial.ky):
        rows.append(
            {
                "ky": float(ky),
                "serial_gamma": float(serial.gamma[idx]),
                "batched_gamma": float(batched.gamma[idx]),
                "gamma_rel_error": float(gamma_rel[idx]),
                "serial_omega": float(serial.omega[idx]),
                "batched_omega": float(batched.omega[idx]),
                "omega_abs_error": float(omega_abs[idx]),
            }
        )

    return _json_clean(
        {
            "case": "Cyclone ITG linear ky-batch scan",
            "source": "SPECTRAX-GK real linear solver; serial ky_batch=1 vs fixed-shape ky_batch>1",
            "physics_anchor": "Cyclone Base Case ITG linear scan",
            "ky_values": ky_values.tolist(),
            "grid": {"Nx": nx, "Ny": ny, "Nz": nz, "Nl": nlaguerre, "Nm": nhermite},
            "time": {"dt": dt, "steps": steps, "sample_stride": 2},
            "serial_batch": serial_batch,
            "parallel_batch": parallel_batch,
            "serial_elapsed_s": serial_elapsed,
            "batched_elapsed_s": batch_elapsed,
            "observed_speedup": speedup,
            "max_gamma_rel_error": max_gamma_rel,
            "max_omega_abs_error": max_omega_abs,
            "gamma_rtol": gamma_rtol,
            "omega_atol": omega_atol,
            "identity_passed": identity_passed,
            "rows": rows,
            "notes": (
                "This is a numerical-identity gate for independent ky batching. "
                "Speedup is reported for engineering tracking but is not the acceptance gate."
            ),
        }
    )


def write_artifacts(summary: dict[str, object], out_prefix: Path) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_prefix.with_suffix(".json")
    csv_path = out_prefix.with_suffix(".csv")
    png_path = out_prefix.with_suffix(".png")
    pdf_path = out_prefix.with_suffix(".pdf")

    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = list(summary["rows"])
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    ky = np.asarray([row["ky"] for row in rows], dtype=float)
    serial_gamma = np.asarray([row["serial_gamma"] for row in rows], dtype=float)
    batched_gamma = np.asarray([row["batched_gamma"] for row in rows], dtype=float)
    serial_omega = np.asarray([row["serial_omega"] for row in rows], dtype=float)
    batched_omega = np.asarray([row["batched_omega"] for row in rows], dtype=float)
    gamma_rel = np.asarray([row["gamma_rel_error"] for row in rows], dtype=float)
    omega_abs = np.asarray([row["omega_abs_error"] for row in rows], dtype=float)

    set_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.8), constrained_layout=True)

    axes[0].plot(ky, serial_gamma, "o-", lw=2.0, color="#1b6ca8", label="serial gamma")
    axes[0].plot(ky, batched_gamma, "s--", lw=1.8, color="#c03a2b", label="batched gamma")
    axes[0].plot(ky, serial_omega, "o-", lw=2.0, color="#5f8f2d", label="serial omega")
    axes[0].plot(ky, batched_omega, "s--", lw=1.8, color="#7d3c98", label="batched omega")
    axes[0].set_xlabel(r"$k_y \rho_i$")
    axes[0].set_ylabel(r"$\gamma, \omega$")
    axes[0].set_title("Cyclone linear scan")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)

    axes[1].semilogy(ky, np.maximum(gamma_rel, 1.0e-16), "o-", lw=2.0, color="#1b6ca8", label=r"$\gamma$ rel.")
    axes[1].semilogy(ky, np.maximum(omega_abs, 1.0e-16), "s-", lw=2.0, color="#c03a2b", label=r"$\omega$ abs.")
    axes[1].axhline(float(summary["gamma_rtol"]), color="#1b6ca8", ls=":", lw=1.2)
    axes[1].axhline(float(summary["omega_atol"]), color="#c03a2b", ls=":", lw=1.2)
    axes[1].set_xlabel(r"$k_y \rho_i$")
    axes[1].set_ylabel("identity error")
    axes[1].set_title("Numerical identity gate")
    axes[1].legend(frameon=False, fontsize=8)

    serial_elapsed = float(summary["serial_elapsed_s"])
    batch_elapsed = float(summary["batched_elapsed_s"])
    axes[2].bar(["serial", "batched"], [serial_elapsed, batch_elapsed], color=["#1b6ca8", "#c03a2b"], alpha=0.82)
    status = "passed" if bool(summary["identity_passed"]) else "failed"
    axes[2].set_ylabel("wall time [s]")
    axes[2].set_title("Batch timing")
    axes[2].text(
        0.04,
        0.95,
        f"speedup = {float(summary['observed_speedup']):.2f}x\nidentity gate: {status}",
        transform=axes[2].transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "0.82", "alpha": 0.92},
    )

    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_PREFIX)
    parser.add_argument("--ky", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.4])
    parser.add_argument("--serial-batch", type=int, default=1)
    parser.add_argument("--parallel-batch", type=int, default=2)
    parser.add_argument("--gamma-rtol", type=float, default=1.0e-8)
    parser.add_argument("--omega-atol", type=float, default=1.0e-8)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--nx", type=int, default=1)
    parser.add_argument("--ny", type=int, default=12)
    parser.add_argument("--nz", type=int, default=16)
    parser.add_argument("--nlaguerre", type=int, default=3)
    parser.add_argument("--nhermite", type=int, default=4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = build_parallel_ky_scan_gate(
        ky_values=np.asarray(args.ky, dtype=float),
        serial_batch=args.serial_batch,
        parallel_batch=args.parallel_batch,
        gamma_rtol=args.gamma_rtol,
        omega_atol=args.omega_atol,
        steps=args.steps,
        dt=args.dt,
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        nlaguerre=args.nlaguerre,
        nhermite=args.nhermite,
    )
    write_artifacts(summary, args.out_prefix)
    print(f"Wrote {args.out_prefix.with_suffix('.png')}")
    if not bool(summary["identity_passed"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
