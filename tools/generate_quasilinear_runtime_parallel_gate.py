#!/usr/bin/env python3
"""Generate a quasilinear runtime-scan worker identity gate."""

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

from spectraxgk.config import GeometryConfig, GridConfig, InitializationConfig, TimeConfig
from spectraxgk.plotting import set_plot_style
from spectraxgk.runtime import RuntimeLinearScanResult, run_runtime_scan
from spectraxgk.runtime_config import (
    RuntimeConfig,
    RuntimeNormalizationConfig,
    RuntimeQuasilinearConfig,
    RuntimeSpeciesConfig,
    RuntimeTermsConfig,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREFIX = REPO_ROOT / "docs" / "_static" / "quasilinear_runtime_parallel_gate"


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


def _default_runtime_config(*, nx: int, ny: int, nz: int) -> RuntimeConfig:
    return RuntimeConfig(
        grid=GridConfig(Nx=nx, Ny=ny, Nz=nz, ntheta=nz, nperiod=1, y0=10.0, boundary="linked"),
        time=TimeConfig(t_max=0.2, dt=0.02, method="rk2", use_diffrax=False, sample_stride=1),
        geometry=GeometryConfig(q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778),
        init=InitializationConfig(init_field="density", init_amp=1.0e-8, gaussian_init=False),
        normalization=RuntimeNormalizationConfig(contract="cyclone", diagnostic_norm="none"),
        terms=RuntimeTermsConfig(hypercollisions=0.0, end_damping=0.0),
        species=(RuntimeSpeciesConfig(name="ion"),),
        quasilinear=RuntimeQuasilinearConfig(
            enabled=True,
            mode="saturated",
            saturation_rule="mixing_length",
            amplitude_normalization="phi_rms",
            csat=1.0,
        ),
    )


def _timed_runtime_scan(
    cfg: RuntimeConfig,
    ky_values: np.ndarray,
    *,
    workers: int,
    solver: str,
    nlaguerre: int,
    nhermite: int,
) -> tuple[RuntimeLinearScanResult, float]:
    start = time.perf_counter()
    result = run_runtime_scan(
        cfg,
        ky_values=ky_values.tolist(),
        Nl=nlaguerre,
        Nm=nhermite,
        solver=solver,
        workers=workers,
        parallel_executor="thread",
        show_progress=False,
    )
    return result, time.perf_counter() - start


def _ql_column(payloads: tuple[dict[str, Any], ...], key: str) -> np.ndarray:
    return np.asarray([float(row.get(key, np.nan)) for row in payloads], dtype=float)


def build_quasilinear_runtime_parallel_gate(
    *,
    ky_values: np.ndarray,
    workers: int,
    rtol: float,
    atol: float,
    solver: str,
    nx: int,
    ny: int,
    nz: int,
    nlaguerre: int,
    nhermite: int,
) -> dict[str, Any]:
    """Compare serial and worker-parallel quasilinear runtime scans."""

    cfg = _default_runtime_config(nx=nx, ny=ny, nz=nz)
    serial, serial_elapsed = _timed_runtime_scan(
        cfg,
        ky_values,
        workers=1,
        solver=solver,
        nlaguerre=nlaguerre,
        nhermite=nhermite,
    )
    parallel, parallel_elapsed = _timed_runtime_scan(
        cfg,
        ky_values,
        workers=workers,
        solver=solver,
        nlaguerre=nlaguerre,
        nhermite=nhermite,
    )
    if serial.quasilinear is None or parallel.quasilinear is None:
        raise RuntimeError("quasilinear runtime scan did not return spectrum payloads")
    if len(serial.quasilinear) != len(parallel.quasilinear):
        raise RuntimeError("serial and worker scans returned different quasilinear spectrum lengths")
    if not np.allclose(serial.ky, parallel.ky, rtol=0.0, atol=0.0):
        raise RuntimeError("serial and worker scans returned different ky ordering")

    columns = (
        "gamma",
        "omega",
        "kperp_eff2",
        "heat_flux_weight_total",
        "particle_flux_weight_total",
        "amplitude2",
        "saturated_heat_flux_total",
        "saturated_particle_flux_total",
    )
    rows: list[dict[str, Any]] = []
    max_abs_error = 0.0
    max_rel_error = 0.0
    for idx, ky in enumerate(serial.ky):
        row: dict[str, Any] = {"ky": float(ky)}
        for column in columns:
            serial_values = _ql_column(serial.quasilinear, column)
            parallel_values = _ql_column(parallel.quasilinear, column)
            serial_value = float(serial_values[idx])
            parallel_value = float(parallel_values[idx])
            abs_error = abs(parallel_value - serial_value)
            denom = max(abs(serial_value), float(atol))
            rel_error = abs_error / denom
            max_abs_error = max(max_abs_error, abs_error)
            max_rel_error = max(max_rel_error, rel_error)
            row[f"serial_{column}"] = serial_value
            row[f"parallel_{column}"] = parallel_value
            row[f"{column}_abs_error"] = abs_error
            row[f"{column}_rel_error"] = rel_error
        rows.append(row)

    identity_passed = bool(max_abs_error <= float(atol) or max_rel_error <= float(rtol))
    speedup = float(serial_elapsed / parallel_elapsed) if parallel_elapsed > 0.0 else math.inf
    return _json_clean(
        {
            "kind": "quasilinear_runtime_parallel_gate",
            "case": "Cyclone ITG runtime quasilinear scan",
            "source": "run_runtime_scan serial workers=1 vs independent ky workers>1",
            "claim_level": "quasilinear_state_extraction_identity_not_speedup_claim",
            "ky_values": ky_values.tolist(),
            "grid": {"Nx": nx, "Ny": ny, "Nz": nz, "Nl": nlaguerre, "Nm": nhermite},
            "solver": solver,
            "workers": int(workers),
            "serial_elapsed_s": serial_elapsed,
            "parallel_elapsed_s": parallel_elapsed,
            "observed_speedup": speedup,
            "rtol": float(rtol),
            "atol": float(atol),
            "max_abs_error": max_abs_error,
            "max_rel_error": max_rel_error,
            "identity_passed": identity_passed,
            "serial_parallel_metadata": parallel.parallel,
            "rows": rows,
            "notes": (
                "Acceptance is exact ordered quasilinear-spectrum identity. "
                "Timing is reported for tracking only and should not be used as a production speedup claim."
            ),
        }
    )


def write_artifacts(summary: dict[str, Any], out_prefix: Path) -> dict[str, str]:
    """Write JSON/CSV/PNG/PDF artifacts for the quasilinear identity gate."""

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_prefix.with_suffix(".json")
    csv_path = out_prefix.with_suffix(".csv")
    png_path = out_prefix.with_suffix(".png")
    pdf_path = out_prefix.with_suffix(".pdf")

    json_path.write_text(json.dumps(_json_clean(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = list(summary["rows"])
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    ky = np.asarray([row["ky"] for row in rows], dtype=float)
    serial_heat = np.asarray([row["serial_heat_flux_weight_total"] for row in rows], dtype=float)
    parallel_heat = np.asarray([row["parallel_heat_flux_weight_total"] for row in rows], dtype=float)
    serial_sat = np.asarray([row["serial_saturated_heat_flux_total"] for row in rows], dtype=float)
    parallel_sat = np.asarray([row["parallel_saturated_heat_flux_total"] for row in rows], dtype=float)
    heat_err = np.asarray([row["heat_flux_weight_total_abs_error"] for row in rows], dtype=float)
    sat_err = np.asarray([row["saturated_heat_flux_total_abs_error"] for row in rows], dtype=float)

    set_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.8), constrained_layout=True)
    axes[0].plot(ky, serial_heat, "o-", lw=2.0, color="#1b6f8f", label="serial weight")
    axes[0].plot(ky, parallel_heat, "s--", lw=1.8, color="#b55a30", label="worker weight")
    axes[0].set_xlabel(r"$k_y \rho_i$")
    axes[0].set_ylabel("linear heat-flux weight")
    axes[0].set_title("QL state extraction")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(ky, serial_sat, "o-", lw=2.0, color="#386641", label="serial saturated")
    axes[1].plot(ky, parallel_sat, "s--", lw=1.8, color="#92400e", label="worker saturated")
    axes[1].set_xlabel(r"$k_y \rho_i$")
    axes[1].set_ylabel("saturated heat-flux estimate")
    axes[1].set_title("Spectrum ordering")
    axes[1].legend(frameon=False, fontsize=8)

    floor = 1.0e-18
    axes[2].semilogy(ky, np.maximum(heat_err, floor), "o-", lw=2.0, color="#1b6f8f", label="weight abs. error")
    axes[2].semilogy(ky, np.maximum(sat_err, floor), "s-", lw=2.0, color="#b55a30", label="sat. abs. error")
    axes[2].axhline(float(summary["atol"]), color="#44403c", ls=":", lw=1.2, label="atol")
    status = "passed" if bool(summary["identity_passed"]) else "failed"
    axes[2].set_xlabel(r"$k_y \rho_i$")
    axes[2].set_ylabel("absolute error")
    axes[2].set_title(f"Identity gate: {status}")
    axes[2].legend(frameon=False, fontsize=8)

    fig.suptitle("Quasilinear runtime scan parallelization identity", y=1.02, fontsize=13, fontweight="bold")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {"json": str(json_path), "csv": str(csv_path), "png": str(png_path), "pdf": str(pdf_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_PREFIX)
    parser.add_argument("--ky", nargs="+", type=float, default=[0.1, 0.2])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--rtol", type=float, default=1.0e-10)
    parser.add_argument("--atol", type=float, default=1.0e-10)
    parser.add_argument("--solver", choices=("krylov", "time", "gx_time", "auto"), default="krylov")
    parser.add_argument("--nx", type=int, default=1)
    parser.add_argument("--ny", type=int, default=8)
    parser.add_argument("--nz", type=int, default=12)
    parser.add_argument("--nlaguerre", type=int, default=2)
    parser.add_argument("--nhermite", type=int, default=2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = build_quasilinear_runtime_parallel_gate(
        ky_values=np.asarray(args.ky, dtype=float),
        workers=args.workers,
        rtol=args.rtol,
        atol=args.atol,
        solver=args.solver,
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        nlaguerre=args.nlaguerre,
        nhermite=args.nhermite,
    )
    paths = write_artifacts(summary, args.out_prefix)
    for path in paths.values():
        print(f"saved {path}")


if __name__ == "__main__":
    main()
