#!/usr/bin/env python3
"""Fail-closed validation for nonlinear runtime NetCDF diagnostics.

This tool is intentionally lightweight: it checks that a completed nonlinear
``.out.nc`` file contains the time axis and heat-flux diagnostics required by
the long-window transport and nonlinear-gradient gates before larger campaign
steps consume the artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

try:
    from .check_matched_nonlinear_transport_matrix_progress import (
        _bundle_paths,
        _read_output_tmax,
        _repo_relative,
    )
except ImportError:  # pragma: no cover - direct script execution
    from check_matched_nonlinear_transport_matrix_progress import (
        _bundle_paths,
        _read_output_tmax,
        _repo_relative,
    )


DEFAULT_HEAT_FLUX_VARIABLE = "Diagnostics/HeatFlux_st"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "outputs", nargs="+", type=Path, help="Nonlinear runtime .out.nc files"
    )
    parser.add_argument("--json-out", type=Path, help="Optional JSON report path")
    parser.add_argument("--heat-flux-variable", default=DEFAULT_HEAT_FLUX_VARIABLE)
    parser.add_argument("--min-samples", type=int, default=2)
    parser.add_argument(
        "--tmin",
        type=float,
        default=None,
        help="Optional required analysis-window start",
    )
    parser.add_argument(
        "--tmax", type=float, default=None, help="Optional required analysis-window end"
    )
    parser.add_argument(
        "--tmax-atol",
        type=float,
        default=None,
        help=(
            "Absolute tolerance for the required tmax coverage check. The default "
            "uses a fraction of the saved diagnostic cadence to tolerate fixed-step "
            "float32 accumulation roundoff."
        ),
    )
    parser.add_argument(
        "--min-window-samples",
        type=int,
        default=None,
        help="Minimum samples inside [tmin,tmax]; defaults to --min-samples when a window is requested.",
    )
    parser.add_argument(
        "--min-abs-window-mean",
        type=float,
        default=None,
        help="Optional fail-closed lower bound on |mean heat flux| in the requested window.",
    )
    return parser


def build_target_time_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check whether one nonlinear NetCDF output bundle reached a target time."
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-time", required=True, type=float)
    parser.add_argument("--time-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--quiet", action="store_true")
    return parser


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_target_time_report(
    *,
    output: Path,
    target_time: float,
    time_tolerance: float,
) -> dict[str, object]:
    bundle = _bundle_paths(output)
    present = {key: path.exists() for key, path in bundle.items()}
    bundle_complete = all(present.values())
    output_tmax = _read_output_tmax(bundle["out"])
    target_confirmed = bool(
        bundle_complete
        and output_tmax is not None
        and float(output_tmax) >= float(target_time) - float(time_tolerance)
    )
    return {
        "kind": "nonlinear_output_target_time_check",
        "output": _repo_relative(output),
        "bundle": {key: _repo_relative(path) for key, path in bundle.items()},
        "present": present,
        "bundle_complete": bundle_complete,
        "output_tmax": output_tmax,
        "target_time": float(target_time),
        "time_tolerance": float(time_tolerance),
        "target_time_confirmed": target_confirmed,
    }


def _netcdf_variable(root: Any, variable_path: str) -> np.ndarray:
    group = root
    parts = variable_path.split("/")
    for part in parts[:-1]:
        if part not in group.groups:
            raise KeyError(f"missing NetCDF group {part!r} in {variable_path!r}")
        group = group.groups[part]
    name = parts[-1]
    if name not in group.variables:
        raise KeyError(f"missing NetCDF variable {name!r} in {variable_path!r}")
    return np.asarray(group.variables[name][:], dtype=float)


def _window_mask(
    time: np.ndarray, *, tmin: float | None, tmax: float | None
) -> np.ndarray:
    mask = np.isfinite(time)
    if tmin is not None:
        mask &= time >= float(tmin)
    if tmax is not None:
        mask &= time <= float(tmax)
    return mask


def _resolved_tmax_atol(time: np.ndarray, requested: float | None) -> float:
    if requested is not None:
        return max(float(requested), 0.0)
    finite = np.asarray(time[np.isfinite(time)], dtype=float)
    if finite.size < 2:
        return 1.0e-6
    positive_diffs = np.diff(finite)
    positive_diffs = positive_diffs[positive_diffs > 0.0]
    if positive_diffs.size == 0:
        return 1.0e-6
    return min(1.0, max(1.0e-6, 0.25 * float(np.median(positive_diffs))))


def validate_output(
    path: Path,
    *,
    heat_flux_variable: str = DEFAULT_HEAT_FLUX_VARIABLE,
    min_samples: int = 2,
    tmin: float | None = None,
    tmax: float | None = None,
    tmax_atol: float | None = None,
    min_window_samples: int | None = None,
    min_abs_window_mean: float | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "path": path.as_posix(),
        "passed": False,
        "failures": [],
        "warnings": [],
    }
    failures = row["failures"]
    warnings = row["warnings"]
    if not path.exists():
        failures.append("missing_output_file")
        return row

    try:
        import netCDF4
    except Exception as exc:  # pragma: no cover - dependency is in pyproject
        failures.append(f"netcdf4_unavailable:{type(exc).__name__}")
        return row

    try:
        with netCDF4.Dataset(path) as root:
            time = _netcdf_variable(root, "Grids/time").reshape(-1)
            heat = _netcdf_variable(root, heat_flux_variable)
    except Exception as exc:
        failures.append(f"read_error:{type(exc).__name__}:{exc}")
        return row

    row["samples"] = int(time.size)
    row["heat_flux_shape"] = list(heat.shape)
    row["time_min"] = float(np.nanmin(time)) if time.size else None
    row["time_max"] = float(np.nanmax(time)) if time.size else None
    row["tmax_atol"] = _resolved_tmax_atol(time, tmax_atol)

    if time.size < int(min_samples):
        failures.append("too_few_time_samples")
    if time.size and not bool(np.all(np.isfinite(time))):
        failures.append("nonfinite_time")
    if time.size > 1 and bool(np.any(np.diff(time) <= 0.0)):
        failures.append("nonmonotone_time")
    if heat.ndim == 0:
        failures.append("scalar_heat_flux_variable")
        heat_total = np.asarray([], dtype=float)
    elif heat.shape[0] != time.size:
        failures.append("heat_flux_time_dimension_mismatch")
        heat_total = np.asarray([], dtype=float)
    else:
        heat_total = heat.reshape((time.size, -1)).sum(axis=1)
        if not bool(np.all(np.isfinite(heat_total))):
            failures.append("nonfinite_heat_flux")
        row["heat_flux_min"] = float(np.nanmin(heat_total)) if heat_total.size else None
        row["heat_flux_max"] = float(np.nanmax(heat_total)) if heat_total.size else None
        row["heat_flux_last"] = float(heat_total[-1]) if heat_total.size else None

    if tmax is not None and (
        not time.size or float(np.nanmax(time)) + float(row["tmax_atol"]) < float(tmax)
    ):
        failures.append("does_not_reach_required_tmax")

    if tmin is not None or tmax is not None:
        required_window_samples = (
            int(min_window_samples)
            if min_window_samples is not None
            else int(min_samples)
        )
        mask = _window_mask(time, tmin=tmin, tmax=tmax)
        window_heat = (
            heat_total[mask] if heat_total.size else np.asarray([], dtype=float)
        )
        row["window"] = {
            "tmin": None if tmin is None else float(tmin),
            "tmax": None if tmax is None else float(tmax),
            "samples": int(window_heat.size),
            "mean_heat_flux": float(np.mean(window_heat)) if window_heat.size else None,
            "sem_heat_flux": (
                float(np.std(window_heat, ddof=1) / np.sqrt(window_heat.size))
                if window_heat.size > 1
                else None
            ),
        }
        if window_heat.size < required_window_samples:
            failures.append("too_few_window_samples")
        if window_heat.size and min_abs_window_mean is not None:
            mean_abs = abs(float(np.mean(window_heat)))
            if mean_abs < float(min_abs_window_mean):
                failures.append("window_mean_heat_flux_below_threshold")
        if window_heat.size and abs(float(np.mean(window_heat))) < 1.0e-12:
            warnings.append("near_zero_window_mean_heat_flux")

    row["passed"] = len(failures) == 0
    return row


def check_outputs(
    outputs: list[Path],
    *,
    heat_flux_variable: str,
    min_samples: int,
    tmin: float | None,
    tmax: float | None,
    tmax_atol: float | None,
    min_window_samples: int | None,
    min_abs_window_mean: float | None,
) -> dict[str, Any]:
    rows = [
        validate_output(
            output,
            heat_flux_variable=heat_flux_variable,
            min_samples=min_samples,
            tmin=tmin,
            tmax=tmax,
            tmax_atol=tmax_atol,
            min_window_samples=min_window_samples,
            min_abs_window_mean=min_abs_window_mean,
        )
        for output in outputs
    ]
    passed = all(bool(row["passed"]) for row in rows)
    return {
        "kind": "nonlinear_runtime_output_diagnostics_gate",
        "passed": passed,
        "heat_flux_variable": heat_flux_variable,
        "config": {
            "min_samples": int(min_samples),
            "tmin": None if tmin is None else float(tmin),
            "tmax": None if tmax is None else float(tmax),
            "tmax_atol": tmax_atol,
            "min_window_samples": min_window_samples,
            "min_abs_window_mean": min_abs_window_mean,
        },
        "summary": {
            "outputs": len(rows),
            "passed": sum(1 for row in rows if bool(row["passed"])),
            "failed": sum(1 for row in rows if not bool(row["passed"])),
        },
        "rows": rows,
    }


def main_target_time(argv: list[str] | None = None) -> int:
    args = build_target_time_parser().parse_args(argv)
    report = build_target_time_report(
        output=args.output,
        target_time=float(args.target_time),
        time_tolerance=float(args.time_tolerance),
    )
    if args.out_json is not None:
        _write_json(args.out_json, report)
    if not args.quiet:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if bool(report["target_time_confirmed"]) else 1


def main(argv: list[str] | None = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    if tokens and tokens[0] == "target-time":
        return main_target_time(tokens[1:])

    args = build_parser().parse_args(tokens)
    payload = check_outputs(
        args.outputs,
        heat_flux_variable=str(args.heat_flux_variable),
        min_samples=int(args.min_samples),
        tmin=args.tmin,
        tmax=args.tmax,
        tmax_atol=args.tmax_atol,
        min_window_samples=args.min_window_samples,
        min_abs_window_mean=args.min_abs_window_mean,
    )
    if args.json_out is not None:
        _write_json(args.json_out, payload)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0 if bool(payload["passed"]) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
