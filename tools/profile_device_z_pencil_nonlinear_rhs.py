#!/usr/bin/env python3
"""Profile the device-z-sharded fused pencil nonlinear RHS route."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_PREFIX = REPO_ROOT / "docs" / "_static" / "nonlinear_device_z_pencil_rhs_cpu4_profile"


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


def _parse_int_tuple(raw: str) -> tuple[int, ...]:
    return tuple(int(item) for item in str(raw).split(",") if str(item).strip())


def _block_until_ready(value: Any) -> Any:
    for leaf in jax.tree_util.tree_leaves(value):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return value


def _time_repeated(fn: Any, *, warmups: int, repeats: int) -> tuple[Any, list[float]]:
    for _ in range(int(warmups)):
        _block_until_ready(fn())
    last: Any = None
    times: list[float] = []
    for _ in range(int(repeats)):
        start = time.perf_counter()
        last = _block_until_ready(fn())
        times.append(float(time.perf_counter() - start))
    return last, times


def _stats(times: list[float]) -> dict[str, float]:
    return {
        "min": float(min(times)),
        "median": float(statistics.median(times)),
        "mean": float(statistics.fmean(times)),
        "max": float(max(times)),
        "std": float(statistics.pstdev(times)) if len(times) > 1 else 0.0,
    }


def _max_abs_rel(candidate: Any, reference: Any, *, floor: float) -> tuple[float, float]:
    candidate_arr = np.asarray(candidate)
    reference_arr = np.asarray(reference)
    max_abs = float(np.max(np.abs(candidate_arr - reference_arr)))
    scale = max(float(np.max(np.abs(reference_arr))), float(floor))
    return max_abs, float(max_abs / scale)


def build_profile(
    *,
    shape: tuple[int, int, int, int, int],
    device_counts: tuple[int, ...],
    warmups: int,
    repeats: int,
    atol: float,
    rtol: float,
    min_speedup: float,
) -> dict[str, Any]:
    from spectraxgk.nonlinear_parallel import (
        _pencil_nonlinear_spectral_rhs,
        _serial_nonlinear_spectral_rhs,
        deterministic_nonlinear_spectral_state,
        device_z_pencil_nonlinear_spectral_rhs,
    )

    state = deterministic_nonlinear_spectral_state(shape)
    devices = tuple(jax.devices())
    serial_jit = jax.jit(lambda item: _serial_nonlinear_spectral_rhs(item)[2])
    serial_out, serial_times = _time_repeated(
        lambda: serial_jit(state),
        warmups=int(warmups),
        repeats=int(repeats),
    )
    serial_stats = _stats(serial_times)
    rows: list[dict[str, Any]] = [
        {
            "device_count": 1,
            "active": True,
            "identity_passed": True,
            "rhs_max_abs_error": 0.0,
            "rhs_max_rel_error": 0.0,
            "median_s": serial_stats["median"],
            "speedup_vs_serial": 1.0,
            "speedup_gate_passed": False,
            "blocked_reasons": [],
            "stats": serial_stats,
        }
    ]

    for count in device_counts:
        count = int(count)
        if count <= 1:
            continue
        if len(devices) < count:
            rows.append(
                {
                    "device_count": count,
                    "active": False,
                    "identity_passed": False,
                    "rhs_max_abs_error": None,
                    "rhs_max_rel_error": None,
                    "median_s": None,
                    "speedup_vs_serial": None,
                    "speedup_gate_passed": False,
                    "blocked_reasons": ["not_enough_devices"],
                    "stats": {},
                }
            )
            continue
        routed_rhs, report = device_z_pencil_nonlinear_spectral_rhs(
            state,
            devices=devices[:count],
            atol=atol,
            rtol=rtol,
        )
        _block_until_ready(routed_rhs)
        if not report.decomposed_path_enabled:
            rows.append(
                {
                    "device_count": count,
                    "active": bool(report.device_sharding_active),
                    "identity_passed": bool(report.identity_passed),
                    "rhs_max_abs_error": report.rhs_max_abs_error,
                    "rhs_max_rel_error": report.rhs_max_rel_error,
                    "median_s": None,
                    "speedup_vs_serial": None,
                    "speedup_gate_passed": False,
                    "blocked_reasons": list(report.blocked_reasons),
                    "stats": {},
                }
            )
            continue

        mesh = Mesh(np.asarray(devices[:count]), ("z",))
        sharding = NamedSharding(mesh, PartitionSpec(None, None, None, None, "z"))
        with mesh:
            sharded_state = jax.device_put(state, sharding)
            sharded_jit = jax.jit(
                lambda item: _pencil_nonlinear_spectral_rhs(item)[2],
                in_shardings=sharding,
                out_shardings=sharding,
            )
            sharded_out, sharded_times = _time_repeated(
                lambda: sharded_jit(sharded_state),
                warmups=int(warmups),
                repeats=int(repeats),
            )
        rhs_abs, rhs_rel = _max_abs_rel(sharded_out, serial_out, floor=atol)
        sharded_stats = _stats(sharded_times)
        speedup = serial_stats["median"] / sharded_stats["median"]
        rows.append(
            {
                "device_count": count,
                "active": True,
                "identity_passed": bool(rhs_abs <= atol and rhs_rel <= rtol),
                "rhs_max_abs_error": rhs_abs,
                "rhs_max_rel_error": rhs_rel,
                "median_s": sharded_stats["median"],
                "speedup_vs_serial": speedup,
                "speedup_gate_passed": bool(speedup >= min_speedup and rhs_abs <= atol and rhs_rel <= rtol),
                "blocked_reasons": [] if speedup >= min_speedup else ["speedup_below_gate"],
                "stats": sharded_stats,
            }
        )

    active_rows = [row for row in rows if row["active"] and row["device_count"] > 1]
    max_speedup = max((float(row["speedup_vs_serial"]) for row in active_rows), default=1.0)
    all_active_identity = all(bool(row["identity_passed"]) for row in active_rows)
    speedup_gate = any(bool(row["speedup_gate_passed"]) for row in active_rows)
    status = "production_speedup_candidate" if speedup_gate else "identity_timed_no_production_speedup"
    if not active_rows:
        status = "skipped_no_multidevice"

    return _json_clean(
        {
            "kind": "nonlinear_device_z_pencil_rhs_profile",
            "claim_scope": (
                "device z-sharded fused pencil nonlinear RHS timing; FFT axes are local per z shard, "
                "no global spectral reconstruction is used, and no production speedup claim is allowed "
                "unless the speedup gate passes"
            ),
            "backend": str(jax.default_backend()),
            "device_count_available": int(len(devices)),
            "shape": shape,
            "warmups": int(warmups),
            "repeats": int(repeats),
            "atol": float(atol),
            "rtol": float(rtol),
            "min_speedup": float(min_speedup),
            "serial_stats_s": serial_stats,
            "rows": rows,
            "summary": {
                "status": status,
                "all_active_identity_passed": bool(all_active_identity),
                "max_speedup_vs_serial": float(max_speedup),
                "production_speedup_claim_allowed": bool(speedup_gate),
            },
        }
    )


def write_artifacts(summary: dict[str, Any], out_prefix: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from spectraxgk.plotting import set_plot_style

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_json = out_prefix.with_suffix(".json")
    out_csv = out_prefix.with_suffix(".csv")
    out_png = out_prefix.with_suffix(".png")
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    rows = list(summary["rows"])
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "device_count",
                "active",
                "identity_passed",
                "rhs_max_abs_error",
                "rhs_max_rel_error",
                "median_s",
                "speedup_vs_serial",
                "speedup_gate_passed",
                "blocked_reasons",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in writer.fieldnames})

    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), constrained_layout=True)
    counts = [int(row["device_count"]) for row in rows]
    speedups = [float(row["speedup_vs_serial"] or 0.0) for row in rows]
    errors = [float(row["rhs_max_abs_error"] or 0.0) for row in rows]
    axes[0].plot(counts, speedups, "o-", lw=2.0, color="#1b6ca8")
    axes[0].axhline(float(summary["min_speedup"]), color="0.25", ls=":", lw=1.2, label="promotion gate")
    axes[0].set_xlabel("logical CPU devices")
    axes[0].set_ylabel("speedup vs serial")
    axes[0].set_title("z-sharded fused pencil RHS")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(True, alpha=0.25)

    axes[1].bar([str(count) for count in counts], np.maximum(errors, 1.0e-16), color="#b65f23")
    axes[1].axhline(float(summary["atol"]), color="0.25", ls=":", lw=1.2, label="atol")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("logical CPU devices")
    axes[1].set_ylabel("max |RHS error|")
    axes[1].set_title("serial-vs-sharded identity")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(True, alpha=0.25, axis="y")
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT_PREFIX)
    parser.add_argument("--shape", type=_parse_int_tuple, default=(4, 16, 96, 96, 32))
    parser.add_argument("--device-counts", type=_parse_int_tuple, default=(1, 2, 4))
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--atol", type=float, default=5.0e-6)
    parser.add_argument("--rtol", type=float, default=1.0e-4)
    parser.add_argument("--min-speedup", type=float, default=1.5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    shape = tuple(int(item) for item in args.shape)
    if len(shape) != 5:
        raise ValueError("--shape must contain five comma-separated integers")
    summary = build_profile(
        shape=shape,  # type: ignore[arg-type]
        device_counts=tuple(int(item) for item in args.device_counts),
        warmups=int(args.warmups),
        repeats=int(args.repeats),
        atol=float(args.atol),
        rtol=float(args.rtol),
        min_speedup=float(args.min_speedup),
    )
    write_artifacts(summary, args.out_prefix)
    print(json.dumps(summary["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
