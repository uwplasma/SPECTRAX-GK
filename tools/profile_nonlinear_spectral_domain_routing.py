#!/usr/bin/env python3
"""Profile the logical nonlinear spectral-domain route after identity gating."""

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
import jax.numpy as jnp
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_PREFIX = REPO_ROOT / "docs" / "_static" / "nonlinear_spectral_domain_routing_profile"


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


def _parse_chunks(raw: str | tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(raw, tuple):
        return tuple(int(item) for item in raw)
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
    if not times:
        raise ValueError("at least one timing repeat is required")
    return {
        "min": float(min(times)),
        "median": float(statistics.median(times)),
        "mean": float(statistics.fmean(times)),
        "max": float(max(times)),
        "std": float(statistics.pstdev(times)) if len(times) > 1 else 0.0,
    }


def _max_abs_rel_error(candidate: Any, reference: Any, *, floor: float = 1.0e-30) -> tuple[float, float]:
    candidate_arr = np.asarray(candidate)
    reference_arr = np.asarray(reference)
    err = candidate_arr - reference_arr
    max_abs = float(np.max(np.abs(err)))
    scale = max(float(np.max(np.abs(reference_arr))), float(floor))
    return max_abs, float(max_abs / scale)


def build_profile(
    *,
    shape: tuple[int, int, int, int, int],
    y_chunks: tuple[int, ...],
    x_chunks: tuple[int, ...],
    steps: int,
    dt: float,
    warmups: int,
    repeats: int,
    min_speedup: float,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    """Return serial-vs-logical routed timing plus identity metrics."""

    from spectraxgk import nonlinear_parallel as npmod
    from spectraxgk.nonlinear_parallel import (
        deterministic_nonlinear_spectral_state,
        integrate_logical_decomposed_nonlinear_spectral,
    )

    state = deterministic_nonlinear_spectral_state(shape)
    routed_state, identity_report = integrate_logical_decomposed_nonlinear_spectral(
        state,
        y_chunks=y_chunks,
        x_chunks=x_chunks,
        steps=int(steps),
        dt=float(dt),
        atol=float(atol),
        rtol=float(rtol),
    )
    serial_reference = state
    dt_array = jnp.asarray(float(dt), dtype=jnp.real(state).dtype)
    for _ in range(int(steps)):
        _field, _bracket, rhs = npmod._serial_nonlinear_spectral_rhs(serial_reference)
        serial_reference = serial_reference + dt_array * rhs
    _block_until_ready((routed_state, serial_reference))

    def serial_route(item: jax.Array) -> jax.Array:
        out = item
        local_dt = jnp.asarray(float(dt), dtype=jnp.real(item).dtype)
        for _ in range(int(steps)):
            _field, _bracket, rhs = npmod._serial_nonlinear_spectral_rhs(out)
            out = out + local_dt * rhs
        return out

    def logical_route(item: jax.Array) -> jax.Array:
        out = item
        local_dt = jnp.asarray(float(dt), dtype=jnp.real(item).dtype)
        for _ in range(int(steps)):
            _reconstructed, _field, _bracket, rhs = npmod._logical_sharded_nonlinear_spectral_rhs(
                out,
                y_chunks=y_chunks,
                x_chunks=x_chunks,
            )
            out = out + local_dt * rhs
        return out

    serial_jit = jax.jit(serial_route)
    logical_jit = jax.jit(logical_route)
    serial_out, serial_times = _time_repeated(
        lambda: serial_jit(state),
        warmups=int(warmups),
        repeats=int(repeats),
    )
    logical_out, logical_times = _time_repeated(
        lambda: logical_jit(state),
        warmups=int(warmups),
        repeats=int(repeats),
    )
    routed_abs, routed_rel = _max_abs_rel_error(logical_out, serial_out, floor=float(atol))
    serial_stats = _stats(serial_times)
    logical_stats = _stats(logical_times)
    speedup = (
        serial_stats["median"] / logical_stats["median"]
        if logical_stats["median"] > 0.0
        else math.nan
    )
    speedup_gate_passed = bool(
        identity_report.identity_passed
        and routed_abs <= float(atol)
        and routed_rel <= float(rtol)
        and math.isfinite(speedup)
        and speedup >= float(min_speedup)
    )
    return _json_clean(
        {
            "kind": "nonlinear_spectral_domain_routing_profile",
            "claim_scope": (
                "diagnostic serial-vs-logical nonlinear spectral-domain timing; "
                "identity-gated local route only, not production distributed FFT or GPU speedup"
            ),
            "backend": str(jax.default_backend()),
            "device_count": int(jax.device_count()),
            "shape": tuple(int(item) for item in shape),
            "y_chunks": y_chunks,
            "x_chunks": x_chunks,
            "steps": int(steps),
            "dt": float(dt),
            "warmups": int(warmups),
            "repeats": int(repeats),
            "identity_passed": bool(identity_report.identity_passed),
            "decomposed_path_enabled": bool(identity_report.decomposed_path_enabled),
            "identity_report": identity_report.to_dict(),
            "timing_identity_max_abs_error": routed_abs,
            "timing_identity_max_rel_error": routed_rel,
            "atol": float(atol),
            "rtol": float(rtol),
            "min_speedup": float(min_speedup),
            "serial_stats_s": serial_stats,
            "logical_domain_stats_s": logical_stats,
            "serial_times_s": serial_times,
            "logical_domain_times_s": logical_times,
            "strong_speedup_vs_serial": speedup,
            "speedup_gate_passed": speedup_gate_passed,
            "production_speedup_claim_allowed": False,
            "status": (
                "diagnostic_speedup_candidate"
                if speedup_gate_passed
                else "identity_timed_no_production_speedup"
            ),
        }
    )


def write_artifacts(summary: dict[str, Any], out_prefix: Path) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from spectraxgk.plotting import set_plot_style

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_prefix.with_suffix(".json")
    csv_path = out_prefix.with_suffix(".csv")
    png_path = out_prefix.with_suffix(".png")
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    rows = [
        {
            "route": "serial",
            "median_s": summary["serial_stats_s"]["median"],
            "mean_s": summary["serial_stats_s"]["mean"],
            "min_s": summary["serial_stats_s"]["min"],
            "max_s": summary["serial_stats_s"]["max"],
        },
        {
            "route": "logical_domain",
            "median_s": summary["logical_domain_stats_s"]["median"],
            "mean_s": summary["logical_domain_stats_s"]["mean"],
            "min_s": summary["logical_domain_stats_s"]["min"],
            "max_s": summary["logical_domain_stats_s"]["max"],
        },
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.4), constrained_layout=True)
    labels = [row["route"].replace("_", "\n") for row in rows]
    medians = np.asarray([float(row["median_s"]) for row in rows])
    axes[0].bar(labels, medians, color=["#495057", "#2a9d8f"])
    axes[0].set_ylabel("median time [s]")
    axes[0].set_title("JIT warm route timing")
    axes[0].grid(True, axis="y", alpha=0.25)

    errors = np.asarray(
        [
            max(float(summary["timing_identity_max_abs_error"]), 1.0e-16),
            max(float(summary["timing_identity_max_rel_error"]), 1.0e-16),
        ]
    )
    axes[1].bar(["abs", "rel"], errors, color="#1b6ca8")
    axes[1].axhline(float(summary["atol"]), color="0.25", ls=":", lw=1.1, label="atol")
    axes[1].axhline(float(summary["rtol"]), color="0.45", ls="--", lw=1.1, label="rtol")
    axes[1].set_yscale("log")
    axes[1].set_title("Serial vs routed identity")
    axes[1].set_ylabel("max error")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(True, axis="y", alpha=0.25)

    speedup = float(summary["strong_speedup_vs_serial"])
    status = str(summary["status"]).replace("_", " ")
    fig.suptitle(f"Logical nonlinear domain route: {speedup:.2f}x ({status})", fontsize=11)
    fig.savefig(png_path, dpi=220)
    plt.close(fig)
    return {"json": str(json_path), "csv": str(csv_path), "png": str(png_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT_PREFIX)
    parser.add_argument("--nl", type=int, default=2)
    parser.add_argument("--nm", type=int, default=4)
    parser.add_argument("--ny", type=int, default=32)
    parser.add_argument("--nx", type=int, default=32)
    parser.add_argument("--nz", type=int, default=4)
    parser.add_argument("--y-chunks", type=_parse_chunks, default=(16, 16))
    parser.add_argument("--x-chunks", type=_parse_chunks, default=(16, 16))
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--min-speedup", type=float, default=1.5)
    parser.add_argument("--atol", type=float, default=5.0e-6)
    parser.add_argument("--rtol", type=float, default=5.0e-6)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = build_profile(
        shape=(args.nl, args.nm, args.ny, args.nx, args.nz),
        y_chunks=tuple(args.y_chunks),
        x_chunks=tuple(args.x_chunks),
        steps=int(args.steps),
        dt=float(args.dt),
        warmups=int(args.warmups),
        repeats=int(args.repeats),
        min_speedup=float(args.min_speedup),
        atol=float(args.atol),
        rtol=float(args.rtol),
    )
    paths = write_artifacts(summary, Path(args.out_prefix))
    print(
        json.dumps(
            {
                "status": summary["status"],
                "identity_passed": summary["identity_passed"],
                "speedup": summary["strong_speedup_vs_serial"],
                "paths": paths,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if bool(summary["identity_passed"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
