#!/usr/bin/env python3
"""Profile the opt-in electrostatic linear-slices parallel RHS route."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any, Callable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.generate_linear_rhs_electrostatic_slices_gate import build_problem  # noqa: E402

DEFAULT_PREFIX = REPO_ROOT / "docs" / "_static" / "linear_rhs_parallel_slices_profile"


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


def _configure_logical_cpu_devices(count: int) -> None:
    if int(count) <= 1:
        return
    flag = f"--xla_force_host_platform_device_count={int(count)}"
    current = os.environ.get("XLA_FLAGS", "")
    if "xla_force_host_platform_device_count" not in current:
        os.environ["XLA_FLAGS"] = f"{current} {flag}".strip()


def _block_until_ready(tree: Any) -> None:
    import jax

    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _terms() -> Any:
    from spectraxgk.linear import LinearTerms

    return LinearTerms(
        streaming=1.0,
        mirror=1.0,
        curvature=1.0,
        gradb=1.0,
        diamagnetic=1.0,
        collisions=0.0,
        hypercollisions=0.0,
        hyperdiffusion=0.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
    )


def _time_callable(fn: Callable[[], Any], *, repeats: int) -> tuple[list[float], Any]:
    samples: list[float] = []
    result: Any = None
    for _ in range(int(repeats)):
        start = time.perf_counter()
        result = fn()
        _block_until_ready(result)
        samples.append(time.perf_counter() - start)
    return samples, result


def profile_linear_rhs_parallel_slices(
    *,
    platform: str,
    requested_devices: int,
    nx: int,
    ny: int,
    nz: int,
    nl: int,
    nm: int,
    warmups: int,
    repeats: int,
    atol: float,
    rtol: float,
) -> dict[str, object]:
    """Time serial and Hermite-sharded electrostatic linear-slices RHS calls."""

    import jax
    import jax.numpy as jnp

    from spectraxgk.linear import linear_rhs_cached, linear_rhs_parallel_cached
    from spectraxgk.workflows.runtime.config import RuntimeParallelConfig

    platform_name = str(platform).lower()
    device_list = list(jax.devices(platform_name))[: int(requested_devices)]
    if len(device_list) < int(requested_devices):
        raise RuntimeError(f"requested {requested_devices} {platform_name} devices, but only {len(device_list)} are available")
    state, cache, params, grid = build_problem(nx=nx, ny=ny, nz=nz, nl=nl, nm=nm)
    terms = _terms()
    parallel_cfg = RuntimeParallelConfig(
        strategy="velocity",
        backend="electrostatic_linear_slices",
        axis="hermite",
        num_devices=len(device_list),
    )

    def serial_call():
        return linear_rhs_cached(state, cache, params, terms=terms, use_jit=True, use_custom_vjp=True)

    def sharded_call():
        return linear_rhs_parallel_cached(
            state,
            cache,
            params,
            terms=terms,
            parallel=parallel_cfg,
            use_custom_vjp=False,
        )

    for _ in range(int(warmups)):
        _block_until_ready(serial_call())
        _block_until_ready(sharded_call())

    serial_samples, serial_result = _time_callable(serial_call, repeats=repeats)
    sharded_samples, sharded_result = _time_callable(sharded_call, repeats=repeats)
    serial_rhs, serial_phi = serial_result
    sharded_rhs, sharded_phi = sharded_result

    abs_err = jnp.max(jnp.abs(sharded_rhs - serial_rhs))
    scale = jnp.max(jnp.abs(serial_rhs))
    rel_err = abs_err / jnp.maximum(scale, jnp.asarray(1.0e-30, dtype=scale.dtype))
    phi_abs_err = jnp.max(jnp.abs(sharded_phi - serial_phi))
    phi_norm = jnp.linalg.norm(serial_phi)
    _block_until_ready((abs_err, rel_err, phi_abs_err, phi_norm))

    serial_median = float(median(serial_samples))
    sharded_median = float(median(sharded_samples))
    speedup = serial_median / sharded_median if sharded_median > 0.0 else math.nan
    identity_passed = bool(float(abs_err) <= float(atol) and float(rel_err) <= float(rtol) and float(phi_abs_err) <= float(atol))

    rows = [
        {"route": "serial", "median_s": serial_median, "samples_s": serial_samples},
        {"route": "sharded", "median_s": sharded_median, "samples_s": sharded_samples},
    ]
    return _json_clean(
        {
            "kind": "linear_rhs_parallel_slices_profile",
            "claim_scope": (
                "engineering timing for the current opt-in electrostatic linear-slices route; "
                "not a publication speedup claim"
            ),
            "state_shape": tuple(int(x) for x in state.shape),
            "grid": {"Nx": int(nx), "Ny_requested": int(ny), "Ny_actual": int(grid.ky.size), "Nz": int(grid.z.size)},
            "platform": platform_name,
            "requested_devices": int(requested_devices),
            "actual_devices": len(device_list),
            "warmups": int(warmups),
            "repeats": int(repeats),
            "serial_median_s": serial_median,
            "sharded_median_s": sharded_median,
            "speedup": float(speedup),
            "identity_passed": identity_passed,
            "max_abs_error": float(np.asarray(abs_err)),
            "max_rel_error": float(np.asarray(rel_err)),
            "max_phi_abs_error": float(np.asarray(phi_abs_err)),
            "phi_norm": float(np.asarray(phi_norm)),
            "atol": float(atol),
            "rtol": float(rtol),
            "rows": rows,
            "notes": (
                "Both routes are warmed before timing. The serial route uses the production JIT path; "
                "the sharded route uses the cached fused Hermite shard-map callable."
            ),
        }
    )


def write_artifacts(summary: dict[str, object], out_prefix: Path) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from spectraxgk.plotting import set_plot_style

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_prefix.with_suffix(".json")
    csv_path = out_prefix.with_suffix(".csv")
    png_path = out_prefix.with_suffix(".png")
    pdf_path = out_prefix.with_suffix(".pdf")

    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = list(summary["rows"])
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["route", "median_s", "samples_s"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    labels = [str(row["route"]) for row in rows]
    medians = np.asarray([row["median_s"] for row in rows], dtype=float)

    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), constrained_layout=True)
    axes[0].bar(labels, medians, color=["#2f6f9f", "#d9792b"])
    axes[0].set_ylabel("median wall time [s]")
    axes[0].set_title("Electrostatic linear RHS timing")
    for idx, value in enumerate(medians):
        axes[0].text(idx, value, f"{value:.3e}s", ha="center", va="bottom", fontsize=8)

    errors = np.asarray(
        [
            float(summary["max_abs_error"]),
            float(summary["max_rel_error"]),
            float(summary["max_phi_abs_error"]),
        ],
        dtype=float,
    )
    axes[1].semilogy(["RHS abs", "RHS rel", "phi abs"], np.maximum(errors, 1.0e-16), "s-", lw=2.0)
    axes[1].axhline(float(summary["atol"]), ls=":", lw=1.2, color="0.25", label="abs gate")
    axes[1].axhline(float(summary["rtol"]), ls="--", lw=1.2, color="0.35", label="rel gate")
    status = "passed" if bool(summary["identity_passed"]) else "failed"
    axes[1].set_ylabel("error")
    axes[1].set_title(f"Identity {status}; speedup={float(summary['speedup']):.2f}x")
    axes[1].legend(frameon=False, fontsize=8)
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
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_PREFIX)
    parser.add_argument("--platform", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--logical-devices", type=int, default=2)
    parser.add_argument("--nl", type=int, default=4)
    parser.add_argument("--nm", type=int, default=16)
    parser.add_argument("--ny", type=int, default=8)
    parser.add_argument("--nx", type=int, default=1)
    parser.add_argument("--nz", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--atol", type=float, default=2.0e-5)
    parser.add_argument("--rtol", type=float, default=2.0e-6)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.platform == "cpu":
        _configure_logical_cpu_devices(args.logical_devices)
    summary = profile_linear_rhs_parallel_slices(
        platform=str(args.platform),
        requested_devices=int(args.logical_devices),
        nx=int(args.nx),
        ny=int(args.ny),
        nz=int(args.nz),
        nl=int(args.nl),
        nm=int(args.nm),
        warmups=int(args.warmups),
        repeats=int(args.repeats),
        atol=float(args.atol),
        rtol=float(args.rtol),
    )
    paths = write_artifacts(summary, args.out_prefix)
    print(json.dumps({"identity_passed": summary["identity_passed"], "speedup": summary["speedup"], "paths": paths}, indent=2))


if __name__ == "__main__":
    main()
