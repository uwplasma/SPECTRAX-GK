#!/usr/bin/env python3
"""Generate a shard-map velocity field-reduction identity gate."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREFIX = REPO_ROOT / "docs" / "_static" / "velocity_field_reduce_gate"


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


def _state(shape: tuple[int, ...]) -> Any:
    import jax.numpy as jnp

    values = jnp.arange(int(np.prod(shape)), dtype=jnp.float32).reshape(shape)
    return values + 0.02j * (values + 1.0)


def build_velocity_field_reduce_gate(
    *,
    shape: tuple[int, int, int, int, int],
    requested_devices: int,
    atol: float,
    rtol: float,
) -> dict[str, object]:
    """Run the Hermite-axis field reduction and compare against reference."""

    import jax
    import jax.numpy as jnp

    from spectraxgk.parallel.velocity import (
        build_velocity_sharding_plan,
        velocity_field_reduce_reference,
        velocity_field_reduce_shard_map,
    )

    device_list = list(jax.devices("cpu"))[: int(requested_devices)]
    if len(device_list) < int(requested_devices):
        raise RuntimeError(f"requested {requested_devices} CPU devices, but only {len(device_list)} are available")
    state = _state(shape)
    plan = build_velocity_sharding_plan(shape, num_devices=len(device_list), axes=("hermite",))
    reference = velocity_field_reduce_reference(state, axis="hermite")
    sharded = velocity_field_reduce_shard_map(state, plan, axis="hermite", devices=device_list)
    _block_until_ready(sharded)

    err = jnp.max(jnp.abs(sharded - reference))
    reference_norm = jnp.max(jnp.abs(reference))
    _block_until_ready((err, reference_norm))
    max_abs_error = float(np.asarray(err))
    max_abs_reference = float(np.asarray(reference_norm))
    max_allowed_error = float(atol) + float(rtol) * max(max_abs_reference, 1.0)
    max_rel_error = max_abs_error / max(max_abs_reference, 1.0e-30)
    identity_passed = bool(max_abs_error <= max_allowed_error)

    sharded_trace = np.asarray(sharded[0, :, 0, 0])
    reference_trace = np.asarray(reference[0, :, 0, 0])
    rows = []
    for ky_idx in range(shape[2]):
        rows.append(
            {
                "ky_index": int(ky_idx),
                "reduced_real": float(np.real(sharded_trace[ky_idx])),
                "reference_real": float(np.real(reference_trace[ky_idx])),
                "abs_error": float(abs(sharded_trace[ky_idx] - reference_trace[ky_idx])),
                "rel_error": float(
                    abs(sharded_trace[ky_idx] - reference_trace[ky_idx])
                    / max(abs(reference_trace[ky_idx]), 1.0e-30)
                ),
            }
        )

    return _json_clean(
        {
            "case": "Velocity-space field-reduction shard_map identity gate",
            "source": "spectraxgk.parallel.velocity.velocity_field_reduce_shard_map",
            "claim_scope": "communication-kernel identity gate, not a nonlinear runtime speedup claim",
            "state_shape": shape,
            "reduction_axis": "hermite",
            "requested_devices": int(requested_devices),
            "actual_devices": len(device_list),
            "plan": plan.to_dict(),
            "max_abs_error": max_abs_error,
            "max_abs_reference": max_abs_reference,
            "max_rel_error": max_rel_error,
            "max_allowed_error": max_allowed_error,
            "atol": float(atol),
            "rtol": float(rtol),
            "identity_passed": identity_passed,
            "rows": rows,
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
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    ky = np.asarray([row["ky_index"] for row in rows], dtype=float)
    reduced = np.asarray([row["reduced_real"] for row in rows], dtype=float)
    reference = np.asarray([row["reference_real"] for row in rows], dtype=float)
    error = np.asarray([row["abs_error"] for row in rows], dtype=float)

    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), constrained_layout=True)
    axes[0].plot(ky, reference, "o-", lw=2.0, label="reference reduction")
    axes[0].plot(ky, reduced, "s--", lw=1.8, label="shard_map reduction")
    axes[0].set_xlabel(r"$k_y$ index")
    axes[0].set_ylabel("real reduced field contribution")
    axes[0].set_title("Velocity reduction")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].semilogy(ky, np.maximum(error, 1.0e-16), "s-", lw=2.0, label="absolute error")
    axes[1].axhline(float(summary["max_allowed_error"]), ls=":", lw=1.2, color="0.25", label="gate")
    status = "passed" if bool(summary["identity_passed"]) else "failed"
    axes[1].set_xlabel(r"$k_y$ index")
    axes[1].set_ylabel("absolute error")
    axes[1].set_title(f"Identity gate {status}")
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
    parser.add_argument("--logical-devices", type=int, default=2)
    parser.add_argument("--nl", type=int, default=2)
    parser.add_argument("--nm", type=int, default=8)
    parser.add_argument("--ny", type=int, default=4)
    parser.add_argument("--nx", type=int, default=1)
    parser.add_argument("--nz", type=int, default=5)
    parser.add_argument("--atol", type=float, default=1.0e-5)
    parser.add_argument("--rtol", type=float, default=1.0e-7)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _configure_logical_cpu_devices(args.logical_devices)
    summary = build_velocity_field_reduce_gate(
        shape=(int(args.nl), int(args.nm), int(args.ny), int(args.nx), int(args.nz)),
        requested_devices=int(args.logical_devices),
        atol=float(args.atol),
        rtol=float(args.rtol),
    )
    paths = write_artifacts(summary, args.out_prefix)
    print(json.dumps({"identity_passed": summary["identity_passed"], "paths": paths}, indent=2))


if __name__ == "__main__":
    main()
