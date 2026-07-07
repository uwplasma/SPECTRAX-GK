#!/usr/bin/env python3
"""Generate a shard-map Hermite streaming-ladder identity gate."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREFIX = REPO_ROOT / "docs" / "_static" / "hermite_streaming_ladder_gate"


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
    return values + 0.01j * (values + 1.0)


def build_hermite_streaming_ladder_gate(
    *,
    shape: tuple[int, int, int, int, int],
    requested_devices: int,
    vth: float,
    atol: float,
    rtol: float,
) -> dict[str, object]:
    """Run the Hermite streaming ladder and compare against reference."""

    import jax
    import jax.numpy as jnp

    from spectraxgk.parallel.velocity import (
        build_velocity_sharding_plan,
        hermite_streaming_ladder_reference,
        hermite_streaming_ladder_shard_map,
        velocity_field_reduce_reference,
        velocity_field_reduce_shard_map,
    )

    device_list = list(jax.devices("cpu"))[: int(requested_devices)]
    if len(device_list) < int(requested_devices):
        raise RuntimeError(
            f"requested {requested_devices} CPU devices, but only {len(device_list)} are available"
        )
    state = _state(shape)
    plan = build_velocity_sharding_plan(
        shape, num_devices=len(device_list), axes=("hermite",)
    )

    ladder_reference = hermite_streaming_ladder_reference(state, vth=float(vth))
    ladder_sharded = hermite_streaming_ladder_shard_map(
        state, plan, vth=float(vth), devices=device_list
    )
    reduction_reference = velocity_field_reduce_reference(state, axis="hermite")
    reduction_sharded = velocity_field_reduce_shard_map(
        state, plan, axis="hermite", devices=device_list
    )
    _block_until_ready((ladder_sharded, reduction_sharded))

    ladder_abs = jnp.max(jnp.abs(ladder_sharded - ladder_reference))
    ladder_scale = jnp.max(jnp.abs(ladder_reference))
    ladder_rel = ladder_abs / jnp.maximum(
        ladder_scale, jnp.asarray(1.0e-30, dtype=ladder_scale.dtype)
    )
    reduction_abs = jnp.max(jnp.abs(reduction_sharded - reduction_reference))
    _block_until_ready((ladder_abs, ladder_rel, reduction_abs))
    max_ladder_abs_error = float(np.asarray(ladder_abs))
    max_ladder_rel_error = float(np.asarray(ladder_rel))
    max_reduction_abs_error = float(np.asarray(reduction_abs))
    identity_passed = bool(
        max_ladder_abs_error <= float(atol) and max_ladder_rel_error <= float(rtol)
    )

    state_trace = np.asarray(state[0, :, 0, 0, 0])
    reference_trace = np.asarray(ladder_reference[0, :, 0, 0, 0])
    sharded_trace = np.asarray(ladder_sharded[0, :, 0, 0, 0])
    rows = []
    for m_idx in range(shape[1]):
        rows.append(
            {
                "m": int(m_idx),
                "state_real": float(np.real(state_trace[m_idx])),
                "ladder_real": float(np.real(sharded_trace[m_idx])),
                "reference_real": float(np.real(reference_trace[m_idx])),
                "abs_error": float(abs(sharded_trace[m_idx] - reference_trace[m_idx])),
            }
        )

    return _json_clean(
        {
            "case": "Hermite streaming-ladder shard_map identity gate",
            "source": "spectraxgk.parallel.velocity.hermite_streaming_ladder_shard_map",
            "claim_scope": "Hermite streaming communication/coefficient gate, not a nonlinear runtime speedup claim",
            "state_shape": shape,
            "vth": float(vth),
            "requested_devices": int(requested_devices),
            "actual_devices": len(device_list),
            "plan": plan.to_dict(),
            "max_ladder_abs_error": max_ladder_abs_error,
            "max_ladder_rel_error": max_ladder_rel_error,
            "max_reduction_abs_error": max_reduction_abs_error,
            "atol": float(atol),
            "rtol": float(rtol),
            "identity_passed": identity_passed,
            "rows": rows,
            "notes": (
                "This gate combines nearest-Hermite shard_map exchange with the sqrt(m+1)/sqrt(m) "
                "streaming-ladder coefficients. It also records the accompanying Hermite field-reduction "
                "error because production velocity-space sharding needs both communication primitives."
            ),
        }
    )


def write_artifacts(summary: dict[str, object], out_prefix: Path) -> dict[str, str]:
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
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=list(rows[0].keys()), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    m = np.asarray([row["m"] for row in rows], dtype=float)
    state = np.asarray([row["state_real"] for row in rows], dtype=float)
    ladder = np.asarray([row["ladder_real"] for row in rows], dtype=float)
    reference = np.asarray([row["reference_real"] for row in rows], dtype=float)
    error = np.asarray([row["abs_error"] for row in rows], dtype=float)

    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), constrained_layout=True)
    axes[0].plot(m, state, "o-", lw=2.0, label="state")
    axes[0].plot(m, reference, "s-", lw=1.8, label="reference ladder")
    axes[0].plot(m, ladder, "^--", lw=1.8, label="shard_map ladder")
    axes[0].set_xlabel("Hermite index m")
    axes[0].set_ylabel("real value")
    axes[0].set_title("Streaming ladder")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].semilogy(
        m, np.maximum(error, 1.0e-16), "s-", lw=2.0, label="absolute error"
    )
    axes[1].axhline(
        float(summary["atol"]), ls=":", lw=1.2, color="0.25", label="abs gate"
    )
    status = "passed" if bool(summary["identity_passed"]) else "failed"
    axes[1].set_xlabel("Hermite index m")
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
    parser.add_argument("--vth", type=float, default=1.7)
    parser.add_argument("--atol", type=float, default=1.0e-5)
    parser.add_argument("--rtol", type=float, default=1.0e-6)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _configure_logical_cpu_devices(args.logical_devices)
    summary = build_hermite_streaming_ladder_gate(
        shape=(int(args.nl), int(args.nm), int(args.ny), int(args.nx), int(args.nz)),
        requested_devices=int(args.logical_devices),
        vth=float(args.vth),
        atol=float(args.atol),
        rtol=float(args.rtol),
    )
    paths = write_artifacts(summary, args.out_prefix)
    print(
        json.dumps(
            {"identity_passed": summary["identity_passed"], "paths": paths}, indent=2
        )
    )


if __name__ == "__main__":
    main()
