#!/usr/bin/env python3
"""Generate a shard-map Hermite ghost-exchange identity gate."""

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
DEFAULT_PREFIX = REPO_ROOT / "docs" / "_static" / "hermite_exchange_gate"


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


def build_hermite_exchange_gate(
    *,
    shape: tuple[int, int, int, int, int],
    requested_devices: int,
    atol: float,
) -> dict[str, object]:
    """Run the Hermite ghost-exchange kernel and compare against reference."""

    import jax
    import jax.numpy as jnp

    from spectraxgk.parallel.velocity import (
        build_velocity_sharding_plan,
        hermite_neighbor_reference,
        hermite_neighbor_shard_map,
    )

    device_list = list(jax.devices("cpu"))[: int(requested_devices)]
    if len(device_list) < int(requested_devices):
        raise RuntimeError(f"requested {requested_devices} CPU devices, but only {len(device_list)} are available")
    state = _state(shape)
    plan = build_velocity_sharding_plan(shape, num_devices=len(device_list), axes=("hermite",))
    lower_ref, upper_ref = hermite_neighbor_reference(state)
    lower_sharded, upper_sharded = hermite_neighbor_shard_map(state, plan, devices=device_list)
    _block_until_ready((lower_sharded, upper_sharded))

    lower_err = jnp.max(jnp.abs(lower_sharded - lower_ref))
    upper_err = jnp.max(jnp.abs(upper_sharded - upper_ref))
    _block_until_ready((lower_err, upper_err))
    max_lower_abs_error = float(np.asarray(lower_err))
    max_upper_abs_error = float(np.asarray(upper_err))
    identity_passed = bool(max(max_lower_abs_error, max_upper_abs_error) <= float(atol))

    center_trace = np.asarray(state[0, :, 0, 0, 0])
    lower_trace = np.asarray(lower_sharded[0, :, 0, 0, 0])
    upper_trace = np.asarray(upper_sharded[0, :, 0, 0, 0])
    lower_ref_trace = np.asarray(lower_ref[0, :, 0, 0, 0])
    upper_ref_trace = np.asarray(upper_ref[0, :, 0, 0, 0])
    rows = []
    for m_idx in range(shape[1]):
        rows.append(
            {
                "m": int(m_idx),
                "center_real": float(np.real(center_trace[m_idx])),
                "lower_real": float(np.real(lower_trace[m_idx])),
                "upper_real": float(np.real(upper_trace[m_idx])),
                "lower_reference_real": float(np.real(lower_ref_trace[m_idx])),
                "upper_reference_real": float(np.real(upper_ref_trace[m_idx])),
                "lower_abs_error": float(abs(lower_trace[m_idx] - lower_ref_trace[m_idx])),
                "upper_abs_error": float(abs(upper_trace[m_idx] - upper_ref_trace[m_idx])),
            }
        )

    return _json_clean(
        {
            "case": "Hermite ghost-exchange shard_map identity gate",
            "source": "spectraxgk.parallel.velocity.hermite_neighbor_shard_map",
            "claim_scope": "communication-kernel identity gate, not a nonlinear runtime speedup claim",
            "state_shape": shape,
            "requested_devices": int(requested_devices),
            "actual_devices": len(device_list),
            "plan": plan.to_dict(),
            "max_lower_abs_error": max_lower_abs_error,
            "max_upper_abs_error": max_upper_abs_error,
            "atol": float(atol),
            "identity_passed": identity_passed,
            "rows": rows,
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

    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = list(summary["rows"])
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    m = np.asarray([row["m"] for row in rows], dtype=float)
    center = np.asarray([row["center_real"] for row in rows], dtype=float)
    lower = np.asarray([row["lower_real"] for row in rows], dtype=float)
    upper = np.asarray([row["upper_real"] for row in rows], dtype=float)
    lower_err = np.asarray([row["lower_abs_error"] for row in rows], dtype=float)
    upper_err = np.asarray([row["upper_abs_error"] for row in rows], dtype=float)

    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), constrained_layout=True)
    axes[0].plot(m, center, "o-", lw=2.0, label="center")
    axes[0].plot(m, lower, "s--", lw=1.8, label="lower neighbor")
    axes[0].plot(m, upper, "^--", lw=1.8, label="upper neighbor")
    axes[0].set_xlabel("Hermite index m")
    axes[0].set_ylabel("real value")
    axes[0].set_title("Nearest-Hermite exchange")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].semilogy(m, np.maximum(lower_err, 1.0e-16), "s-", lw=2.0, label="lower error")
    axes[1].semilogy(m, np.maximum(upper_err, 1.0e-16), "^-", lw=2.0, label="upper error")
    axes[1].axhline(float(summary["atol"]), ls=":", lw=1.2, color="0.25", label="gate")
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
    parser.add_argument("--atol", type=float, default=1.0e-7)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _configure_logical_cpu_devices(args.logical_devices)
    summary = build_hermite_exchange_gate(
        shape=(int(args.nl), int(args.nm), int(args.ny), int(args.nx), int(args.nz)),
        requested_devices=int(args.logical_devices),
        atol=float(args.atol),
    )
    paths = write_artifacts(summary, args.out_prefix)
    print(json.dumps({"identity_passed": summary["identity_passed"], "paths": paths}, indent=2))


if __name__ == "__main__":
    main()
