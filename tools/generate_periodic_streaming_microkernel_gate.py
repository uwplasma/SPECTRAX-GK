#!/usr/bin/env python3
"""Generate a periodic streaming microkernel identity gate."""

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
DEFAULT_PREFIX = REPO_ROOT / "docs" / "_static" / "periodic_streaming_microkernel_gate"


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


def _state(shape: tuple[int, ...]) -> tuple[Any, Any]:
    import jax.numpy as jnp

    nz = int(shape[-1])
    z = jnp.linspace(0.0, 2.0 * jnp.pi, nz, endpoint=False)
    state = jnp.zeros(shape, dtype=jnp.complex64)
    state = state.at[0, 0, 1, 0, 0, :].set(jnp.exp(1j * z))
    state = state.at[0, 1, 3, 1, 0, :].set(0.3 * jnp.exp(2j * z))
    return state, z


def _production_streaming_term(state: Any, *, kz: Any, vth: float) -> Any:
    import jax.numpy as jnp

    from spectraxgk.basis import hermite_ladder_coeffs
    from spectraxgk.terms.operators import streaming_term

    nm = int(state.shape[-4])
    sqrt_p, sqrt_m = hermite_ladder_coeffs(nm - 1)
    sqrt_p = sqrt_p[:nm].reshape((1, 1, nm, 1, 1, 1))
    sqrt_m = sqrt_m[:nm].reshape((1, 1, nm, 1, 1, 1))
    return streaming_term(
        state,
        kz=kz,
        vth=jnp.asarray(vth, dtype=jnp.float32).reshape((1, 1, 1, 1, 1, 1)),
        sqrt_p=sqrt_p,
        sqrt_m=sqrt_m,
    )


def build_periodic_streaming_microkernel_gate(
    *,
    shape: tuple[int, int, int, int, int, int],
    requested_devices: int,
    vth: float,
    atol: float,
    rtol: float,
) -> dict[str, object]:
    """Compare the shard-map periodic streaming microkernel to production."""

    import jax
    import jax.numpy as jnp

    from spectraxgk.parallel.velocity import (
        build_velocity_sharding_plan,
        periodic_streaming_reference,
        periodic_streaming_shard_map,
    )

    device_list = list(jax.devices("cpu"))[: int(requested_devices)]
    if len(device_list) < int(requested_devices):
        raise RuntimeError(f"requested {requested_devices} CPU devices, but only {len(device_list)} are available")
    state, z = _state(shape)
    dz = float(np.asarray(z[1] - z[0]))
    kz = 2.0 * jnp.pi * jnp.fft.fftfreq(int(shape[-1]), d=dz)
    plan = build_velocity_sharding_plan(shape, num_devices=len(device_list), axes=("hermite",))

    production = _production_streaming_term(state, kz=kz, vth=float(vth))
    reference = periodic_streaming_reference(state, kz=kz, vth=jnp.asarray([vth], dtype=jnp.float32))
    sharded = periodic_streaming_shard_map(state, plan, kz=kz, vth=jnp.asarray([vth], dtype=jnp.float32), devices=device_list)
    _block_until_ready((production, reference, sharded))

    ref_abs = jnp.max(jnp.abs(reference - production))
    sharded_abs = jnp.max(jnp.abs(sharded - production))
    scale = jnp.max(jnp.abs(production))
    sharded_rel = sharded_abs / jnp.maximum(scale, jnp.asarray(1.0e-30, dtype=scale.dtype))
    _block_until_ready((ref_abs, sharded_abs, sharded_rel))
    max_reference_abs_error = float(np.asarray(ref_abs))
    max_sharded_abs_error = float(np.asarray(sharded_abs))
    max_sharded_rel_error = float(np.asarray(sharded_rel))
    identity_passed = bool(max_sharded_abs_error <= float(atol) and max_sharded_rel_error <= float(rtol))

    production_trace = np.asarray(production[0, 0, :, 0, 0, 1])
    sharded_trace = np.asarray(sharded[0, 0, :, 0, 0, 1])
    state_trace = np.asarray(state[0, 0, :, 0, 0, 1])
    rows = []
    for m_idx in range(shape[2]):
        rows.append(
            {
                "m": int(m_idx),
                "state_abs": float(abs(state_trace[m_idx])),
                "production_abs": float(abs(production_trace[m_idx])),
                "sharded_abs": float(abs(sharded_trace[m_idx])),
                "abs_error": float(abs(sharded_trace[m_idx] - production_trace[m_idx])),
            }
        )

    return _json_clean(
        {
            "case": "Periodic streaming microkernel shard_map identity gate",
            "source": "spectraxgk.parallel.velocity.periodic_streaming_shard_map",
            "reference_source": "spectraxgk.terms.operators.streaming_term",
            "claim_scope": "linear streaming microkernel identity gate, not a full RHS or nonlinear speedup claim",
            "state_shape": shape,
            "vth": float(vth),
            "requested_devices": int(requested_devices),
            "actual_devices": len(device_list),
            "plan": plan.to_dict(),
            "max_reference_abs_error": max_reference_abs_error,
            "max_sharded_abs_error": max_sharded_abs_error,
            "max_sharded_rel_error": max_sharded_rel_error,
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
    production = np.asarray([row["production_abs"] for row in rows], dtype=float)
    sharded = np.asarray([row["sharded_abs"] for row in rows], dtype=float)
    error = np.asarray([row["abs_error"] for row in rows], dtype=float)

    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), constrained_layout=True)
    axes[0].plot(m, production, "s-", lw=1.8, label="production streaming")
    axes[0].plot(m, sharded, "^--", lw=1.8, label="shard_map streaming")
    axes[0].set_xlabel("Hermite index m")
    axes[0].set_ylabel("absolute value")
    axes[0].set_title("Periodic streaming")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].semilogy(m, np.maximum(error, 1.0e-16), "s-", lw=2.0, label="absolute error")
    axes[1].axhline(float(summary["atol"]), ls=":", lw=1.2, color="0.25", label="abs gate")
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
    parser.add_argument("--ns", type=int, default=1)
    parser.add_argument("--nl", type=int, default=2)
    parser.add_argument("--nm", type=int, default=8)
    parser.add_argument("--ny", type=int, default=2)
    parser.add_argument("--nx", type=int, default=1)
    parser.add_argument("--nz", type=int, default=16)
    parser.add_argument("--vth", type=float, default=1.7)
    parser.add_argument("--atol", type=float, default=1.0e-5)
    parser.add_argument("--rtol", type=float, default=1.0e-6)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _configure_logical_cpu_devices(args.logical_devices)
    summary = build_periodic_streaming_microkernel_gate(
        shape=(int(args.ns), int(args.nl), int(args.nm), int(args.ny), int(args.nx), int(args.nz)),
        requested_devices=int(args.logical_devices),
        vth=float(args.vth),
        atol=float(args.atol),
        rtol=float(args.rtol),
    )
    paths = write_artifacts(summary, args.out_prefix)
    print(json.dumps({"identity_passed": summary["identity_passed"], "paths": paths}, indent=2))


if __name__ == "__main__":
    main()
