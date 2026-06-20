#!/usr/bin/env python3
"""Generate a logical-CPU independent-scan parallelization identity gate."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREFIX = REPO_ROOT / "docs" / "_static" / "logical_cpu_parallel_scan_gate"


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


def _block_until_ready(tree: Any) -> None:
    import jax

    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _timed_scan_model(
    ky_values: np.ndarray,
    *,
    batch_size: int,
    devices: list[Any],
) -> tuple[dict[str, np.ndarray], float]:
    """Evaluate a tiny JAX-native linear-scan model through ``batch_map``."""

    import jax
    import jax.numpy as jnp

    from spectraxgk.parallel import batch_map

    ky_jax = jnp.asarray(ky_values, dtype=jnp.float64 if jax.config.jax_enable_x64 else jnp.float32)

    def model(ky):
        kperp2 = 0.08 + ky**2
        drive = 0.52 * ky * jnp.exp(-0.75 * ky**2)
        damping = 0.045 + 0.02 * ky**2
        operator = jnp.asarray(
            [
                [drive - damping - 0.18j * ky, 0.08 + 0.02j * ky],
                [0.04 - 0.01j * ky, -0.12 - 0.04 * kperp2 + 0.06j * ky],
            ],
            dtype=jnp.complex128 if jax.config.jax_enable_x64 else jnp.complex64,
        )
        eigvals = jnp.linalg.eigvals(operator)
        mode = eigvals[jnp.argmax(jnp.real(eigvals))]
        gamma = jnp.real(mode)
        omega = jnp.imag(mode)
        return {
            "gamma": gamma,
            "omega": omega,
            "kperp2": kperp2,
            "ql_proxy": jnp.maximum(gamma, 0.0) / kperp2,
        }

    start = time.perf_counter()
    output = batch_map(model, ky_jax, batch_size=batch_size, devices=devices)
    _block_until_ready(output)
    elapsed = time.perf_counter() - start
    arrays = jax.tree_util.tree_map(lambda leaf: np.asarray(leaf, dtype=float), output)
    return dict(arrays), elapsed


def _select_devices(requested_devices: int) -> list[Any]:
    import jax

    devices = list(jax.devices("cpu"))
    if not devices:
        raise RuntimeError("No CPU JAX devices are available")
    return devices[: max(1, min(int(requested_devices), len(devices)))]


def build_logical_cpu_parallel_scan_gate(
    *,
    ky_values: np.ndarray,
    serial_batch: int,
    parallel_batch: int,
    requested_devices: int,
    gamma_rtol: float,
    omega_atol: float,
    ql_rtol: float,
) -> dict[str, object]:
    """Compare serial and logical-CPU device-batched independent scans."""

    from spectraxgk.workflows.runtime.config import RuntimeParallelConfig

    devices = _select_devices(requested_devices)
    serial_config = RuntimeParallelConfig(strategy="serial", axis="ky", batch_size=serial_batch, num_devices=1)
    parallel_config = RuntimeParallelConfig(
        strategy="device_batch",
        axis="ky",
        batch_size=parallel_batch,
        num_devices=len(devices),
        strict_identity=True,
    )
    serial, serial_elapsed = _timed_scan_model(ky_values, batch_size=serial_batch, devices=devices[:1])
    batched, batch_elapsed = _timed_scan_model(ky_values, batch_size=parallel_batch, devices=devices)

    gamma_denom = np.maximum(np.abs(serial["gamma"]), 1.0e-12)
    ql_denom = np.maximum(np.abs(serial["ql_proxy"]), 1.0e-12)
    gamma_rel = np.abs(batched["gamma"] - serial["gamma"]) / gamma_denom
    omega_abs = np.abs(batched["omega"] - serial["omega"])
    ql_rel = np.abs(batched["ql_proxy"] - serial["ql_proxy"]) / ql_denom

    max_gamma_rel = float(np.max(gamma_rel))
    max_omega_abs = float(np.max(omega_abs))
    max_ql_rel = float(np.max(ql_rel))
    identity_passed = bool(max_gamma_rel <= gamma_rtol and max_omega_abs <= omega_atol and max_ql_rel <= ql_rtol)
    speedup = float(serial_elapsed / batch_elapsed) if batch_elapsed > 0.0 else math.inf

    rows = []
    for idx, ky in enumerate(np.asarray(ky_values, dtype=float)):
        rows.append(
            {
                "ky": float(ky),
                "serial_gamma": float(serial["gamma"][idx]),
                "batched_gamma": float(batched["gamma"][idx]),
                "gamma_rel_error": float(gamma_rel[idx]),
                "serial_omega": float(serial["omega"][idx]),
                "batched_omega": float(batched["omega"][idx]),
                "omega_abs_error": float(omega_abs[idx]),
                "serial_ql_proxy": float(serial["ql_proxy"][idx]),
                "batched_ql_proxy": float(batched["ql_proxy"][idx]),
                "ql_rel_error": float(ql_rel[idx]),
            }
        )

    return _json_clean(
        {
            "case": "Logical-CPU independent ky scan",
            "source": "JAX-native tiny non-Hermitian linear scan model through spectraxgk.batch_map",
            "claim_scope": "parallel interface and numerical identity gate, not a gyrokinetic physics validation",
            "ky_values": np.asarray(ky_values, dtype=float),
            "serial_parallel_config": serial_config.to_dict(),
            "device_parallel_config": parallel_config.to_dict(),
            "requested_devices": int(requested_devices),
            "actual_devices": len(devices),
            "serial_elapsed_s": serial_elapsed,
            "batched_elapsed_s": batch_elapsed,
            "observed_speedup": speedup,
            "max_gamma_rel_error": max_gamma_rel,
            "max_omega_abs_error": max_omega_abs,
            "max_ql_rel_error": max_ql_rel,
            "gamma_rtol": gamma_rtol,
            "omega_atol": omega_atol,
            "ql_rtol": ql_rtol,
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

    ky = np.asarray([row["ky"] for row in rows], dtype=float)
    serial_gamma = np.asarray([row["serial_gamma"] for row in rows], dtype=float)
    batched_gamma = np.asarray([row["batched_gamma"] for row in rows], dtype=float)
    serial_ql = np.asarray([row["serial_ql_proxy"] for row in rows], dtype=float)
    batched_ql = np.asarray([row["batched_ql_proxy"] for row in rows], dtype=float)
    gamma_rel = np.asarray([row["gamma_rel_error"] for row in rows], dtype=float)
    ql_rel = np.asarray([row["ql_rel_error"] for row in rows], dtype=float)

    set_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.8), constrained_layout=True)
    axes[0].plot(ky, serial_gamma, "o-", lw=2.0, label="serial gamma")
    axes[0].plot(ky, batched_gamma, "s--", lw=1.8, label="parallel gamma")
    axes[0].set_xlabel(r"$k_y \rho_i$")
    axes[0].set_ylabel(r"$\gamma$")
    axes[0].set_title("Linear-scan observable")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(ky, serial_ql, "o-", lw=2.0, label="serial QL proxy")
    axes[1].plot(ky, batched_ql, "s--", lw=1.8, label="parallel QL proxy")
    axes[1].set_xlabel(r"$k_y \rho_i$")
    axes[1].set_ylabel(r"$\max(\gamma,0)/k_\perp^2$")
    axes[1].set_title("Structured pytree output")
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].semilogy(ky, np.maximum(gamma_rel, 1.0e-16), "o-", lw=2.0, label=r"$\gamma$ rel.")
    axes[2].semilogy(ky, np.maximum(ql_rel, 1.0e-16), "s-", lw=2.0, label="QL rel.")
    axes[2].axhline(float(summary["gamma_rtol"]), ls=":", lw=1.1)
    axes[2].axhline(float(summary["ql_rtol"]), ls=":", lw=1.1)
    status = "passed" if bool(summary["identity_passed"]) else "failed"
    axes[2].set_xlabel(r"$k_y \rho_i$")
    axes[2].set_ylabel("identity error")
    axes[2].set_title(f"Gate {status}, {float(summary['observed_speedup']):.2f}x")
    axes[2].legend(frameon=False, fontsize=8)

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
    parser.add_argument("--ky", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    parser.add_argument("--serial-batch", type=int, default=1)
    parser.add_argument("--parallel-batch", type=int, default=2)
    parser.add_argument("--logical-devices", type=int, default=4)
    parser.add_argument("--gamma-rtol", type=float, default=1.0e-6)
    parser.add_argument("--omega-atol", type=float, default=1.0e-6)
    parser.add_argument("--ql-rtol", type=float, default=1.0e-6)
    return parser


def _configure_logical_cpu_devices(count: int) -> None:
    if int(count) <= 1:
        return
    flag = f"--xla_force_host_platform_device_count={int(count)}"
    current = os.environ.get("XLA_FLAGS", "")
    if "xla_force_host_platform_device_count" not in current:
        os.environ["XLA_FLAGS"] = f"{current} {flag}".strip()


def main() -> None:
    args = build_parser().parse_args()
    _configure_logical_cpu_devices(args.logical_devices)
    summary = build_logical_cpu_parallel_scan_gate(
        ky_values=np.asarray(args.ky, dtype=float),
        serial_batch=args.serial_batch,
        parallel_batch=args.parallel_batch,
        requested_devices=args.logical_devices,
        gamma_rtol=args.gamma_rtol,
        omega_atol=args.omega_atol,
        ql_rtol=args.ql_rtol,
    )
    paths = write_artifacts(summary, args.out_prefix)
    print(json.dumps({"identity_passed": summary["identity_passed"], "paths": paths}, indent=2))


if __name__ == "__main__":
    main()
