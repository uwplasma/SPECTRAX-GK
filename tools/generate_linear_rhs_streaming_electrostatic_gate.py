#!/usr/bin/env python3
"""Generate an electrostatic streaming linear-RHS identity gate."""

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
DEFAULT_PREFIX = REPO_ROOT / "docs" / "_static" / "linear_rhs_streaming_electrostatic_gate"


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


def _streaming_only_terms() -> Any:
    from spectraxgk.linear import LinearTerms

    return LinearTerms(
        streaming=1.0,
        mirror=0.0,
        curvature=0.0,
        gradb=0.0,
        diamagnetic=0.0,
        collisions=0.0,
        hypercollisions=0.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
    )


def build_problem(
    *,
    nx: int,
    ny: int,
    nz: int,
    nl: int,
    nm: int,
) -> tuple[Any, Any, Any, Any]:
    """Build a periodic electrostatic streaming problem with nonzero phi."""

    import jax.numpy as jnp

    from spectraxgk.config import CycloneBaseCase, GridConfig
    from spectraxgk.geometry import SAlphaGeometry
    from spectraxgk.grids import build_spectral_grid
    from spectraxgk.linear import LinearParams, build_linear_cache

    cfg = CycloneBaseCase(grid=GridConfig(Nx=int(nx), Ny=int(ny), Nz=int(nz), Lx=6.0, Ly=6.0, boundary="periodic"))
    grid = build_spectral_grid(cfg.grid)
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = LinearParams(beta=0.0, fapar=0.0)
    cache = build_linear_cache(grid, geom, params, int(nl), int(nm))
    z = jnp.linspace(0.0, 2.0 * jnp.pi, grid.z.size, endpoint=False)
    state = jnp.zeros((int(nl), int(nm), grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64)
    ky_index = min(1, grid.ky.size - 1)
    state = state.at[0, 0, ky_index, 0, :].set(0.2 * jnp.exp(1j * z))
    if int(nm) > 3 and int(nl) > 1:
        state = state.at[1, 2, ky_index, 0, :].set(0.1 * jnp.exp(2j * z))
    return state, cache, params, grid


def build_linear_rhs_streaming_electrostatic_gate(
    *,
    requested_devices: int,
    nx: int,
    ny: int,
    nz: int,
    nl: int,
    nm: int,
    atol: float,
    rtol: float,
) -> dict[str, object]:
    """Compare serial streaming+phi RHS with the explicit velocity route."""

    import jax
    import jax.numpy as jnp

    from spectraxgk.linear import linear_rhs_cached, linear_rhs_parallel_cached
    from spectraxgk.runtime_config import RuntimeParallelConfig
    from spectraxgk.velocity_sharding import build_velocity_sharding_plan

    device_list = list(jax.devices("cpu"))[: int(requested_devices)]
    if len(device_list) < int(requested_devices):
        raise RuntimeError(f"requested {requested_devices} CPU devices, but only {len(device_list)} are available")
    state, cache, params, grid = build_problem(nx=nx, ny=ny, nz=nz, nl=nl, nm=nm)
    terms = _streaming_only_terms()
    serial, phi_serial = linear_rhs_cached(state, cache, params, terms=terms, use_jit=True)
    parallel_cfg = RuntimeParallelConfig(
        strategy="velocity",
        backend="streaming_electrostatic",
        axis="hermite",
        num_devices=len(device_list),
    )
    sharded, phi_sharded = linear_rhs_parallel_cached(
        state,
        cache,
        params,
        terms=terms,
        parallel=parallel_cfg,
        use_custom_vjp=True,
    )
    _block_until_ready((serial, phi_serial, sharded, phi_sharded))

    abs_err = jnp.max(jnp.abs(sharded - serial))
    scale = jnp.max(jnp.abs(serial))
    rel_err = abs_err / jnp.maximum(scale, jnp.asarray(1.0e-30, dtype=scale.dtype))
    phi_abs_err = jnp.max(jnp.abs(phi_sharded - phi_serial))
    phi_norm = jnp.linalg.norm(phi_serial)
    _block_until_ready((abs_err, rel_err, phi_abs_err, phi_norm))
    max_abs_error = float(np.asarray(abs_err))
    max_rel_error = float(np.asarray(rel_err))
    max_phi_abs_error = float(np.asarray(phi_abs_err))
    phi_norm_float = float(np.asarray(phi_norm))
    identity_passed = bool(max_abs_error <= float(atol) and max_rel_error <= float(rtol) and max_phi_abs_error <= float(atol))

    ky_index = min(1, grid.ky.size - 1)
    serial_trace = np.asarray(serial[0, :, ky_index, 0, 1])
    sharded_trace = np.asarray(sharded[0, :, ky_index, 0, 1])
    rows = []
    for m_idx in range(int(nm)):
        rows.append(
            {
                "m": int(m_idx),
                "serial_abs": float(abs(serial_trace[m_idx])),
                "sharded_abs": float(abs(sharded_trace[m_idx])),
                "abs_error": float(abs(sharded_trace[m_idx] - serial_trace[m_idx])),
            }
        )

    plan = build_velocity_sharding_plan(state.shape, num_devices=len(device_list), axes=("hermite",))
    return _json_clean(
        {
            "case": "Electrostatic streaming linear-RHS shard-map identity gate",
            "source": "spectraxgk.linear.linear_rhs_parallel_cached backend=streaming_electrostatic",
            "reference_source": "spectraxgk.linear.linear_rhs_cached with only streaming enabled",
            "claim_scope": "streaming plus electrostatic field-solve call-graph identity, not a full-RHS or nonlinear speedup claim",
            "state_shape": tuple(int(x) for x in state.shape),
            "grid": {"Nx": int(nx), "Ny_requested": int(ny), "Ny_actual": int(grid.ky.size), "Nz": int(grid.z.size)},
            "requested_devices": int(requested_devices),
            "actual_devices": len(device_list),
            "plan": plan.to_dict(),
            "phi_norm": phi_norm_float,
            "max_phi_abs_error": max_phi_abs_error,
            "max_abs_error": max_abs_error,
            "max_rel_error": max_rel_error,
            "atol": float(atol),
            "rtol": float(rtol),
            "identity_passed": identity_passed,
            "rows": rows,
            "notes": (
                "Only streaming is enabled. The state includes an m=0 density perturbation, so phi is nonzero. "
                "The electrostatic field solve uses the Hermite-sharded field-reduction path gated separately by "
                "tools/generate_electrostatic_field_reduce_gate.py."
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
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    m = np.asarray([row["m"] for row in rows], dtype=float)
    serial = np.asarray([row["serial_abs"] for row in rows], dtype=float)
    sharded = np.asarray([row["sharded_abs"] for row in rows], dtype=float)
    error = np.asarray([row["abs_error"] for row in rows], dtype=float)

    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), constrained_layout=True)
    axes[0].plot(m, serial, "s-", lw=1.8, label="serial streaming+phi")
    axes[0].plot(m, sharded, "^--", lw=1.8, label="velocity route")
    axes[0].set_xlabel("Hermite index m")
    axes[0].set_ylabel("absolute value")
    axes[0].set_title("Electrostatic streaming RHS")
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
    parser.add_argument("--nl", type=int, default=2)
    parser.add_argument("--nm", type=int, default=8)
    parser.add_argument("--ny", type=int, default=4)
    parser.add_argument("--nx", type=int, default=1)
    parser.add_argument("--nz", type=int, default=16)
    parser.add_argument("--atol", type=float, default=2.0e-5)
    parser.add_argument("--rtol", type=float, default=2.0e-6)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _configure_logical_cpu_devices(args.logical_devices)
    summary = build_linear_rhs_streaming_electrostatic_gate(
        requested_devices=int(args.logical_devices),
        nx=int(args.nx),
        ny=int(args.ny),
        nz=int(args.nz),
        nl=int(args.nl),
        nm=int(args.nm),
        atol=float(args.atol),
        rtol=float(args.rtol),
    )
    paths = write_artifacts(summary, args.out_prefix)
    print(json.dumps({"identity_passed": summary["identity_passed"], "paths": paths}, indent=2))


if __name__ == "__main__":
    main()
