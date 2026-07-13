#!/usr/bin/env python3
"""Profile the opt-in electrostatic linear-slices parallel RHS route."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any, Callable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.artifacts.generate_linear_rhs_parallel_gates import build_problem  # noqa: E402

DEFAULT_PREFIX = REPO_ROOT / "docs" / "_static" / "linear_rhs_parallel_slices_profile"
DEFAULT_SWEEP_PREFIX = (
    REPO_ROOT / "docs" / "_static" / "linear_rhs_parallel_slices_sweep"
)


def _git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


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


def _parse_int_list(text: str) -> list[int]:
    values = [int(part.strip()) for part in str(text).split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError(
            "expected a comma-separated list of positive integers"
        )
    if any(value < 1 for value in values):
        raise argparse.ArgumentTypeError("all values must be positive")
    return values


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


def _build_species_problem(*, nx: int, ny: int, nz: int, nl: int, nm: int):
    import jax.numpy as jnp

    from spectraxgk.config import CycloneBaseCase, GridConfig
    from spectraxgk.core.grid import build_spectral_grid
    from spectraxgk.geometry import SAlphaGeometry
    from spectraxgk.linear import LinearParams, build_linear_cache

    cfg = CycloneBaseCase(
        grid=GridConfig(Nx=nx, Ny=ny, Nz=nz, Lx=6.0, Ly=6.0, boundary="periodic")
    )
    grid = build_spectral_grid(cfg.grid)
    params = LinearParams(
        charge_sign=jnp.asarray([1.0, -1.0]),
        density=jnp.asarray([1.0, 1.0]),
        mass=jnp.asarray([1.0, 1.0 / 1836.0]),
        temp=jnp.asarray([1.0, 1.0]),
        vth=jnp.asarray([1.0, 42.0]),
        rho=jnp.asarray([1.0, 0.023]),
        R_over_Ln=jnp.asarray([2.2, 2.2]),
        R_over_LTi=jnp.asarray([6.9, 0.0]),
        R_over_LTe=jnp.asarray([0.0, 6.9]),
        tz=jnp.asarray([1.0, -1.0]),
        tau_e=0.0,
        beta=0.0,
        fapar=0.0,
    )
    cache = build_linear_cache(
        grid, SAlphaGeometry.from_config(cfg.geometry), params, Nl=nl, Nm=nm
    )
    shape = (2, nl, nm, grid.ky.size, grid.kx.size, grid.z.size)
    rng = np.random.default_rng(20260712)
    state = jnp.asarray(
        1.0e-3
        * (
            rng.standard_normal(shape, dtype=np.float32)
            + 1j * rng.standard_normal(shape, dtype=np.float32)
        ),
        dtype=jnp.complex64,
    )
    return state, cache, params, grid


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
    axis: str = "hermite",
) -> dict[str, object]:
    """Time serial and velocity-sharded electrostatic linear-slices RHS calls."""

    import jax
    from spectraxgk.linear import linear_rhs_cached, linear_rhs_parallel_cached
    from spectraxgk.workflows.runtime.config import RuntimeParallelConfig

    platform_name = str(platform).lower()
    device_list = list(jax.devices(platform_name))[: int(requested_devices)]
    if len(device_list) < int(requested_devices):
        raise RuntimeError(
            f"requested {requested_devices} {platform_name} devices, but only {len(device_list)} are available"
        )
    axis_name = str(axis).strip().lower().replace("-", "_")
    if axis_name not in {"hermite", "species"}:
        raise ValueError("axis must be 'hermite' or 'species'")
    if axis_name == "species" and int(requested_devices) != 2:
        raise ValueError("species profiling requires exactly two devices")
    builder = _build_species_problem if axis_name == "species" else build_problem
    state, cache, params, grid = builder(nx=nx, ny=ny, nz=nz, nl=nl, nm=nm)
    terms = _terms()
    parallel_cfg = RuntimeParallelConfig(
        strategy="velocity",
        backend="auto" if axis_name == "species" else "electrostatic_linear_slices",
        axis=axis_name,
        num_devices=len(device_list),
    )

    serial_state = state
    sharded_state = state
    serial_cache, serial_params = cache, params
    sharded_cache, sharded_params = cache, params
    if axis_name == "species":
        from spectraxgk.solvers.linear.parallel_electrostatic import (
            prepare_electrostatic_species_inputs,
        )

        sharded_state, sharded_cache, sharded_params = (
            prepare_electrostatic_species_inputs(
                state, cache, params, devices=device_list
            )
        )
        serial_compiled = jax.jit(
            lambda value: linear_rhs_cached(
                value,
                serial_cache,
                serial_params,
                terms=terms,
                use_jit=False,
                use_custom_vjp=False,
                force_electrostatic_fields=True,
            )
        )
        sharded_compiled = jax.jit(
            lambda value: linear_rhs_parallel_cached(
                value,
                sharded_cache,
                sharded_params,
                terms=terms,
                parallel=parallel_cfg,
                use_custom_vjp=False,
            )
        )

        def serial_call():
            return serial_compiled(serial_state)

        def sharded_call():
            return sharded_compiled(sharded_state)

    else:

        def serial_call():
            return linear_rhs_cached(
                serial_state,
                serial_cache,
                serial_params,
                terms=terms,
                use_jit=True,
                use_custom_vjp=True,
            )

        def sharded_call():
            return linear_rhs_parallel_cached(
                sharded_state,
                sharded_cache,
                sharded_params,
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

    serial_rhs_host = np.asarray(serial_rhs)
    sharded_rhs_host = np.asarray(sharded_rhs)
    serial_phi_host = np.asarray(serial_phi)
    sharded_phi_host = np.asarray(sharded_phi)
    abs_err = float(np.max(np.abs(sharded_rhs_host - serial_rhs_host)))
    scale = float(np.max(np.abs(serial_rhs_host)))
    rel_err = abs_err / max(scale, 1.0e-30)
    phi_abs_err = float(np.max(np.abs(sharded_phi_host - serial_phi_host)))
    phi_norm = float(np.linalg.norm(serial_phi_host))

    serial_median = float(median(serial_samples))
    sharded_median = float(median(sharded_samples))
    speedup = serial_median / sharded_median if sharded_median > 0.0 else math.nan
    identity_passed = bool(
        abs_err <= float(atol) and rel_err <= float(rtol) and phi_abs_err <= float(atol)
    )

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
            "decomposition_axis": axis_name,
            "grid": {
                "Nx": int(nx),
                "Ny_requested": int(ny),
                "Ny_actual": int(grid.ky.size),
                "Nz": int(grid.z.size),
            },
            "platform": platform_name,
            "git_revision": _git_revision(),
            "jax_version": jax.__version__,
            "python_version": sys.version.split()[0],
            "requested_devices": int(requested_devices),
            "actual_devices": len(device_list),
            "warmups": int(warmups),
            "repeats": int(repeats),
            "serial_median_s": serial_median,
            "sharded_median_s": sharded_median,
            "speedup": float(speedup),
            "identity_passed": identity_passed,
            "max_abs_error": abs_err,
            "max_rel_error": rel_err,
            "max_phi_abs_error": phi_abs_err,
            "phi_norm": phi_norm,
            "atol": float(atol),
            "rtol": float(rtol),
            "rows": rows,
            "notes": (
                "Both routes are warmed before timing. The serial route uses the production JIT path; "
                + (
                    "the species route compiles a host-prepared two-device shard-map callable."
                    if axis_name == "species"
                    else "the sharded route uses the cached fused Hermite shard-map callable."
                )
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
            fh, fieldnames=["route", "median_s", "samples_s"], lineterminator="\n"
        )
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
    axes[1].semilogy(
        ["RHS abs", "RHS rel", "phi abs"], np.maximum(errors, 1.0e-16), "s-", lw=2.0
    )
    axes[1].axhline(
        float(summary["atol"]), ls=":", lw=1.2, color="0.25", label="abs gate"
    )
    axes[1].axhline(
        float(summary["rtol"]), ls="--", lw=1.2, color="0.35", label="rel gate"
    )
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


def run_sweep(
    *,
    platform: str,
    devices: list[int],
    nms: list[int],
    nx: int,
    ny: int,
    nz: int,
    nl: int,
    warmups: int,
    repeats: int,
    atol: float,
    rtol: float,
) -> dict[str, object]:
    """Run a bounded engineering timing sweep over devices and Hermite size."""

    rows: list[dict[str, object]] = []
    for nm in nms:
        for requested_devices in devices:
            summary = profile_linear_rhs_parallel_slices(
                platform=platform,
                requested_devices=requested_devices,
                nx=nx,
                ny=ny,
                nz=nz,
                nl=nl,
                nm=nm,
                warmups=warmups,
                repeats=repeats,
                atol=atol,
                rtol=rtol,
            )
            rows.append(
                {
                    "platform": platform,
                    "requested_devices": int(requested_devices),
                    "nm": int(nm),
                    "state_shape": tuple(summary["state_shape"]),
                    "serial_median_s": float(summary["serial_median_s"]),
                    "sharded_median_s": float(summary["sharded_median_s"]),
                    "speedup": float(summary["speedup"]),
                    "identity_passed": bool(summary["identity_passed"]),
                    "max_abs_error": float(summary["max_abs_error"]),
                    "max_rel_error": float(summary["max_rel_error"]),
                    "max_phi_abs_error": float(summary["max_phi_abs_error"]),
                }
            )
    passed = all(bool(row["identity_passed"]) for row in rows)
    return _json_clean(
        {
            "kind": "linear_rhs_parallel_slices_sweep",
            "claim_scope": (
                "engineering sweep over the opt-in electrostatic linear-slices route; "
                "not a publication speedup claim"
            ),
            "platform": platform,
            "devices": [int(x) for x in devices],
            "nms": [int(x) for x in nms],
            "grid": {
                "Nx": int(nx),
                "Ny_requested": int(ny),
                "Nz": int(nz),
                "Nl": int(nl),
            },
            "warmups": int(warmups),
            "repeats": int(repeats),
            "atol": float(atol),
            "rtol": float(rtol),
            "identity_passed": passed,
            "rows": rows,
            "notes": (
                "Use this plot to find useful CPU/GPU regimes before promoting any speedup claim. "
                "The release correctness gate remains the small composed identity artifact."
            ),
        }
    )


def write_sweep_artifacts(
    summary: dict[str, object], out_prefix: Path
) -> dict[str, str]:
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
    fieldnames = [
        "platform",
        "requested_devices",
        "nm",
        "state_shape",
        "serial_median_s",
        "sharded_median_s",
        "speedup",
        "identity_passed",
        "max_abs_error",
        "max_rel_error",
        "max_phi_abs_error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.9), constrained_layout=True)
    nms = sorted({int(row["nm"]) for row in rows})
    palette = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, max(len(nms), 1)))
    center = 0.5 * (len(nms) - 1)
    for series_index, (color, nm) in enumerate(zip(palette, nms, strict=True)):
        subset = sorted(
            (row for row in rows if int(row["nm"]) == nm),
            key=lambda row: int(row["requested_devices"]),
        )
        x = np.asarray([int(row["requested_devices"]) for row in subset], dtype=float)
        x_visible = x + 0.05 * (series_index - center)
        speedup = np.asarray([float(row["speedup"]) for row in subset], dtype=float)
        rel_error = np.asarray(
            [float(row["max_rel_error"]) for row in subset], dtype=float
        )
        axes[0].plot(x, speedup, "o-", lw=2.0, color=color, label=f"Nm={nm}")
        axes[1].semilogy(
            x_visible,
            np.maximum(rel_error, 1.0e-16),
            "s-",
            lw=2.0,
            color=color,
            label=f"Nm={nm}",
        )
    axes[0].axhline(1.0, color="0.35", ls="--", lw=1.1)
    axes[0].set_xlabel("devices")
    axes[0].set_ylabel("serial / sharded median time")
    axes[0].set_title("Electrostatic RHS engineering speedup")
    axes[1].axhline(
        float(summary["rtol"]), color="0.35", ls="--", lw=1.1, label="relative gate"
    )
    axes[1].set_xlabel("devices")
    axes[1].set_ylabel("max relative RHS error")
    status = "passed" if bool(summary["identity_passed"]) else "failed"
    axes[1].set_title(f"Identity {status}")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
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
    parser.add_argument("--axis", choices=("hermite", "species"), default="hermite")
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


def build_sweep_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep electrostatic linear-slices RHS timings over devices and Hermite size."
    )
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_SWEEP_PREFIX)
    parser.add_argument("--platform", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--devices", type=_parse_int_list, default=[1, 2, 4, 8])
    parser.add_argument("--nms", type=_parse_int_list, default=[64, 128])
    parser.add_argument("--nl", type=int, default=4)
    parser.add_argument("--ny", type=int, default=32)
    parser.add_argument("--nx", type=int, default=1)
    parser.add_argument("--nz", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--atol", type=float, default=2.0e-5)
    parser.add_argument("--rtol", type=float, default=1.0e-5)
    return parser


def main_profile(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
        axis=str(args.axis),
    )
    paths = write_artifacts(summary, args.out_prefix)
    print(
        json.dumps(
            {
                "identity_passed": summary["identity_passed"],
                "speedup": summary["speedup"],
                "paths": paths,
            },
            indent=2,
        )
    )
    return 0


def main_sweep(argv: list[str] | None = None) -> int:
    args = build_sweep_parser().parse_args(argv)
    devices = list(args.devices)
    if args.platform == "cpu":
        _configure_logical_cpu_devices(max(devices))
    summary = run_sweep(
        platform=str(args.platform),
        devices=devices,
        nms=list(args.nms),
        nx=int(args.nx),
        ny=int(args.ny),
        nz=int(args.nz),
        nl=int(args.nl),
        warmups=int(args.warmups),
        repeats=int(args.repeats),
        atol=float(args.atol),
        rtol=float(args.rtol),
    )
    paths = write_sweep_artifacts(summary, args.out_prefix)
    print(
        json.dumps(
            {"identity_passed": summary["identity_passed"], "paths": paths}, indent=2
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    if tokens and tokens[0] == "sweep":
        return main_sweep(tokens[1:])
    return main_profile(tokens)


if __name__ == "__main__":
    raise SystemExit(main())
