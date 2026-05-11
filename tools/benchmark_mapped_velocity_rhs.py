#!/usr/bin/env python
"""Benchmark mapped velocity-basis RHS readiness on runtime linear states."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import jax
import jax.numpy as jnp
import numpy as np

from spectraxgk.geometry import apply_gx_geometry_grid_defaults
from spectraxgk.grids import build_spectral_grid
from spectraxgk.io import load_runtime_from_toml
from spectraxgk.linear import build_linear_cache
from spectraxgk.runtime import (
    _build_initial_condition,
    _select_nonlinear_mode_indices,
    build_runtime_geometry,
    build_runtime_linear_params,
    build_runtime_term_config,
)
from spectraxgk.terms.assembly import _is_static_zero, assemble_rhs_cached
from spectraxgk.velocity_maps import VelocityMapConfig, map_regularization


DEFAULT_MAP_SPECS = (
    "identity:0.0:0.0:0.0",
    "parallel_shift:0.15:0.0:0.0",
    "parallel_scale:0.0:-0.08:0.0",
    "perp_scale:0.0:0.0:0.06",
)

CSV_FIELDS = (
    "case",
    "backend",
    "state",
    "Nl",
    "Nm",
    "vector_size",
    "map_label",
    "parallel_shift",
    "parallel_log_scale",
    "perpendicular_log_scale",
    "parallel_scale",
    "perpendicular_scale",
    "compile_execute_seconds",
    "warm_seconds",
    "baseline_warm_seconds",
    "warm_over_baseline",
    "rhs_norm",
    "phi_norm",
    "rhs_rel_error_vs_unmapped",
    "gamma_proxy",
    "omega_proxy",
    "gamma_proxy_abs_error_vs_unmapped",
    "omega_proxy_abs_error_vs_unmapped",
    "dense_gamma",
    "dense_omega",
    "dense_gamma_abs_error_vs_unmapped",
    "dense_omega_abs_error_vs_unmapped",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear_miller.toml"),
    )
    parser.add_argument("--ky", type=float, default=0.3)
    parser.add_argument("--kx", type=float, default=None)
    parser.add_argument("--Nl", type=str, default="4", help="Comma-separated Laguerre resolutions.")
    parser.add_argument("--Nm", type=str, default="6,8,12", help="Comma-separated Hermite resolutions.")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--state", choices=("initial", "z_wave"), default="z_wave")
    parser.add_argument("--z-mode", type=int, default=1)
    parser.add_argument("--z-wave-amplitude", type=float, default=1.0e-3)
    parser.add_argument(
        "--map",
        dest="maps",
        action="append",
        default=None,
        help=(
            "Velocity-map spec as label:parallel_shift:parallel_log_scale:perpendicular_log_scale. "
            "May be repeated."
        ),
    )
    parser.add_argument(
        "--dense-eigen-max-size",
        type=int,
        default=256,
        help="Compute a dense dominant eigenvalue only when state.size is at most this value; 0 disables it.",
    )
    parser.add_argument("--identity-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--out-json", type=Path, default=Path("docs/_static/mapped_velocity_rhs_readiness.json"))
    parser.add_argument("--out-csv", type=Path, default=Path("docs/_static/mapped_velocity_rhs_readiness.csv"))
    return parser.parse_args()


def _parse_int_list(value: str, *, name: str) -> list[int]:
    out = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not out:
        raise ValueError(f"{name} must contain at least one integer")
    if any(item < 1 for item in out):
        raise ValueError(f"{name} values must be >= 1")
    return out


def _parse_map_spec(spec: str) -> tuple[str, VelocityMapConfig]:
    parts = [part.strip() for part in spec.split(":")]
    if len(parts) != 4 or not parts[0]:
        raise ValueError(
            "map specs must be label:parallel_shift:parallel_log_scale:perpendicular_log_scale"
        )
    label = parts[0]
    return (
        label,
        VelocityMapConfig(
            parallel_shift=float(parts[1]),
            parallel_log_scale=float(parts[2]),
            perpendicular_log_scale=float(parts[3]),
        ),
    )


def _parse_map_specs(specs: list[str] | None) -> list[tuple[str, VelocityMapConfig]]:
    return [_parse_map_spec(spec) for spec in (specs if specs is not None else list(DEFAULT_MAP_SPECS))]


def _block_tree(tree: Any) -> None:
    for leaf in jax.tree_util.tree_leaves(tree):
        try:
            jax.block_until_ready(leaf)
        except TypeError:
            continue


def _time_jitted(fn: Callable[..., Any], *args: Any, repeats: int) -> tuple[float, float, Any]:
    t0 = time.perf_counter()
    out = fn(*args)
    _block_tree(out)
    compile_execute = time.perf_counter() - t0

    warm_t0 = time.perf_counter()
    for _ in range(int(repeats)):
        out = fn(*args)
        _block_tree(out)
    warm_seconds = (time.perf_counter() - warm_t0) / float(repeats)
    return compile_execute, warm_seconds, out


def _z_variation_norm(state: jnp.ndarray) -> float:
    mean_z = jnp.mean(state, axis=-1, keepdims=True)
    return float(np.asarray(jnp.linalg.norm(state - mean_z)))


def _inject_z_wave(
    state: jnp.ndarray,
    *,
    ky_index: int,
    kx_index: int,
    amplitude: float,
    z_mode: int,
) -> jnp.ndarray:
    """Inject a deterministic parallel wave so linked-z streaming paths are active."""

    state = jnp.asarray(state)
    nz = state.shape[-1]
    nm = state.shape[-4]
    m_index = min(max(1, nm - 1), 3)
    z = jnp.arange(nz, dtype=jnp.float32)
    phase = 2.0 * jnp.pi * float(z_mode) * z / float(nz)
    wave = amplitude * jnp.exp(1j * phase).astype(state.dtype)
    perturbation = jnp.zeros_like(state)
    if state.ndim == 6:
        perturbation = perturbation.at[:, 0, m_index, ky_index, kx_index, :].set(wave)
    elif state.ndim == 5:
        perturbation = perturbation.at[0, m_index, ky_index, kx_index, :].set(wave)
    else:  # pragma: no cover - runtime state builder controls dimensionality.
        raise ValueError("state must have 5 or 6 dimensions")
    return state + perturbation


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0.0:
        return None
    value = float(numerator) / float(denominator)
    return value if math.isfinite(value) else None


def _relative_l2(lhs: jnp.ndarray, rhs: jnp.ndarray) -> float:
    rhs_norm = float(np.asarray(jnp.linalg.norm(rhs)))
    diff_norm = float(np.asarray(jnp.linalg.norm(lhs - rhs)))
    if rhs_norm == 0.0:
        return 0.0 if diff_norm == 0.0 else math.inf
    return diff_norm / rhs_norm


def _rayleigh_frequency_proxy(state: jnp.ndarray, rhs: jnp.ndarray) -> tuple[float, float]:
    denom = jnp.vdot(state, state)
    lam = jnp.where(jnp.abs(denom) > 0.0, jnp.vdot(state, rhs) / denom, jnp.nan + 1j * jnp.nan)
    return float(np.asarray(jnp.real(lam))), float(np.asarray(-jnp.imag(lam)))


def _dense_dominant_frequency(
    rhs_fn: Callable[[jnp.ndarray], tuple[jnp.ndarray, Any]],
    template: jnp.ndarray,
    *,
    max_size: int,
) -> tuple[float, float] | None:
    """Return dominant dense eigen gamma/omega for tiny states, otherwise ``None``."""

    n = int(template.size)
    if int(max_size) <= 0 or n > int(max_size):
        return None

    eye = jnp.eye(n, dtype=template.dtype)

    def _apply(flat_state: jnp.ndarray) -> jnp.ndarray:
        rhs, _fields = rhs_fn(flat_state.reshape(template.shape))
        return rhs.reshape(-1)

    columns = jax.vmap(_apply)(eye)
    matrix = columns.T
    eigvals = jnp.linalg.eigvals(matrix)
    eig = eigvals[jnp.argmax(jnp.real(eigvals))]
    return float(np.asarray(jnp.real(eig))), float(np.asarray(-jnp.imag(eig)))


def _map_regularization_floats(cfg: VelocityMapConfig) -> dict[str, float]:
    reg = map_regularization(cfg)
    return {key: float(np.asarray(value)) for key, value in reg.items()}


def _build_summary(
    rows: list[dict[str, Any]],
    *,
    config: str,
    backend: str,
    repeats: int,
    state: str,
    z_variation_norm: float,
    nonlinear_weight: float,
    identity_tolerance: float,
    dense_eigen_max_size: int,
) -> dict[str, Any]:
    identity_rows = [row for row in rows if row["map_label"] == "identity"]
    mapped_rows = [row for row in rows if row["map_label"] != "unmapped"]
    ratios = [
        float(row["warm_over_baseline"])
        for row in mapped_rows
        if row.get("warm_over_baseline") is not None and math.isfinite(float(row["warm_over_baseline"]))
    ]
    identity_rhs_errors = [float(row["rhs_rel_error_vs_unmapped"]) for row in identity_rows]
    identity_gamma_errors = [float(row["gamma_proxy_abs_error_vs_unmapped"]) for row in identity_rows]
    identity_omega_errors = [float(row["omega_proxy_abs_error_vs_unmapped"]) for row in identity_rows]

    sensitivity = _resolution_sensitivity(rows)
    max_identity_rhs_error = max(identity_rhs_errors, default=None)
    max_identity_gamma_error = max(identity_gamma_errors, default=None)
    max_identity_omega_error = max(identity_omega_errors, default=None)
    identity_pass = (
        max_identity_rhs_error is not None
        and max_identity_rhs_error <= float(identity_tolerance)
        and max(identity_gamma_errors, default=0.0) <= float(identity_tolerance)
        and max(identity_omega_errors, default=0.0) <= float(identity_tolerance)
    )
    return {
        "kind": "mapped_velocity_rhs_readiness",
        "case": Path(config).stem,
        "config": config,
        "backend": backend,
        "repeats": int(repeats),
        "state": state,
        "z_variation_norm": float(z_variation_norm),
        "nonlinear_input_weight": float(nonlinear_weight),
        "nonlinear_terms_forced_linear": bool(float(nonlinear_weight) != 0.0),
        "dense_eigen_max_size": int(dense_eigen_max_size),
        "identity_tolerance": float(identity_tolerance),
        "row_count": len(rows),
        "max_identity_rhs_rel_error": max_identity_rhs_error,
        "max_identity_gamma_proxy_abs_error": max_identity_gamma_error,
        "max_identity_omega_proxy_abs_error": max_identity_omega_error,
        "max_mapped_warm_over_unmapped": max(ratios, default=None),
        "resolution_sensitivity": sensitivity,
        "readiness": {
            "identity_map_matches_unmapped": bool(identity_pass),
            "all_proxy_metrics_finite": all(
                math.isfinite(float(row["gamma_proxy"])) and math.isfinite(float(row["omega_proxy"]))
                for row in rows
            ),
        },
        "rows": rows,
        "claim_scope": (
            "Mapped velocity-basis readiness for cached linear RHS assembly. Timings are single-state "
            "JAX warm-call measurements; gamma/omega values are Rayleigh-quotient proxies unless dense "
            "eigen fields are populated. Nonlinear real-space mapped-basis support is not asserted here, "
            "and any input nonlinear term is forced off for this benchmark."
        ),
    }


def _resolution_sensitivity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    labels = sorted({str(row["map_label"]) for row in rows})
    for label in labels:
        label_rows = sorted(
            [row for row in rows if row["map_label"] == label],
            key=lambda row: (int(row["Nl"]), int(row["Nm"])),
        )
        if len(label_rows) < 2:
            continue
        coarse = label_rows[0]
        fine = label_rows[-1]
        out.append(
            {
                "map_label": label,
                "coarse_Nl": int(coarse["Nl"]),
                "coarse_Nm": int(coarse["Nm"]),
                "fine_Nl": int(fine["Nl"]),
                "fine_Nm": int(fine["Nm"]),
                "gamma_proxy_abs_delta": abs(float(coarse["gamma_proxy"]) - float(fine["gamma_proxy"])),
                "omega_proxy_abs_delta": abs(float(coarse["omega_proxy"]) - float(fine["omega_proxy"])),
                "warm_seconds_delta": float(coarse["warm_seconds"]) - float(fine["warm_seconds"]),
            }
        )
    return out


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in CSV_FIELDS} for row in rows])


def _row_for_result(
    *,
    case: str,
    backend: str,
    state: str,
    nl: int,
    nm: int,
    vector_size: int,
    label: str,
    cfg: VelocityMapConfig | None,
    compile_execute_seconds: float,
    warm_seconds: float,
    baseline_warm_seconds: float,
    rhs: jnp.ndarray,
    phi: jnp.ndarray,
    baseline_rhs: jnp.ndarray,
    baseline_gamma: float,
    baseline_omega: float,
    gamma_proxy: float,
    omega_proxy: float,
    dense_freq: tuple[float, float] | None,
    baseline_dense_freq: tuple[float, float] | None,
) -> dict[str, Any]:
    if cfg is None:
        shift = log_parallel = log_perp = 0.0
        parallel_scale = perpendicular_scale = 1.0
    else:
        reg = _map_regularization_floats(cfg)
        shift = float(np.asarray(cfg.parallel_shift))
        log_parallel = float(np.asarray(cfg.parallel_log_scale))
        log_perp = float(np.asarray(cfg.perpendicular_log_scale))
        parallel_scale = reg["parallel_scale"]
        perpendicular_scale = reg["perpendicular_scale"]

    dense_gamma = None if dense_freq is None else dense_freq[0]
    dense_omega = None if dense_freq is None else dense_freq[1]
    baseline_dense_gamma = None if baseline_dense_freq is None else baseline_dense_freq[0]
    baseline_dense_omega = None if baseline_dense_freq is None else baseline_dense_freq[1]
    return {
        "case": case,
        "backend": backend,
        "state": state,
        "Nl": int(nl),
        "Nm": int(nm),
        "vector_size": int(vector_size),
        "map_label": label,
        "parallel_shift": shift,
        "parallel_log_scale": log_parallel,
        "perpendicular_log_scale": log_perp,
        "parallel_scale": parallel_scale,
        "perpendicular_scale": perpendicular_scale,
        "compile_execute_seconds": float(compile_execute_seconds),
        "warm_seconds": float(warm_seconds),
        "baseline_warm_seconds": float(baseline_warm_seconds),
        "warm_over_baseline": _safe_ratio(warm_seconds, baseline_warm_seconds),
        "rhs_norm": float(np.asarray(jnp.linalg.norm(rhs))),
        "phi_norm": float(np.asarray(jnp.linalg.norm(phi))),
        "rhs_rel_error_vs_unmapped": _relative_l2(rhs, baseline_rhs),
        "gamma_proxy": float(gamma_proxy),
        "omega_proxy": float(omega_proxy),
        "gamma_proxy_abs_error_vs_unmapped": abs(float(gamma_proxy) - float(baseline_gamma)),
        "omega_proxy_abs_error_vs_unmapped": abs(float(omega_proxy) - float(baseline_omega)),
        "dense_gamma": dense_gamma,
        "dense_omega": dense_omega,
        "dense_gamma_abs_error_vs_unmapped": (
            None if dense_gamma is None or baseline_dense_gamma is None else abs(dense_gamma - baseline_dense_gamma)
        ),
        "dense_omega_abs_error_vs_unmapped": (
            None if dense_omega is None or baseline_dense_omega is None else abs(dense_omega - baseline_dense_omega)
        ),
    }


def main() -> None:
    args = _parse_args()
    nl_values = _parse_int_list(args.Nl, name="Nl")
    nm_values = _parse_int_list(args.Nm, name="Nm")
    map_specs = _parse_map_specs(args.maps)
    repeats = max(int(args.repeats), 1)

    cfg, _ = load_runtime_from_toml(args.config)
    geom = build_runtime_geometry(cfg)
    grid_cfg = apply_gx_geometry_grid_defaults(geom, cfg.grid)
    grid = build_spectral_grid(grid_cfg)
    term_cfg = build_runtime_term_config(cfg)
    linear_terms = replace(term_cfg, nonlinear=0.0)
    force_electrostatic_fields = _is_static_zero(linear_terms.apar) and _is_static_zero(linear_terms.bpar)
    ky_index, kx_index = _select_nonlinear_mode_indices(
        grid,
        ky_target=args.ky,
        kx_target=args.kx,
        use_dealias_mask=bool(cfg.time.nonlinear_dealias),
    )

    rows: list[dict[str, Any]] = []
    last_z_variation_norm = 0.0
    backend = jax.default_backend()
    case = Path(args.config).stem

    for nl in nl_values:
        for nm in nm_values:
            params = build_runtime_linear_params(cfg, Nm=nm, geom=geom)
            cache = build_linear_cache(grid, geom, params, nl, nm)
            state0 = _build_initial_condition(
                grid,
                geom,
                cfg,
                ky_index=ky_index,
                kx_index=kx_index,
                Nl=nl,
                Nm=nm,
                nspecies=len(cfg.species),
            )
            state = jnp.asarray(state0)
            if args.state == "z_wave":
                state = _inject_z_wave(
                    state,
                    ky_index=int(ky_index),
                    kx_index=int(kx_index),
                    amplitude=float(args.z_wave_amplitude),
                    z_mode=int(args.z_mode),
                )
            last_z_variation_norm = _z_variation_norm(state)

            baseline_fn = jax.jit(
                lambda state_arg: assemble_rhs_cached(
                    state_arg,
                    cache,
                    params,
                    terms=linear_terms,
                    force_electrostatic_fields=force_electrostatic_fields,
                )
            )
            mapped_fn = jax.jit(
                lambda state_arg, map_cfg: assemble_rhs_cached(
                    state_arg,
                    cache,
                    params,
                    terms=linear_terms,
                    velocity_map=map_cfg,
                    force_electrostatic_fields=force_electrostatic_fields,
                )
            )

            baseline_compile, baseline_warm, baseline_out = _time_jitted(baseline_fn, state, repeats=repeats)
            baseline_rhs, baseline_fields = baseline_out
            baseline_gamma, baseline_omega = _rayleigh_frequency_proxy(state, baseline_rhs)
            baseline_dense = _dense_dominant_frequency(
                baseline_fn,
                state,
                max_size=int(args.dense_eigen_max_size),
            )
            rows.append(
                _row_for_result(
                    case=case,
                    backend=backend,
                    state=args.state,
                    nl=nl,
                    nm=nm,
                    vector_size=int(state.size),
                    label="unmapped",
                    cfg=None,
                    compile_execute_seconds=baseline_compile,
                    warm_seconds=baseline_warm,
                    baseline_warm_seconds=baseline_warm,
                    rhs=baseline_rhs,
                    phi=baseline_fields.phi,
                    baseline_rhs=baseline_rhs,
                    baseline_gamma=baseline_gamma,
                    baseline_omega=baseline_omega,
                    gamma_proxy=baseline_gamma,
                    omega_proxy=baseline_omega,
                    dense_freq=baseline_dense,
                    baseline_dense_freq=baseline_dense,
                )
            )

            for label, map_cfg in map_specs:
                mapped_compile, mapped_warm, mapped_out = _time_jitted(
                    mapped_fn,
                    state,
                    map_cfg,
                    repeats=repeats,
                )
                mapped_rhs, mapped_fields = mapped_out
                gamma_proxy, omega_proxy = _rayleigh_frequency_proxy(state, mapped_rhs)
                dense_freq = _dense_dominant_frequency(
                    lambda state_arg, map_cfg=map_cfg: mapped_fn(state_arg, map_cfg),
                    state,
                    max_size=int(args.dense_eigen_max_size),
                )
                rows.append(
                    _row_for_result(
                        case=case,
                        backend=backend,
                        state=args.state,
                        nl=nl,
                        nm=nm,
                        vector_size=int(state.size),
                        label=label,
                        cfg=map_cfg,
                        compile_execute_seconds=mapped_compile,
                        warm_seconds=mapped_warm,
                        baseline_warm_seconds=baseline_warm,
                        rhs=mapped_rhs,
                        phi=mapped_fields.phi,
                        baseline_rhs=baseline_rhs,
                        baseline_gamma=baseline_gamma,
                        baseline_omega=baseline_omega,
                        gamma_proxy=gamma_proxy,
                        omega_proxy=omega_proxy,
                        dense_freq=dense_freq,
                        baseline_dense_freq=baseline_dense,
                    )
                )
                print(
                    f"Nl={nl} Nm={nm} map={label} "
                    f"warm={mapped_warm:.6e}s ratio={_safe_ratio(mapped_warm, baseline_warm):.3f} "
                    f"rhs_rel={rows[-1]['rhs_rel_error_vs_unmapped']:.3e} "
                    f"gamma={gamma_proxy:.6e} omega={omega_proxy:.6e}"
                )

    summary = _build_summary(
        rows,
        config=str(args.config),
        backend=backend,
        repeats=repeats,
        state=args.state,
        z_variation_norm=last_z_variation_norm,
        nonlinear_weight=float(term_cfg.nonlinear),
        identity_tolerance=float(args.identity_tolerance),
        dense_eigen_max_size=int(args.dense_eigen_max_size),
    )
    _write_json(args.out_json, summary)
    _write_csv(args.out_csv, rows)
    print(f"saved {args.out_json}")
    print(f"saved {args.out_csv}")


if __name__ == "__main__":
    main()
