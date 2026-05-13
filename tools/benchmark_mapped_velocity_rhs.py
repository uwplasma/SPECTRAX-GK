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

from spectraxgk.config import GridConfig
from spectraxgk.geometry import SAlphaGeometry, apply_gx_geometry_grid_defaults
from spectraxgk.grids import build_spectral_grid, select_ky_grid
from spectraxgk.io import load_runtime_from_toml
from spectraxgk.linear import LinearParams, LinearTerms, build_linear_cache
from spectraxgk.linear_krylov import dominant_eigenpair
from spectraxgk.runtime import (
    _build_initial_condition,
    _select_nonlinear_mode_indices,
    build_runtime_geometry,
    build_runtime_linear_params,
    build_runtime_term_config,
)
from spectraxgk.terms.assembly import _is_static_zero, assemble_rhs_cached
from spectraxgk.terms.config import TermConfig
from spectraxgk.velocity_maps import VelocityMapConfig, map_regularization


DEFAULT_MAP_SPECS = (
    "identity:0.0:0.0:0.0",
    "parallel_shift:0.15:0.0:0.0",
    "parallel_scale:0.0:-0.08:0.0",
    "perp_scale:0.0:0.0:0.06",
)
KRYLOV_RETURNED_VECTOR_RESIDUAL_TOLERANCE = 1.0e-4

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
    parser.add_argument(
        "--eigen-scorecard-max-size",
        type=int,
        default=128,
        help=(
            "Build a tiny actual assembled-RHS dense-eigen scorecard up to this matrix size; "
            "0 disables the scorecard."
        ),
    )
    parser.add_argument(
        "--krylov-scorecard-max-size",
        type=int,
        default=128,
        help=(
            "Build a tiny mapped-RHS Krylov scorecard up to this vector size; "
            "0 disables the scorecard."
        ),
    )
    parser.add_argument("--krylov-scorecard-dim", type=int, default=24)
    parser.add_argument("--krylov-scorecard-restarts", type=int, default=1)
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
    return (
        parts[0],
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


def _dense_operator_matrix(
    rhs_fn: Callable[[jnp.ndarray], tuple[jnp.ndarray, Any]],
    template: jnp.ndarray,
    *,
    max_size: int,
) -> jnp.ndarray | None:
    """Materialize a tiny matrix-free RHS operator by applying it to basis vectors."""

    n = int(template.size)
    if int(max_size) <= 0 or n > int(max_size):
        return None

    eye = jnp.eye(n, dtype=template.dtype)

    def _apply(flat_state: jnp.ndarray) -> jnp.ndarray:
        rhs, _fields = rhs_fn(flat_state.reshape(template.shape))
        return rhs.reshape(-1)

    columns = jax.vmap(_apply)(eye)
    return columns.T


def _dense_dominant_frequency(
    rhs_fn: Callable[[jnp.ndarray], tuple[jnp.ndarray, Any]],
    template: jnp.ndarray,
    *,
    max_size: int,
) -> tuple[float, float] | None:
    """Return dominant dense eigen gamma/omega for tiny states, otherwise ``None``."""

    matrix = _dense_operator_matrix(rhs_fn, template, max_size=max_size)
    if matrix is None:
        return None
    eigvals = jnp.linalg.eigvals(matrix)
    eig = eigvals[jnp.argmax(jnp.real(eigvals))]
    return float(np.asarray(jnp.real(eig))), float(np.asarray(-jnp.imag(eig)))


def _dominant_frequency_from_matrix(matrix: np.ndarray) -> tuple[float, float]:
    eigvals = np.linalg.eigvals(np.asarray(matrix))
    eig = eigvals[int(np.argmax(np.real(eigvals)))]
    return float(np.real(eig)), float(-np.imag(eig))


def _spectral_radius(matrix: np.ndarray) -> float:
    eigvals = np.linalg.eigvals(np.asarray(matrix))
    return float(np.max(np.abs(eigvals)))


def _relative_matrix_error(lhs: np.ndarray, rhs: np.ndarray) -> float:
    rhs_norm = float(np.linalg.norm(rhs))
    diff_norm = float(np.linalg.norm(lhs - rhs))
    if rhs_norm == 0.0:
        return 0.0 if diff_norm == 0.0 else math.inf
    return diff_norm / rhs_norm


def _is_identity_velocity_map(cfg: VelocityMapConfig) -> bool:
    return (
        float(np.asarray(cfg.parallel_shift)) == 0.0
        and float(np.asarray(cfg.parallel_log_scale)) == 0.0
        and float(np.asarray(cfg.perpendicular_log_scale)) == 0.0
    )


def _dedupe_scorecard_maps(
    map_specs: list[tuple[str, VelocityMapConfig]],
) -> list[tuple[str, VelocityMapConfig | None]]:
    specs: list[tuple[str, VelocityMapConfig | None]] = [("unmapped", None), ("identity", VelocityMapConfig())]
    seen = {"unmapped", "identity"}
    for label, cfg in map_specs:
        if label in seen or _is_identity_velocity_map(cfg):
            continue
        specs.append((label, cfg))
        seen.add(label)
    return specs


def _build_tiny_eigen_scorecard(
    map_specs: list[tuple[str, VelocityMapConfig]],
    *,
    max_size: int,
    identity_tolerance: float,
) -> dict[str, Any] | None:
    """Build a compact dense-eigen artifact from the actual mapped RHS assembly."""

    nl, nm = 2, 3
    grid_full = build_spectral_grid(GridConfig(Nx=1, Ny=4, Nz=4, Lx=6.28, Ly=6.28))
    grid = select_ky_grid(grid_full, 1)
    geom = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778, drift_scale=1.0)
    params = LinearParams(
        R_over_Ln=0.8,
        R_over_LTi=2.49,
        R_over_LTe=0.0,
        omega_d_scale=1.0,
        omega_star_scale=1.0,
        rho_star=1.0,
        kpar_scale=float(geom.gradpar()),
        nu=0.0,
    )
    terms = TermConfig(
        streaming=1.0,
        mirror=0.0,
        curvature=1.0,
        gradb=1.0,
        diamagnetic=0.0,
        collisions=0.0,
        hypercollisions=0.0,
        hyperdiffusion=0.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
    )
    template = jnp.zeros((nl, nm, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64)
    if int(max_size) <= 0 or int(template.size) > int(max_size):
        return None

    cache = build_linear_cache(grid, geom, params, nl, nm)

    def _rhs(state: jnp.ndarray, map_cfg: VelocityMapConfig | None) -> tuple[jnp.ndarray, Any]:
        return assemble_rhs_cached(
            state,
            cache,
            params,
            terms=terms,
            use_custom_vjp=False,
            velocity_map=map_cfg,
            force_electrostatic_fields=True,
        )

    matrices: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for label, map_cfg in _dedupe_scorecard_maps(map_specs):
        matrix = _dense_operator_matrix(
            lambda state_arg, map_cfg=map_cfg: _rhs(state_arg, map_cfg),
            template,
            max_size=max_size,
        )
        if matrix is None:  # pragma: no cover - guarded by the template size check above.
            return None
        matrices[label] = np.asarray(matrix)

    baseline_matrix = matrices["unmapped"]
    baseline_gamma, baseline_omega = _dominant_frequency_from_matrix(baseline_matrix)
    for label, matrix in matrices.items():
        gamma, omega = _dominant_frequency_from_matrix(matrix)
        rows.append(
            {
                "map_label": label,
                "matrix_size": int(template.size),
                "Nl": nl,
                "Nm": nm,
                "Nz": int(grid.z.size),
                "dense_gamma": gamma,
                "dense_omega": omega,
                "dense_gamma_abs_error_vs_unmapped": abs(gamma - baseline_gamma),
                "dense_omega_abs_error_vs_unmapped": abs(omega - baseline_omega),
                "spectral_radius": _spectral_radius(matrix),
                "matrix_rel_error_vs_unmapped": _relative_matrix_error(matrix, baseline_matrix),
            }
        )

    identity_rows = [row for row in rows if row["map_label"] == "identity"]
    max_identity_matrix_error = max(
        (float(row["matrix_rel_error_vs_unmapped"]) for row in identity_rows),
        default=None,
    )
    max_identity_gamma_error = max(
        (float(row["dense_gamma_abs_error_vs_unmapped"]) for row in identity_rows),
        default=None,
    )
    max_identity_omega_error = max(
        (float(row["dense_omega_abs_error_vs_unmapped"]) for row in identity_rows),
        default=None,
    )
    identity_pass = (
        max_identity_matrix_error is not None
        and max_identity_matrix_error <= float(identity_tolerance)
        and max_identity_gamma_error is not None
        and max_identity_gamma_error <= float(identity_tolerance)
        and max_identity_omega_error is not None
        and max_identity_omega_error <= float(identity_tolerance)
    )
    return {
        "kind": "mapped_velocity_rhs_tiny_dense_eigen_scorecard",
        "matrix_source": "spectraxgk.terms.assembly.assemble_rhs_cached",
        "matrix_size": int(template.size),
        "max_size": int(max_size),
        "identity_tolerance": float(identity_tolerance),
        "max_identity_matrix_rel_error": max_identity_matrix_error,
        "max_identity_dense_gamma_abs_error": max_identity_gamma_error,
        "max_identity_dense_omega_abs_error": max_identity_omega_error,
        "readiness": {
            "identity_dense_operator_matches_unmapped": bool(identity_pass),
            "all_dense_metrics_finite": all(
                math.isfinite(float(row["dense_gamma"]))
                and math.isfinite(float(row["dense_omega"]))
                and math.isfinite(float(row["spectral_radius"]))
                and math.isfinite(float(row["matrix_rel_error_vs_unmapped"]))
                for row in rows
            ),
        },
        "rows": rows,
        "claim_scope": (
            "Tiny actual assembled-RHS dense matrix materialized from matrix-free basis-vector applies. "
            "This validates mapped identity eigen/operator consistency on a compact linear grid; it is "
            "not a production-size spectrum."
        ),
    }


def _tiny_eigen_case() -> tuple[
    jnp.ndarray,
    Any,
    LinearParams,
    TermConfig,
    LinearTerms,
]:
    nl, nm = 2, 3
    grid_full = build_spectral_grid(GridConfig(Nx=1, Ny=4, Nz=4, Lx=6.28, Ly=6.28))
    grid = select_ky_grid(grid_full, 1)
    geom = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778, drift_scale=1.0)
    params = LinearParams(
        R_over_Ln=0.8,
        R_over_LTi=2.49,
        R_over_LTe=0.0,
        omega_d_scale=1.0,
        omega_star_scale=1.0,
        rho_star=1.0,
        kpar_scale=float(geom.gradpar()),
        nu=0.0,
    )
    terms = LinearTerms(
        streaming=1.0,
        mirror=0.0,
        curvature=1.0,
        gradb=1.0,
        diamagnetic=0.0,
        collisions=0.0,
        hypercollisions=0.0,
        hyperdiffusion=0.0,
        end_damping=0.0,
        apar=0.0,
        bpar=0.0,
    )
    term_cfg = TermConfig(
        streaming=terms.streaming,
        mirror=terms.mirror,
        curvature=terms.curvature,
        gradb=terms.gradb,
        diamagnetic=terms.diamagnetic,
        collisions=terms.collisions,
        hypercollisions=terms.hypercollisions,
        hyperdiffusion=terms.hyperdiffusion,
        end_damping=terms.end_damping,
        apar=terms.apar,
        bpar=terms.bpar,
    )
    template = jnp.zeros((nl, nm, grid.ky.size, grid.kx.size, grid.z.size), dtype=jnp.complex64)
    cache = build_linear_cache(grid, geom, params, nl, nm)
    return template, cache, params, term_cfg, terms


def _nearest_eigen_abs_error(eig: complex, matrix: np.ndarray) -> float:
    eigvals = np.linalg.eigvals(np.asarray(matrix))
    return float(np.min(np.abs(eigvals - eig)))


def _operator_residual_rel(matrix: np.ndarray, eig: complex, vec: np.ndarray) -> float:
    flat = np.asarray(vec).reshape(-1)
    mat = np.asarray(matrix)
    residual = mat @ flat - complex(eig) * flat
    denominator = (float(np.linalg.norm(mat, ord=2)) + abs(complex(eig))) * float(np.linalg.norm(flat))
    if denominator == 0.0:
        return 0.0 if float(np.linalg.norm(residual)) == 0.0 else math.inf
    return float(np.linalg.norm(residual) / denominator)


def _build_tiny_krylov_scorecard(
    map_specs: list[tuple[str, VelocityMapConfig]],
    *,
    max_size: int,
    identity_tolerance: float,
    krylov_dim: int,
    restarts: int,
) -> dict[str, Any] | None:
    """Exercise the mapped RHS through the actual Krylov eigen-solver path."""

    template, cache, params, term_cfg, terms = _tiny_eigen_case()
    if int(max_size) <= 0 or int(template.size) > int(max_size):
        return None

    rng = np.random.default_rng(12345)
    v0_np = rng.normal(size=template.shape) + 1j * rng.normal(size=template.shape)
    v0 = jnp.asarray(v0_np, dtype=template.dtype)
    krylov_dim_use = min(max(int(krylov_dim), 1), int(template.size))
    restarts_use = max(int(restarts), 1)

    rows: list[dict[str, Any]] = []
    baseline_row: dict[str, Any] | None = None
    for label, map_cfg in _dedupe_scorecard_maps(map_specs):
        params_use = params if map_cfg is None else replace(params, velocity_map=map_cfg)

        def _rhs(state_arg: jnp.ndarray, map_cfg=map_cfg) -> tuple[jnp.ndarray, Any]:
            return assemble_rhs_cached(
                state_arg,
                cache,
                params,
                terms=term_cfg,
                use_custom_vjp=False,
                velocity_map=map_cfg,
                force_electrostatic_fields=True,
            )

        matrix = _dense_operator_matrix(_rhs, template, max_size=max_size)
        if matrix is None:  # pragma: no cover - guarded by the template size check above.
            return None
        matrix_np = np.asarray(matrix)
        eig, vec = dominant_eigenpair(
            v0,
            cache,
            params_use,
            terms=terms,
            method="arnoldi",
            krylov_dim=krylov_dim_use,
            restarts=restarts_use,
            omega_min_factor=-1.0e6,
            omega_cap_factor=1.0e6,
        )
        eig_complex = complex(np.asarray(eig))
        vec_np = np.asarray(vec)
        row = {
            "map_label": label,
            "matrix_size": int(template.size),
            "krylov_dim": int(krylov_dim_use),
            "restarts": int(restarts_use),
            "krylov_gamma": float(eig_complex.real),
            "krylov_omega": float(-eig_complex.imag),
            "nearest_dense_eigen_abs_error": _nearest_eigen_abs_error(eig_complex, matrix_np),
            "operator_residual_rel": _operator_residual_rel(matrix_np, eig_complex, vec_np),
            "vector_norm": float(np.linalg.norm(vec_np.reshape(-1))),
        }
        if label == "unmapped":
            baseline_row = row
            row["krylov_eigen_abs_error_vs_unmapped"] = 0.0
        elif baseline_row is not None:
            baseline_eig = complex(
                float(baseline_row["krylov_gamma"]),
                -float(baseline_row["krylov_omega"]),
            )
            row["krylov_eigen_abs_error_vs_unmapped"] = abs(eig_complex - baseline_eig)
        else:
            row["krylov_eigen_abs_error_vs_unmapped"] = None
        rows.append(row)

    identity_rows = [row for row in rows if row["map_label"] == "identity"]
    residuals = [float(row["operator_residual_rel"]) for row in rows]
    nearest_errors = [float(row["nearest_dense_eigen_abs_error"]) for row in rows]
    identity_errors = [
        float(row["krylov_eigen_abs_error_vs_unmapped"])
        for row in identity_rows
        if row.get("krylov_eigen_abs_error_vs_unmapped") is not None
    ]
    return {
        "kind": "mapped_velocity_rhs_tiny_krylov_scorecard",
        "matrix_source": "spectraxgk.terms.assembly.assemble_rhs_cached",
        "krylov_source": "spectraxgk.linear_krylov.dominant_eigenpair",
        "matrix_size": int(template.size),
        "max_size": int(max_size),
        "identity_tolerance": float(identity_tolerance),
        "operator_residual_tolerance": KRYLOV_RETURNED_VECTOR_RESIDUAL_TOLERANCE,
        "residual_operator": "same materialized compact operator used for dense eigenvalue matching",
        "krylov_dim": int(krylov_dim_use),
        "restarts": int(restarts_use),
        "max_operator_residual_rel": max(residuals, default=None),
        "max_nearest_dense_eigen_abs_error": max(nearest_errors, default=None),
        "max_identity_krylov_eigen_abs_error": max(identity_errors, default=None),
        "readiness": {
            "all_krylov_metrics_finite": all(
                math.isfinite(float(row["krylov_gamma"]))
                and math.isfinite(float(row["krylov_omega"]))
                and math.isfinite(float(row["nearest_dense_eigen_abs_error"]))
                and math.isfinite(float(row["operator_residual_rel"]))
                for row in rows
            ),
            "all_krylov_eigenvalues_close_to_dense": all(value <= 1.0e-4 for value in nearest_errors),
            "all_krylov_returned_vectors_validate_materialized_operator": all(
                value <= KRYLOV_RETURNED_VECTOR_RESIDUAL_TOLERANCE for value in residuals
            ),
            "identity_krylov_matches_unmapped": bool(identity_errors)
            and max(identity_errors) <= float(identity_tolerance),
        },
        "rows": rows,
        "claim_scope": (
            "Tiny actual mapped-RHS Krylov scorecard using the same LinearParams.velocity_map path "
            "used by runtime Krylov solves. Eigenvalues are compared against the nearest dense eigenvalue "
            "of the materialized compact operator, and returned Krylov vectors are checked as right "
            "eigenvectors against that same compact operator. This is an eigen-solver plumbing gate, "
            "not a production-size spectrum."
        ),
    }


def _map_regularization_floats(cfg: VelocityMapConfig) -> dict[str, float]:
    reg = map_regularization(cfg)
    return {key: float(np.asarray(value)) for key, value in reg.items()}


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
    eigen_scorecard: dict[str, Any] | None,
    krylov_scorecard: dict[str, Any] | None,
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
    max_identity_rhs_error = max(identity_rhs_errors, default=None)
    max_identity_gamma_error = max(identity_gamma_errors, default=None)
    max_identity_omega_error = max(identity_omega_errors, default=None)
    identity_pass = (
        max_identity_rhs_error is not None
        and max_identity_rhs_error <= float(identity_tolerance)
        and max(identity_gamma_errors, default=0.0) <= float(identity_tolerance)
        and max(identity_omega_errors, default=0.0) <= float(identity_tolerance)
    )
    eigen_identity_pass = None
    if eigen_scorecard is not None:
        eigen_identity_pass = bool(
            eigen_scorecard["readiness"]["identity_dense_operator_matches_unmapped"]
        )
    krylov_identity_pass = None
    krylov_eigen_pass = None
    krylov_vector_residual_pass = None
    if krylov_scorecard is not None:
        krylov_identity_pass = bool(
            krylov_scorecard["readiness"]["identity_krylov_matches_unmapped"]
        )
        krylov_eigen_pass = bool(
            krylov_scorecard["readiness"]["all_krylov_eigenvalues_close_to_dense"]
        )
        krylov_vector_residual_pass = bool(
            krylov_scorecard["readiness"][
                "all_krylov_returned_vectors_validate_materialized_operator"
            ]
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
        "resolution_sensitivity": _resolution_sensitivity(rows),
        "eigen_scorecard": eigen_scorecard,
        "krylov_scorecard": krylov_scorecard,
        "readiness": {
            "identity_map_matches_unmapped": bool(identity_pass),
            "all_proxy_metrics_finite": all(
                math.isfinite(float(row["gamma_proxy"])) and math.isfinite(float(row["omega_proxy"]))
                for row in rows
            ),
            "tiny_dense_eigen_scorecard_available": eigen_scorecard is not None,
            "identity_dense_operator_matches_unmapped": eigen_identity_pass,
            "tiny_krylov_scorecard_available": krylov_scorecard is not None,
            "identity_krylov_matches_unmapped": krylov_identity_pass,
            "all_krylov_eigenvalues_close_to_dense": krylov_eigen_pass,
            "all_krylov_returned_vectors_validate_materialized_operator": krylov_vector_residual_pass,
        },
        "rows": rows,
        "claim_scope": (
            "Mapped velocity-basis readiness for cached linear RHS assembly. Timings are single-state "
            "JAX warm-call measurements; gamma/omega values are Rayleigh-quotient proxies unless dense "
            "eigen fields are populated. The eigen_scorecard field adds a tiny actual assembled-RHS dense "
            "operator check for mapped identity consistency. The krylov_scorecard field additionally runs "
            "the mapped operator through the production Krylov eigen wrapper on the same compact grid and "
            "compares Krylov eigenvalues with the nearest dense eigenvalues, then validates returned-vector "
            "residuals against that same materialized operator. Nonlinear real-space mapped-basis support "
            "is not asserted here, and any input nonlinear term is forced off for this benchmark."
        ),
    }


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

    eigen_scorecard = _build_tiny_eigen_scorecard(
        map_specs,
        max_size=int(args.eigen_scorecard_max_size),
        identity_tolerance=float(args.identity_tolerance),
    )
    krylov_scorecard = _build_tiny_krylov_scorecard(
        map_specs,
        max_size=int(args.krylov_scorecard_max_size),
        identity_tolerance=float(args.identity_tolerance),
        krylov_dim=int(args.krylov_scorecard_dim),
        restarts=int(args.krylov_scorecard_restarts),
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
        eigen_scorecard=eigen_scorecard,
        krylov_scorecard=krylov_scorecard,
    )
    _write_json(args.out_json, summary)
    _write_csv(args.out_csv, rows)
    print(f"saved {args.out_json}")
    print(f"saved {args.out_csv}")


if __name__ == "__main__":
    main()
