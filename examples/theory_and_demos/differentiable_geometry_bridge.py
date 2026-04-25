#!/usr/bin/env python3
"""Differentiable geometry bridge validation and inverse-design demo.

The example validates the Phase-A geometry contract used by
``vmec_jax -> booz_xform_jax -> SPECTRAX-GK`` workflows:

1. a boundary-parameter vector produces solver-ready flux-tube geometry arrays;
2. SPECTRAX-GK differentiates geometry observables through that in-memory
   contract;
3. autodiff sensitivities are checked against central finite differences;
4. a small two-parameter inverse problem recovers target geometry observables
   and reports local UQ covariance.

When ``vmec_jax`` is available, the figure also includes an independent
boundary-aspect derivative through the real ``vmec_jax`` boundary Fourier API.
The Boozer bridge is audited by checking that ``booz_xform_jax`` exposes the
JAX-native functional transform API used by high-fidelity follow-up workflows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from spectraxgk.autodiff_validation import covariance_diagnostics
from spectraxgk.geometry.differentiable import (
    discover_differentiable_geometry_backends,
    finite_difference_jacobian,
    flux_tube_geometry_from_mapping,
    flux_tube_geometry_observables,
    geometry_inverse_design_report,
    geometry_observable_names,
    geometry_sensitivity_report,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PNG = REPO_ROOT / "docs" / "_static" / "differentiable_geometry_bridge.png"
DEFAULT_JSON = REPO_ROOT / "docs" / "_static" / "differentiable_geometry_bridge.json"


def _mapping_from_boundary_params(params: jnp.ndarray, *, ntheta: int = 96) -> dict[str, Any]:
    """Analytic field-line map with VMEC-like boundary controls.

    ``params[0]`` controls a mirror/ripple amplitude and ``params[1]`` controls
    shaping/elongation. The returned arrays are exactly the solver-ready
    contract that a high-fidelity ``vmec_jax`` / ``booz_xform_jax`` pipeline
    must supply.
    """

    theta = jnp.linspace(-jnp.pi, jnp.pi, int(ntheta), endpoint=False)
    ripple, elongation = params
    ones = jnp.ones_like(theta)
    shear = 0.35 + 0.08 * elongation
    field_line_shift = shear * theta - 0.18 * elongation * jnp.sin(theta)
    bmag = 1.0 + ripple * jnp.cos(theta) + 0.025 * jnp.cos(2.0 * theta)
    cv = jnp.cos(theta) + field_line_shift * jnp.sin(theta)
    cv0 = -shear * jnp.sin(theta)
    gradpar = (0.82 + 0.015 * ripple) * ones
    return {
        "theta": theta,
        "gradpar": gradpar,
        "bmag": bmag,
        "bgrad": -ripple * jnp.sin(theta),
        "gds2": 1.0 + field_line_shift**2,
        "gds21": -shear * field_line_shift,
        "gds22": (shear * shear) * ones,
        "cvdrift": cv,
        "gbdrift": cv,
        "cvdrift0": cv0,
        "gbdrift0": cv0,
        "jacobian": 1.0 / (gradpar * bmag),
        "grho": ones,
        "q": 1.4 + 0.04 * elongation,
        "s_hat": shear,
        "epsilon": ripple,
        "R0": 1.0,
        "nfp": 5,
    }


def _observable_fn(params: jnp.ndarray) -> jnp.ndarray:
    geom = flux_tube_geometry_from_mapping(
        _mapping_from_boundary_params(params),
        source_model="vmec_jax:demo-contract",
        validate_finite=False,
    )
    return flux_tube_geometry_observables(geom)


def _inverse_design(
    initial: jnp.ndarray,
    target_params: jnp.ndarray,
    *,
    observable_indices: tuple[int, int] = (1, 2),
    nsteps: int = 8,
    damping: float = 2.0e-6,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    target = _observable_fn(target_params)[jnp.asarray(observable_indices)]
    params = jnp.asarray(initial, dtype=jnp.float64)
    history: list[dict[str, Any]] = []
    for step in range(int(nsteps) + 1):
        obs = _observable_fn(params)[jnp.asarray(observable_indices)]
        residual = obs - target
        objective = 0.5 * jnp.dot(residual, residual)
        history.append(
            {
                "step": step,
                "params": np.asarray(params).tolist(),
                "objective": float(objective),
                "residual_norm": float(jnp.linalg.norm(residual)),
            }
        )
        if step == int(nsteps):
            break
        jac = jax.jacfwd(lambda p: _observable_fn(p)[jnp.asarray(observable_indices)])(params)
        normal = jac.T @ jac + float(damping) * jnp.eye(int(params.shape[0]))
        delta = jnp.linalg.solve(normal, jac.T @ residual)
        params = params - delta
    return np.asarray(params), history


def _vmec_boundary_panel(params: jnp.ndarray) -> dict[str, Any]:
    info = discover_differentiable_geometry_backends()
    if not info.get("vmec_jax_boundary_api_available", False):
        return {
            "available": False,
            "aspect": None,
            "grad_ad": None,
            "grad_fd": None,
            "max_abs_ad_fd_error": None,
        }

    import vmec_jax as vj  # type: ignore[import-not-found]

    modes = vj.vmec_mode_table(2, 0)
    grid = vj.make_angle_grid(96, 1, 1)
    basis = vj.build_helical_basis(modes, grid)

    def aspect_fn(x: jnp.ndarray) -> jnp.ndarray:
        ripple, elongation = x
        r0 = 1.0
        minor = 0.22 * (1.0 + 0.5 * ripple)
        r_cos = jnp.zeros(modes.K, dtype=jnp.float64).at[0].set(r0).at[1].set(minor)
        z_sin = jnp.zeros(modes.K, dtype=jnp.float64).at[1].set(minor * (1.0 + elongation))
        zeros = jnp.zeros_like(r_cos)
        boundary = vj.BoundaryCoeffs(R_cos=r_cos, R_sin=zeros, Z_cos=zeros, Z_sin=z_sin)
        return vj.boundary_aspect_ratio(boundary, basis)

    grad_ad = jax.grad(aspect_fn)(params)
    grad_fd = finite_difference_jacobian(lambda x: jnp.asarray([aspect_fn(x)]), params, step=2.0e-5)[0]
    return {
        "available": True,
        "aspect": float(aspect_fn(params)),
        "grad_ad": np.asarray(grad_ad).tolist(),
        "grad_fd": np.asarray(grad_fd).tolist(),
        "max_abs_ad_fd_error": float(np.max(np.abs(np.asarray(grad_ad - grad_fd)))),
    }


def _covariance_ellipse(cov: np.ndarray, center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 0.0)
    angle = np.linspace(0.0, 2.0 * np.pi, 200)
    circle = np.vstack([np.cos(angle), np.sin(angle)])
    ellipse = eigvecs @ (np.sqrt(eigvals)[:, None] * circle) + center[:, None]
    return ellipse[0], ellipse[1]


def make_figure(payload: dict[str, Any], out_png: Path) -> None:
    names = payload["observable_names"]
    sensitivity = np.asarray(payload["sensitivity"]["jacobian_ad"], dtype=float)
    history = payload["inverse_history"]
    params_hist = np.asarray([row["params"] for row in history], dtype=float)
    obj = np.asarray([row["objective"] for row in history], dtype=float)
    cov = np.asarray(payload["uq"]["covariance"], dtype=float)
    final_params = np.asarray(payload["inverse_final_params"], dtype=float)
    target_params = np.asarray(payload["target_params"], dtype=float)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.4), constrained_layout=True)

    theta = np.linspace(0.0, 2.0 * np.pi, 240)
    for params, label, color in (
        (np.asarray(payload["initial_params"], dtype=float), "initial", "#657786"),
        (target_params, "target", "#d1495b"),
        (final_params, "recovered", "#00798c"),
    ):
        minor = 0.22 * (1.0 + 0.5 * params[0])
        elong = 1.0 + params[1]
        r = 1.0 + minor * np.cos(theta)
        z = minor * elong * np.sin(theta)
        axes[0, 0].plot(r, z, label=label, color=color, linewidth=2.0)
    axes[0, 0].set_aspect("equal", adjustable="box")
    axes[0, 0].set_xlabel("R / R0")
    axes[0, 0].set_ylabel("Z / R0")
    axes[0, 0].set_title("Boundary controls")
    axes[0, 0].legend(frameon=True, fontsize=9)

    scale = np.maximum(np.max(np.abs(sensitivity), axis=1, keepdims=True), 1.0e-14)
    im = axes[0, 1].imshow(sensitivity / scale, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
    axes[0, 1].set_xticks([0, 1], ["ripple", "elongation"])
    axes[0, 1].set_yticks(np.arange(len(names)), names, fontsize=8)
    axes[0, 1].set_title("Normalized AD sensitivity map")
    fig.colorbar(im, ax=axes[0, 1], shrink=0.86, label="row-normalized derivative")

    axes[1, 0].semilogy(np.arange(obj.size), np.maximum(obj, 1.0e-30), marker="o", color="#edae49")
    axes[1, 0].set_xlabel("Gauss-Newton step")
    axes[1, 0].set_ylabel("geometry objective")
    axes[1, 0].set_title("Two-parameter inverse design")
    axp = axes[1, 0].twinx()
    axp.plot(params_hist[:, 0], color="#d1495b", linestyle="--", label="ripple")
    axp.plot(params_hist[:, 1], color="#00798c", linestyle=":", label="elongation")
    axp.set_ylabel("parameters")
    axp.legend(frameon=True, fontsize=8, loc="upper right")

    xell, yell = _covariance_ellipse(cov, final_params)
    x_scale = 1.0e6
    y_scale = 1.0e6
    axes[1, 1].plot(
        (xell - final_params[0]) * x_scale,
        (yell - final_params[1]) * y_scale,
        color="#2a9d8f",
        linewidth=2.0,
        label="1 sigma covariance",
    )
    axes[1, 1].scatter(
        [(target_params[0] - final_params[0]) * x_scale],
        [(target_params[1] - final_params[1]) * y_scale],
        facecolors="none",
        edgecolors="#d1495b",
        linewidths=2.0,
        label="target",
        zorder=4,
    )
    axes[1, 1].scatter([0.0], [0.0], color="#00798c", label="recovered", zorder=3)
    axes[1, 1].set_xlabel(r"ripple $-$ recovered [$10^{-6}$]")
    axes[1, 1].set_ylabel(r"elongation $-$ recovered [$10^{-6}$]")
    axes[1, 1].set_title("Local UQ covariance")
    axes[1, 1].legend(frameon=True, fontsize=9)

    fig.suptitle("Differentiable geometry bridge: AD, inverse design, and UQ", fontsize=15)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-png", type=Path, default=DEFAULT_PNG)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_JSON)
    args = parser.parse_args(argv)

    initial = jnp.asarray([0.035, 0.12], dtype=jnp.float64)
    target_params = jnp.asarray([0.085, 0.34], dtype=jnp.float64)
    final_params, history = _inverse_design(initial, target_params)

    sensitivity = geometry_sensitivity_report(_mapping_from_boundary_params, final_params, fd_step=2.0e-5)
    target_obs = np.asarray(_observable_fn(target_params))[[1, 2]]
    final_obs = np.asarray(_observable_fn(jnp.asarray(final_params)))[[1, 2]]
    residual = final_obs - target_obs
    jac = np.asarray(jax.jacfwd(lambda p: _observable_fn(p)[jnp.asarray([1, 2])])(jnp.asarray(final_params)))
    uq = covariance_diagnostics(jac, residual, regularization=1.0e-8)
    workflow_report = geometry_inverse_design_report(
        _mapping_from_boundary_params,
        initial,
        jnp.asarray(target_obs),
        observable_indices=(1, 2),
        max_steps=8,
        damping=2.0e-6,
        fd_step=2.0e-5,
    )
    backend_info = discover_differentiable_geometry_backends()
    vmec_boundary = _vmec_boundary_panel(jnp.asarray(final_params))

    payload: dict[str, Any] = {
        "backend_info": backend_info,
        "booz_xform_jax_api_available": bool(backend_info.get("booz_xform_jax_api_available", False)),
        "vmec_boundary": vmec_boundary,
        "observable_names": list(geometry_observable_names()),
        "initial_params": np.asarray(initial).tolist(),
        "target_params": np.asarray(target_params).tolist(),
        "inverse_final_params": np.asarray(final_params).tolist(),
        "inverse_history": history,
        "inverse_target_observables": target_obs.tolist(),
        "inverse_final_observables": final_obs.tolist(),
        "inverse_observable_residual": residual.tolist(),
        "sensitivity": sensitivity,
        "uq": uq,
        "geometry_inverse_design_report": workflow_report,
        "notes": (
            "This is a bounded differentiable-geometry bridge validation. The high-fidelity VMEC/Boozer "
            "pipeline must provide the same solver-ready field-line arrays; this artifact validates the "
            "JAX tracing, AD-vs-FD sensitivities, inverse recovery, and UQ machinery at that contract boundary."
        ),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    make_figure(payload, args.out_png)
    print(f"Wrote {args.out_png}")
    print(f"Wrote {args.out_json}")
    print(f"max AD/FD relative error: {sensitivity['max_rel_ad_fd_error']:.3e}")
    print(f"final residual norm: {history[-1]['residual_norm']:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
