#!/usr/bin/env python3
"""Build the backend-free stellarator objective-portfolio validation panel."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from spectraxgk.plotting import set_plot_style  # noqa: E402
from spectraxgk.objectives.stellarator_portfolio import (  # noqa: E402
    aggregate_objective_portfolio,
    portfolio_objective_weight_vector,
    portfolio_sample_weight_tensor,
    validate_objective_portfolio_contract,
)


DEFAULT_OUT = ROOT / "docs" / "_static" / "stellarator_objective_portfolio_gate.png"
OBJECTIVE_NAMES = ("growth_proxy", "quasilinear_flux_proxy", "nonlinear_window_proxy")


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
        return value if np.isfinite(value) else None
    return value


def _fixture_rows(params: jnp.ndarray) -> jnp.ndarray:
    surface = jnp.asarray([0.25, 0.55])[:, None, None]
    alpha = jnp.asarray([0.0, 0.65])[None, :, None]
    ky = jnp.asarray([0.25, 0.45, 0.75])[None, None, :]
    ripple = params[0]
    shear = params[1]
    pressure = params[2]

    growth = 0.030 + 0.19 * ripple**2 + 0.035 * ky + 0.018 * jnp.cos(alpha + shear) + 0.010 * surface
    ql_flux = growth * (0.80 + 0.35 * pressure**2 + 0.12 * ky) / (0.55 + ky)
    window_flux = ql_flux * (1.0 + 0.08 * jnp.sin(alpha) + 0.18 * surface) + 0.015 * ripple * pressure
    return jnp.stack([growth, ql_flux, window_flux], axis=-1)


def stellarator_objective_portfolio_gate_payload(
    *,
    fd_step: float = 1.0e-3,
    rtol: float = 2.0e-3,
    atol: float = 2.0e-5,
) -> dict[str, object]:
    """Return a deterministic AD/JVP/FD validation payload for the reducer."""

    params = jnp.asarray([0.34, -0.21, 0.27])
    direction = jnp.asarray([0.40, -0.30, 0.20])
    surface_weights = jnp.asarray([1.0, 1.8])
    alpha_weights = jnp.asarray([1.4, 1.0])
    ky_weights = jnp.asarray([0.8, 1.2, 1.5])
    objective_weights = jnp.asarray([0.65, 1.0, 1.35])

    def scalar_objective(x: jnp.ndarray) -> jnp.ndarray:
        return aggregate_objective_portfolio(
            _fixture_rows(x),
            surface_weights=surface_weights,
            alpha_weights=alpha_weights,
            ky_weights=ky_weights,
            objective_weights=objective_weights,
        )

    rows = _fixture_rows(params)
    contract = validate_objective_portfolio_contract(
        rows,
        surface_weights=surface_weights,
        alpha_weights=alpha_weights,
        ky_weights=ky_weights,
        objective_weights=objective_weights,
    )
    sample_weights = portfolio_sample_weight_tensor(
        rows,
        surface_weights=surface_weights,
        alpha_weights=alpha_weights,
        ky_weights=ky_weights,
    )
    normalized_objective_weights = portfolio_objective_weight_vector(rows, objective_weights=objective_weights)
    sample_scalars = jnp.sum(rows * normalized_objective_weights, axis=-1)
    value = scalar_objective(params)
    gradient = jax.grad(scalar_objective)(params)
    _base, tangent = jax.jvp(scalar_objective, (params,), (direction,))
    gradient_directional = jnp.vdot(gradient, direction)
    step = float(fd_step)
    finite_difference = (scalar_objective(params + step * direction) - scalar_objective(params - step * direction)) / (
        2.0 * step
    )
    tangent_error = float(abs(tangent - gradient_directional))
    fd_error = float(abs(tangent - finite_difference))
    fd_tolerance = float(atol + rtol * abs(float(finite_difference)))
    passed = bool(tangent_error <= float(atol) and fd_error <= fd_tolerance)

    samples: list[dict[str, object]] = []
    surface_labels = [0.25, 0.55]
    alpha_labels = [0.0, 0.65]
    ky_labels = [0.25, 0.45, 0.75]
    rows_np = np.asarray(rows, dtype=float)
    sample_weights_np = np.asarray(sample_weights, dtype=float)
    sample_scalars_np = np.asarray(sample_scalars, dtype=float)
    for surface_index, surface_label in enumerate(surface_labels):
        for alpha_index, alpha_label in enumerate(alpha_labels):
            for ky_index, ky_label in enumerate(ky_labels):
                objectives = rows_np[surface_index, alpha_index, ky_index, :]
                samples.append(
                    {
                        "surface_index": surface_index,
                        "surface_label": surface_label,
                        "alpha_index": alpha_index,
                        "alpha": alpha_label,
                        "ky_index": ky_index,
                        "ky": ky_label,
                        "sample_weight": float(sample_weights_np[surface_index, alpha_index, ky_index]),
                        "scalar_value": float(sample_scalars_np[surface_index, alpha_index, ky_index]),
                        **{
                            objective_name: float(objectives[objective_index])
                            for objective_index, objective_name in enumerate(OBJECTIVE_NAMES)
                        },
                    }
                )

    return {
        "kind": "stellarator_objective_portfolio_gate",
        "passed": passed,
        "claim_scope": (
            "backend-free reduced objective portfolio contract over fixed surface/alpha/ky rows; "
            "not a VMEC/Boozer or nonlinear turbulent transport optimization claim"
        ),
        "contract": contract.to_dict(),
        "objective_names": list(OBJECTIVE_NAMES),
        "parameter_names": ["ripple_proxy", "shear_proxy", "pressure_proxy"],
        "parameters": np.asarray(params, dtype=float).tolist(),
        "direction": np.asarray(direction, dtype=float).tolist(),
        "surface_weights": np.asarray(surface_weights, dtype=float).tolist(),
        "alpha_weights": np.asarray(alpha_weights, dtype=float).tolist(),
        "ky_weights": np.asarray(ky_weights, dtype=float).tolist(),
        "objective_weights": np.asarray(objective_weights, dtype=float).tolist(),
        "normalized_objective_weights": np.asarray(normalized_objective_weights, dtype=float).tolist(),
        "value": float(value),
        "gradient": np.asarray(gradient, dtype=float).tolist(),
        "jvp": float(tangent),
        "gradient_directional": float(gradient_directional),
        "finite_difference_directional": float(finite_difference),
        "fd_step": step,
        "rtol": float(rtol),
        "atol": float(atol),
        "jvp_grad_abs_error": tangent_error,
        "jvp_fd_abs_error": fd_error,
        "jvp_fd_tolerance": fd_tolerance,
        "samples": samples,
        "next_action": (
            "Use the same reducer around VMEC/Boozer-produced objective tables, then gate real optimizer steps "
            "with branch-continuity, AD/JVP, finite-difference, and held-out surface/alpha/ky checks."
        ),
    }


def write_stellarator_objective_portfolio_artifacts(
    payload: dict[str, object],
    *,
    out: str | Path = DEFAULT_OUT,
) -> dict[str, str]:
    """Write JSON/CSV/PNG/PDF companions for a portfolio-gate payload."""

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_path.with_suffix(".json")
    csv_path = out_path.with_suffix(".csv")
    pdf_path = out_path.with_suffix(".pdf")
    json_path.write_text(json.dumps(_json_clean(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    fieldnames = [
        "surface_index",
        "surface_label",
        "alpha_index",
        "alpha",
        "ky_index",
        "ky",
        "sample_weight",
        "scalar_value",
        *OBJECTIVE_NAMES,
    ]
    samples = payload.get("samples", [])
    rows = samples if isinstance(samples, list) else []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            if isinstance(row, dict):
                writer.writerow({field: row.get(field, "") for field in fieldnames})

    sample_labels = [
        f"s{row.get('surface_index')}/a{row.get('alpha_index')}/k{row.get('ky_index')}"
        for row in rows
        if isinstance(row, dict)
    ]
    sample_scalars = np.asarray(
        [float(row.get("scalar_value", np.nan)) for row in rows if isinstance(row, dict)],
        dtype=float,
    )
    sample_weights = np.asarray(
        [float(row.get("sample_weight", np.nan)) for row in rows if isinstance(row, dict)],
        dtype=float,
    )
    derivative_names = ["JVP", "grad dot d", "central FD"]
    derivative_values = np.asarray(
        [
            float(payload.get("jvp", np.nan)),
            float(payload.get("gradient_directional", np.nan)),
            float(payload.get("finite_difference_directional", np.nan)),
        ],
        dtype=float,
    )

    set_plot_style()
    fig, (ax_samples, ax_derivatives) = plt.subplots(
        1,
        2,
        figsize=(12.5, 5.1),
        gridspec_kw={"width_ratios": [1.25, 1.0]},
    )
    x = np.arange(sample_scalars.size)
    colors = plt.cm.viridis(sample_weights / max(float(np.max(sample_weights)), 1.0e-12))
    ax_samples.bar(x, sample_scalars, color=colors, edgecolor="#202020", linewidth=0.5)
    ax_samples.set_xticks(x, sample_labels, rotation=35, ha="right")
    ax_samples.set_ylabel("objective-weighted row value")
    ax_samples.set_title("Surface/alpha/ky portfolio rows")
    ax_samples.grid(axis="y", alpha=0.25)

    ax_derivatives.bar(derivative_names, derivative_values, color=["#219ebc", "#2a9d8f", "#f4a261"])
    ax_derivatives.set_title("Directional derivative parity")
    ax_derivatives.set_ylabel("d objective / d direction")
    ax_derivatives.grid(axis="y", alpha=0.25)
    status = "passed" if payload.get("passed") else "open"
    summary = "\n".join(
        [
            f"status: {status}",
            f"value: {float(payload.get('value', np.nan)):.6g}",
            f"JVP-grad error: {float(payload.get('jvp_grad_abs_error', np.nan)):.3e}",
            f"JVP-FD error: {float(payload.get('jvp_fd_abs_error', np.nan)):.3e}",
            f"FD tolerance: {float(payload.get('jvp_fd_tolerance', np.nan)):.3e}",
            "scope: backend-free reducer contract",
        ]
    )
    ax_derivatives.text(
        0.02,
        0.03,
        summary,
        transform=ax_derivatives.transAxes,
        va="bottom",
        ha="left",
        family="monospace",
        fontsize=9.2,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.86},
    )
    fig.suptitle(f"Stellarator objective portfolio reducer gate: {status}")
    fig.text(
        0.5,
        0.015,
        "Aggregates fixed per-surface/per-alpha/per-ky reduced objectives with validated normalized weights. "
        "This panel does not exercise optional VMEC/Boozer backends.",
        ha="center",
        fontsize=8.2,
        color="#333333",
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.28, wspace=0.28)
    fig.savefig(out_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    return {"png": str(out_path), "pdf": str(pdf_path), "json": str(json_path), "csv": str(csv_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--fd-step", type=float, default=1.0e-3)
    parser.add_argument("--rtol", type=float, default=2.0e-3)
    parser.add_argument("--atol", type=float, default=2.0e-5)
    parser.add_argument("--json-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = stellarator_objective_portfolio_gate_payload(
        fd_step=args.fd_step,
        rtol=args.rtol,
        atol=args.atol,
    )
    if args.json_only:
        print(json.dumps(_json_clean(payload), indent=2, sort_keys=True))
        return 0
    paths = write_stellarator_objective_portfolio_artifacts(payload, out=args.out)
    print(json.dumps(paths, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
