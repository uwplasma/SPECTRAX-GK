#!/usr/bin/env python3
"""Build the solver-objective geometry-gradient validation artifact."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from spectraxgk.plotting import set_plot_style  # noqa: E402
from spectraxgk.solver_objective_gradients import linear_solver_geometry_gradient_report  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "docs" / "_static" / "solver_objective_gradient_gate.png"


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, np.generic):
        return _json_clean(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def write_solver_objective_gradient_artifacts(
    payload: dict[str, object],
    *,
    out: str | Path = DEFAULT_OUT,
) -> dict[str, str]:
    """Write JSON/CSV/PNG/PDF companions for a solver-gradient payload."""

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_path.with_suffix(".json")
    csv_path = out_path.with_suffix(".csv")
    pdf_path = out_path.with_suffix(".pdf")
    json_path.write_text(json.dumps(_json_clean(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    raw_rows = payload.get("objective_gates", [])
    rows = raw_rows if isinstance(raw_rows, list) else []
    fieldnames = ["objective", "parameter", "implicit", "finite_difference", "abs_error", "rel_error", "passed"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            if isinstance(row, dict):
                writer.writerow({name: row.get(name, "") for name in fieldnames})

    raw_objectives = payload.get("objective_names", [])
    raw_parameters = payload.get("parameter_names", [])
    objectives = [str(name) for name in raw_objectives] if isinstance(raw_objectives, list) else []
    parameters = [str(name) for name in raw_parameters] if isinstance(raw_parameters, list) else []
    objective_labels = {
        "gamma": r"$\gamma$",
        "omega": r"$\omega$",
        "kperp_eff2": r"$\langle k_\perp^2\rangle$",
        "linear_heat_flux_weight": r"$Q_i$ weight",
        "linear_particle_flux_weight": r"$\Gamma_i$ weight",
        "mixing_length_heat_flux_proxy": r"$\gamma Q_i/k_\perp^2$",
    }
    parameter_labels = {
        "bmag_ripple": r"$|B|$ ripple",
        "curvature_drift_scale": "drift scale",
        "Rcos_mid_surface_m1": r"$R_{m=1}$",
    }
    raw_gate = payload.get("eigenpair_gate", {})
    gate = raw_gate if isinstance(raw_gate, dict) else {}
    implicit = np.array(gate.get("jacobian_implicit", []), dtype=np.float64)
    finite_difference = np.array(gate.get("jacobian_fd", []), dtype=np.float64)
    err = np.abs(implicit - finite_difference)
    rel = err / np.maximum(np.abs(finite_difference), float(gate.get("atol", 2.0e-3)))

    set_plot_style()
    fig, (ax_scatter, ax_heat) = plt.subplots(1, 2, figsize=(12.4, 5.1), gridspec_kw={"width_ratios": [1.0, 1.15]})
    flat_fd = finite_difference.ravel()
    flat_impl = implicit.ravel()
    ax_scatter.scatter(flat_fd, flat_impl, s=42, c="#2a9d8f", edgecolor="#202020", linewidth=0.6)
    lim = float(max(np.max(np.abs(flat_fd)), np.max(np.abs(flat_impl)), 1.0e-12))
    ax_scatter.plot([-lim, lim], [-lim, lim], color="#333333", lw=1.1, ls="--")
    ax_scatter.set_xlabel("central finite difference")
    ax_scatter.set_ylabel("implicit eigenpair sensitivity")
    ax_scatter.set_title("AD/FD gradient agreement")
    ax_scatter.grid(alpha=0.25)

    image = ax_heat.imshow(np.log10(np.maximum(rel, 1.0e-8)), cmap="viridis", aspect="auto", vmin=-4.0, vmax=0.0)
    ax_heat.set_xticks(
        np.arange(len(parameters)),
        [parameter_labels.get(str(name), str(name)) for name in parameters],
        rotation=15,
        ha="right",
    )
    ax_heat.set_yticks(np.arange(len(objectives)), [objective_labels.get(str(name), str(name)) for name in objectives])
    ax_heat.set_title(r"$\log_{10}$ relative error by observable")
    for i in range(rel.shape[0]):
        for j in range(rel.shape[1]):
            ax_heat.text(j, i, f"{rel[i, j]:.1e}", ha="center", va="center", color="white", fontsize=7.4)
    cbar = fig.colorbar(image, ax=ax_heat, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\log_{10}$ relative error")

    source_scope = str(payload.get("source_scope", "solver_ready_geometry_contract"))
    status = "passed" if payload.get("passed") else "open"
    if source_scope == "mode21_vmec_boozer_state":
        fig.suptitle(f"VMEC/Boozer state-to-solver frequency-gradient gate: {status}")
    else:
        fig.suptitle(f"Solver-objective geometry-gradient gate: {status}")
    if source_scope == "mode21_vmec_boozer_state":
        caption = (
            "Actual linear-RHS eigenfrequency is differentiated through vmec_jax state coefficients, "
            "booz_xform_jax mode-21 Boozer geometry, and the solver cache."
        )
    else:
        caption = (
            "Actual linear-RHS eigenpair observables are differentiated with respect to solver-ready geometry arrays. "
            "This is a production solver gate, not yet the full VMEC/Boozer state-gradient claim."
        )
    fig.text(0.5, 0.02, caption, ha="center", fontsize=8.2, color="#333333")
    fig.subplots_adjust(left=0.09, right=0.97, top=0.86, bottom=0.22, wspace=0.30)
    fig.savefig(out_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    return {"png": str(out_path), "pdf": str(pdf_path), "json": str(json_path), "csv": str(csv_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--fd-step", type=float, default=1.0e-3)
    parser.add_argument("--rtol", type=float, default=1.0e-1)
    parser.add_argument("--atol", type=float, default=2.0e-3)
    parser.add_argument("--json-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = linear_solver_geometry_gradient_report(fd_step=args.fd_step, rtol=args.rtol, atol=args.atol)
    if args.json_only:
        print(json.dumps(_json_clean(payload), indent=2, sort_keys=True))
        return 0
    paths = write_solver_objective_gradient_artifacts(payload, out=args.out)
    print(json.dumps(paths, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
