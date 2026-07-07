#!/usr/bin/env python3
"""Generate the nonlinear spectral communication identity gate artifact."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, cast

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_JSON = (
    REPO_ROOT
    / "docs"
    / "_static"
    / "nonlinear_spectral_communication_identity_gate.json"
)
DEFAULT_OUT_PNG = (
    REPO_ROOT
    / "docs"
    / "_static"
    / "nonlinear_spectral_communication_identity_gate.png"
)


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


def build_nonlinear_spectral_communication_gate(
    *,
    shape: tuple[int, int, int, int, int],
    y_chunks: tuple[int, ...],
    x_chunks: tuple[int, ...],
    steps: int,
    dt: float,
    atol: float,
    rtol: float,
) -> dict[str, object]:
    """Run deterministic spectral communication, RHS, and integrator gates."""

    from spectraxgk.operators.nonlinear.parallel import (
        deterministic_nonlinear_spectral_state,
        nonlinear_spectral_communication_identity_gate,
        nonlinear_spectral_integrator_identity_gate,
        nonlinear_spectral_pencil_rhs_identity_gate,
        nonlinear_spectral_pencil_transport_window_identity_gate,
        nonlinear_spectral_rhs_identity_gate,
    )

    state = deterministic_nonlinear_spectral_state(shape)
    pencil_rtol = max(float(rtol), 1.0e-5)
    communication_report = nonlinear_spectral_communication_identity_gate(
        state,
        y_chunks=y_chunks,
        x_chunks=x_chunks,
        atol=atol,
        rtol=rtol,
    )
    rhs_report = nonlinear_spectral_rhs_identity_gate(
        state,
        y_chunks=y_chunks,
        x_chunks=x_chunks,
        atol=atol,
        rtol=rtol,
    )
    integrator_report = nonlinear_spectral_integrator_identity_gate(
        state,
        y_chunks=y_chunks,
        x_chunks=x_chunks,
        steps=steps,
        dt=dt,
        atol=atol,
        rtol=rtol,
    )
    pencil_rhs_report = nonlinear_spectral_pencil_rhs_identity_gate(
        state,
        y_chunks=y_chunks,
        x_chunks=x_chunks,
        atol=atol,
        rtol=pencil_rtol,
    )
    pencil_window_report = nonlinear_spectral_pencil_transport_window_identity_gate(
        state,
        y_chunks=y_chunks,
        x_chunks=x_chunks,
        steps=steps,
        dt=dt,
        atol=atol,
        rtol=pencil_rtol,
    )
    error_rows = [
        {
            "operator": "fft_forward_inverse",
            "stage": "communication",
            "max_abs_error": communication_report.fft_max_abs_error,
            "max_rel_error": communication_report.fft_max_rel_error,
            "identity_passed": bool(
                communication_report.fft_max_abs_error <= communication_report.atol
                and communication_report.fft_max_rel_error <= communication_report.rtol
            ),
        },
        {
            "operator": "nonlinear_bracket",
            "stage": "communication",
            "max_abs_error": communication_report.bracket_max_abs_error,
            "max_rel_error": communication_report.bracket_max_rel_error,
            "identity_passed": bool(
                communication_report.bracket_max_abs_error <= communication_report.atol
                and communication_report.bracket_max_rel_error
                <= communication_report.rtol
            ),
        },
        {
            "operator": "spectral_field_solve_layout",
            "stage": "communication",
            "max_abs_error": communication_report.field_max_abs_error,
            "max_rel_error": communication_report.field_max_rel_error,
            "identity_passed": bool(
                communication_report.field_max_abs_error <= communication_report.atol
                and communication_report.field_max_rel_error
                <= communication_report.rtol
            ),
        },
        {
            "operator": "logical_sharded_rhs",
            "stage": "rhs",
            "max_abs_error": rhs_report.rhs_max_abs_error,
            "max_rel_error": rhs_report.rhs_max_rel_error,
            "identity_passed": bool(
                rhs_report.rhs_max_abs_error <= rhs_report.atol
                and rhs_report.rhs_max_rel_error <= rhs_report.rtol
            ),
        },
        {
            "operator": "logical_integrator_final_state",
            "stage": "integrator",
            "max_abs_error": integrator_report.final_state_max_abs_error,
            "max_rel_error": integrator_report.final_state_max_rel_error,
            "identity_passed": bool(
                integrator_report.final_state_max_abs_error <= integrator_report.atol
                and integrator_report.final_state_max_rel_error
                <= integrator_report.rtol
            ),
        },
        {
            "operator": "logical_integrator_flux_proxy_trace",
            "stage": "integrator",
            "max_abs_error": integrator_report.flux_proxy_trace_max_abs_error,
            "max_rel_error": integrator_report.flux_proxy_trace_max_rel_error,
            "identity_passed": bool(
                integrator_report.flux_proxy_trace_max_abs_error
                <= integrator_report.atol
                and integrator_report.flux_proxy_trace_max_rel_error
                <= integrator_report.rtol
            ),
        },
        {
            "operator": "pencil_fused_rhs",
            "stage": "pencil_rhs",
            "max_abs_error": pencil_rhs_report.rhs_max_abs_error,
            "max_rel_error": pencil_rhs_report.rhs_max_rel_error,
            "identity_passed": bool(pencil_rhs_report.identity_passed),
        },
        {
            "operator": "pencil_physical_transport_window",
            "stage": "pencil_transport",
            "max_abs_error": pencil_window_report.final_state_max_abs_error,
            "max_rel_error": pencil_window_report.final_state_max_rel_error,
            "identity_passed": bool(pencil_window_report.identity_passed),
        },
    ]
    combined_gate = {
        "identity_passed": bool(
            communication_report.identity_passed
            and rhs_report.identity_passed
            and integrator_report.identity_passed
            and pencil_rhs_report.identity_passed
            and pencil_window_report.identity_passed
        ),
        "communication_identity_passed": bool(communication_report.identity_passed),
        "rhs_identity_passed": bool(rhs_report.identity_passed),
        "integrator_identity_passed": bool(integrator_report.identity_passed),
        "pencil_rhs_identity_passed": bool(pencil_rhs_report.identity_passed),
        "pencil_transport_window_identity_passed": bool(
            pencil_window_report.identity_passed
        ),
        "pencil_work_model_speedup_feasible": bool(
            pencil_window_report.work_model.production_speedup_feasible
        ),
        "decomposed_path_enabled": bool(
            communication_report.decomposed_path_enabled
            and rhs_report.decomposed_path_enabled
            and integrator_report.decomposed_path_enabled
            and pencil_rhs_report.decomposed_path_enabled
            and pencil_window_report.decomposed_path_enabled
        ),
        "atol": float(atol),
        "rtol": float(rtol),
        "pencil_rtol": float(pencil_rtol),
        "steps": int(steps),
        "dt": float(dt),
        "claim_scope": (
            "diagnostic nonlinear spectral communication, RHS, fixed-step "
            "integrator, pencil fused-bracket, and physical transport-window "
            "identity gate; no production distributed FFT routing or speedup claim"
        ),
        "blocked_reasons": sorted(
            {
                *communication_report.blocked_reasons,
                *rhs_report.blocked_reasons,
                *integrator_report.blocked_reasons,
                *pencil_rhs_report.blocked_reasons,
                *pencil_window_report.blocked_reasons,
            }
        ),
    }
    return _json_clean(
        {
            "case": "Nonlinear spectral decomposition identity gate",
            "source": "spectraxgk.operators.nonlinear.parallel nonlinear-spectral communication utilities",
            "claim_scope": combined_gate["claim_scope"],
            "kind": "nonlinear_spectral_communication_identity_gate",
            "communication_decomposition": {
                "y_chunks": communication_report.y_chunks,
                "y_offsets": communication_report.y_offsets,
                "x_chunks": communication_report.x_chunks,
                "x_offsets": communication_report.x_offsets,
                "tile_bounds": rhs_report.tile_bounds,
                "blocked_reasons": combined_gate["blocked_reasons"],
            },
            "gate": combined_gate,
            "communication_gate": communication_report.to_dict(),
            "rhs_gate": rhs_report.to_dict(),
            "integrator_gate": integrator_report.to_dict(),
            "pencil_rhs_gate": pencil_rhs_report.to_dict(),
            "pencil_transport_window_gate": pencil_window_report.to_dict(),
            "rows": error_rows,
        }
    )


def write_artifacts(summary: dict[str, object], out_json: Path, out_png: Path) -> None:
    """Write the JSON report and compact communication-identity plot."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from spectraxgk.artifacts.plotting import set_plot_style

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    rows = cast(list[dict[str, Any]], summary["rows"])
    operators = [str(row["operator"]).replace("_", "\n") for row in rows]
    abs_errors = np.asarray([float(row["max_abs_error"]) for row in rows])
    rel_errors = np.asarray([float(row["max_rel_error"]) for row in rows])
    gate = cast(dict[str, Any], summary["gate"])
    atol = float(gate["atol"])
    rtol = float(gate["rtol"])

    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.8), constrained_layout=True)
    x = np.arange(len(rows))
    axes[0].bar(x, np.maximum(abs_errors, 1.0e-16), color="#1b6ca8")
    axes[0].axhline(atol, color="0.2", ls=":", lw=1.2, label="atol")
    axes[0].set_yscale("log")
    axes[0].set_xticks(x, operators)
    axes[0].set_ylabel("max absolute error")
    axes[0].set_title("Communication, RHS, integrator identity")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].bar(x, np.maximum(rel_errors, 1.0e-16), color="#b65f23")
    axes[1].axhline(rtol, color="0.2", ls=":", lw=1.2, label="rtol")
    axes[1].set_yscale("log")
    axes[1].set_xticks(x, operators)
    axes[1].set_ylabel("max relative error")
    axes[1].set_title("All logical routes gated")
    axes[1].legend(frameon=False, fontsize=8)

    for ax in axes:
        ax.grid(True, alpha=0.25, axis="y")
    fig.suptitle("Nonlinear spectral decomposition identity gate", fontsize=12)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def _parse_chunks(raw: str) -> tuple[int, ...]:
    return tuple(int(item) for item in raw.split(",") if item.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-png", type=Path, default=DEFAULT_OUT_PNG)
    parser.add_argument("--nl", type=int, default=2)
    parser.add_argument("--nm", type=int, default=3)
    parser.add_argument("--ny", type=int, default=6)
    parser.add_argument("--nx", type=int, default=4)
    parser.add_argument("--nz", type=int, default=2)
    parser.add_argument("--y-chunks", type=_parse_chunks, default=(2, 2, 2))
    parser.add_argument("--x-chunks", type=_parse_chunks, default=(2, 2))
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--atol", type=float, default=5.0e-6)
    parser.add_argument("--rtol", type=float, default=5.0e-6)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = build_nonlinear_spectral_communication_gate(
        shape=(args.nl, args.nm, args.ny, args.nx, args.nz),
        y_chunks=args.y_chunks,
        x_chunks=args.x_chunks,
        steps=int(args.steps),
        dt=float(args.dt),
        atol=args.atol,
        rtol=args.rtol,
    )
    write_artifacts(summary, args.out_json, args.out_png)
    print(json.dumps(summary["gate"], indent=2, sort_keys=True))
    gate = cast(dict[str, Any], summary["gate"])
    return 0 if bool(gate["identity_passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
