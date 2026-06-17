#!/usr/bin/env python3
"""Generate the nonlinear state-domain decomposition identity gate artifact."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_JSON = REPO_ROOT / "docs" / "_static" / "nonlinear_domain_parallel_identity_gate.json"
DEFAULT_OUT_PNG = REPO_ROOT / "docs" / "_static" / "nonlinear_domain_parallel_identity_gate.png"


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


def build_nonlinear_domain_parallel_gate(
    *,
    shape: tuple[int, int],
    axis: int,
    num_domains: int,
    dt: float,
    transport_steps: int,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    """Run the deterministic serial-vs-decomposed nonlinear identity gate."""

    import jax.numpy as jnp

    from spectraxgk.operators.nonlinear.parallel import (
        build_nonlinear_domain_decomposition_plan,
        deterministic_nonlinear_domain_state,
        nonlinear_domain_parallel_identity_gate,
        nonlinear_domain_transport_window_identity_gate,
        prototype_nonlinear_domain_decomposed_step,
        prototype_nonlinear_domain_serial_step,
    )

    state = deterministic_nonlinear_domain_state(shape)
    plan = build_nonlinear_domain_decomposition_plan(
        tuple(state.shape),
        axis=axis,
        num_domains=num_domains,
    )
    gated_state, report = nonlinear_domain_parallel_identity_gate(
        state,
        plan,
        dt=dt,
        atol=atol,
        rtol=rtol,
    )
    transport_report = nonlinear_domain_transport_window_identity_gate(
        state,
        plan,
        dt=dt,
        steps=transport_steps,
        atol=atol,
        rtol=rtol,
    )
    serial = prototype_nonlinear_domain_serial_step(state, axis=plan.axis, dt=dt)
    decomposed = prototype_nonlinear_domain_decomposed_step(state, plan, dt=dt)

    serial_axis_trace = np.asarray(jnp.real(serial[:, 0] if plan.axis == 0 else serial[0, :]))
    decomposed_axis_trace = np.asarray(
        jnp.real(decomposed[:, 0] if plan.axis == 0 else decomposed[0, :])
    )
    abs_error_trace = np.asarray(
        jnp.abs(
            (decomposed[:, 0] - serial[:, 0])
            if plan.axis == 0
            else (decomposed[0, :] - serial[0, :])
        )
    )
    rows = []
    for idx, (serial_value, decomposed_value, abs_error) in enumerate(
        zip(serial_axis_trace, decomposed_axis_trace, abs_error_trace, strict=True)
    ):
        rows.append(
            {
                "axis_index": int(idx),
                "serial_real": float(serial_value),
                "decomposed_real": float(decomposed_value),
                "abs_error": float(abs_error),
            }
        )

    return _json_clean(
        {
            "case": "Nonlinear state-domain decomposition identity gate",
            "source": "spectraxgk.operators.nonlinear.parallel nonlinear-domain prototype utilities",
            "claim_scope": report.claim_scope,
            "state_shape": tuple(int(item) for item in state.shape),
            "decomposition": plan.decomposition_metadata(),
            "boundary_identity": {
                "indices": report.boundary_indices,
                "max_abs_error": report.boundary_max_abs_error,
                "max_rel_error": report.boundary_max_rel_error,
            },
            "transport_window": {
                "gate": transport_report.to_dict(),
                "metrics": [
                    {
                        "metric": "mass_trace",
                        "max_abs_error": transport_report.mass_trace_max_abs_error,
                        "max_rel_error": transport_report.mass_trace_max_rel_error,
                        "serial_drift": transport_report.serial_mass_drift,
                        "decomposed_drift": transport_report.decomposed_mass_drift,
                        "identity_passed": bool(
                            transport_report.mass_trace_max_abs_error <= transport_report.atol
                            and transport_report.mass_trace_max_rel_error <= transport_report.rtol
                        ),
                    },
                    {
                        "metric": "free_energy_trace",
                        "max_abs_error": transport_report.free_energy_trace_max_abs_error,
                        "max_rel_error": transport_report.free_energy_trace_max_rel_error,
                        "serial_drift": transport_report.serial_free_energy_drift,
                        "decomposed_drift": transport_report.decomposed_free_energy_drift,
                        "identity_passed": bool(
                            transport_report.free_energy_trace_max_abs_error
                            <= transport_report.atol
                            and transport_report.free_energy_trace_max_rel_error
                            <= transport_report.rtol
                        ),
                    },
                    {
                        "metric": "boundary_flux_proxy_trace",
                        "max_abs_error": transport_report.flux_proxy_trace_max_abs_error,
                        "max_rel_error": transport_report.flux_proxy_trace_max_rel_error,
                        "serial_drift": None,
                        "decomposed_drift": None,
                        "identity_passed": bool(
                            transport_report.flux_proxy_trace_max_abs_error
                            <= transport_report.atol
                            and transport_report.flux_proxy_trace_max_rel_error
                            <= transport_report.rtol
                        ),
                    },
                ],
            },
            "dt": float(dt),
            "gate": report.to_dict(),
            "gated_state_matches_decomposed": bool(
                np.allclose(np.asarray(gated_state), np.asarray(decomposed), atol=atol, rtol=rtol)
            ),
            "gated_state_matches_serial": bool(
                np.allclose(np.asarray(gated_state), np.asarray(serial), atol=atol, rtol=rtol)
            ),
            "rows": rows,
        }
    )


def write_artifacts(summary: dict[str, Any], out_json: Path, out_png: Path) -> None:
    """Write the JSON report and compact identity-gate plot."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from spectraxgk.plotting import set_plot_style

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    rows = list(summary["rows"])
    x = np.asarray([row["axis_index"] for row in rows], dtype=float)
    serial = np.asarray([row["serial_real"] for row in rows], dtype=float)
    decomposed = np.asarray([row["decomposed_real"] for row in rows], dtype=float)
    abs_error = np.asarray([row["abs_error"] for row in rows], dtype=float)
    gate = dict(summary["gate"])

    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.8), constrained_layout=True)
    axes[0].plot(x, serial, "o-", lw=2.0, label="serial")
    axes[0].plot(x, decomposed, "s--", lw=1.8, label="decomposed")
    axes[0].set_xlabel("domain-axis index")
    axes[0].set_ylabel("real(state)")
    axes[0].set_title("Prototype nonlinear step")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].semilogy(x, np.maximum(abs_error, 1.0e-16), "o-", lw=2.0)
    axes[1].axhline(float(gate["atol"]), color="0.25", ls=":", lw=1.2, label="atol")
    status = "passed" if bool(gate["identity_passed"]) else "failed closed"
    axes[1].set_xlabel("domain-axis index")
    axes[1].set_ylabel("absolute error")
    axes[1].set_title(f"Identity gate {status}")
    axes[1].legend(frameon=False, fontsize=8)

    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-png", type=Path, default=DEFAULT_OUT_PNG)
    parser.add_argument("--nx", type=int, default=6)
    parser.add_argument("--ny", type=int, default=4)
    parser.add_argument("--axis", type=int, default=0)
    parser.add_argument("--num-domains", type=int, default=2)
    parser.add_argument("--dt", type=float, default=0.025)
    parser.add_argument("--transport-steps", type=int, default=5)
    parser.add_argument("--atol", type=float, default=1.0e-6)
    parser.add_argument("--rtol", type=float, default=1.0e-6)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = build_nonlinear_domain_parallel_gate(
        shape=(args.nx, args.ny),
        axis=args.axis,
        num_domains=args.num_domains,
        dt=args.dt,
        transport_steps=args.transport_steps,
        atol=args.atol,
        rtol=args.rtol,
    )
    write_artifacts(summary, args.out_json, args.out_png)
    print(json.dumps(summary["gate"], indent=2, sort_keys=True))
    return 0 if bool(dict(summary["gate"])["identity_passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
