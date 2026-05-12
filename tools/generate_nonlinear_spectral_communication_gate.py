#!/usr/bin/env python3
"""Generate the nonlinear spectral communication identity gate artifact."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_JSON = (
    REPO_ROOT / "docs" / "_static" / "nonlinear_spectral_communication_identity_gate.json"
)
DEFAULT_OUT_PNG = (
    REPO_ROOT / "docs" / "_static" / "nonlinear_spectral_communication_identity_gate.png"
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
    atol: float,
    rtol: float,
) -> dict[str, object]:
    """Run the deterministic split/reassemble spectral communication gate."""

    from spectraxgk.nonlinear_parallel import (
        deterministic_nonlinear_spectral_state,
        nonlinear_spectral_communication_identity_gate,
    )

    state = deterministic_nonlinear_spectral_state(shape)
    report = nonlinear_spectral_communication_identity_gate(
        state,
        y_chunks=y_chunks,
        x_chunks=x_chunks,
        atol=atol,
        rtol=rtol,
    )
    error_rows = [
        {
            "operator": "fft_forward_inverse",
            "max_abs_error": report.fft_max_abs_error,
            "max_rel_error": report.fft_max_rel_error,
            "identity_passed": bool(
                report.fft_max_abs_error <= report.atol
                and report.fft_max_rel_error <= report.rtol
            ),
        },
        {
            "operator": "nonlinear_bracket",
            "max_abs_error": report.bracket_max_abs_error,
            "max_rel_error": report.bracket_max_rel_error,
            "identity_passed": bool(
                report.bracket_max_abs_error <= report.atol
                and report.bracket_max_rel_error <= report.rtol
            ),
        },
        {
            "operator": "spectral_field_solve_layout",
            "max_abs_error": report.field_max_abs_error,
            "max_rel_error": report.field_max_rel_error,
            "identity_passed": bool(
                report.field_max_abs_error <= report.atol
                and report.field_max_rel_error <= report.rtol
            ),
        },
    ]
    return _json_clean(
        {
            "case": "Nonlinear spectral communication identity gate",
            "source": "spectraxgk.nonlinear_parallel nonlinear-spectral communication utilities",
            "claim_scope": report.claim_scope,
            "kind": "nonlinear_spectral_communication_identity_gate",
            "gate": report.to_dict(),
            "rows": error_rows,
        }
    )


def write_artifacts(summary: dict[str, object], out_json: Path, out_png: Path) -> None:
    """Write the JSON report and compact communication-identity plot."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from spectraxgk.plotting import set_plot_style

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    rows = list(summary["rows"])
    operators = [str(row["operator"]).replace("_", "\n") for row in rows]
    abs_errors = np.asarray([float(row["max_abs_error"]) for row in rows])
    rel_errors = np.asarray([float(row["max_rel_error"]) for row in rows])
    gate = dict(summary["gate"])
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
    axes[0].set_title("Split/reassemble identity")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].bar(x, np.maximum(rel_errors, 1.0e-16), color="#b65f23")
    axes[1].axhline(rtol, color="0.2", ls=":", lw=1.2, label="rtol")
    axes[1].set_yscale("log")
    axes[1].set_xticks(x, operators)
    axes[1].set_ylabel("max relative error")
    axes[1].set_title("Communication gate passed")
    axes[1].legend(frameon=False, fontsize=8)

    for ax in axes:
        ax.grid(True, alpha=0.25, axis="y")
    fig.suptitle("Nonlinear spectral communication identity gate", fontsize=12)
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
    parser.add_argument("--atol", type=float, default=5.0e-6)
    parser.add_argument("--rtol", type=float, default=5.0e-6)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = build_nonlinear_spectral_communication_gate(
        shape=(args.nl, args.nm, args.ny, args.nx, args.nz),
        y_chunks=args.y_chunks,
        x_chunks=args.x_chunks,
        atol=args.atol,
        rtol=args.rtol,
    )
    write_artifacts(summary, args.out_json, args.out_png)
    print(json.dumps(summary["gate"], indent=2, sort_keys=True))
    return 0 if bool(dict(summary["gate"])["identity_passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
