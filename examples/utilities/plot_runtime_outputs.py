#!/usr/bin/env python3
"""Plot diagnostics from a nonlinear *.out.nc file."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    import netCDF4
except ModuleNotFoundError as exc:  # pragma: no cover - runtime helper.
    raise SystemExit("netCDF4 is required: pip install netCDF4") from exc


def _load_diag_var(group, name: str):
    if name in group.variables:
        return np.asarray(group.variables[name][:])
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Path to *.out.nc file")
    parser.add_argument("--out", type=Path, default=Path("tools_out/diagnostics_plot.png"))
    parser.add_argument("--title", type=str, default="SPECTRAX-GK Diagnostics")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with netCDF4.Dataset(args.path) as root:
        diag = root.groups["Diagnostics"]
        t = np.asarray(diag.variables["t"][:])
        phi2 = _load_diag_var(diag, "Phi2_t")
        Wg = _load_diag_var(diag, "Wg_st")
        Wphi = _load_diag_var(diag, "Wphi_st")
        heat = _load_diag_var(diag, "HeatFlux_st")

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    ax = axes.ravel()

    if phi2 is not None:
        ax[0].plot(t, phi2, color="#1f77b4")
        ax[0].set_title("Phi2_t")
    if Wg is not None:
        ax[1].plot(t, Wg[..., 0], color="#ff7f0e")
        ax[1].set_title("Wg_st (s0)")
    if Wphi is not None:
        ax[2].plot(t, Wphi[..., 0], color="#2ca02c")
        ax[2].set_title("Wphi_st (s0)")
    if heat is not None:
        ax[3].plot(t, heat[..., 0], color="#d62728")
        ax[3].set_title("HeatFlux_st (s0)")

    for axis in ax:
        axis.set_xlabel("t")
        axis.grid(True, alpha=0.3)

    fig.suptitle(args.title)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
