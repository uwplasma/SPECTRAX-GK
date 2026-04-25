#!/usr/bin/env python3
"""Plot CPU/GPU nonlinear RHS kernel split profiles."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


DEFAULT_INPUTS = {
    "CPU grid": Path("docs/_static/nonlinear_rhs_profile_cpu.csv"),
    "CPU spectral": Path("docs/_static/nonlinear_rhs_profile_cpu_spectral.csv"),
    "GPU grid": Path("docs/_static/nonlinear_rhs_profile_gpu.csv"),
    "GPU spectral": Path("docs/_static/nonlinear_rhs_profile_gpu_spectral.csv"),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("docs/_static/nonlinear_rhs_profile.png"))
    return parser.parse_args()


def _read_profile(path: Path) -> dict[str, float]:
    with path.open(newline="") as f:
        return {row["kernel"]: float(row["seconds"]) for row in csv.DictReader(f)}


def main() -> int:
    args = _parse_args()
    profiles = {label: _read_profile(path) for label, path in DEFAULT_INPUTS.items() if path.exists()}
    if not profiles:
        raise FileNotFoundError("no nonlinear RHS profile CSV files were found")

    kernels = ["field_solve", "linear_rhs", "nonlinear_bracket", "full_rhs"]
    labels = list(profiles)
    x = np.arange(len(kernels))
    width = min(0.18, 0.75 / max(len(labels), 1))
    offsets = (np.arange(len(labels)) - (len(labels) - 1) / 2.0) * width

    import matplotlib.pyplot as plt

    colors = ["#52616b", "#8c6d31", "#2f6f73", "#c46a3a"]
    fig, ax = plt.subplots(figsize=(9.0, 4.2), constrained_layout=True)
    for idx, label in enumerate(labels):
        values = [profiles[label].get(kernel, np.nan) for kernel in kernels]
        ax.bar(x + offsets[idx], values, width=width, label=label, color=colors[idx % len(colors)], edgecolor="0.18")

    ax.set_yscale("log")
    ax.set_ylabel("seconds per compiled kernel call")
    ax.set_title("Nonlinear RHS kernel profile: Cyclone short case")
    ax.set_xticks(x, [kernel.replace("_", "\n") for kernel in kernels])
    ax.grid(axis="y", which="major", alpha=0.25)
    ax.legend(frameon=False, ncols=2, fontsize=8)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=220)
    if args.out.suffix.lower() != ".pdf":
        fig.savefig(args.out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
