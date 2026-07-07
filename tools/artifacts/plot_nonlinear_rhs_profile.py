#!/usr/bin/env python3
"""Plot CPU/GPU nonlinear RHS kernel split profiles."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_INPUTS = {
    "CPU grid": Path("docs/_static/nonlinear_rhs_profile_cpu.csv"),
    "CPU spectral": Path("docs/_static/nonlinear_rhs_profile_cpu_spectral.csv"),
    "GPU grid": Path("docs/_static/nonlinear_rhs_profile_gpu.csv"),
    "GPU spectral": Path("docs/_static/nonlinear_rhs_profile_gpu_spectral.csv"),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=Path("docs/_static/nonlinear_rhs_profile.png")
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional JSON summary path. Defaults to the plot path with a .json suffix.",
    )
    parser.add_argument(
        "--input",
        action="append",
        default=None,
        metavar="LABEL=CSV",
        help="Optional labeled profile CSV. May be repeated. Defaults to the shipped short Cyclone inputs.",
    )
    parser.add_argument(
        "--case",
        default="cyclone_short",
        help="Case label written to the JSON summary.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional plot title suffix. Defaults to a title from --case.",
    )
    return parser.parse_args()


def _parse_input_arg(value: str) -> tuple[str, Path]:
    label, sep, path = value.partition("=")
    if not sep or not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError("--input must have the form LABEL=CSV")
    return label.strip(), Path(path.strip())


def _read_profile(path: Path) -> dict[str, float]:
    with path.open(newline="") as f:
        return {row["kernel"]: float(row["seconds"]) for row in csv.DictReader(f)}


def _case_title(case: str) -> str:
    titles = {
        "cyclone_short": "Cyclone short case",
        "cyclone_miller_benchmark_size": "Cyclone Miller benchmark-size case",
    }
    return titles.get(case, case.replace("_", " "))


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    ratio = float(numerator) / float(denominator)
    return ratio if math.isfinite(ratio) else None


def _build_summary(
    profiles: dict[str, dict[str, float]], *, case: str = "cyclone_short"
) -> dict[str, Any]:
    """Return a machine-readable summary of RHS split-profile CSV files."""

    rows: dict[str, dict[str, Any]] = {}
    for label, profile in profiles.items():
        full_rhs = profile.get("full_rhs")
        timed_kernels = {
            key: value for key, value in profile.items() if key != "full_rhs"
        }
        dominant = (
            max(timed_kernels, key=lambda key: timed_kernels[key])
            if timed_kernels
            else None
        )
        rows[label] = {
            "seconds": profile,
            "dominant_measured_kernel": dominant,
            "field_solve_fraction_of_full_rhs": _safe_ratio(
                profile.get("field_solve"), full_rhs
            ),
            "linear_rhs_fraction_of_full_rhs": _safe_ratio(
                profile.get("linear_rhs"), full_rhs
            ),
            "nonlinear_bracket_fraction_of_full_rhs": _safe_ratio(
                profile.get("nonlinear_bracket"), full_rhs
            ),
        }

    spectral_speedups: dict[str, dict[str, float | None]] = {}
    for backend in ("CPU", "GPU"):
        grid = profiles.get(f"{backend} grid")
        spectral = profiles.get(f"{backend} spectral")
        if grid is None or spectral is None:
            continue
        spectral_speedups[backend.lower()] = {
            "full_rhs_grid_over_spectral": _safe_ratio(
                grid.get("full_rhs"), spectral.get("full_rhs")
            ),
            "nonlinear_bracket_grid_over_spectral": _safe_ratio(
                grid.get("nonlinear_bracket"), spectral.get("nonlinear_bracket")
            ),
        }

    full_rhs_candidates = [
        (profile["full_rhs"], label)
        for label, profile in profiles.items()
        if "full_rhs" in profile and math.isfinite(float(profile["full_rhs"]))
    ]
    fastest = min(
        full_rhs_candidates, default=(None, None), key=lambda item: float(item[0])
    )
    return {
        "kind": "nonlinear_rhs_profile_summary",
        "case": case,
        "rows": rows,
        "spectral_speedups": spectral_speedups,
        "fastest_full_rhs_label": fastest[1],
        "fastest_full_rhs_seconds": fastest[0],
        "claim_scope": (
            "RHS split profile for hot-path localization. Treat speedups as bounded engineering "
            "measurements; production runtime claims require matched benchmark-size CPU/GPU sweeps."
        ),
    }


def _write_summary_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = _parse_args()
    input_paths = (
        dict(_parse_input_arg(item) for item in args.input)
        if args.input is not None
        else DEFAULT_INPUTS
    )
    profiles = {
        label: _read_profile(path)
        for label, path in input_paths.items()
        if path.exists()
    }
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
        ax.bar(
            x + offsets[idx],
            values,
            width=width,
            label=label,
            color=colors[idx % len(colors)],
            edgecolor="0.18",
        )

    ax.set_yscale("log")
    ax.set_ylabel("seconds per compiled kernel call")
    ax.set_title(
        f"Nonlinear RHS kernel profile: {args.title or _case_title(str(args.case))}"
    )
    ax.set_xticks(x, [kernel.replace("_", "\n") for kernel in kernels])
    ax.grid(axis="y", which="major", alpha=0.25)
    ax.legend(frameon=False, ncols=2, fontsize=8)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=220)
    if args.out.suffix.lower() != ".pdf":
        fig.savefig(args.out.with_suffix(".pdf"))
    plt.close(fig)
    summary_path = (
        args.summary_json
        if args.summary_json is not None
        else args.out.with_suffix(".json")
    )
    _write_summary_json(_build_summary(profiles, case=str(args.case)), summary_path)
    print(f"saved {args.out}")
    print(f"saved {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
