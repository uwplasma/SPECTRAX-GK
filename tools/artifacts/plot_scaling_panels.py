#!/usr/bin/env python3
"""Plot scaling and parallelization panels from tracked JSON/CSV artifacts.

Subcommands:
  diffrax-speedup       Two-device diffrax scaling from scaling_speedup_data.csv.
  independent-ky        Independent ky CPU/GPU scan strong-scaling panel.
  nonlinear-sharding    Nonlinear whole-state sharding diagnostic scaling panel.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIFFRAX_DATA = REPO_ROOT / "docs" / "_static" / "scaling_speedup_data.csv"
DEFAULT_DIFFRAX_PNG = REPO_ROOT / "docs" / "_static" / "scaling_speedup.png"
DEFAULT_INDEPENDENT_KY_INPUTS = [
    REPO_ROOT / "docs" / "_static" / "independent_ky_scan_scaling_cpu_large.json",
    REPO_ROOT / "docs" / "_static" / "independent_ky_scan_scaling_gpu_large.json",
]
DEFAULT_INDEPENDENT_KY_PREFIX = (
    REPO_ROOT / "docs" / "_static" / "independent_ky_scan_scaling_large"
)
DEFAULT_NONLINEAR_SHARDING_INPUTS = [
    REPO_ROOT / "docs" / "_static" / "nonlinear_sharding_strong_scaling_cpu_large.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "nonlinear_sharding_strong_scaling_gpu_xlarge.json",
]
DEFAULT_NONLINEAR_SHARDING_PREFIX = (
    REPO_ROOT / "docs" / "_static" / "nonlinear_sharding_strong_scaling_large"
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


def _save_figure(fig, png_path: Path) -> dict[str, str]:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = png_path.with_suffix(".pdf")
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    return {"png": str(png_path), "pdf": str(pdf_path)}


def write_diffrax_speedup_panel(
    data_path: Path = DEFAULT_DIFFRAX_DATA,
    out_png: Path = DEFAULT_DIFFRAX_PNG,
) -> dict[str, str]:
    """Plot two-device diffrax speedup from the tracked CSV sweep."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from spectraxgk.artifacts.plotting import set_plot_style

    df = pd.read_csv(data_path)
    set_plot_style()
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.6))

    for backend, color in [("cpu", "#1f77b4"), ("cuda", "#ff7f0e")]:
        sub = df[df["backend"] == backend]
        steps = sorted(sub["steps"].unique())
        speedups = []
        for step in steps:
            sub_step = sub[sub["steps"] == step]
            t1 = float(sub_step[sub_step["devices"] == 1]["elapsed_s"].iloc[0])
            t2 = float(sub_step[sub_step["devices"] == 2]["elapsed_s"].iloc[0])
            speedups.append(t1 / t2)
        ax.plot(
            steps,
            speedups,
            marker="o",
            linewidth=2.4,
            color=color,
            label=f"{backend.upper()} 2 devices",
        )

    ax.axhline(2.0, color="#444444", linestyle=":", linewidth=1.0, label="ideal")
    ax.set_ylabel("Speedup (1x / 2x)")
    ax.set_xlabel("Linear integration steps per run")
    ax.set_xticks(sorted(df[df["backend"].isin(["cpu", "cuda"])]["steps"].unique()))
    ax.set_title("Two-device diffrax scaling (Ny=64, Nz=128, Nl=6, Nm=6)")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    paths = _save_figure(fig, out_png)
    plt.close(fig)
    return paths


def _independent_ky_grid_label(payload: dict[str, Any]) -> str:
    grid = payload["grid"]
    return (
        f"Nx={int(grid['Nx'])}, Ny={int(grid['Ny'])}, Nz={int(grid['Nz'])}, "
        f"Nl={int(grid['Nl'])}, Nm={int(grid['Nm'])}"
    )


def load_independent_ky_summary(paths: list[Path]) -> dict[str, Any]:
    """Load independent-ky scaling rows from CPU/GPU JSON summaries."""

    rows: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        inputs.append(
            {
                "path": str(path),
                "backend": str(payload["backend"]),
                "grid": payload["grid"],
                "time": payload["time"],
                "identity_passed": bool(payload["identity_passed"]),
            }
        )
        for row in payload["rows"]:
            item = dict(row)
            item["backend"] = str(payload["backend"])
            item["source"] = str(path)
            item["grid_label"] = _independent_ky_grid_label(payload)
            rows.append(item)
    return _json_clean(
        {
            "kind": "independent_ky_scan_scaling_combined",
            "claim_scope": (
                "solver-backed independent ky scan strong-scaling artifact for CPU processes "
                "and GPU workers; not a nonlinear domain-decomposition speedup claim"
            ),
            "identity_passed": all(item["identity_passed"] for item in inputs),
            "inputs": inputs,
            "rows": rows,
        }
    )


def write_independent_ky_artifacts(
    summary: dict[str, Any], out_prefix: Path
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from spectraxgk.artifacts.plotting import set_plot_style

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_prefix.with_suffix(".json")
    csv_path = out_prefix.with_suffix(".csv")
    png_path = out_prefix.with_suffix(".png")
    pdf_path = out_prefix.with_suffix(".pdf")
    json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows = list(summary["rows"])
    fieldnames = [
        "backend",
        "requested_devices",
        "actual_workers",
        "timed_wall_s",
        "strong_speedup_vs_1_device",
        "parallel_efficiency",
        "max_gamma_rel_error",
        "max_omega_abs_error",
        "identity_gate_pass",
        "grid_label",
        "source",
        "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    set_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.0), constrained_layout=True)
    style = {
        "cpu": {"color": "#276b8e", "marker": "o", "label": "CPU workers"},
        "gpu": {"color": "#c45a14", "marker": "s", "label": "GPU workers"},
    }
    max_devices = max(int(row["requested_devices"]) for row in rows)
    for backend in sorted({str(row["backend"]) for row in rows}):
        subset = sorted(
            (row for row in rows if str(row["backend"]) == backend),
            key=lambda row: int(row["requested_devices"]),
        )
        x = np.asarray([int(row["requested_devices"]) for row in subset], dtype=float)
        speedup = np.asarray(
            [float(row["strong_speedup_vs_1_device"]) for row in subset], dtype=float
        )
        elapsed = np.asarray(
            [float(row["timed_wall_s"]) for row in subset], dtype=float
        )
        gamma_rel = np.asarray(
            [float(row["max_gamma_rel_error"]) for row in subset], dtype=float
        )
        omega_abs = np.asarray(
            [float(row["max_omega_abs_error"]) for row in subset], dtype=float
        )
        color = style.get(backend, {}).get("color")
        marker = style.get(backend, {}).get("marker", "o")
        label = style.get(backend, {}).get("label", backend)
        axes[0].plot(x, speedup, marker=marker, lw=2.2, color=color, label=label)
        axes[1].semilogy(
            x,
            np.maximum(elapsed, 1.0e-16),
            marker=marker,
            lw=2.2,
            color=color,
            label=label,
        )
        axes[2].semilogy(
            x,
            np.maximum(gamma_rel, 1.0e-16),
            marker=marker,
            lw=2.0,
            color=color,
            label=label + r" $\gamma$",
        )
        axes[2].semilogy(
            x + 0.04,
            np.maximum(omega_abs, 1.0e-16),
            marker=marker,
            ls="--",
            lw=1.8,
            color=color,
            label=label + r" $\omega$",
        )
    ideal = np.arange(1, max_devices + 1)
    axes[0].plot(ideal, ideal, ":", color="0.35", lw=1.4, label="ideal")
    axes[0].set_xlabel("workers/devices")
    axes[0].set_ylabel("speedup vs one worker")
    axes[0].set_title("Independent ky strong scaling")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].set_xlabel("workers/devices")
    axes[1].set_ylabel("median scan time [s]")
    axes[1].set_title("Solver throughput")
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].set_xlabel("workers/devices")
    axes[2].set_ylabel("identity error")
    axes[2].set_title("Gamma/omega identity")
    axes[2].legend(frameon=False, fontsize=7, ncol=2)
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "png": str(png_path),
        "pdf": str(pdf_path),
    }


def _nonlinear_sharding_grid_label(grid: dict[str, Any]) -> str:
    return (
        f"Nx={int(grid['Nx'])}, Ny={int(grid['Ny_requested'])}, Nz={int(grid['Nz'])}, "
        f"Nl={int(grid['Nl'])}, Nm={int(grid['Nm'])}"
    )


def load_nonlinear_sharding_summary(paths: list[Path]) -> dict[str, Any]:
    """Load nonlinear whole-state sharding rows from CPU/GPU JSON summaries."""

    rows: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        backend = str(payload["backend"])
        inputs.append(
            {
                "path": str(path),
                "backend": backend,
                "grid": payload["grid"],
                "identity_passed": bool(payload["identity_passed"]),
                "speedup_passed": bool(payload.get("speedup_passed", False)),
                "status": str(payload.get("status", "diagnostic_identity_only")),
                "speedup_blockers": list(payload.get("speedup_blockers", [])),
                "claim_scope": payload.get("claim_scope", ""),
            }
        )
        for row in payload["rows"]:
            item = dict(row)
            item["source"] = str(path)
            item["grid_label"] = _nonlinear_sharding_grid_label(payload["grid"])
            rows.append(item)
    return _json_clean(
        {
            "kind": "nonlinear_sharding_strong_scaling_combined",
            "claim_scope": (
                "large CPU/GPU nonlinear whole-state sharding artifact. It is a numerical-identity "
                "and profiler-direction result; it is not a production speedup claim unless speedup_passed "
                "is true and the separate production gate passes."
            ),
            "identity_passed": all(bool(item["identity_passed"]) for item in inputs),
            "speedup_passed": all(bool(item["speedup_passed"]) for item in inputs),
            "status": (
                "identity_and_speedup"
                if inputs and all(bool(item["speedup_passed"]) for item in inputs)
                else "diagnostic_identity_only"
            ),
            "speedup_blockers": [
                f"{item['backend']}:{blocker}"
                for item in inputs
                for blocker in item.get("speedup_blockers", [])
            ],
            "inputs": inputs,
            "rows": rows,
        }
    )


def write_nonlinear_sharding_artifacts(
    summary: dict[str, Any], out_prefix: Path
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from spectraxgk.artifacts.plotting import set_plot_style

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_prefix.with_suffix(".json")
    csv_path = out_prefix.with_suffix(".csv")
    png_path = out_prefix.with_suffix(".png")
    pdf_path = out_prefix.with_suffix(".pdf")

    json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows = list(summary["rows"])
    fieldnames = [
        "backend",
        "requested_devices",
        "actual_devices",
        "grid_label",
        "best_spec",
        "state_sharding_active",
        "identity_gate_pass",
        "parallel_median_s",
        "strong_speedup_vs_1_device",
        "same_process_speedup",
        "max_rel_state_error",
        "source",
        "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.1), constrained_layout=True)
    style = {
        "cpu": {"color": "#276b8e", "marker": "o", "label": "CPU logical devices"},
        "gpu": {"color": "#c45a14", "marker": "s", "label": "GPU devices"},
    }
    for backend in sorted({str(row["backend"]) for row in rows}):
        subset = sorted(
            (row for row in rows if str(row["backend"]) == backend),
            key=lambda row: int(row["requested_devices"]),
        )
        x = np.asarray([int(row["requested_devices"]) for row in subset], dtype=float)
        y = np.asarray(
            [float(row["strong_speedup_vs_1_device"]) for row in subset], dtype=float
        )
        t = np.asarray([float(row["parallel_median_s"]) for row in subset], dtype=float)
        label = f"{style.get(backend, {}).get('label', backend)}"
        color = style.get(backend, {}).get("color", None)
        marker = style.get(backend, {}).get("marker", "o")
        axes[0].plot(x, y, marker=marker, lw=2.2, color=color, label=label)
        axes[1].semilogy(
            x, np.maximum(t, 1.0e-16), marker=marker, lw=2.2, color=color, label=label
        )
    xmax = max(int(row["requested_devices"]) for row in rows)
    ideal = np.arange(1, xmax + 1)
    axes[0].plot(ideal, ideal, ":", color="0.35", lw=1.4, label="ideal")
    axes[0].axhline(1.0, color="0.5", ls="--", lw=1.0)
    axes[0].set_xlabel("devices")
    axes[0].set_ylabel("speedup vs one device")
    axes[0].set_title("Large nonlinear whole-state sharding")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].set_xlabel("devices")
    axes[1].set_ylabel("median fixed-step time [s]")
    axes[1].set_title("Identity-preserving timing")
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "png": str(png_path),
        "pdf": str(pdf_path),
    }


def build_diffrax_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot two-device diffrax scaling.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DIFFRAX_DATA)
    parser.add_argument("--out-png", type=Path, default=DEFAULT_DIFFRAX_PNG)
    return parser


def build_independent_ky_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot independent-ky scan scaling.")
    parser.add_argument(
        "--inputs", type=Path, nargs="+", default=DEFAULT_INDEPENDENT_KY_INPUTS
    )
    parser.add_argument(
        "--out-prefix", type=Path, default=DEFAULT_INDEPENDENT_KY_PREFIX
    )
    return parser


def build_nonlinear_sharding_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot nonlinear whole-state sharding strong-scaling diagnostics."
    )
    parser.add_argument(
        "--inputs", type=Path, nargs="+", default=DEFAULT_NONLINEAR_SHARDING_INPUTS
    )
    parser.add_argument(
        "--out-prefix", type=Path, default=DEFAULT_NONLINEAR_SHARDING_PREFIX
    )
    return parser


def main_diffrax_speedup(argv: list[str] | None = None) -> int:
    args = build_diffrax_parser().parse_args(argv)
    paths = write_diffrax_speedup_panel(args.data, args.out_png)
    print(f"Wrote {paths['png']}")
    return 0


def main_independent_ky(argv: list[str] | None = None) -> int:
    args = build_independent_ky_parser().parse_args(argv)
    summary = load_independent_ky_summary([Path(path) for path in args.inputs])
    paths = write_independent_ky_artifacts(summary, Path(args.out_prefix))
    print(
        json.dumps(
            {"identity_passed": summary["identity_passed"], "paths": paths}, indent=2
        )
    )
    return 0


def main_nonlinear_sharding(argv: list[str] | None = None) -> int:
    args = build_nonlinear_sharding_parser().parse_args(argv)
    summary = load_nonlinear_sharding_summary([Path(path) for path in args.inputs])
    paths = write_nonlinear_sharding_artifacts(summary, Path(args.out_prefix))
    print(
        json.dumps(
            {
                "identity_passed": summary["identity_passed"],
                "speedup_passed": summary["speedup_passed"],
                "status": summary["status"],
                "paths": paths,
            },
            indent=2,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    if not tokens:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument(
            "command",
            choices=("diffrax-speedup", "independent-ky", "nonlinear-sharding"),
        )
        parser.print_help()
        return 2
    command, rest = tokens[0], tokens[1:]
    if command == "diffrax-speedup":
        return main_diffrax_speedup(rest)
    if command == "independent-ky":
        return main_independent_ky(rest)
    if command == "nonlinear-sharding":
        return main_nonlinear_sharding(rest)
    raise SystemExit(f"unknown command: {command}")


if __name__ == "__main__":
    raise SystemExit(main())
