#!/usr/bin/env python3
"""Combine quasilinear/UQ ensemble CPU/GPU scaling artifacts into one panel."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = [
    REPO_ROOT / "docs" / "_static" / "quasilinear_uq_ensemble_scaling_cpu_large.json",
    REPO_ROOT / "docs" / "_static" / "quasilinear_uq_ensemble_scaling_gpu_large.json",
]
DEFAULT_PREFIX = REPO_ROOT / "docs" / "_static" / "quasilinear_uq_ensemble_scaling_large"


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


def _grid_label(payload: dict[str, Any]) -> str:
    grid = payload["grid"]
    return (
        f"Nx={int(grid['Nx'])}, Ny={int(grid['Ny'])}, Nz={int(grid['Nz'])}, "
        f"Nl={int(grid['Nl'])}, Nm={int(grid['Nm'])}"
    )


def load_summary(paths: list[Path]) -> dict[str, Any]:
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
                "claim_scope": str(payload.get("claim_scope", "")),
            }
        )
        for row in payload["rows"]:
            item = dict(row)
            item["backend"] = str(payload["backend"])
            item["source"] = str(path)
            item["grid_label"] = _grid_label(payload)
            rows.append(item)
    return _json_clean(
        {
            "kind": "quasilinear_uq_ensemble_scaling_combined",
            "claim_scope": (
                "solver-backed quasilinear/UQ ensemble strong-scaling artifact for "
                "independent CPU processes and GPU workers; the observable is a "
                "reduced mixing-length feature from real late-time linear scans, "
                "not a promoted absolute nonlinear heat-flux predictor"
            ),
            "identity_passed": all(item["identity_passed"] for item in inputs),
            "inputs": inputs,
            "rows": rows,
        }
    )


def write_artifacts(summary: dict[str, Any], out_prefix: Path) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from spectraxgk.plotting import set_plot_style

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_prefix.with_suffix(".json")
    csv_path = out_prefix.with_suffix(".csv")
    png_path = out_prefix.with_suffix(".png")
    pdf_path = out_prefix.with_suffix(".pdf")
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = list(summary["rows"])
    fieldnames = [
        "backend",
        "requested_devices",
        "actual_workers",
        "timed_wall_s",
        "strong_speedup_vs_1_device",
        "parallel_efficiency",
        "ensemble_mean_heat_flux_proxy",
        "ensemble_std_heat_flux_proxy",
        "max_heat_flux_proxy_rel_error",
        "max_gamma_abs_error",
        "identity_gate_pass",
        "grid_label",
        "source",
        "error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
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
        subset = sorted((row for row in rows if str(row["backend"]) == backend), key=lambda row: int(row["requested_devices"]))
        x = np.asarray([int(row["requested_devices"]) for row in subset], dtype=float)
        speedup = np.asarray([float(row["strong_speedup_vs_1_device"]) for row in subset], dtype=float)
        elapsed = np.asarray([float(row["timed_wall_s"]) for row in subset], dtype=float)
        heat_rel = np.asarray([float(row["max_heat_flux_proxy_rel_error"]) for row in subset], dtype=float)
        gamma_abs = np.asarray([float(row["max_gamma_abs_error"]) for row in subset], dtype=float)
        color = style.get(backend, {}).get("color")
        marker = style.get(backend, {}).get("marker", "o")
        label = style.get(backend, {}).get("label", backend)
        axes[0].plot(x, speedup, marker=marker, lw=2.2, color=color, label=label)
        axes[1].semilogy(x, np.maximum(elapsed, 1.0e-16), marker=marker, lw=2.2, color=color, label=label)
        axes[2].semilogy(x, np.maximum(heat_rel, 1.0e-16), marker=marker, lw=2.0, color=color, label=label + " QL")
        axes[2].semilogy(
            x + 0.04,
            np.maximum(gamma_abs, 1.0e-16),
            marker=marker,
            ls="--",
            lw=1.8,
            color=color,
            label=label + r" $\gamma$",
        )
    ideal = np.arange(1, max_devices + 1)
    axes[0].plot(ideal, ideal, ":", color="0.35", lw=1.4, label="ideal")
    axes[0].set_xlabel("workers/devices")
    axes[0].set_ylabel("speedup vs one worker")
    axes[0].set_title("Quasilinear/UQ ensemble scaling")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].set_xlabel("workers/devices")
    axes[1].set_ylabel("median ensemble time [s]")
    axes[1].set_title("Late-time linear solves")
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].set_xlabel("workers/devices")
    axes[2].set_ylabel("identity error")
    axes[2].set_title("QL/growth identity")
    axes[2].legend(frameon=False, fontsize=7, ncol=2)
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    return {"json": str(json_path), "csv": str(csv_path), "png": str(png_path), "pdf": str(pdf_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", default=DEFAULT_INPUTS)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_PREFIX)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = load_summary([Path(path) for path in args.inputs])
    paths = write_artifacts(summary, Path(args.out_prefix))
    print(json.dumps({"identity_passed": summary["identity_passed"], "paths": paths}, indent=2))


if __name__ == "__main__":
    main()
