#!/usr/bin/env python3
"""Combine nonlinear strong-scaling sweep JSON files into one publication panel."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUTS = [
    REPO_ROOT / "docs" / "_static" / "nonlinear_sharding_strong_scaling_cpu_large.json",
    REPO_ROOT
    / "docs"
    / "_static"
    / "nonlinear_sharding_strong_scaling_gpu_xlarge.json",
]
DEFAULT_PREFIX = (
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


def load_summary(paths: list[Path]) -> dict[str, Any]:
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
            item["grid_label"] = _grid_label(payload["grid"])
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


def _grid_label(grid: dict[str, Any]) -> str:
    return (
        f"Nx={int(grid['Nx'])}, Ny={int(grid['Ny_requested'])}, Nz={int(grid['Nz'])}, "
        f"Nl={int(grid['Nl'])}, Nm={int(grid['Nm'])}"
    )


def write_artifacts(summary: dict[str, Any], out_prefix: Path) -> dict[str, str]:
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", default=DEFAULT_INPUTS)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_PREFIX)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = load_summary([Path(path) for path in args.inputs])
    paths = write_artifacts(summary, Path(args.out_prefix))
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


if __name__ == "__main__":
    main()
