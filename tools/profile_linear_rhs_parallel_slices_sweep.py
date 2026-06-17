#!/usr/bin/env python3
"""Sweep electrostatic linear-slices RHS timings over devices and Hermite size."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.profile_linear_rhs_parallel_slices import (  # noqa: E402
    _configure_logical_cpu_devices,
    profile_linear_rhs_parallel_slices,
)

DEFAULT_PREFIX = REPO_ROOT / "docs" / "_static" / "linear_rhs_parallel_slices_sweep"


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


def _parse_int_list(text: str) -> list[int]:
    values = [int(part.strip()) for part in str(text).split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated list of positive integers")
    if any(value < 1 for value in values):
        raise argparse.ArgumentTypeError("all values must be positive")
    return values


def run_sweep(
    *,
    platform: str,
    devices: list[int],
    nms: list[int],
    nx: int,
    ny: int,
    nz: int,
    nl: int,
    warmups: int,
    repeats: int,
    atol: float,
    rtol: float,
) -> dict[str, object]:
    """Run the bounded engineering timing sweep."""

    rows: list[dict[str, object]] = []
    for nm in nms:
        for requested_devices in devices:
            summary = profile_linear_rhs_parallel_slices(
                platform=platform,
                requested_devices=requested_devices,
                nx=nx,
                ny=ny,
                nz=nz,
                nl=nl,
                nm=nm,
                warmups=warmups,
                repeats=repeats,
                atol=atol,
                rtol=rtol,
            )
            rows.append(
                {
                    "platform": platform,
                    "requested_devices": int(requested_devices),
                    "nm": int(nm),
                    "state_shape": tuple(summary["state_shape"]),
                    "serial_median_s": float(summary["serial_median_s"]),
                    "sharded_median_s": float(summary["sharded_median_s"]),
                    "speedup": float(summary["speedup"]),
                    "identity_passed": bool(summary["identity_passed"]),
                    "max_abs_error": float(summary["max_abs_error"]),
                    "max_rel_error": float(summary["max_rel_error"]),
                    "max_phi_abs_error": float(summary["max_phi_abs_error"]),
                }
            )
    passed = all(bool(row["identity_passed"]) for row in rows)
    return _json_clean(
        {
            "kind": "linear_rhs_parallel_slices_sweep",
            "claim_scope": (
                "engineering sweep over the opt-in electrostatic linear-slices route; "
                "not a publication speedup claim"
            ),
            "platform": platform,
            "devices": [int(x) for x in devices],
            "nms": [int(x) for x in nms],
            "grid": {"Nx": int(nx), "Ny_requested": int(ny), "Nz": int(nz), "Nl": int(nl)},
            "warmups": int(warmups),
            "repeats": int(repeats),
            "atol": float(atol),
            "rtol": float(rtol),
            "identity_passed": passed,
            "rows": rows,
            "notes": (
                "Use this plot to find useful CPU/GPU regimes before promoting any speedup claim. "
                "The release correctness gate remains the small composed identity artifact."
            ),
        }
    )


def write_artifacts(summary: dict[str, object], out_prefix: Path) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from spectraxgk.artifacts.plotting import set_plot_style

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_prefix.with_suffix(".json")
    csv_path = out_prefix.with_suffix(".csv")
    png_path = out_prefix.with_suffix(".png")
    pdf_path = out_prefix.with_suffix(".pdf")

    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = list(summary["rows"])
    fieldnames = [
        "platform",
        "requested_devices",
        "nm",
        "state_shape",
        "serial_median_s",
        "sharded_median_s",
        "speedup",
        "identity_passed",
        "max_abs_error",
        "max_rel_error",
        "max_phi_abs_error",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.9), constrained_layout=True)
    nms = sorted({int(row["nm"]) for row in rows})
    palette = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, max(len(nms), 1)))
    center = 0.5 * (len(nms) - 1)
    for series_index, (color, nm) in enumerate(zip(palette, nms, strict=True)):
        subset = sorted((row for row in rows if int(row["nm"]) == nm), key=lambda row: int(row["requested_devices"]))
        x = np.asarray([int(row["requested_devices"]) for row in subset], dtype=float)
        x_visible = x + 0.05 * (series_index - center)
        speedup = np.asarray([float(row["speedup"]) for row in subset], dtype=float)
        rel_error = np.asarray([float(row["max_rel_error"]) for row in subset], dtype=float)
        axes[0].plot(x, speedup, "o-", lw=2.0, color=color, label=f"Nm={nm}")
        axes[1].semilogy(
            x_visible,
            np.maximum(rel_error, 1.0e-16),
            "s-",
            lw=2.0,
            color=color,
            label=f"Nm={nm}",
        )
    axes[0].axhline(1.0, color="0.35", ls="--", lw=1.1)
    axes[0].set_xlabel("devices")
    axes[0].set_ylabel("serial / sharded median time")
    axes[0].set_title("Electrostatic RHS engineering speedup")
    axes[1].axhline(float(summary["rtol"]), color="0.35", ls="--", lw=1.1, label="relative gate")
    axes[1].set_xlabel("devices")
    axes[1].set_ylabel("max relative RHS error")
    status = "passed" if bool(summary["identity_passed"]) else "failed"
    axes[1].set_title(f"Identity {status}")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    return {"json": str(json_path), "csv": str(csv_path), "png": str(png_path), "pdf": str(pdf_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_PREFIX)
    parser.add_argument("--platform", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--devices", type=_parse_int_list, default=[1, 2, 4, 8])
    parser.add_argument("--nms", type=_parse_int_list, default=[64, 128])
    parser.add_argument("--nl", type=int, default=4)
    parser.add_argument("--ny", type=int, default=32)
    parser.add_argument("--nx", type=int, default=1)
    parser.add_argument("--nz", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--atol", type=float, default=2.0e-5)
    parser.add_argument("--rtol", type=float, default=1.0e-5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    devices = list(args.devices)
    if args.platform == "cpu":
        _configure_logical_cpu_devices(max(devices))
    summary = run_sweep(
        platform=str(args.platform),
        devices=devices,
        nms=list(args.nms),
        nx=int(args.nx),
        ny=int(args.ny),
        nz=int(args.nz),
        nl=int(args.nl),
        warmups=int(args.warmups),
        repeats=int(args.repeats),
        atol=float(args.atol),
        rtol=float(args.rtol),
    )
    paths = write_artifacts(summary, args.out_prefix)
    print(json.dumps({"identity_passed": summary["identity_passed"], "paths": paths}, indent=2))


if __name__ == "__main__":
    main()
