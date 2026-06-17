#!/usr/bin/env python3
"""Compare quasilinear and nonlinear ky-spectrum shapes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any
import sys

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from spectraxgk.artifacts.plotting import set_plot_style  # noqa: E402


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, np.generic):
        return _json_clean(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _netcdf_variable(root: Any, path: str) -> Any:
    current = root
    parts = [part for part in path.strip("/").split("/") if part]
    if not parts:
        raise ValueError("NetCDF variable path must not be empty")
    for group in parts[:-1]:
        if group not in current.groups:
            raise KeyError(f"NetCDF group '{group}' not found in '{path}'")
        current = current.groups[group]
    name = parts[-1]
    if name not in current.variables:
        raise KeyError(f"NetCDF variable '{name}' not found in '{path}'")
    return current.variables[name]


def _l1_distribution(values: np.ndarray, *, floor: float) -> np.ndarray:
    clean = np.where(np.isfinite(values), np.abs(values), 0.0)
    total = float(np.sum(clean))
    if total <= floor:
        raise ValueError("spectrum has zero total amplitude after cleaning")
    return clean / total


def build_spectrum_shape_report(
    *,
    spectrum_csv: str | Path,
    nonlinear_netcdf: str | Path,
    ql_column: str = "heat_flux_weight_total",
    nonlinear_variable: str = "Diagnostics/HeatFlux_kyst",
    time_min: float | None = None,
    time_max: float | None = None,
    species_index: int = 0,
    tv_gate: float = 0.2,
    cosine_gate: float = 0.95,
    floor: float = 1.0e-300,
) -> dict[str, Any]:
    """Build a normalized quasilinear-vs-nonlinear spectrum-shape report."""

    try:
        import netCDF4
    except ImportError as exc:  # pragma: no cover - exercised only without optional dependency.
        raise RuntimeError("netCDF4 is required to read resolved nonlinear spectra") from exc

    spectrum_path = Path(spectrum_csv)
    nonlinear_path = Path(nonlinear_netcdf)
    ql_data = np.genfromtxt(spectrum_path, delimiter=",", names=True)
    if ql_data.shape == ():
        ql_data = np.asarray([ql_data], dtype=ql_data.dtype)
    names = set(ql_data.dtype.names or ())
    if "ky" not in names:
        raise ValueError(f"{spectrum_path} is missing required column 'ky'")
    if ql_column not in names:
        raise ValueError(f"{spectrum_path} is missing quasilinear column '{ql_column}'")
    ql_ky = np.asarray(ql_data["ky"], dtype=float)
    ql_values = np.asarray(ql_data[ql_column], dtype=float)
    finite = np.isfinite(ql_ky) & np.isfinite(ql_values)
    if not np.any(finite):
        raise ValueError(f"{spectrum_path} contains no finite quasilinear samples")
    ql_ky = ql_ky[finite]
    ql_values = ql_values[finite]

    with netCDF4.Dataset(nonlinear_path) as root:
        ky = np.asarray(_netcdf_variable(root, "Grids/ky")[:], dtype=float)
        time = np.asarray(_netcdf_variable(root, "Grids/time")[:], dtype=float)
        resolved = np.asarray(_netcdf_variable(root, nonlinear_variable)[:], dtype=float)

    if resolved.ndim != 3:
        raise ValueError(f"{nonlinear_variable} must have shape (time, species, ky)")
    if species_index < 0 or species_index >= resolved.shape[1]:
        raise ValueError(f"species_index {species_index} is out of bounds for {resolved.shape[1]} species")
    if resolved.shape[0] != time.size or resolved.shape[2] != ky.size:
        raise ValueError("resolved nonlinear spectrum dimensions do not match time/ky grids")
    tmask = np.isfinite(time)
    if time_min is not None:
        tmask &= time >= float(time_min)
    if time_max is not None:
        tmask &= time <= float(time_max)
    if not np.any(tmask):
        raise ValueError("selected nonlinear time window contains no samples")
    nonlinear_spectrum = np.nanmean(np.abs(resolved[tmask, species_index, :]), axis=0)

    matched = []
    matched_indices = []
    for target in ql_ky:
        idx = int(np.argmin(np.abs(ky - target)))
        matched_indices.append(idx)
        matched.append(float(nonlinear_spectrum[idx]))
    matched_values = np.asarray(matched, dtype=float)
    ql_dist = _l1_distribution(ql_values, floor=floor)
    nl_dist = _l1_distribution(matched_values, floor=floor)
    residual = ql_dist - nl_dist
    tv_distance = float(0.5 * np.sum(np.abs(residual)))
    cosine = float(np.dot(ql_dist, nl_dist) / (np.linalg.norm(ql_dist) * np.linalg.norm(nl_dist)))
    passed = bool(tv_distance <= float(tv_gate) and cosine >= float(cosine_gate))
    return {
        "kind": "quasilinear_spectrum_shape_gate",
        "passed": passed,
        "spectrum_csv": str(spectrum_path),
        "nonlinear_netcdf": str(nonlinear_path),
        "ql_column": str(ql_column),
        "nonlinear_variable": str(nonlinear_variable),
        "time_min": None if time_min is None else float(time_min),
        "time_max": None if time_max is None else float(time_max),
        "time_samples": int(np.count_nonzero(tmask)),
        "species_index": int(species_index),
        "tv_gate": float(tv_gate),
        "cosine_gate": float(cosine_gate),
        "total_variation_distance": tv_distance,
        "cosine_similarity": cosine,
        "ky": ql_ky.tolist(),
        "nonlinear_ky": ky[np.asarray(matched_indices, dtype=int)].tolist(),
        "quasilinear_raw": ql_values.tolist(),
        "nonlinear_raw": matched_values.tolist(),
        "quasilinear_distribution": ql_dist.tolist(),
        "nonlinear_distribution": nl_dist.tolist(),
        "distribution_residual": residual.tolist(),
    }


def write_spectrum_shape_figure(report: dict[str, Any], *, out: str | Path, title: str) -> dict[str, str]:
    """Write PNG/PDF/JSON artifacts for a spectrum-shape report."""

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ky = np.asarray(report["ky"], dtype=float)
    ql = np.asarray(report["quasilinear_distribution"], dtype=float)
    nl = np.asarray(report["nonlinear_distribution"], dtype=float)
    residual = np.asarray(report["distribution_residual"], dtype=float)

    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    ax0, ax1 = axes
    ax0.plot(ky, ql, marker="o", linewidth=2.0, label="quasilinear weight", color="#0f4c81")
    ax0.plot(ky, nl, marker="s", linewidth=2.0, label="nonlinear resolved heat flux", color="#d1495b")
    ax0.set_xlabel(r"$k_y\rho_i$")
    ax0.set_ylabel("normalized spectrum")
    ax0.set_title("Shape comparison")
    ax0.legend(loc="best")
    ax0.grid(True, alpha=0.25)

    width = 0.75 * float(np.min(np.diff(ky))) if ky.size > 1 else 0.02
    ax1.bar(ky, residual, width=width, color=np.where(residual >= 0.0, "#2a9d8f", "#f97316"))
    ax1.axhline(0.0, color="0.2", linewidth=1.0)
    ax1.set_xlabel(r"$k_y\rho_i$")
    ax1.set_ylabel("QL - nonlinear")
    ax1.set_title("Distribution residual")
    ax1.text(
        0.03,
        0.95,
        f"passed = {report['passed']}\n"
        f"TV = {float(report['total_variation_distance']):.3f} <= {float(report['tv_gate']):.3f}\n"
        f"cosine = {float(report['cosine_similarity']):.3f} >= {float(report['cosine_gate']):.3f}",
        transform=ax1.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.75", "alpha": 0.92},
    )
    ax1.grid(True, axis="y", alpha=0.25)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps(_json_clean(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"png": str(out_path), "pdf": str(pdf_path), "json": str(json_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spectrum", required=True, help="Input quasilinear spectrum CSV")
    parser.add_argument("--nonlinear", required=True, help="Input nonlinear NetCDF with resolved diagnostics")
    parser.add_argument("--out", required=True, help="Output PNG path")
    parser.add_argument("--ql-column", default="heat_flux_weight_total")
    parser.add_argument("--nonlinear-variable", default="Diagnostics/HeatFlux_kyst")
    parser.add_argument("--time-min", type=float, default=None)
    parser.add_argument("--time-max", type=float, default=None)
    parser.add_argument("--species-index", type=int, default=0)
    parser.add_argument("--tv-gate", type=float, default=0.2)
    parser.add_argument("--cosine-gate", type=float, default=0.95)
    parser.add_argument("--title", default="Quasilinear/nonlinear ky-spectrum shape gate")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_spectrum_shape_report(
        spectrum_csv=args.spectrum,
        nonlinear_netcdf=args.nonlinear,
        ql_column=args.ql_column,
        nonlinear_variable=args.nonlinear_variable,
        time_min=args.time_min,
        time_max=args.time_max,
        species_index=args.species_index,
        tv_gate=args.tv_gate,
        cosine_gate=args.cosine_gate,
    )
    paths = write_spectrum_shape_figure(report, out=args.out, title=args.title)
    print(f"saved {paths['png']}")
    print(f"saved {paths['pdf']}")
    print(f"saved {paths['json']}")
    print(
        "passed={passed} total_variation_distance={tv:.6g} cosine_similarity={cos:.6g}".format(
            passed=report["passed"],
            tv=report["total_variation_distance"],
            cos=report["cosine_similarity"],
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
