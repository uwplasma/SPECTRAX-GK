#!/usr/bin/env python3
"""Plot a quasilinear ky spectrum from ``*.quasilinear_spectrum.csv``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from spectraxgk.artifacts.plotting import set_plot_style  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "docs" / "_static" / "quasilinear_cyclone_spectrum.png"


def _json_clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, np.generic):
        return _json_clean(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def load_quasilinear_spectrum(path: str | Path) -> dict[str, np.ndarray]:
    """Load a quasilinear spectrum CSV written by the runtime scan path."""

    arr = np.genfromtxt(path, delimiter=",", names=True)
    if arr.shape == ():
        arr = np.asarray([arr], dtype=arr.dtype)
    required = {
        "ky",
        "gamma",
        "omega",
        "kperp_eff2",
        "heat_flux_weight_total",
        "particle_flux_weight_total",
        "amplitude2",
        "saturated_heat_flux_total",
    }
    missing = sorted(required - set(arr.dtype.names or ()))
    if missing:
        raise ValueError(f"missing quasilinear spectrum column(s): {missing}")
    return {name: np.asarray(arr[name], dtype=float) for name in arr.dtype.names or ()}


def plot_quasilinear_spectrum(
    spectrum: dict[str, np.ndarray],
    *,
    title: str = "Quasilinear Transport Spectrum",
) -> tuple[plt.Figure, np.ndarray]:
    """Create a publication-style quasilinear spectrum panel."""

    set_plot_style()
    ky = np.asarray(spectrum["ky"], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.4), constrained_layout=True)
    ax00, ax01, ax10, ax11 = axes.ravel()

    gamma_line = ax00.plot(ky, spectrum["gamma"], marker="o", linewidth=2.2, color="#0f4c81", label=r"$\gamma$")
    ax00.axhline(0.0, color="#444444", linewidth=1.0, alpha=0.6)
    ax00.set_xlabel(r"$k_y \rho_i$")
    ax00.set_ylabel(r"$\gamma$", color="#0f4c81")
    ax00.tick_params(axis="y", labelcolor="#0f4c81")
    ax00.set_title("Linear mode")
    ax00r = ax00.twinx()
    omega_line = ax00r.plot(ky, spectrum["omega"], marker="s", linewidth=2.0, color="#c44e52", label=r"$\omega$")
    ax00r.set_ylabel(r"$\omega$", color="#c44e52")
    ax00r.tick_params(axis="y", labelcolor="#c44e52")
    lines = gamma_line + omega_line
    ax00.legend(lines, [line.get_label() for line in lines], frameon=False, loc="upper left")

    ax01.plot(ky, spectrum["kperp_eff2"], marker="o", linewidth=2.2, color="#2a9d8f")
    ax01.set_xlabel(r"$k_y \rho_i$")
    ax01.set_ylabel(r"$k_{\perp,\mathrm{eff}}^2$")
    ax01.set_title("Eigenfunction-weighted scale")

    ax10.plot(
        ky,
        spectrum["heat_flux_weight_total"],
        marker="o",
        linewidth=2.2,
        color="#0f4c81",
        label="heat",
    )
    ax10.plot(
        ky,
        spectrum["particle_flux_weight_total"],
        marker="s",
        linewidth=2.0,
        color="#f4a261",
        label="particle",
    )
    ax10.axhline(0.0, color="#444444", linewidth=1.0, alpha=0.6)
    ax10.set_xlabel(r"$k_y \rho_i$")
    ax10.set_ylabel("linear flux weight")
    ax10.set_title("Amplitude-normalized weights")
    ax10.legend(frameon=False)

    sat_heat = np.asarray(spectrum["saturated_heat_flux_total"], dtype=float)
    amp2 = np.asarray(spectrum["amplitude2"], dtype=float)
    heat_lines = []
    if np.isfinite(sat_heat).any():
        heat_lines = ax11.plot(ky, sat_heat, marker="o", linewidth=2.2, color="#c44e52", label="heat estimate")
    ax11.axhline(0.0, color="#444444", linewidth=1.0, alpha=0.6)
    ax11.set_xlabel(r"$k_y \rho_i$")
    ax11.set_ylabel("heat estimate", color="#c44e52")
    ax11.tick_params(axis="y", labelcolor="#c44e52")
    ax11.set_title("Uncalibrated rule output")
    amp_lines = []
    if np.isfinite(amp2).any():
        ax11r = ax11.twinx()
        amp_lines = ax11r.plot(ky, amp2, marker="^", linewidth=1.8, color="#6c757d", linestyle="--", label=r"$A^2$")
        ax11r.set_ylabel(r"$A^2$", color="#6c757d")
        ax11r.tick_params(axis="y", labelcolor="#6c757d")
    lines = heat_lines + amp_lines
    if lines:
        ax11.legend(lines, [line.get_label() for line in lines], frameon=False, loc="upper right")

    fig.suptitle(title, fontsize=14)
    return fig, axes


def write_quasilinear_spectrum_figure(
    spectrum_csv: str | Path,
    *,
    out: str | Path = DEFAULT_OUT,
    title: str = "Quasilinear Transport Spectrum",
) -> dict[str, str]:
    """Load a spectrum CSV and write PNG/PDF/JSON plot artifacts."""

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    spectrum = load_quasilinear_spectrum(spectrum_csv)
    fig, _axes = plot_quasilinear_spectrum(spectrum, title=title)
    fig.savefig(out_path, dpi=220)
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path)
    plt.close(fig)
    meta = {
        "kind": "quasilinear_spectrum_figure",
        "source": str(spectrum_csv),
        "png": str(out_path),
        "pdf": str(pdf_path),
        "n_ky": int(np.asarray(spectrum["ky"]).size),
        "ky_min": float(np.min(spectrum["ky"])),
        "ky_max": float(np.max(spectrum["ky"])),
    }
    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps(_json_clean(meta), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"png": str(out_path), "pdf": str(pdf_path), "json": str(json_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spectrum", required=True, help="Input *.quasilinear_spectrum.csv")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output PNG path")
    parser.add_argument("--title", default="Quasilinear Transport Spectrum")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = write_quasilinear_spectrum_figure(args.spectrum, out=args.out, title=args.title)
    print(f"saved {paths['png']}")
    print(f"saved {paths['pdf']}")
    print(f"saved {paths['json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
