"""Generate publication-ready figures for docs and README."""

from __future__ import annotations

from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spectraxgk.benchmarking.shared import (
    load_cyclone_reference,
    load_etg_reference,
    load_kbm_reference,
    LinearScanResult,
)
from spectraxgk.runtime import run_runtime_scan
from spectraxgk.workflows.runtime.toml import load_runtime_from_toml
from spectraxgk.artifacts.plotting import (
    cyclone_comparison_figure,
    cyclone_reference_figure,
    scan_comparison_figure,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate validation figures.")
    parser.add_argument(
        "--case",
        choices=["all", "cyclone", "etg"],
        default="all",
        help="Limit figure generation to a specific case.",
    )
    parser.add_argument("--quiet", action="store_true", help="Disable verbose logging.")
    parser.add_argument(
        "--no-progress", action="store_true", help="Disable tqdm progress bars."
    )
    return parser.parse_args(argv)


def _run_etg_figures(*, outdir: Path, verbose: bool, progress: bool) -> None:
    etg_ref = load_etg_reference()
    mismatch_csv = outdir / "etg_mismatch_table.csv"
    if mismatch_csv.exists():
        etg_scan = _load_spectrax_scan_from_mismatch(mismatch_csv)
    else:
        cfg_etg, _ = load_runtime_from_toml(
            ROOT / "examples/linear/axisymmetric/etg.toml"
        )
        etg_ky = np.asarray(etg_ref.ky)
        etg_scan = run_runtime_scan(
            cfg_etg,
            etg_ky,
            Nl=24,
            Nm=8,
            solver="time",
            batch_ky=True,
            method=cfg_etg.time.method,
            dt=cfg_etg.time.dt,
            steps=int(round(cfg_etg.time.t_max / cfg_etg.time.dt)),
            sample_stride=cfg_etg.time.sample_stride,
            auto_window=False,
            tmin=1.0,
            tmax=cfg_etg.time.t_max,
            fit_signal="phi",
            mode_method="z_index",
            show_progress=progress,
        )
    fig, _axes = scan_comparison_figure(
        etg_scan.ky,
        etg_scan.gamma,
        etg_scan.omega,
        r"$k_y \rho_i$",
        "ETG Benchmark Scan",
        x_ref=etg_ref.ky,
        gamma_ref=etg_ref.gamma,
        omega_ref=etg_ref.omega,
        label="SPECTRAX-GK",
        ref_label="Reference",
        log_x=True,
    )
    fig.savefig(outdir / "etg_comparison.png", dpi=200)
    fig.savefig(outdir / "etg_comparison.pdf")


def _run_kbm_figures(*, outdir: Path) -> None:
    kbm_ref = load_kbm_reference()
    mismatch_csv = outdir / "kbm_mismatch_table.csv"
    if not mismatch_csv.exists():
        raise FileNotFoundError(
            f"missing {mismatch_csv}; generate the KBM mismatch table first"
        )
    kbm_scan = _load_spectrax_scan_from_mismatch(mismatch_csv)
    fig, _axes = scan_comparison_figure(
        kbm_scan.ky,
        kbm_scan.gamma,
        kbm_scan.omega,
        r"$\beta$",
        "KBM Benchmark Scan",
        x_ref=kbm_ref.ky,
        gamma_ref=kbm_ref.gamma,
        omega_ref=kbm_ref.omega,
        label="SPECTRAX-GK",
        ref_label="Reference",
        log_x=False,
    )
    fig.savefig(outdir / "kbm_comparison.png", dpi=200)
    fig.savefig(outdir / "kbm_comparison.pdf")


def _load_spectrax_scan_from_mismatch(
    csv_path: Path, *, x_col: str = "ky"
) -> LinearScanResult:
    df = pd.read_csv(csv_path).sort_values(x_col)
    return LinearScanResult(
        ky=df[x_col].to_numpy(dtype=float),
        gamma=df["gamma_spectrax"].to_numpy(dtype=float),
        omega=df["omega_spectrax"].to_numpy(dtype=float),
    )


def _load_reference_from_mismatch(csv_path: Path, *, x_col: str) -> LinearScanResult:
    df = pd.read_csv(csv_path).sort_values(x_col)
    return LinearScanResult(
        ky=df[x_col].to_numpy(dtype=float),
        gamma=df["gamma_ref"].to_numpy(dtype=float),
        omega=df["omega_ref"].to_numpy(dtype=float),
    )


def _load_cyclone_scan_from_rows(csv_path: Path) -> LinearScanResult:
    df = pd.read_csv(csv_path).sort_values("ky")
    return LinearScanResult(
        ky=df["ky"].to_numpy(dtype=float),
        gamma=df["gamma_spectrax"].to_numpy(dtype=float),
        omega=df["omega_spectrax"].to_numpy(dtype=float),
    )


def _cyclone_refresh_reference(ref: LinearScanResult) -> LinearScanResult:
    keep = np.asarray(ref.ky) <= 0.45 + 1.0e-12
    return LinearScanResult(
        ky=np.asarray(ref.ky)[keep],
        gamma=np.asarray(ref.gamma)[keep],
        omega=np.asarray(ref.omega)[keep],
    )


def main() -> int:
    args = _parse_args()
    verbose = not args.quiet
    progress = not args.no_progress

    outdir = ROOT / "docs" / "_static"
    outdir.mkdir(parents=True, exist_ok=True)
    if args.case == "etg":
        _run_etg_figures(outdir=outdir, verbose=verbose, progress=progress)
        return 0

    # Cyclone reference (adiabatic electrons)
    ref_full = load_cyclone_reference()
    ref = _cyclone_refresh_reference(ref_full)
    fig, _axes = cyclone_reference_figure(ref)
    fig.savefig(outdir / "cyclone_reference.png", dpi=200)
    fig.savefig(outdir / "cyclone_reference.pdf")

    scan = _load_cyclone_scan_from_rows(outdir / "cyclone_mismatch_table.csv")
    fig, _axes = cyclone_comparison_figure(ref, scan)
    fig.savefig(outdir / "cyclone_comparison.png", dpi=200)
    fig.savefig(outdir / "cyclone_comparison.pdf")
    if args.case == "cyclone":
        return 0

    _run_etg_figures(outdir=outdir, verbose=verbose, progress=progress)
    _run_kbm_figures(outdir=outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
