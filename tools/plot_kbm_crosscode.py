#!/usr/bin/env python3
"""Plot KBM cross-code comparison from GS2/stella mismatch CSV tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gs2-csv", type=Path, required=True)
    p.add_argument("--stella-csv", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("docs/_static/kbm_gs2_stella_comparison.png"))
    return p


def main() -> None:
    args = _build_parser().parse_args()
    gs2 = pd.read_csv(args.gs2_csv).sort_values("beta")
    stella = pd.read_csv(args.stella_csv).sort_values("beta")

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6.0, 5.0))
    ax0, ax1 = axes
    beta = gs2["beta"].to_numpy()

    ax0.plot(beta, gs2["gamma_spectrax"], "o-", label="SPECTRAX")
    ax0.plot(beta, gs2["gamma_ref"], "s--", label="GS2")
    ax0.plot(stella["beta"], stella["gamma_ref"], "d--", label="stella")
    ax0.set_ylabel(r"$\gamma$")
    ax0.set_title("KBM cross-code comparison (ky=0.3, matched inputs)")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="best")

    ax1.plot(beta, gs2["omega_spectrax"], "o-", label="SPECTRAX")
    ax1.plot(beta, gs2["omega_ref"], "s--", label="GS2")
    ax1.plot(stella["beta"], stella["omega_ref"], "d--", label="stella")
    ax1.set_xlabel(r"$\beta_{ref}$")
    ax1.set_ylabel(r"$\omega$")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    fig.savefig(args.out.with_suffix(".pdf"))
    print(f"saved figure: {args.out}")


if __name__ == "__main__":
    main()
