"""Tests for the tokamak GX validation panel builder."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def _write_plot(path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([0.0, 1.0], [0.0, 1.0], linewidth=2.0)
    ax.set_title(title)
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)


def _write_linear_csv(path: Path) -> None:
    pd.DataFrame(
        {
            "ky": [0.05, 0.10, 0.20],
            "gamma": [0.01, 0.03, 0.08],
            "omega": [0.03, 0.06, 0.13],
            "gamma_gx": [0.011, 0.029, 0.079],
            "omega_gx": [0.031, 0.061, 0.129],
        }
    ).to_csv(path, index=False)


def test_tokamak_panel_builder_renders_png(tmp_path: Path) -> None:
    cyclone_linear = tmp_path / "cyclone_linear.csv"
    kbm_linear = tmp_path / "kbm_linear.csv"
    cyclone_nl = tmp_path / "cyclone_nl.png"
    kbm_nl = tmp_path / "kbm_nl.png"
    out = tmp_path / "tokamak_panel.png"

    _write_linear_csv(cyclone_linear)
    _write_linear_csv(kbm_linear)
    _write_plot(cyclone_nl, "Cyclone")
    _write_plot(kbm_nl, "KBM")

    subprocess.run(
        [
            sys.executable,
            str(TOOLS / "make_gx_cyclone_kbm_panel.py"),
            "--cyclone-linear",
            str(cyclone_linear),
            "--kbm-linear",
            str(kbm_linear),
            "--cyclone-nonlinear-panel",
            str(cyclone_nl),
            "--kbm-nonlinear-panel",
            str(kbm_nl),
            "--out",
            str(out),
        ],
        check=True,
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(TOOLS)},
    )

    assert out.exists()
    assert out.stat().st_size > 0
