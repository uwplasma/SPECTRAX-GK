"""Tests for the publication-facing GX validation panel builder."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def _write_plot(path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.plot([0.0, 1.0], [0.0, 1.0], linewidth=2.0)
    ax.set_title(title)
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)


def _write_linear_csv(path: Path) -> None:
    pd.DataFrame(
        {
            "ky": [0.1, 0.2, 0.3],
            "mean_abs_omega": [1.0e-5, 2.0e-5, 3.0e-5],
            "mean_rel_omega": [1.0e-3, 2.0e-3, 3.0e-3],
            "mean_abs_gamma": [4.0e-5, 5.0e-5, 6.0e-5],
            "mean_rel_gamma": [4.0e-2, 5.0e-2, 6.0e-2],
            "mean_rel_Wg": [1.0e-2, 2.0e-2, 3.0e-2],
            "mean_rel_Wphi": [1.5e-2, 2.5e-2, 3.5e-2],
            "mean_rel_Wapar": [0.0, 0.0, 0.0],
        }
    ).to_csv(path, index=False)


def test_publication_panel_builder_renders_png_and_pdf(tmp_path: Path) -> None:
    cyclone = tmp_path / "cyclone_kbm.png"
    w7x = tmp_path / "w7x.png"
    hsx = tmp_path / "hsx.png"
    w7x_csv = tmp_path / "w7x.csv"
    hsx_csv = tmp_path / "hsx.csv"
    out_png = tmp_path / "publication_panel.png"
    out_pdf = tmp_path / "publication_panel.pdf"

    _write_plot(cyclone, "Cyclone KBM")
    _write_plot(w7x, "W7-X")
    _write_plot(hsx, "HSX")
    _write_linear_csv(w7x_csv)
    _write_linear_csv(hsx_csv)

    subprocess.run(
        [
            sys.executable,
            str(TOOLS / "make_gx_publication_panel.py"),
            "--cyclone-kbm-panel",
            str(cyclone),
            "--w7x-panel",
            str(w7x),
            "--hsx-panel",
            str(hsx),
            "--w7x-linear-csv",
            str(w7x_csv),
            "--hsx-linear-csv",
            str(hsx_csv),
            "--out",
            str(out_png),
            "--pdf-out",
            str(out_pdf),
        ],
        check=True,
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(TOOLS)},
    )

    assert out_png.exists()
    assert out_pdf.exists()
    assert out_png.stat().st_size > 0
    assert out_pdf.stat().st_size > 0
