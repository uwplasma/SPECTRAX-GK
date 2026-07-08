"""Tests for tracked reference-comparison panel builders."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT / "tools" / "comparison"
sys.path.insert(0, str(TOOLS))

from make_reference_panels import (  # noqa: E402
    STATIC,
    _autocrop_image,
    _linear_table_rows,
    _load_imported_linear,
    _load_imported_linear_lastvalue,
    _load_secondary,
    _plot_imported_linear,
    _plot_secondary,
    _secondary_table_rows,
    build_parser,
)


def _write_plot(path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([0.0, 1.0], [0.0, 1.0], linewidth=2.0)
    ax.set_title(title)
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)


def _write_imported_linear_csv(path: Path) -> None:
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


def _write_imported_lastvalue_csv(path: Path) -> None:
    pd.DataFrame(
        {
            "ky": [0.1, 0.2, 0.3],
            "rel_gamma": [5.0e-3, 4.0e-3, 3.0e-3],
            "rel_omega": [6.0e-4, 5.0e-4, 4.0e-4],
            "gamma": [0.1, 0.2, 0.3],
            "gamma_gx": [0.1, 0.2, 0.3],
            "omega": [-0.2, -0.3, -0.4],
            "omega_gx": [-0.2, -0.3, -0.4],
        }
    ).to_csv(path, index=False)


def _write_linear_mismatch_csv(path: Path) -> None:
    pd.DataFrame(
        {
            "ky": [0.05, 0.10, 0.20],
            "gamma": [0.01, 0.03, 0.08],
            "omega": [0.03, 0.06, 0.13],
            "gamma_gx": [0.011, 0.029, 0.079],
            "omega_gx": [0.031, 0.061, 0.129],
        }
    ).to_csv(path, index=False)


def _write_secondary_csv(path: Path) -> None:
    pd.DataFrame(
        {
            "ky": [0.1],
            "kx": [0.05],
            "gamma_gx": [4.9],
            "gamma_sp": [4.91],
            "rel_gamma": [2.0e-3],
            "omega_gx": [-1.6e-4],
            "omega_sp": [3.0e-7],
        }
    ).to_csv(path, index=False)


def test_load_secondary_adds_abs_omega_when_missing(tmp_path: Path) -> None:
    path = tmp_path / "secondary.csv"
    _write_secondary_csv(path)
    df = _load_secondary(path)
    assert "abs_omega" in df.columns
    assert float(df.loc[0, "abs_omega"]) == pytest.approx(1.603e-4)


def test_secondary_table_rows_format_expected_values(tmp_path: Path) -> None:
    path = tmp_path / "secondary.csv"
    pd.DataFrame(
        {
            "ky": [0.0],
            "kx": [-0.05],
            "gamma_gx": [4.901835],
            "gamma_sp": [4.901937],
            "rel_gamma": [2.1e-5],
            "omega_gx": [-1.6e-4],
            "omega_sp": [2.6e-7],
            "abs_omega": [1.6026e-4],
        }
    ).to_csv(path, index=False)
    rows = _secondary_table_rows(_load_secondary(path))
    assert rows == [
        [
            "(0.00, -0.05)",
            "4.901835",
            "4.901937",
            "2.10e-05",
            "-1.60e-04",
            "2.60e-07",
            "1.60e-04",
        ]
    ]


def test_parser_defaults_to_repository_static_assets() -> None:
    args = build_parser().parse_args(["summary"])
    assert args.secondary_csv == STATIC / "comparison" / "secondary_reference_out_compare.csv"
    assert STATIC == ROOT / "docs" / "_static"


def test_load_imported_linear_requires_expected_columns(tmp_path: Path) -> None:
    path = tmp_path / "linear.csv"
    pd.DataFrame({"ky": [0.1], "mean_rel_omega": [1.0e-3]}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        _load_imported_linear(path)


def test_load_imported_linear_lastvalue_requires_expected_columns(
    tmp_path: Path,
) -> None:
    path = tmp_path / "linear_lastvalue.csv"
    pd.DataFrame({"ky": [0.1], "rel_gamma": [1.0e-3]}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        _load_imported_linear_lastvalue(path)


def test_linear_table_rows_formats_scan_metrics(tmp_path: Path) -> None:
    path = tmp_path / "linear.csv"
    _write_imported_linear_csv(path)
    rows = _linear_table_rows(_load_imported_linear(path))
    assert rows[0] == ["0.100", "1.00e-05", "4.00e-05", "1.00e-02", "1.50e-02"]


def test_plot_imported_linear_adds_lines_and_log_axis() -> None:
    df = pd.DataFrame(
        {
            "ky": [0.1, 0.2],
            "mean_rel_omega": [1.0e-3, 2.0e-3],
            "mean_rel_gamma": [3.0e-2, 4.0e-2],
            "mean_rel_Wg": [5.0e-5, 6.0e-5],
            "mean_rel_Wphi": [7.0e-5, 8.0e-5],
            "mean_rel_Wapar": [0.0, 0.0],
        }
    )
    lastvalue = pd.DataFrame(
        {
            "ky": [0.1, 0.2],
            "rel_gamma": [5.0e-3, 6.0e-3],
            "rel_omega": [7.0e-4, 8.0e-4],
            "gamma": [0.1, 0.2],
            "gamma_gx": [0.1, 0.2],
            "omega": [-0.2, -0.3],
            "omega_gx": [-0.2, -0.3],
        }
    )
    fig, ax = plt.subplots()
    try:
        _plot_imported_linear(
            ax, df, "Imported", lastvalue=lastvalue, note="late-time closure"
        )
        assert ax.get_yscale() == "log"
        assert len(ax.lines) == 6
        assert any(text.get_text() == "late-time closure" for text in ax.texts)
    finally:
        plt.close(fig)


def test_plot_secondary_adds_gamma_lines_and_abs_omega_bars() -> None:
    df = pd.DataFrame(
        {
            "ky": [0.0, 0.1],
            "kx": [-0.05, 0.05],
            "gamma_gx": [4.9, 4.9],
            "gamma_sp": [4.9, 4.9],
            "abs_omega": [1.0e-4, 2.0e-4],
        }
    )
    fig, ax = plt.subplots()
    try:
        _plot_secondary(ax, df, "Secondary")
        assert len(ax.lines) == 2
        assert len(fig.axes) == 2
    finally:
        plt.close(fig)


def test_autocrop_image_trims_uniform_border() -> None:
    image = np.ones((10, 12, 3), dtype=float)
    image[3:7, 4:9, :] = 0.0
    cropped = _autocrop_image(image, white_threshold=0.95, pad_pixels=0)
    assert cropped.shape == (4, 5, 3)


def test_reference_panel_subcommands_render_outputs(tmp_path: Path) -> None:
    cyclone_linear = tmp_path / "cyclone_linear.csv"
    kbm_linear = tmp_path / "kbm_linear.csv"
    cyclone_nl = tmp_path / "cyclone_nl.png"
    kbm_nl = tmp_path / "kbm_nl.png"
    cyclone_kbm = tmp_path / "cyclone_kbm.png"
    w7x = tmp_path / "w7x.png"
    hsx = tmp_path / "hsx.png"
    w7x_csv = tmp_path / "w7x.csv"
    hsx_csv = tmp_path / "hsx.csv"
    w7x_lastvalue_csv = tmp_path / "w7x_last.csv"
    hsx_lastvalue_csv = tmp_path / "hsx_last.csv"
    secondary = tmp_path / "secondary.csv"

    _write_linear_mismatch_csv(cyclone_linear)
    _write_linear_mismatch_csv(kbm_linear)
    _write_plot(cyclone_nl, "Cyclone")
    _write_plot(kbm_nl, "KBM")
    _write_plot(cyclone_kbm, "Cyclone KBM")
    _write_plot(w7x, "W7-X")
    _write_plot(hsx, "HSX")
    _write_imported_linear_csv(w7x_csv)
    _write_imported_linear_csv(hsx_csv)
    _write_imported_lastvalue_csv(w7x_lastvalue_csv)
    _write_imported_lastvalue_csv(hsx_lastvalue_csv)
    _write_secondary_csv(secondary)

    tokamak_out = tmp_path / "tokamak_panel.png"
    publication_png = tmp_path / "publication_panel.png"
    publication_pdf = tmp_path / "publication_panel.pdf"
    summary_out = tmp_path / "summary_panel.png"
    base_env = {**os.environ, "PYTHONPATH": str(TOOLS)}

    subprocess.run(
        [
            sys.executable,
            str(TOOLS / "make_reference_panels.py"),
            "tokamak",
            "--cyclone-linear",
            str(cyclone_linear),
            "--kbm-linear",
            str(kbm_linear),
            "--cyclone-nonlinear-panel",
            str(cyclone_nl),
            "--kbm-nonlinear-panel",
            str(kbm_nl),
            "--out",
            str(tokamak_out),
        ],
        check=True,
        cwd=ROOT,
        env=base_env,
    )
    subprocess.run(
        [
            sys.executable,
            str(TOOLS / "make_reference_panels.py"),
            "publication",
            "--cyclone-kbm-panel",
            str(cyclone_kbm),
            "--w7x-panel",
            str(w7x),
            "--hsx-panel",
            str(hsx),
            "--w7x-linear-csv",
            str(w7x_csv),
            "--hsx-linear-csv",
            str(hsx_csv),
            "--w7x-linear-lastvalue-csv",
            str(w7x_lastvalue_csv),
            "--hsx-linear-lastvalue-csv",
            str(hsx_lastvalue_csv),
            "--out",
            str(publication_png),
            "--pdf-out",
            str(publication_pdf),
        ],
        check=True,
        cwd=ROOT,
        env=base_env,
    )
    subprocess.run(
        [
            sys.executable,
            str(TOOLS / "make_reference_panels.py"),
            "summary",
            "--cyclone-kbm-panel",
            str(cyclone_kbm),
            "--w7x-panel",
            str(w7x),
            "--hsx-panel",
            str(hsx),
            "--w7x-linear-csv",
            str(w7x_csv),
            "--hsx-linear-csv",
            str(hsx_csv),
            "--w7x-linear-lastvalue-csv",
            str(w7x_lastvalue_csv),
            "--hsx-linear-lastvalue-csv",
            str(hsx_lastvalue_csv),
            "--secondary-csv",
            str(secondary),
            "--out",
            str(summary_out),
        ],
        check=True,
        cwd=ROOT,
        env=base_env,
    )

    for path in [tokamak_out, publication_png, publication_pdf, summary_out]:
        assert path.exists()
        assert path.stat().st_size > 0
