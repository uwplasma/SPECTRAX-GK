"""Tests for the tracked GX summary panel builder."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from make_gx_summary_panel import (
    STATIC,
    _autocrop_image,
    _linear_table_rows,
    _load_imported_linear,
    _load_imported_linear_lastvalue,
    _plot_imported_linear,
    _plot_secondary,
    _load_secondary,
    _secondary_table_rows,
    build_parser,
)


def test_load_secondary_adds_abs_omega_when_missing(tmp_path: Path) -> None:
    path = tmp_path / "secondary.csv"
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
    assert rows == [["(0.00, -0.05)", "4.901835", "4.901937", "2.10e-05", "-1.60e-04", "2.60e-07", "1.60e-04"]]


def test_parser_defaults_to_real_secondary_out_nc_asset() -> None:
    args = build_parser().parse_args([])
    assert args.secondary_csv == STATIC / "secondary_gx_out_compare.csv"


def test_load_imported_linear_requires_expected_columns(tmp_path: Path) -> None:
    path = tmp_path / "linear.csv"
    pd.DataFrame({"ky": [0.1], "mean_rel_omega": [1.0e-3]}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        _load_imported_linear(path)


def test_load_imported_linear_lastvalue_requires_expected_columns(tmp_path: Path) -> None:
    path = tmp_path / "linear_lastvalue.csv"
    pd.DataFrame({"ky": [0.1], "rel_gamma": [1.0e-3]}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        _load_imported_linear_lastvalue(path)


def test_linear_table_rows_formats_scan_metrics(tmp_path: Path) -> None:
    path = tmp_path / "linear.csv"
    pd.DataFrame(
        {
            "ky": [0.1],
            "mean_abs_omega": [1.0e-6],
            "mean_rel_omega": [2.0e-3],
            "mean_abs_gamma": [3.0e-5],
            "mean_rel_gamma": [4.0e-2],
            "mean_rel_Wg": [5.0e-6],
            "mean_rel_Wphi": [6.0e-6],
            "mean_rel_Wapar": [0.0],
        }
    ).to_csv(path, index=False)
    rows = _linear_table_rows(_load_imported_linear(path))
    assert rows == [["0.100", "1.00e-06", "3.00e-05", "5.00e-06", "6.00e-06"]]


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
        _plot_imported_linear(ax, df, "Imported", lastvalue=lastvalue, note="late-time closure")
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
