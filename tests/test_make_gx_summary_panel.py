"""Tests for the tracked GX summary panel builder."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from make_gx_summary_panel import _load_secondary, _secondary_table_rows


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
