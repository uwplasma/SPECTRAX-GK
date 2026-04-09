from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from derive_imported_linear_lastvalue_table import _build_lastvalue_table, _load_scan


def test_build_lastvalue_table_converts_scan_columns() -> None:
    df = pd.DataFrame(
        {
            "ky": [0.1, 0.05],
            "gamma_last": [0.032, 0.012],
            "omega_last": [0.058, 0.029],
            "gamma_ref_last": [0.031, 0.011],
            "omega_ref_last": [0.059, 0.028],
        }
    )

    out = _build_lastvalue_table(df)

    assert list(out.columns) == ["ky", "gamma", "omega", "gamma_gx", "omega_gx", "rel_gamma", "rel_omega"]
    assert list(out["ky"]) == [0.05, 0.1]
    row = out.iloc[0]
    assert row["gamma"] == pytest.approx(0.012)
    assert row["gamma_gx"] == pytest.approx(0.011)
    assert row["rel_gamma"] == pytest.approx((0.012 - 0.011) / 0.011)


def test_load_scan_requires_lastvalue_columns(tmp_path: Path) -> None:
    path = tmp_path / "scan.csv"
    pd.DataFrame({"ky": [0.1], "gamma_last": [0.2]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        _load_scan(path)
