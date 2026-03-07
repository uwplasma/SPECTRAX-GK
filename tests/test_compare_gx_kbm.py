"""Tests for the deterministic GX KBM comparison harness."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def test_compare_gx_kbm_checkpoints_partial_rows(tmp_path: Path) -> None:
    tools_dir = Path(__file__).resolve().parents[1] / "tools"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_kbm as mod
    finally:
        sys.path.remove(str(tools_dir))

    out = tmp_path / "kbm.csv"
    rows = [
        {"ky": 0.2, "solver": "gx_time", "gamma": 1.0},
        {"ky": 0.3, "solver": "gx_time", "gamma": 2.0},
    ]

    mod._write_rows(out, rows[:1])
    first = pd.read_csv(out)
    assert list(first["ky"]) == [0.2]

    mod._write_rows(out, rows)
    second = pd.read_csv(out)
    assert list(second["ky"]) == [0.2, 0.3]
    assert list(second["gamma"]) == [1.0, 2.0]
