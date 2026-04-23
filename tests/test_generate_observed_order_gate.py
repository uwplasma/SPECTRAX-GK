from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


def _load_tool_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "generate_observed_order_gate.py"
    spec = importlib.util.spec_from_file_location("generate_observed_order_gate", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_convergence_series_from_resolution_column(tmp_path: Path) -> None:
    mod = _load_tool_module()
    path = tmp_path / "conv.csv"
    pd.DataFrame({"N": [16, 4, 8], "error": [1.0 / 16**2, 1.0 / 4**2, 1.0 / 8**2]}).to_csv(
        path,
        index=False,
    )

    h, err, rows = mod.load_convergence_series(
        path,
        step_column=None,
        resolution_column="N",
        error_column="error",
    )

    assert list(h) == pytest.approx([0.25, 0.125, 0.0625])
    assert list(err) == pytest.approx([1.0 / 4**2, 1.0 / 8**2, 1.0 / 16**2])
    assert rows[0]["step_source"] == "1/N"


def test_generate_observed_order_gate_main_writes_json_and_plot(tmp_path: Path) -> None:
    mod = _load_tool_module()
    csv_path = tmp_path / "conv.csv"
    out_json = tmp_path / "conv.json"
    out_png = tmp_path / "conv.png"
    pd.DataFrame({"h": [0.4, 0.2, 0.1], "err": [0.16, 0.04, 0.01]}).to_csv(
        csv_path,
        index=False,
    )

    assert (
        mod.main(
            [
                "--csv",
                str(csv_path),
                "--step-column",
                "h",
                "--error-column",
                "err",
                "--min-order",
                "1.9",
                "--max-final-error",
                "0.02",
                "--out-json",
                str(out_json),
                "--out-png",
                str(out_png),
            ]
        )
        == 0
    )

    payload = json.loads(out_json.read_text())
    assert payload["gate_passed"] is True
    assert payload["asymptotic_order"] == pytest.approx(2.0)
    assert out_png.exists()
    assert out_png.with_suffix(".pdf").exists()


def test_load_convergence_series_requires_one_step_source(tmp_path: Path) -> None:
    mod = _load_tool_module()
    path = tmp_path / "conv.csv"
    pd.DataFrame({"h": [0.1, 0.05], "N": [10, 20], "err": [0.01, 0.0025]}).to_csv(
        path,
        index=False,
    )

    with pytest.raises(ValueError, match="exactly one"):
        mod.load_convergence_series(path, step_column="h", resolution_column="N", error_column="err")
