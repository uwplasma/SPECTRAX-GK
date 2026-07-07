from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def _load_tool_module():
    path = (
        Path(__file__).resolve().parents[3]
        / "tools"
        / "artifacts"
        / "build_zonal_flow_objective_gate.py"
    )
    spec = importlib.util.spec_from_file_location(
        "build_zonal_flow_objective_gate", path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_build_zonal_flow_objective_gate_writes_diagnostic_artifacts(
    tmp_path: Path,
) -> None:
    mod = _load_tool_module()
    summary = tmp_path / "summary.csv"
    comparison = tmp_path / "compare.csv"
    _write_csv(
        summary,
        [
            {
                "kx_target": 0.05,
                "residual_level": 0.20,
                "residual_std": 0.03,
                "gam_damping_rate": "",
            },
            {
                "kx_target": 0.10,
                "residual_level": 0.40,
                "residual_std": 0.02,
                "gam_damping_rate": 0.04,
            },
        ],
    )
    _write_csv(
        comparison,
        [
            {"kx": 0.05, "tail_std": 0.12, "reference_tail_std": 0.03},
            {"kx": 0.10, "tail_std": 0.05, "reference_tail_std": 0.05},
        ],
    )
    out_json = tmp_path / "gate.json"
    out_csv = tmp_path / "gate.csv"
    out_png = tmp_path / "gate.png"

    rc = mod.main(
        [
            "--summary-csv",
            str(summary),
            "--comparison-csv",
            str(comparison),
            "--out-json",
            str(out_json),
            "--out-csv",
            str(out_csv),
            "--out-png",
            str(out_png),
            "--recurrence-source",
            "tail_std_ratio",
            "--missing-damping-policy",
            "zero",
            "--recurrence-weight",
            "0.5",
        ]
    )

    assert rc == 0
    assert out_json.exists()
    assert out_csv.exists()
    assert out_png.exists()
    assert out_png.with_suffix(".pdf").exists()
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["validation_status"] == "diagnostic"
    assert payload["promotion_ready"] is False
    assert payload["missing_damping_count"] == 1
    assert payload["sample_count"] == 2
    assert payload["recurrence_source"] == "tail_std_ratio"
    assert payload["gate_index_include"] is False
    recurrences = [row["recurrence_amplitude"] for row in payload["row_table"]]
    np.testing.assert_allclose(recurrences, [4.0, 1.0])
    json.dumps(payload, allow_nan=False)


def test_build_zonal_flow_objective_gate_fail_policy_rejects_missing_damping(
    tmp_path: Path,
) -> None:
    mod = _load_tool_module()
    summary = tmp_path / "summary.csv"
    _write_csv(
        summary,
        [
            {
                "kx_target": 0.05,
                "residual_level": 0.20,
                "residual_std": 0.03,
                "gam_damping_rate": "",
            }
        ],
    )

    with pytest.raises(ValueError, match="missing finite damping_rate"):
        mod.main(
            [
                "--summary-csv",
                str(summary),
                "--comparison-csv",
                str(tmp_path / "missing.csv"),
                "--missing-damping-policy",
                "fail",
            ]
        )
