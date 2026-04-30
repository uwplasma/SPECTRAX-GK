"""Tests for the VMEC/Boozer parity-matrix artifact builder."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "build_vmec_boozer_parity_matrix.py"


def _load_tool_module():
    spec = importlib.util.spec_from_file_location("build_vmec_boozer_parity_matrix", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fake_report(**kwargs: object) -> dict[str, object]:
    assert int(kwargs["mboz"]) >= 21
    assert int(kwargs["nboz"]) >= 21
    return {
        "available": True,
        "case_name": kwargs["case_name"],
        "status": "diagnostic_open",
        "mboz": kwargs["mboz"],
        "nboz": kwargs["nboz"],
        "equal_arc_core_worst_normalized_max_abs": 4.0e-3,
        "equal_arc_core_worst_scalar_rel": 2.0e-3,
        "equal_arc_derivative_worst_normalized_max_abs": 2.0e-2,
        "equal_arc_metric_worst_normalized_max_abs": 3.0e-2,
        "equal_arc_drift_worst_normalized_max_abs": 7.0e-2,
        "equal_arc_core_tolerance": 1.0e-2,
        "equal_arc_derivative_tolerance": 3.0e-2,
        "equal_arc_metric_tolerance": 8.0e-2,
        "equal_arc_drift_tolerance": 8.0e-2,
        "equal_arc_core_passed": True,
        "equal_arc_derivative_passed": True,
        "equal_arc_metric_passed": True,
        "equal_arc_drift_passed": True,
        "production_parity_passed": False,
        "worst_core_normalized_max_abs": 2.0e-1,
        "worst_scalar_rel": 1.0e-3,
    }


def test_build_parity_matrix_uses_mode21_floor_and_summarizes_rows() -> None:
    mod = _load_tool_module()
    cases = (
        mod.ParityCase("nfp4_QH_warm_start", "QH", "stellarator", 16),
        mod.ParityCase("nfp3_QI_fixed_resolution_final", "QI", "stellarator", 8),
    )

    payload = mod.build_parity_matrix(cases=cases, reporter=_fake_report)

    assert payload["kind"] == "vmec_boozer_parity_matrix"
    assert payload["minimum_boozer_mode_count"] == 21
    assert payload["summary"]["n_cases"] == 2
    assert payload["summary"]["n_equal_arc_passed"] == 2
    assert payload["summary"]["all_equal_arc_passed"] is True
    assert all(row["mode_floor_passed"] for row in payload["rows"])
    assert payload["rows"][0]["equal_arc_drift_worst_normalized_max_abs"] == pytest.approx(7.0e-2)


def test_build_parity_matrix_rejects_underresolved_boozer_modes() -> None:
    mod = _load_tool_module()
    cases = (mod.ParityCase("nfp3_QI_fixed_resolution_final", "QI", "stellarator", 8, mboz=20, nboz=21),)

    with pytest.raises(ValueError, match="mboz and nboz"):
        mod.build_parity_matrix(cases=cases, reporter=_fake_report)


def test_write_parity_matrix_artifacts_writes_companions(tmp_path: Path) -> None:
    mod = _load_tool_module()
    payload = mod.build_parity_matrix(
        cases=(mod.ParityCase("shaped_tokamak_pressure", "tokamak", "axisymmetric", 8),),
        reporter=_fake_report,
    )

    paths = mod.write_parity_matrix_artifacts(payload, out=tmp_path / "parity.png")

    for path in paths.values():
        assert Path(path).exists()
    saved = json.loads((tmp_path / "parity.json").read_text(encoding="utf-8"))
    assert saved["summary"]["n_equal_arc_passed"] == 1
    assert "shaped_tokamak_pressure" in (tmp_path / "parity.csv").read_text(encoding="utf-8")
