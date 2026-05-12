"""Tests for the bounded VMEC/Boozer QI robustness scan artifact."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "build_vmec_boozer_qi_robustness.py"


def _load_tool_module():
    spec = importlib.util.spec_from_file_location("build_vmec_boozer_qi_robustness", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fake_report(**kwargs: object) -> dict[str, object]:
    ntheta = int(kwargs["ntheta"])
    drift = 8.1879e-2 if ntheta == 8 else 7.020560597703801e-2
    passed = drift <= 8.0e-2
    return {
        "available": True,
        "case_name": kwargs["case_name"],
        "status": "diagnostic_open",
        "mboz": kwargs["mboz"],
        "nboz": kwargs["nboz"],
        "equal_arc_core_worst_normalized_max_abs": 6.3e-3,
        "equal_arc_core_worst_scalar_rel": 1.1e-3,
        "equal_arc_derivative_worst_normalized_max_abs": 9.3e-3,
        "equal_arc_metric_worst_normalized_max_abs": 8.8e-3,
        "equal_arc_drift_worst_normalized_max_abs": drift,
        "equal_arc_core_tolerance": 1.0e-2,
        "equal_arc_derivative_tolerance": 3.0e-2,
        "equal_arc_metric_tolerance": 8.0e-2,
        "equal_arc_drift_tolerance": 8.0e-2,
        "equal_arc_core_passed": True,
        "equal_arc_derivative_passed": True,
        "equal_arc_metric_passed": True,
        "equal_arc_drift_passed": passed,
    }


def test_qi_robustness_scan_selects_ntheta16_floor_without_relaxing_drift_gate() -> None:
    mod = _load_tool_module()

    payload = mod.build_qi_robustness_scan(
        scan_points=(
            mod.ScanPoint("nfp3_QI_fixed_resolution_final", "QI", 8),
            mod.ScanPoint("nfp3_QI_fixed_resolution_final", "QI", 16),
        ),
        reporter=_fake_report,
        parity_json=None,
        known_rerun_drift=0.081879,
        live=True,
    )

    assert payload["summary"]["robustness_status"] == "floor_selected"
    assert payload["summary"]["n_drift_only_failures"] == 1
    assert payload["selected_floor"]["ntheta"] == 16
    assert payload["selected_floor"]["mboz"] == 21
    assert payload["selected_floor"]["equal_arc_drift_worst_normalized_max_abs"] == pytest.approx(
        7.020560597703801e-2
    )
    failed = [row for row in payload["rows"] if row["ntheta"] == 8 and row["source"] == "live_scan"]
    assert failed[0]["equal_arc_drift_passed"] is False
    assert failed[0]["equal_arc_drift_worst_normalized_max_abs"] == pytest.approx(0.081879)


def test_qi_robustness_artifact_reads_tracked_parity_json(tmp_path: Path) -> None:
    mod = _load_tool_module()
    parity_json = tmp_path / "parity.json"
    parity_json.write_text(
        json.dumps(
            {
                "qi_seed_robustness": {
                    "rows": [
                        {
                            "case_name": "nfp3_QI_fixed_resolution_final",
                            "label": "QI fixed resolution, ntheta=16",
                            "ntheta": 16,
                            "mboz": 21,
                            "nboz": 21,
                            "available": True,
                            "qi_validation_evaluated": True,
                            "equal_arc_all_passed": True,
                            "equal_arc_core_passed": True,
                            "equal_arc_bgrad_passed": True,
                            "equal_arc_metric_passed": True,
                            "equal_arc_drift_passed": True,
                            "equal_arc_core_worst_normalized_max_abs": 6.2e-3,
                            "equal_arc_core_worst_scalar_rel": 1.0e-3,
                            "equal_arc_derivative_worst_normalized_max_abs": 9.2e-3,
                            "equal_arc_metric_worst_normalized_max_abs": 8.7e-3,
                            "equal_arc_drift_worst_normalized_max_abs": 7.1e-2,
                            "equal_arc_core_tolerance": 1.0e-2,
                            "equal_arc_derivative_tolerance": 3.0e-2,
                            "equal_arc_metric_tolerance": 8.0e-2,
                            "equal_arc_drift_tolerance": 8.0e-2,
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    payload = mod.build_qi_robustness_scan(parity_json=parity_json, live=False)
    paths = mod.write_qi_robustness_artifact(payload, out=tmp_path / "qi.json")

    assert Path(paths["json"]).exists()
    saved = json.loads((tmp_path / "qi.json").read_text(encoding="utf-8"))
    assert saved["selected_floor"]["ntheta"] == 16
    assert saved["summary"]["selected_floor_passed"] is True
