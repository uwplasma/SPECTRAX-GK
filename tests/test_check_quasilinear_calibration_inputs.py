"""Tests for quasilinear calibration input validation gates."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


def _load_tool_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "check_quasilinear_calibration_inputs.py"
    spec = importlib.util.spec_from_file_location("check_quasilinear_calibration_inputs", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_report(path: Path, artifact: str, *, split: str = "holdout") -> None:
    payload = {
        "kind": "quasilinear_calibration_report",
        "points": [
            {
                "case": "synthetic",
                "split": split,
                "predicted_heat_flux": 1.0,
                "observed_heat_flux": 1.1,
                "saturation_rule": "linear_weight",
                "nonlinear_artifact": artifact,
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_audit_passes_when_required_point_matches_passed_gate(tmp_path: Path) -> None:
    mod = _load_tool_module()
    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps(
            {
                "case": "synthetic_nonlinear_window",
                "spectrax": "tools_out/synthetic.csv",
                "gate_report": {"case": "synthetic_nonlinear_window", "passed": True, "gates": []},
            }
        ),
        encoding="utf-8",
    )
    report = tmp_path / "report.json"
    _write_report(report, "tools_out/synthetic.csv")

    paths = mod.write_audit([report], gate_patterns=[str(gate)], out_json=tmp_path / "audit.json", no_plot=True)

    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["reports"][0]["points"][0]["reason"] == "matched passed nonlinear gate"


def test_audit_normalizes_absolute_artifact_paths_from_other_checkouts(tmp_path: Path) -> None:
    mod = _load_tool_module()
    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps(
            {
                "case": "synthetic_nonlinear_window",
                "spectrax": "tools_out/synthetic.csv",
                "gate_report": {"case": "synthetic_nonlinear_window", "passed": True, "gates": []},
            }
        ),
        encoding="utf-8",
    )
    report = tmp_path / "report.json"
    _write_report(report, "/Users/example/local/SPECTRAX-GK/tools_out/synthetic.csv")

    paths = mod.write_audit([report], gate_patterns=[str(gate)], out_json=tmp_path / "audit.json", no_plot=True)

    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    point = payload["reports"][0]["points"][0]
    assert payload["passed"] is True
    assert point["nonlinear_artifact"] == "tools_out/synthetic.csv"
    assert point["reason"] == "matched passed nonlinear gate"


def test_audit_fails_when_required_point_uses_failed_gate(tmp_path: Path) -> None:
    mod = _load_tool_module()
    gate = tmp_path / "external_gate.json"
    gate.write_text(
        json.dumps(
            {
                "case": "external_cth_like",
                "promotion_gate": {"passed": False},
                "runs": [{"csv": "docs/_static/external_vmec_cth_like_nonlinear_t150_pilot.traces.csv"}],
            }
        ),
        encoding="utf-8",
    )
    report = tmp_path / "report.json"
    _write_report(report, "docs/_static/external_vmec_cth_like_nonlinear_t150_pilot.traces.csv")

    paths = mod.write_audit([report], gate_patterns=[str(gate)], out_json=tmp_path / "audit.json", no_plot=True)

    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert payload["reports"][0]["points"][0]["reason"] == "matching nonlinear gate is not passed"


def test_audit_fails_when_required_point_has_no_gate(tmp_path: Path) -> None:
    mod = _load_tool_module()
    report = tmp_path / "report.json"
    _write_report(report, "tools_out/missing.csv")

    paths = mod.write_audit([report], gate_patterns=[], out_json=tmp_path / "audit.json", no_plot=True)

    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert payload["reports"][0]["points"][0]["reason"] == "no matching nonlinear validation/convergence gate"


def test_audit_accepts_nested_high_grid_admission_input_artifact(tmp_path: Path) -> None:
    mod = _load_tool_module()
    gate = tmp_path / "high_grid_admission.json"
    gate.write_text(
        json.dumps(
            {
                "kind": "external_vmec_high_grid_admission_gate",
                "case": "synthetic high-grid admission",
                "inputs": {
                    "replicate_ensemble_gate": "docs/_static/replicate/ensemble_gate.json",
                },
                "promotion_gate": {"passed": True},
            }
        ),
        encoding="utf-8",
    )
    report = tmp_path / "report.json"
    _write_report(report, "docs/_static/replicate/ensemble_gate.json")

    paths = mod.write_audit([report], gate_patterns=[str(gate)], out_json=tmp_path / "audit.json", no_plot=True)

    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    point = payload["reports"][0]["points"][0]
    assert payload["passed"] is True
    assert point["matched_gate"]["case"] == "synthetic high-grid admission"


def test_audit_ignores_non_required_audit_split_without_gate(tmp_path: Path) -> None:
    mod = _load_tool_module()
    report = tmp_path / "report.json"
    _write_report(report, "tools_out/missing.csv", split="audit")

    paths = mod.write_audit([report], gate_patterns=[], out_json=tmp_path / "audit.json", no_plot=True)

    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["reports"][0]["points"][0]["reason"] == "not required split"


def test_tracked_quasilinear_train_holdout_reports_use_passed_nonlinear_gates() -> None:
    mod = _load_tool_module()
    root = Path(__file__).resolve().parents[1]
    reports = [
        root / "docs/_static/quasilinear_cyclone_miller_train_holdout_report.json",
        root / "docs/_static/quasilinear_hsx_train_holdout_report.json",
        root / "docs/_static/quasilinear_w7x_train_holdout_report.json",
        root / "docs/_static/quasilinear_stellarator_train_holdout_report.json",
    ]

    payload = mod.audit_calibration_inputs(reports)

    assert payload["passed"] is True
    required_rows = [
        point
        for report in payload["reports"]
        for point in report["points"]
        if point["required"]
    ]
    assert len(required_rows) == 17
    assert all(point["matched_gate"] is not None for point in required_rows)
    matched_cases = {point["matched_gate"]["case"] for point in required_rows}
    assert matched_cases == {
        "cyclone_nonlinear_long_window",
        "cyclone_miller_nonlinear_window",
        "hsx_nonlinear_window",
        "w7x_nonlinear_window",
        "D-shaped tokamak external VMEC nonlinear t250 high-grid convergence",
        "ITERModel external VMEC nonlinear t350 high-grid convergence",
        "updown_asym_external_vmec_t450",
        "circular_external_vmec_t450",
        "CTH-like external VMEC modified-protocol high-grid admission",
    }
    external_rows = [
        point for point in required_rows if "external_vmec" in str(point["matched_gate"]["artifact"])
    ]
    assert [point["case"] for point in external_rows] == [
        "dshape_external_vmec_t250_window",
        "itermodel_external_vmec_t350_window",
        "updown_asym_external_vmec_t450_window",
        "circular_external_vmec_t450_window",
        "cth_like_external_vmec_t700_high_grid_window",
    ]
