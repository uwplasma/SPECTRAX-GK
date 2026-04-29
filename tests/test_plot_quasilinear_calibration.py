"""Tests for quasilinear calibration plotting."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_tool_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "plot_quasilinear_calibration.py"
    spec = importlib.util.spec_from_file_location("plot_quasilinear_calibration", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plot_quasilinear_calibration_writes_artifacts(tmp_path: Path) -> None:
    mod = _load_tool_module()
    report = {
        "kind": "quasilinear_calibration_report",
        "claim_level": "training_or_audit_only",
        "passed": False,
        "holdout_mean_rel_gate": 0.35,
        "observed_floor": 1.0e-12,
        "points": [
            {
                "case": "cyclone",
                "split": "audit",
                "predicted_heat_flux": 0.1,
                "observed_heat_flux": 1.0,
                "observed_heat_flux_std": 0.2,
            }
        ],
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    paths = mod.write_calibration_figure(report_path, out=tmp_path / "calibration.png", title="QL audit")

    assert Path(paths["png"]).exists()
    assert Path(paths["pdf"]).exists()
    meta = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert meta["claim_level"] == "training_or_audit_only"
    assert meta["n_points"] == 1
    assert meta["mean_abs_relative_error"] == 0.9
