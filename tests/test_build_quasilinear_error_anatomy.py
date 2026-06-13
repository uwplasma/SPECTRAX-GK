from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys


def _load_tool_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "tools" / "build_quasilinear_error_anatomy.py"
    spec = importlib.util.spec_from_file_location("build_quasilinear_error_anatomy", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_error_anatomy_locks_current_fail_closed_residual_story() -> None:
    mod = _load_tool_module()

    report = mod.build_error_anatomy_report()

    assert report["kind"] == "quasilinear_error_anatomy"
    assert report["claim_level"] == "model_development_residual_anatomy_not_absolute_flux_promotion"
    assert report["case_count"] == 12
    assert report["holdout_count"] == 10
    assert report["promotion_gate"]["passed"] is False
    assert "case_residuals_exceed_transport_gate" in report["promotion_gate"]["blockers"]
    assert 0.697 < report["candidate_mean_abs_relative_error"] < 0.698
    assert report["rows"][0]["case"] == "solovev_reference_repair_dt002_amp1em5_n48_t250"
    assert report["rows"][0]["above_transport_gate"] is True
    assert report["rows"][0]["overpredicts"] is True
    groups = {row["geometry_group"]: row for row in report["geometry_group_summary"]}
    assert groups["external axisymmetric VMEC"]["error_budget_fraction"] > 0.82
    assert groups["stellarator benchmark"]["mean_abs_relative_error"] < 0.35
    policy = report["frozen_ledger_policy"]
    assert policy["additional_holdout_collection_active"] is False
    assert policy["ledger_case_count"] == 12
    assert "passing scoped core portfolio" in policy["active_next_step"]
    assert report["dominant_residuals"][0]["case"] == "solovev_reference_repair_dt002_amp1em5_n48_t250"
    assert any("external-axisymmetric residual budget" in item for item in report["model_development_requirements"])
    core = report["core_portfolio_gate"]
    assert core["passed"] is True
    assert core["core_case_count"] == 10
    assert core["core_holdout_count"] == 8
    assert core["excluded_case_count"] == 2
    assert 0.27 < core["core_mean_abs_relative_error"] < 0.29
    assert 0.27 < core["core_holdout_mean_abs_relative_error"] < 0.29
    assert core["core_prediction_interval_coverage"] == 1.0
    assert core["screening_gate_passed"] is False
    assert {row["case"] for row in core["excluded_cases"]} == {
        "solovev_reference_repair_dt002_amp1em5_n48_t250",
        "shaped_tokamak_pressure_external_vmec_t650_high_grid_window",
    }


def test_error_anatomy_cli_writes_sidecars_and_fails_closed(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "ql_error_anatomy.png"

    completed = subprocess.run(
        [
            sys.executable,
            str(root / "tools" / "build_quasilinear_error_anatomy.py"),
            "--out",
            str(out),
        ],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 2
    assert "promotion_passed=False" in completed.stdout
    assert out.exists()
    payload = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["promotion_gate"]["passed"] is False
    assert payload["core_portfolio_gate"]["passed"] is True
    assert payload["frozen_ledger_policy"]["additional_holdout_collection_active"] is False
    csv_text = out.with_suffix(".csv").read_text(encoding="utf-8")
    assert csv_text.startswith("case,label,split,geometry")
