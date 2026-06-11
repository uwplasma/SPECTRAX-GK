"""Tests for the stellarator quasilinear usefulness figure/report."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "plot_quasilinear_stellarator_usefulness.py"
    spec = importlib.util.spec_from_file_location("plot_quasilinear_stellarator_usefulness", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stellarator_usefulness_report_keeps_claim_scoped() -> None:
    module = _load_module()
    report = module.build_report()

    assert report["kind"] == "quasilinear_stellarator_usefulness"
    assert "not_runtime_absolute_flux_predictor" in report["claim_level"]
    assert report["models"]["spectral_envelope_ridge"]["accepted"] is False
    assert report["models"]["spectral_envelope_ridge"]["mean_abs_relative_error"] > 0.35
    assert report["models"]["positive_mixing_length"]["accepted"] is False
    assert report["models"]["positive_mixing_length"]["holdout_mean_abs_relative_error"] > 1.0
    assert "universal" in report["readme_sentence"]
    assert "rank-screening" in report["readme_sentence"]


def test_stellarator_rows_show_simple_rule_failure_and_scope_statuses() -> None:
    module = _load_module()
    report = module.build_report()
    rows = {row["case"]: row for row in report["rows"]}

    for case in ("hsx_nonlinear_window", "w7x_nonlinear_window"):
        row = rows[case]
        assert row["observed_heat_flux"] > 0.0
        assert row["positive_mixing_length_prediction"] == 0.0
        assert row["positive_mixing_length_relative_error"] == 1.0
        assert row["spectral_envelope_ridge_relative_error"] < 0.2
        assert row["stellarator_family"] is True

    qa = report["stellarator_status"]["QA"]
    assert qa["baseline_heat_flux"] > qa["optimized_heat_flux"]
    assert qa["relative_reduction"] > 0.05
    assert "audit only" in qa["status"]

    qh = report["stellarator_status"]["QH"]
    assert qh["high_grid_gate_passed"] is False
    assert qh["least_window_pairwise_heat_flux_symmetric_relative_difference"] > 0.15
    assert "excluded" in qh["status"]


def test_stellarator_usefulness_writer_creates_sidecars(tmp_path: Path) -> None:
    module = _load_module()
    report = module.build_report()
    out = tmp_path / "ql_usefulness.png"

    paths = module.write_figure(report, out=out, title="test", dpi=80)

    for key in ("png", "pdf", "json", "csv"):
        assert Path(paths[key]).exists()
    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert payload["kind"] == "quasilinear_stellarator_usefulness"
    assert Path(paths["csv"]).read_text(encoding="utf-8").startswith("case,label,geometry")
