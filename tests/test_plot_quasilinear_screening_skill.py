"""Tests for quasilinear screening/ranking skill artifacts."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "plot_quasilinear_screening_skill.py"
    spec = importlib.util.spec_from_file_location("plot_quasilinear_screening_skill", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_screening_skill_keeps_spectral_envelope_fail_closed() -> None:
    module = _load_module()
    report = module.build_report()
    models = {row["model"]: row for row in report["models"]}

    assert report["kind"] == "quasilinear_screening_skill"
    assert report["claim_level"] == "screening_correlation_model_development_not_absolute_flux_promotion"
    assert report["gates"]["accepted_screening_models"] == []
    assert report["gates"]["accepted_holdout_screening_models"] == []
    assert report["gates"]["best_screening_model"] == "spectral_envelope_ridge"
    assert report["gates"]["best_holdout_screening_model"] == "spectral_envelope_ridge"
    assert report["gates"]["mean_error_gate_models"] == []
    assert report["gates"]["accepted_absolute_flux_models"] == []
    assert report["gates"]["absolute_flux_promotion_passed"] is False
    assert report["gates"]["screening_correlation_passed"] is False
    assert report["gates"]["holdout_screening_correlation_passed"] is False

    spectral = models["spectral_envelope_ridge"]
    assert spectral["screening_gate_passed"] is False
    assert spectral["holdout_screening_gate_passed"] is False
    assert 0.65 < spectral["spearman"] < 0.75
    assert 0.70 < spectral["pairwise_order_accuracy"] < 0.75
    assert 0.55 < spectral["holdout_spearman"] < 0.75
    assert 0.65 < spectral["holdout_pairwise_order_accuracy"] < 0.75
    assert spectral["holdout_mean_abs_relative_error"] > 0.35

    assert models["positive_mixing_length"]["screening_gate_passed"] is False
    assert models["linear_weight"]["screening_gate_passed"] is False
    assert models["absolute_growth_mixing_length"]["screening_gate_passed"] is False


def test_screening_skill_writer_creates_sidecars(tmp_path: Path) -> None:
    module = _load_module()
    report = module.build_report()
    paths = module.write_figure(report, out=tmp_path / "screening.png", title="test", dpi=80)

    for key in ("png", "pdf", "json", "csv"):
        assert Path(paths[key]).exists()
    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert payload["gates"]["best_screening_model"] == "spectral_envelope_ridge"
    assert payload["gates"]["best_holdout_screening_model"] == "spectral_envelope_ridge"
    assert Path(paths["csv"]).read_text(encoding="utf-8").startswith("model,label")
