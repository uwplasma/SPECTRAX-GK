"""Tests for the quasilinear saturation-rule sweep tool."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


def _load_tool_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "plot_quasilinear_saturation_rule_sweep.py"
    spec = importlib.util.spec_from_file_location("plot_quasilinear_saturation_rule_sweep", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_case(tmp_path: Path, name: str, *, observed: float, gamma_sign: float = 1.0) -> tuple[Path, Path]:
    spectrum = tmp_path / f"{name}_ql.csv"
    spectrum.write_text(
        "ky,gamma,kperp_eff2,heat_flux_weight_total,saturated_heat_flux_total\n"
        f"0.1,{0.2 * gamma_sign},0.5,2.0,{0.8 if gamma_sign > 0.0 else 0.0}\n"
        f"0.2,{0.1 * gamma_sign},1.0,1.0,{0.1 if gamma_sign > 0.0 else 0.0}\n",
        encoding="utf-8",
    )
    diag = tmp_path / f"{name}_diag.csv"
    diag.write_text(f"t,heat_flux\n0.0,{observed}\n1.0,{observed}\n", encoding="utf-8")
    summary = tmp_path / f"{name}_summary.json"
    summary.write_text(
        json.dumps(
            {
                "case": name,
                "spectrax": str(diag),
                "gate_report": {"case": name, "passed": True, "gates": []},
            }
        ),
        encoding="utf-8",
    )
    return spectrum, summary


def test_saturation_rule_sweep_fits_train_scale_and_scores_holdout(tmp_path: Path) -> None:
    mod = _load_tool_module()
    train_spectrum, train_summary = _write_case(tmp_path, "train", observed=9.0)
    holdout_spectrum, holdout_summary = _write_case(tmp_path, "holdout", observed=4.5, gamma_sign=-1.0)
    cases = (
        mod.SaturationCase(
            case="train",
            split="train",
            geometry="cyclone",
            spectrum=train_spectrum,
            nonlinear_summary=train_summary,
        ),
        mod.SaturationCase(
            case="holdout",
            split="holdout",
            geometry="hsx",
            spectrum=holdout_spectrum,
            nonlinear_summary=holdout_summary,
        ),
    )

    report = mod.build_saturation_rule_sweep(cases)

    assert report["claim_level"] == "model_comparison_not_validated_transport"
    assert report["input_validation"]["passed"] is True
    assert report["rules"]["positive_mixing_length"]["scale"] == pytest.approx(10.0)
    assert report["rules"]["positive_mixing_length"]["predicted_heat_flux"][0] == pytest.approx(9.0)
    assert report["rules"]["positive_mixing_length"]["predicted_heat_flux"][1] == pytest.approx(0.0)
    assert report["rules"]["linear_weight"]["scale"] == pytest.approx(3.0)
    assert report["rules"]["linear_weight"]["predicted_heat_flux"][1] == pytest.approx(9.0)
    assert report["null_training_mean_baseline"]["predicted_heat_flux"] == pytest.approx([9.0, 9.0])
    assert report["null_training_mean_baseline"]["holdout_mean_abs_relative_error"] == pytest.approx(1.0)
    assert report["promotion_gate"]["passed"] is False
    assert report["promotion_gate"]["accepted_rules"] == []


def test_saturation_rule_sweep_writes_artifacts(tmp_path: Path) -> None:
    mod = _load_tool_module()
    spectrum, summary = _write_case(tmp_path, "train", observed=9.0)
    cases = (
        mod.SaturationCase(
            case="train",
            split="train",
            geometry="cyclone",
            spectrum=spectrum,
            nonlinear_summary=summary,
        ),
    )
    report = mod.build_saturation_rule_sweep(cases)

    paths = mod.write_saturation_rule_sweep_figure(report, out=tmp_path / "sweep.png", title="Sweep")

    assert Path(paths["png"]).exists()
    assert Path(paths["pdf"]).exists()
    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert payload["kind"] == "quasilinear_saturation_rule_sweep"
    assert "null_training_mean_baseline" in payload
    assert "promotion_gate" in payload


def test_saturation_rule_sweep_rejects_failed_nonlinear_summary_gate(tmp_path: Path) -> None:
    mod = _load_tool_module()
    spectrum, summary = _write_case(tmp_path, "train", observed=9.0)
    data = json.loads(summary.read_text(encoding="utf-8"))
    data["gate_report"]["passed"] = False
    summary.write_text(json.dumps(data), encoding="utf-8")
    cases = (
        mod.SaturationCase(
            case="train",
            split="train",
            geometry="cyclone",
            spectrum=spectrum,
            nonlinear_summary=summary,
        ),
    )

    with pytest.raises(ValueError, match="unvalidated nonlinear train/holdout input"):
        mod.build_saturation_rule_sweep(cases)
