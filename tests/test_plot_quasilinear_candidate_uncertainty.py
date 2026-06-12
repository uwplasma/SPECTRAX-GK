"""Tests for uncertainty-aware quasilinear candidate diagnostics."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


def _load_tool_module():
    tools_dir = Path(__file__).resolve().parents[1] / "tools"
    sys.path.insert(0, str(tools_dir))
    path = tools_dir / "plot_quasilinear_candidate_uncertainty.py"
    spec = importlib.util.spec_from_file_location(
        "plot_quasilinear_candidate_uncertainty", path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_case(
    tmp_path: Path, name: str, *, observed: float, weight: float
) -> tuple[Path, Path]:
    spectrum = tmp_path / f"{name}_ql.csv"
    spectrum.write_text(
        "ky,gamma,kperp_eff2,heat_flux_weight_total,saturated_heat_flux_total\n"
        f"0.1,0.1,0.5,{weight},0.0\n"
        f"0.2,0.1,1.0,{0.5 * weight},0.0\n",
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


def test_candidate_uncertainty_report_and_figure_are_replayable(tmp_path: Path) -> None:
    mod = _load_tool_module()
    cases = []
    for name, observed, weight in [("a", 3.0, 1.0), ("b", 6.0, 2.0), ("c", 9.0, 3.0)]:
        spectrum, summary = _write_case(
            tmp_path, name, observed=observed, weight=weight
        )
        cases.append(mod.SaturationCase(name, "holdout", name, spectrum, summary, None))

    report = mod.build_candidate_uncertainty_report(
        tuple(cases), candidates=("linear_weight",)
    )
    paths = mod.write_candidate_uncertainty_figure(
        report,
        out=tmp_path / "candidate.png",
        title="Candidate",
        dpi=80,
        write_pdf=False,
    )

    assert report["kind"] == "quasilinear_candidate_uncertainty_report"
    assert report["input_validation"]["passed"] is True
    assert "linear_weight" in report["candidates"]
    assert report["promotion_gate"]["requires_interval_coverage"] is True
    assert Path(paths["png"]).exists()
    assert "pdf" not in paths
    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert payload["claim_level"] == "candidate_model_development_not_runtime_option"


def test_linear_state_ridge_candidate_reports_under_sampled_gate(
    tmp_path: Path,
) -> None:
    mod = _load_tool_module()
    cases = []
    for name, observed, weight in [
        ("a", 3.0, 1.0),
        ("b", 6.0, 2.0),
        ("c", 7.0, 2.4),
    ]:
        spectrum, summary = _write_case(
            tmp_path, name, observed=observed, weight=weight
        )
        cases.append(mod.SaturationCase(name, "holdout", name, spectrum, summary, None))

    report = mod.build_candidate_uncertainty_report(
        tuple(cases), candidates=("linear_state_ridge",)
    )
    ridge = report["candidates"]["linear_state_ridge"]

    assert ridge["promotion_eligible"] is False
    assert ridge["eligibility_failures"] == ["insufficient_train_to_parameter_ratio"]
    assert report["promotion_gate"]["accepted_candidates"] == []
    assert report["promotion_gate"]["requires_candidate_eligibility"] is True
    assert ridge["rows"][0]["feature_names"] == list(mod.STATE_FEATURE_NAMES)


def test_candidate_uncertainty_parallel_workers_match_serial(tmp_path: Path) -> None:
    mod = _load_tool_module()
    cases = []
    for name, observed, weight in [
        ("a", 3.0, 1.0),
        ("b", 6.0, 2.0),
        ("c", 9.0, 3.0),
        ("d", 12.0, 4.0),
    ]:
        spectrum, summary = _write_case(
            tmp_path, name, observed=observed, weight=weight
        )
        cases.append(mod.SaturationCase(name, "holdout", name, spectrum, summary, None))

    serial = mod.build_candidate_uncertainty_report(
        tuple(cases), candidates=("linear_weight",), workers=1
    )
    parallel = mod.build_candidate_uncertainty_report(
        tuple(cases), candidates=("linear_weight",), workers=3
    )

    assert parallel["parallel"]["workers"] == 3
    assert (
        parallel["null_training_mean_baseline"] == serial["null_training_mean_baseline"]
    )
    assert parallel["candidates"] == serial["candidates"]
    assert parallel["promotion_gate"] == serial["promotion_gate"]


def test_candidate_uncertainty_negative_controls_and_schema_guards(
    tmp_path: Path,
) -> None:
    mod = _load_tool_module()
    good_spectrum, good_summary = _write_case(
        tmp_path, "good", observed=3.0, weight=1.0
    )
    bad_spectrum = tmp_path / "bad_ql.csv"
    bad_spectrum.write_text(
        "ky,kperp_eff2,heat_flux_weight_total,saturated_heat_flux_total\n0.1,0.5,1.0,0.0\n",
        encoding="utf-8",
    )
    bad_summary = tmp_path / "bad_summary.json"
    bad_diag = tmp_path / "bad_diag.csv"
    bad_diag.write_text("t,heat_flux\n0.0,3.0\n1.0,3.0\n", encoding="utf-8")
    bad_summary.write_text(
        json.dumps(
            {"case": "bad", "spectrax": str(bad_diag), "gate_report": {"passed": True}}
        ),
        encoding="utf-8",
    )
    cases = (
        mod.SaturationCase(
            "good", "holdout", "good", good_spectrum, good_summary, None
        ),
        mod.SaturationCase("bad", "holdout", "bad", bad_spectrum, bad_summary, None),
    )

    with pytest.raises(ValueError, match="required column 'gamma'"):
        mod.build_candidate_uncertainty_report(
            cases, candidates=("linear_state_ridge",)
        )
    with pytest.raises(ValueError, match="unknown candidate"):
        mod.build_candidate_uncertainty_report(
            cases[:1], candidates=("not_a_candidate",)
        )


def test_candidate_uncertainty_can_be_run_as_unvalidated_development_audit(
    tmp_path: Path,
) -> None:
    mod = _load_tool_module()
    cases = []
    for name, observed, weight in [("a", 3.0, 1.0), ("b", 6.0, 2.0)]:
        spectrum, summary = _write_case(
            tmp_path, name, observed=observed, weight=weight
        )
        payload = json.loads(summary.read_text(encoding="utf-8"))
        payload["gate_report"]["passed"] = False
        summary.write_text(json.dumps(payload), encoding="utf-8")
        cases.append(mod.SaturationCase(name, "holdout", name, spectrum, summary, None))

    report = mod.build_candidate_uncertainty_report(
        tuple(cases),
        candidates=("linear_weight",),
        require_validated_inputs=False,
    )

    assert report["input_validation"]["passed"] is None
    assert report["claim_level"] == "candidate_model_development_not_runtime_option"
    assert report["promotion_gate"]["requires_candidate_eligibility"] is True


def test_tracked_candidate_uncertainty_sidecar_is_fail_closed_near_miss() -> None:
    """Lock the scoped quasilinear model-development near miss to the artifact."""

    root = Path(__file__).resolve().parents[1]
    payload = json.loads(
        (root / "docs/_static/quasilinear_candidate_uncertainty.json").read_text(encoding="utf-8")
    )
    gate = payload["promotion_gate"]
    candidates = payload["candidates"]
    spectral = candidates["spectral_envelope_ridge"]
    linear = candidates["linear_weight"]
    state = candidates["linear_state_ridge"]
    null = payload["null_training_mean_baseline"]

    assert payload["claim_level"] == "candidate_model_development_not_runtime_option"
    assert gate["passed"] is False
    assert gate["accepted_candidates"] == []
    assert spectral["promotion_eligible"] is True
    assert spectral["mean_abs_relative_error"] > gate["transport_mean_relative_error_gate"]
    assert spectral["mean_abs_relative_error"] < linear["mean_abs_relative_error"]
    assert spectral["mean_abs_relative_error"] < null["mean_abs_relative_error"]
    assert spectral["prediction_interval_coverage"] >= gate["interval_coverage_gate"]
    assert state["promotion_eligible"] is True
    assert state["eligibility_failures"] == []
    assert state["mean_abs_relative_error"] > null["mean_abs_relative_error"]
    assert state["mean_abs_relative_error"] > linear["mean_abs_relative_error"]
    assert state["mean_abs_relative_error"] > spectral["mean_abs_relative_error"]
