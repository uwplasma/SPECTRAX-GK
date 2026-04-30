"""Tests for uncertainty-aware quasilinear candidate diagnostics."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


def _load_tool_module():
    tools_dir = Path(__file__).resolve().parents[1] / "tools"
    sys.path.insert(0, str(tools_dir))
    path = tools_dir / "plot_quasilinear_candidate_uncertainty.py"
    spec = importlib.util.spec_from_file_location("plot_quasilinear_candidate_uncertainty", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_case(tmp_path: Path, name: str, *, observed: float, weight: float) -> tuple[Path, Path]:
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
        spectrum, summary = _write_case(tmp_path, name, observed=observed, weight=weight)
        cases.append(mod.SaturationCase(name, "holdout", name, spectrum, summary, None))

    report = mod.build_candidate_uncertainty_report(tuple(cases), candidates=("linear_weight",))
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


def test_linear_state_ridge_candidate_reports_under_sampled_gate(tmp_path: Path) -> None:
    mod = _load_tool_module()
    cases = []
    for name, observed, weight in [
        ("a", 3.0, 1.0),
        ("b", 6.0, 2.0),
        ("c", 7.0, 2.4),
    ]:
        spectrum, summary = _write_case(tmp_path, name, observed=observed, weight=weight)
        cases.append(mod.SaturationCase(name, "holdout", name, spectrum, summary, None))

    report = mod.build_candidate_uncertainty_report(tuple(cases), candidates=("linear_state_ridge",))
    ridge = report["candidates"]["linear_state_ridge"]

    assert ridge["promotion_eligible"] is False
    assert ridge["eligibility_failures"] == ["insufficient_train_to_parameter_ratio"]
    assert report["promotion_gate"]["accepted_candidates"] == []
    assert report["promotion_gate"]["requires_candidate_eligibility"] is True
    assert ridge["rows"][0]["feature_names"] == list(mod.STATE_FEATURE_NAMES)
