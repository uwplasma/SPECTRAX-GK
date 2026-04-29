"""Tests for the shape-aware quasilinear saturation diagnostic."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


def _load_tool_module():
    tools_dir = Path(__file__).resolve().parents[1] / "tools"
    sys.path.insert(0, str(tools_dir))
    path = tools_dir / "plot_quasilinear_shape_aware_saturation.py"
    spec = importlib.util.spec_from_file_location("plot_quasilinear_shape_aware_saturation", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_case(
    tmp_path: Path,
    name: str,
    *,
    observed: float,
    nonlinear_dist: tuple[float, float, float],
) -> tuple[Path, Path, Path]:
    spectrum = tmp_path / f"{name}_ql.csv"
    spectrum.write_text(
        "ky,gamma,kperp_eff2,heat_flux_weight_total,saturated_heat_flux_total\n"
        "0.1,0.1,0.5,1.0,0.2\n"
        "0.2,0.1,0.5,1.0,0.2\n"
        "0.4,0.1,0.5,1.0,0.2\n",
        encoding="utf-8",
    )
    diag = tmp_path / f"{name}_diag.csv"
    diag.write_text(f"t,heat_flux\n0.0,{observed}\n1.0,{observed}\n", encoding="utf-8")
    summary = tmp_path / f"{name}_summary.json"
    summary.write_text(json.dumps({"case": name, "spectrax": str(diag)}), encoding="utf-8")
    ql_dist = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    shape = tmp_path / f"{name}_shape.json"
    shape.write_text(
        json.dumps(
            {
                "passed": True,
                "ky": [0.1, 0.2, 0.4],
                "quasilinear_distribution": ql_dist,
                "nonlinear_distribution": nonlinear_dist,
                "total_variation_distance": 0.1,
                "cosine_similarity": 0.98,
            }
        ),
        encoding="utf-8",
    )
    return spectrum, summary, shape


def test_fit_power_law_shape_exponent_uses_case_intercepts(tmp_path: Path) -> None:
    mod = _load_tool_module()
    cases = []
    for name, dist in [
        ("a", (0.2, 0.3, 0.5)),
        ("b", (0.1, 0.3, 0.6)),
    ]:
        spectrum, summary, shape = _write_case(tmp_path, name, observed=1.0, nonlinear_dist=dist)
        cases.append(mod.SaturationCase(name, "train", name, spectrum, summary, shape))

    fit = mod.fit_power_law_shape_exponent(tuple(cases))

    assert fit["n_samples"] == 6
    assert fit["exponent"] > 0.0
    assert set(fit["used_cases"]) == {"a", "b"}


def test_shape_aware_report_and_figure_are_replayable(tmp_path: Path) -> None:
    mod = _load_tool_module()
    cases = []
    for name, observed, dist in [
        ("a", 1.0, (0.2, 0.3, 0.5)),
        ("b", 1.2, (0.2, 0.3, 0.5)),
        ("c", 0.8, (0.1, 0.3, 0.6)),
    ]:
        spectrum, summary, shape = _write_case(tmp_path, name, observed=observed, nonlinear_dist=dist)
        cases.append(mod.SaturationCase(name, "holdout", name, spectrum, summary, shape))

    report = mod.build_shape_aware_saturation_report(tuple(cases))
    paths = mod.write_shape_aware_saturation_figure(report, out=tmp_path / "shape.png", title="Shape")

    assert report["kind"] == "quasilinear_shape_aware_saturation_report"
    assert len(report["leave_one_out"]) == 3
    assert Path(paths["png"]).exists()
    assert Path(paths["pdf"]).exists()
    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert payload["claim_level"] == "leave_one_geometry_out_model_development"


def test_shape_aware_report_rejects_missing_shape_gate(tmp_path: Path) -> None:
    mod = _load_tool_module()
    spectrum, summary, _shape = _write_case(tmp_path, "a", observed=1.0, nonlinear_dist=(0.2, 0.3, 0.5))
    case = mod.SaturationCase("a", "train", "a", spectrum, summary, None)

    with pytest.raises(ValueError, match="missing a tracked shape-gate"):
        mod.build_shape_aware_saturation_report((case,))
