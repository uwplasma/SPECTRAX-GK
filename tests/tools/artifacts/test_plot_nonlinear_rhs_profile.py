from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "tools" / "plot_nonlinear_rhs_profile.py"
spec = importlib.util.spec_from_file_location("plot_nonlinear_rhs_profile", SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def test_build_summary_reports_dominant_kernel_and_speedups() -> None:
    payload = mod._build_summary(
        {
            "CPU grid": {
                "field_solve": 1.0,
                "linear_rhs": 4.0,
                "nonlinear_bracket": 2.0,
                "full_rhs": 8.0,
            },
            "CPU spectral": {
                "field_solve": 1.0,
                "linear_rhs": 4.0,
                "nonlinear_bracket": 1.0,
                "full_rhs": 5.0,
            },
        }
    )

    assert payload["kind"] == "nonlinear_rhs_profile_summary"
    assert payload["rows"]["CPU grid"]["dominant_measured_kernel"] == "linear_rhs"
    assert payload["rows"]["CPU grid"]["linear_rhs_fraction_of_full_rhs"] == 0.5
    assert payload["spectral_speedups"]["cpu"]["full_rhs_grid_over_spectral"] == 1.6
    assert payload["spectral_speedups"]["cpu"]["nonlinear_bracket_grid_over_spectral"] == 2.0
    assert payload["fastest_full_rhs_label"] == "CPU spectral"


def test_write_summary_json_roundtrips(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    mod._write_summary_json({"kind": "x", "value": 1.0}, path)

    assert json.loads(path.read_text(encoding="utf-8")) == {"kind": "x", "value": 1.0}


def test_parse_input_arg_and_case_label() -> None:
    label, path = mod._parse_input_arg("GPU spectral=docs/_static/example.csv")
    payload = mod._build_summary({"GPU spectral": {"full_rhs": 2.0}}, case="larger_case")

    assert label == "GPU spectral"
    assert str(path) == "docs/_static/example.csv"
    assert payload["case"] == "larger_case"
    assert mod._case_title("cyclone_miller_benchmark_size") == "Cyclone Miller benchmark-size case"
