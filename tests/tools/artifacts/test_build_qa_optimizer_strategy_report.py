from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "tools" / "build_qa_optimizer_strategy_report.py"
spec = importlib.util.spec_from_file_location("build_qa_optimizer_strategy_report", SCRIPT)
assert spec is not None
assert spec.loader is not None
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def _write_panel(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "qa_baseline_scipy",
                        "label": "QA baseline",
                        "gate_passed": True,
                        "diagnostic_gate_passed": True,
                        "gate_blockers": [],
                        "setup": {
                            "transport_kind": "nonlinear_window_heat_flux",
                            "spectrax_weight": 0.05,
                            "optimizer": {"method": "scipy"},
                        },
                        "history": {
                            "objective_initial": 100.0,
                            "objective_final": 0.01,
                            "aspect_final": 5.0,
                            "iota_final": 0.4102,
                            "qs_final": 1.0e-5,
                            "nfev": 48,
                        },
                    },
                    {
                        "case_id": "growth_from_strict_baseline",
                        "label": "growth from strict QA",
                        "gate_passed": False,
                        "diagnostic_gate_passed": True,
                        "gate_blockers": ["mean_iota"],
                        "setup": {
                            "transport_kind": "growth",
                            "spectrax_weight": 0.1,
                            "optimizer": {"method": "scalar_trust"},
                        },
                        "history": {
                            "objective_initial": 1.0,
                            "objective_final": 0.25,
                            "aspect_final": 5.004,
                            "iota_final": 0.4099,
                            "qs_final": 5.0e-4,
                            "transport_metric_final": 0.07,
                            "message": "radius too small",
                            "nfev": 16,
                        },
                    },
                ]
            }
        ),
        encoding="utf-8",
    )


def _write_landscape(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "label": "m0p1",
                        "relative_fraction": -0.1,
                        "coefficient_value": 0.9,
                        "reduced_metrics": {"growth": 0.2, "quasilinear_flux_mixing_length": 0.4},
                    },
                    {
                        "label": "0",
                        "relative_fraction": 0.0,
                        "coefficient_value": 1.0,
                        "reduced_metrics": {"growth": 0.3, "quasilinear_flux_mixing_length": 0.5},
                    },
                    {
                        "label": "p0p1",
                        "relative_fraction": 0.1,
                        "coefficient_value": 1.1,
                        "reduced_metrics": {"growth": 0.4, "quasilinear_flux_mixing_length": 0.6},
                    },
                ],
                "nonlinear_ensemble_points": [
                    {"coefficient_value": 0.9, "mean": 11.0, "sem": 0.2, "passed": True},
                    {"coefficient_value": 1.0, "mean": 10.0, "sem": 0.1, "passed": True},
                    {"coefficient_value": 1.1, "mean": 7.0, "sem": 0.3, "passed": True},
                ],
            }
        ),
        encoding="utf-8",
    )


def test_strategy_report_keeps_nonlinear_optimization_fail_closed(tmp_path: Path) -> None:
    panel = tmp_path / "panel.json"
    landscape = tmp_path / "landscape.json"
    _write_panel(panel)
    _write_landscape(landscape)

    report = mod.build_report(panel, landscape)

    assert report["kind"] == "vmec_jax_qa_optimizer_strategy_report"
    assert report["gates"]["deterministic_transport_rows_all_strict_gates_pass"] is False
    assert report["gates"]["has_converged_long_window_landscape"] is True
    assert report["gates"]["has_admitted_long_window_landscape"] is False
    assert report["gates"]["has_material_landscape_reduction_direction"] is True
    assert report["gates"]["nonlinear_absolute_optimization_promoted"] is False
    assert report["cases"][1]["iota_shortfall"] > 0.0
    assert report["landscape"]["n_converged_nonlinear_points"] == 3
    assert report["landscape"]["best_point"]["label"] == "p0p1"
    assert "noise/convergence diagnostic" in report["claim_scope"]

    methods = {item["method"] for item in report["optimizer_recommendations"]}
    assert "vmec_jax_exact_discrete_adjoint_least_squares" in methods
    assert "spsa_common_random_numbers_then_cma_es_or_bo_for_low_dimensional_projected_controls" in methods


def test_strategy_report_exports_public_qa_transport_claim_boundaries(tmp_path: Path) -> None:
    panel = tmp_path / "panel.json"
    landscape = tmp_path / "landscape.json"
    _write_panel(panel)
    _write_landscape(landscape)

    report = mod.build_report(panel, landscape)
    boundaries = {row["transport_kind"]: row for row in report["claim_boundaries"]}

    assert set(boundaries) == {"growth", "quasilinear_flux", "nonlinear_window_heat_flux"}
    assert all(row["nonlinear_turbulent_flux_claim"] is False for row in boundaries.values())
    assert "linear growth-rate residual" in boundaries["growth"]["claim_boundary"]
    assert "not an absolute flux predictor" in boundaries["quasilinear_flux"]["claim_boundary"]
    assert "not a converged nonlinear transport average" in boundaries["nonlinear_window_heat_flux"]["claim_boundary"]
    assert all(
        any("matched" in requirement for requirement in row["promotion_requires"])
        for row in boundaries.values()
    )

    readme = (ROOT / "examples" / "optimization" / "README.md").read_text(encoding="utf-8")
    assert "Claim boundary" in readme
    for kind, row in boundaries.items():
        script = ROOT / row["script"]
        text = script.read_text(encoding="utf-8")

        assert script.name in readme
        assert f'SPECTRAX_KIND = "{kind}"' in text
        assert "WRITE_LONG_NONLINEAR_AUDIT_CONFIGS = True" in text
        assert "RUN_LONG_NONLINEAR_AUDIT_COMMANDS = False" in text
        assert 'NONLINEAR_AUDIT_HORIZONS = "700,1100,1500"' in text
        assert "NONLINEAR_AUDIT_WINDOW_TMIN = 1100.0" in text
        assert "NONLINEAR_AUDIT_WINDOW_TMAX = 1500.0" in text
        assert "build_matched_nonlinear_transport_comparison.py" in text


def test_strategy_report_cli_writes_artifacts(tmp_path: Path) -> None:
    panel = tmp_path / "panel.json"
    landscape = tmp_path / "landscape.json"
    out_prefix = tmp_path / "strategy"
    _write_panel(panel)
    _write_landscape(landscape)

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--panel-json",
            str(panel),
            "--landscape-json",
            str(landscape),
            "--out-prefix",
            str(out_prefix),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )

    status = json.loads(completed.stdout)
    assert Path(status["out_json"]).exists()
    assert out_prefix.with_suffix(".csv").exists()
    assert out_prefix.with_suffix(".png").exists()
