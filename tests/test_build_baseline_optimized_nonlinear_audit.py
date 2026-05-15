from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "build_baseline_optimized_nonlinear_audit.py"


def _load_tool_module():
    spec = importlib.util.spec_from_file_location(
        "build_baseline_optimized_nonlinear_audit", SCRIPT
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _ensemble_payload(*, case: str, mean: float, sem: float = 0.1) -> dict[str, object]:
    return {
        "kind": "nonlinear_window_ensemble_report",
        "claim_level": "replicated_nonlinear_window_uncertainty_gate_not_simulation_claim",
        "case": case,
        "comparison": f"{case}_seed_timestep_replicates",
        "passed": True,
        "gate_report": {"passed": True},
        "statistics": {
            "n_reports": 3,
            "ensemble_mean": mean,
            "combined_sem": sem,
            "combined_sem_rel": sem / abs(mean),
            "mean_spread": 0.2,
            "mean_rel_spread": 0.02,
        },
        "rows": [
            {"case": f"{case}_seed31", "late_mean": mean - 0.1, "sem": sem},
            {"case": f"{case}_seed32", "late_mean": mean + 0.1, "sem": sem},
            {"case": f"{case}_dt", "late_mean": mean, "sem": sem},
        ],
    }


def _selected_audit_payload(optimized_path: Path) -> dict[str, object]:
    return {
        "kind": "production_nonlinear_turbulent_flux_optimization_guard",
        "production_nonlinear_optimization_promoted": True,
        "promotion_gate": {"passed": True, "blockers": []},
        "optimized_equilibrium_artifacts": [
            {
                "path": str(optimized_path),
                "qualifies_for_production_optimization": True,
            }
        ],
    }


def test_baseline_optimized_audit_writes_json_csv_and_png(tmp_path: Path) -> None:
    mod = _load_tool_module()
    baseline = tmp_path / "baseline.json"
    optimized = tmp_path / "optimized.json"
    selected = tmp_path / "selected.json"
    baseline.write_text(
        json.dumps(_ensemble_payload(case="baseline", mean=20.0, sem=0.2)),
        encoding="utf-8",
    )
    optimized.write_text(
        json.dumps(_ensemble_payload(case="optimized_equilibrium_final", mean=12.0, sem=0.2)),
        encoding="utf-8",
    )
    selected.write_text(json.dumps(_selected_audit_payload(optimized)), encoding="utf-8")
    out_json = tmp_path / "audit.json"
    out_csv = tmp_path / "audit.csv"
    out_png = tmp_path / "audit.png"

    rc = mod.main(
        [
            "--baseline-ensemble",
            str(baseline),
            "--optimized-ensemble",
            str(optimized),
            "--selected-optimized-audit",
            str(selected),
            "--out-json",
            str(out_json),
            "--out-csv",
            str(out_csv),
            "--out-png",
            str(out_png),
            "--min-relative-reduction",
            "0.25",
        ]
    )

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert rc == 0
    assert out_csv.exists()
    assert out_png.exists()
    assert payload["passed"] is True
    assert payload["comparison"]["baseline_mean"] == 20.0
    assert payload["comparison"]["optimized_mean"] == 12.0
    assert payload["comparison"]["relative_reduction"] == 0.4
    assert payload["selected_optimized_audit"]["optimized_ensemble_selected"] is True


def test_baseline_optimized_audit_fails_closed_when_baseline_missing(tmp_path: Path) -> None:
    mod = _load_tool_module()
    optimized = tmp_path / "optimized.json"
    optimized.write_text(
        json.dumps(_ensemble_payload(case="optimized_equilibrium_final", mean=12.0)),
        encoding="utf-8",
    )
    selected = tmp_path / "selected.json"
    selected.write_text(json.dumps(_selected_audit_payload(optimized)), encoding="utf-8")
    out_json = tmp_path / "audit.json"

    rc = mod.main(
        [
            "--baseline-ensemble",
            str(tmp_path / "missing_baseline.json"),
            "--optimized-ensemble",
            str(optimized),
            "--selected-optimized-audit",
            str(selected),
            "--out-json",
            str(out_json),
        ]
    )

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert rc == 1
    assert payload["passed"] is False
    assert payload["baseline_ensemble"]["present"] is False
    assert "baseline_ensemble_missing" in payload["blockers"]
    assert any("missing_baseline.json" in blocker for blocker in payload["blockers"])


def test_baseline_optimized_audit_rejects_unselected_optimized_audit(tmp_path: Path) -> None:
    mod = _load_tool_module()
    baseline = tmp_path / "baseline.json"
    optimized = tmp_path / "optimized.json"
    selected = tmp_path / "selected.json"
    baseline.write_text(json.dumps(_ensemble_payload(case="baseline", mean=20.0)), encoding="utf-8")
    optimized.write_text(
        json.dumps(_ensemble_payload(case="optimized_equilibrium_final", mean=12.0)),
        encoding="utf-8",
    )
    selected.write_text(
        json.dumps(
            {
                "promotion_gate": {"passed": True},
                "optimized_equilibrium_artifacts": [
                    {
                        "path": str(tmp_path / "different_optimized.json"),
                        "qualifies_for_production_optimization": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    out_json = tmp_path / "audit.json"

    rc = mod.main(
        [
            "--baseline-ensemble",
            str(baseline),
            "--optimized-ensemble",
            str(optimized),
            "--selected-optimized-audit",
            str(selected),
            "--out-json",
            str(out_json),
        ]
    )

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert rc == 1
    assert payload["passed"] is False
    assert "optimized_ensemble_not_selected_by_audit" in payload["blockers"]
