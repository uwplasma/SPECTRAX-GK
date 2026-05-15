from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "build_vmec_boozer_aggregate_line_search_comparison.py"
spec = importlib.util.spec_from_file_location("build_vmec_boozer_aggregate_line_search_comparison", SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def _line_search_payload(objective: str, derivative: float, initial: float, final: float) -> dict[str, object]:
    return {
        "kind": "vmec_boozer_aggregate_scalar_objective_line_search_report",
        "passed": True,
        "objective": objective,
        "reduction": "mean",
        "n_samples": 2,
        "accepted_steps": 1,
        "max_steps": 1,
        "initial_delta": 0.0,
        "final_delta": -1.0e-8 if derivative > 0.0 else 1.0e-8,
        "initial_objective": initial,
        "final_objective": final,
        "relative_reduction": (initial - final) / initial,
        "stop_reason": "max_steps",
        "samples": [
            {"surface_index": None, "alpha": 0.0, "selected_ky_index": 1, "weight": 0.5},
            {"surface_index": None, "alpha": 0.0, "selected_ky_index": 2, "weight": 0.5},
        ],
        "history": [
            {
                "step": 0,
                "delta": 0.0,
                "objective": initial,
                "central_derivative": derivative,
                "curvature_ratio": 0.02,
                "accepted": True,
                "candidate_delta": -1.0e-8 if derivative > 0.0 else 1.0e-8,
                "candidate_objective": final,
            }
        ],
    }


def _comparison_payload() -> dict[str, object]:
    reports = {
        "growth": _line_search_payload("growth", 2.0, 0.30, 0.29),
        "quasilinear_flux": _line_search_payload("quasilinear_flux", 5.0, 0.90, 0.88),
    }
    return {
        "kind": "vmec_boozer_aggregate_line_search_comparison",
        "passed": True,
        "case_name": "nfp4_QH_warm_start",
        "objectives": ["growth", "quasilinear_flux"],
        "reduction": "mean",
        "n_samples": 2,
        "same_sample_set": True,
        "all_line_searches_passed": True,
        "same_initial_update_direction": True,
        "final_delta_spread": 0.0,
        "relative_reduction_spread": 0.011,
        "rows": [
            {
                "objective": "growth",
                "passed": True,
                "n_samples": 2,
                "initial_objective": 0.30,
                "final_objective": 0.29,
                "absolute_reduction": 0.01,
                "relative_reduction": 0.0333333333,
                "initial_central_derivative": 2.0,
                "initial_update_direction": "negative_delta",
                "accepted_steps": 1,
                "max_steps": 1,
                "initial_delta": 0.0,
                "final_delta": -1.0e-8,
                "stop_reason": "max_steps",
            },
            {
                "objective": "quasilinear_flux",
                "passed": True,
                "n_samples": 2,
                "initial_objective": 0.90,
                "final_objective": 0.88,
                "absolute_reduction": 0.02,
                "relative_reduction": 0.0222222222,
                "initial_central_derivative": 5.0,
                "initial_update_direction": "negative_delta",
                "accepted_steps": 1,
                "max_steps": 1,
                "initial_delta": 0.0,
                "final_delta": -1.0e-8,
                "stop_reason": "max_steps",
            },
        ],
        "reports": reports,
    }


def test_build_vmec_boozer_aggregate_line_search_comparison_report_uses_same_sample_set(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_report(**kwargs):  # noqa: ANN003, ANN202
        calls.append(dict(kwargs))
        objective = str(kwargs["objective"])
        if objective == "growth":
            return _line_search_payload(objective, 2.0, 0.30, 0.29)
        return _line_search_payload(objective, 5.0, 0.90, 0.88)

    monkeypatch.setattr(mod, "vmec_boozer_aggregate_scalar_objective_line_search_report", fake_report)

    payload = mod.build_vmec_boozer_aggregate_line_search_comparison_report(
        selected_ky_indices=(1, 2),
        surface_indices=(None,),
        alphas=(0.0,),
        max_steps=1,
        ntheta=4,
    )

    assert payload["passed"] is True
    assert payload["same_sample_set"] is True
    assert payload["same_initial_update_direction"] is True
    assert [call["objective"] for call in calls] == ["growth", "quasilinear_flux"]
    assert all(call["selected_ky_indices"] == (1, 2) for call in calls)
    assert all(call["ntheta"] == 4 for call in calls)


def test_write_vmec_boozer_aggregate_line_search_comparison_artifacts(tmp_path: Path) -> None:
    paths = mod.write_vmec_boozer_aggregate_line_search_comparison_artifacts(
        _comparison_payload(),
        out=tmp_path / "comparison.png",
    )

    for suffix in ("png", "pdf", "json", "csv"):
        assert Path(paths[suffix]).exists()
    data = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert data["same_initial_update_direction"] is True
    assert "initial_update_direction" in Path(paths["csv"]).read_text(encoding="utf-8")


def test_vmec_boozer_aggregate_line_search_comparison_main_uses_report(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def fake_report(**kwargs):  # noqa: ANN003, ANN202
        calls.append(dict(kwargs))
        objective = str(kwargs["objective"])
        return _line_search_payload(objective, 1.0, 1.0, 0.9)

    monkeypatch.setattr(mod, "vmec_boozer_aggregate_scalar_objective_line_search_report", fake_report)

    result = mod.main(
        [
            "--out",
            str(tmp_path / "comparison.png"),
            "--selected-ky-indices",
            "1",
            "2",
            "--max-steps",
            "2",
            "--json-only",
        ]
    )

    assert result == 0
    assert [call["objective"] for call in calls] == ["growth", "quasilinear_flux"]
    assert all(call["selected_ky_indices"] == (1, 2) for call in calls)
    assert all(call["max_steps"] == 2 for call in calls)
    assert all(call["mboz"] == 21 for call in calls)
