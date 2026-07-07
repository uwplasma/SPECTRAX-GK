from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "tools" / "build_vmec_boozer_second_equilibrium_aggregate_gate.py"
spec = importlib.util.spec_from_file_location(
    "build_vmec_boozer_second_equilibrium_aggregate_gate",
    SCRIPT,
)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def _fd_payload() -> dict[str, object]:
    return {
        "kind": "vmec_boozer_aggregate_scalar_objective_finite_difference_report",
        "passed": True,
        "case_name": "li383_low_res",
        "objective": "quasilinear_flux",
        "reduction": "mean",
        "n_samples": 2,
        "minus_value": 9.80,
        "base_value": 9.79,
        "plus_value": 9.78,
        "central_derivative": -1.0e5,
        "response_abs": 0.02,
        "curvature_ratio": 0.01,
        "samples": [
            {"surface_index": None, "alpha": 0.0, "selected_ky_index": 1, "weight": 0.5},
            {"surface_index": None, "alpha": 0.0, "selected_ky_index": 2, "weight": 0.5},
        ],
    }


def _line_payload() -> dict[str, object]:
    return {
        "kind": "vmec_boozer_aggregate_scalar_objective_line_search_report",
        "passed": True,
        "case_name": "li383_low_res",
        "objective": "quasilinear_flux",
        "reduction": "mean",
        "n_samples": 2,
        "accepted_steps": 1,
        "initial_objective": 9.79,
        "final_objective": 9.78,
        "relative_reduction": 1.0e-3,
        "stop_reason": "max_steps",
        "history": [
            {
                "step": 0,
                "delta": 0.0,
                "objective": 9.79,
                "central_derivative": -1.0e5,
                "finite_difference_passed": True,
                "curvature_ratio": 0.01,
                "accepted": True,
                "candidate_delta": 1.0e-8,
                "candidate_objective": 9.78,
            }
        ],
    }


def test_build_second_equilibrium_payload_passes_with_mode21_defaults(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_fd(**kwargs):  # noqa: ANN003, ANN202
        calls["fd"] = kwargs
        return _fd_payload()

    def fake_line(**kwargs):  # noqa: ANN003, ANN202
        calls["line"] = kwargs
        return _line_payload()

    monkeypatch.setattr(mod, "vmec_boozer_aggregate_scalar_objective_finite_difference_report", fake_fd)
    monkeypatch.setattr(mod, "vmec_boozer_aggregate_scalar_objective_line_search_report", fake_line)

    payload = mod.build_vmec_boozer_second_equilibrium_aggregate_payload(max_wall_seconds=0.0)

    assert payload["passed"] is True
    assert payload["feasible"] is True
    assert payload["case_name"] == "li383_low_res"
    assert payload["mode_bound"] == {"mboz": 21, "nboz": 21, "minimum_required": 21, "passed": True}
    assert payload["coverage"]["selected_ky_indices"] == [1, 2]
    assert payload["finite_difference_summary"]["central_derivative"] == -1.0e5
    assert payload["line_search_summary"]["accepted_steps"] == 1
    assert calls["fd"]["mboz"] == 21
    assert calls["fd"]["nboz"] == 21
    assert calls["line"]["case_name"] == "li383_low_res"


def test_build_second_equilibrium_payload_fails_closed_on_backend_error(monkeypatch) -> None:
    def fake_fd(**_kwargs):  # noqa: ANN003, ANN202
        raise RuntimeError("vmec_jax example fixture missing")

    monkeypatch.setattr(mod, "vmec_boozer_aggregate_scalar_objective_finite_difference_report", fake_fd)

    payload = mod.build_vmec_boozer_second_equilibrium_aggregate_payload(max_wall_seconds=0.0)

    assert payload["passed"] is False
    assert payload["feasible"] is False
    assert payload["blocker_type"] == "RuntimeError"
    assert payload["blocker_message"] == "vmec_jax example fixture missing"
    assert payload["mode_bound"]["passed"] is True


def test_write_second_equilibrium_artifacts(tmp_path: Path) -> None:
    payload = {
        "kind": "vmec_boozer_second_equilibrium_aggregate_gate",
        "passed": True,
        "feasible": True,
        "case_name": "li383_low_res",
        "objective": "quasilinear_flux",
        "mode_bound": {"mboz": 21, "nboz": 21, "minimum_required": 21, "passed": True},
        "sample_bound": {"n_samples_requested": 2, "max_samples": 4, "passed": True},
        "bounded_runtime": {"max_wall_seconds": 300.0, "elapsed_wall_seconds": 41.2, "passed": True},
        "finite_difference_passed": True,
        "line_search_passed": True,
        "finite_difference_summary": {
            "minus_value": 9.80,
            "base_value": 9.79,
            "plus_value": 9.78,
            "central_derivative": -1.0e5,
            "response_abs": 0.02,
            "curvature_ratio": 0.01,
            "n_samples": 2,
        },
        "line_search_summary": {
            "accepted_steps": 1,
            "initial_objective": 9.79,
            "final_objective": 9.78,
            "relative_reduction": 1.0e-3,
            "stop_reason": "max_steps",
        },
    }

    paths = mod.write_vmec_boozer_second_equilibrium_aggregate_artifacts(
        payload,
        out=tmp_path / "second_gate.png",
    )

    for suffix in ("png", "pdf", "json", "csv"):
        assert Path(paths[suffix]).exists()
    saved = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert saved["kind"] == "vmec_boozer_second_equilibrium_aggregate_gate"
    assert "fd_central_derivative" in Path(paths["csv"]).read_text(encoding="utf-8")


def test_main_json_only_uses_reports(monkeypatch, capsys) -> None:
    calls: dict[str, object] = {}

    def fake_fd(**kwargs):  # noqa: ANN003, ANN202
        calls["fd"] = kwargs
        return _fd_payload()

    def fake_line(**kwargs):  # noqa: ANN003, ANN202
        calls["line"] = kwargs
        return _line_payload()

    monkeypatch.setattr(mod, "vmec_boozer_aggregate_scalar_objective_finite_difference_report", fake_fd)
    monkeypatch.setattr(mod, "vmec_boozer_aggregate_scalar_objective_line_search_report", fake_line)

    result = mod.main(
        [
            "--case-name",
            "nfp3_QI_fixed_resolution_final",
            "--selected-ky-indices",
            "1",
            "2",
            "--max-wall-seconds",
            "0",
            "--json-only",
        ]
    )

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is True
    assert calls["fd"]["case_name"] == "nfp3_QI_fixed_resolution_final"
    assert calls["line"]["selected_ky_indices"] == (1, 2)
