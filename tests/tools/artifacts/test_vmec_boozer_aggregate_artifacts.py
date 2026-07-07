from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.artifacts import build_vmec_boozer_aggregate_alpha_holdout_gate as alpha_gate
from tools.artifacts import (
    build_vmec_boozer_aggregate_line_search_comparison as comparison_gate,
)
from tools.artifacts import build_vmec_boozer_aggregate_line_search_gate as line_gate
from tools.artifacts import build_vmec_boozer_aggregate_objective_gate as objective_gate
from tools.artifacts import (
    build_vmec_boozer_aggregate_surface_holdout_gate as surface_gate,
)
from tools.artifacts import (
    build_vmec_boozer_multi_point_objective_gate as multi_point_gate,
)
from tools.artifacts import (
    build_vmec_boozer_second_equilibrium_aggregate_gate as second_gate,
)


def _assert_artifacts(paths: dict[str, str], *, json_key: str, csv_token: str) -> dict:
    for suffix in ("png", "pdf", "json", "csv"):
        assert Path(paths[suffix]).exists()
    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert json_key in payload
    assert csv_token in Path(paths["csv"]).read_text(encoding="utf-8")
    return payload


def _alpha_holdout_payload() -> dict[str, object]:
    return {
        "kind": "vmec_boozer_aggregate_line_search_holdout_report",
        "passed": True,
        "objective": "quasilinear_flux",
        "reduction": "mean",
        "initial_delta": 0.0,
        "final_delta": -1.0e-8,
        "training_passed": True,
        "heldout_passed": True,
        "training_initial_objective": 0.9,
        "training_final_objective": 0.88,
        "training_relative_reduction": 0.0222222222,
        "heldout_initial_objective": 0.91,
        "heldout_final_objective": 0.909,
        "heldout_relative_reduction": 0.0010989011,
        "training_samples": [
            {"surface_index": None, "alpha": 0.0, "selected_ky_index": 1},
            {"surface_index": None, "alpha": 0.0, "selected_ky_index": 2},
        ],
        "heldout_samples": [
            {"surface_index": None, "alpha": 0.5, "selected_ky_index": 1},
            {"surface_index": None, "alpha": 0.5, "selected_ky_index": 2},
        ],
    }


def _surface_holdout_payload() -> dict[str, object]:
    return {
        "kind": "vmec_boozer_aggregate_line_search_holdout_report",
        "passed": True,
        "objective": "quasilinear_flux",
        "reduction": "mean",
        "initial_delta": 0.0,
        "final_delta": 1.0e-8,
        "training_passed": True,
        "heldout_passed": True,
        "training_initial_objective": 0.8045688627,
        "training_final_objective": 0.8035154075,
        "training_relative_reduction": 0.0013093413,
        "heldout_initial_objective": 0.7205574540,
        "heldout_final_objective": 0.7202268146,
        "heldout_relative_reduction": 0.0004588662,
        "training_samples": [
            {"surface_index": 18, "alpha": 0.0, "selected_ky_index": 1},
            {"surface_index": 18, "alpha": 0.0, "selected_ky_index": 2},
        ],
        "heldout_samples": [
            {"surface_index": 19, "alpha": 0.0, "selected_ky_index": 1},
            {"surface_index": 19, "alpha": 0.0, "selected_ky_index": 2},
        ],
    }


def _objective_payload() -> dict[str, object]:
    return {
        "kind": "vmec_boozer_aggregate_scalar_objective_finite_difference_report",
        "passed": True,
        "objective": "quasilinear_flux",
        "reduction": "mean",
        "n_samples": 2,
        "base_value": 0.9,
        "minus_value": 0.85,
        "plus_value": 0.95,
        "central_derivative": 5.0,
        "response_abs": 0.1,
        "curvature_ratio": 0.02,
        "samples": [
            {
                "surface_index": None,
                "alpha": 0.0,
                "selected_ky_index": 1,
                "weight": 0.5,
            },
            {
                "surface_index": None,
                "alpha": 0.0,
                "selected_ky_index": 2,
                "weight": 0.5,
            },
        ],
        "minus_sample_values": [0.7, 1.0],
        "base_sample_values": [0.8, 1.0],
        "plus_sample_values": [0.9, 1.0],
    }


def _multi_point_payload() -> dict[str, object]:
    samples = [
        {"surface_index": None, "alpha": 0.0, "selected_ky_index": 1, "weight": 0.25},
        {"surface_index": None, "alpha": 0.0, "selected_ky_index": 2, "weight": 0.25},
        {"surface_index": None, "alpha": 0.5, "selected_ky_index": 1, "weight": 0.25},
        {"surface_index": None, "alpha": 0.5, "selected_ky_index": 2, "weight": 0.25},
    ]
    payload = _objective_payload()
    payload.update(
        n_samples=len(samples),
        samples=samples,
        minus_sample_values=[0.70, 0.95, 0.75, 1.00],
        base_sample_values=[0.80, 1.00, 0.85, 0.95],
        plus_sample_values=[0.90, 1.05, 0.95, 0.90],
    )
    return payload


def _line_search_payload(
    objective: str = "quasilinear_flux",
    derivative: float = 5.0,
    initial: float = 0.90,
    final: float = 0.88,
) -> dict[str, object]:
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
            {
                "surface_index": None,
                "alpha": 0.0,
                "selected_ky_index": 1,
                "weight": 0.5,
            },
            {
                "surface_index": None,
                "alpha": 0.0,
                "selected_ky_index": 2,
                "weight": 0.5,
            },
        ],
        "history": [
            {
                "step": 0,
                "delta": 0.0,
                "objective": initial,
                "central_derivative": derivative,
                "finite_difference_passed": True,
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


def _second_fd_payload() -> dict[str, object]:
    payload = _objective_payload()
    payload.update(
        case_name="li383_low_res",
        minus_value=9.80,
        base_value=9.79,
        plus_value=9.78,
        central_derivative=-1.0e5,
        response_abs=0.02,
        curvature_ratio=0.01,
    )
    return payload


def _second_line_payload() -> dict[str, object]:
    payload = _line_search_payload("quasilinear_flux", -1.0e5, 9.79, 9.78)
    payload.update(case_name="li383_low_res")
    return payload


def test_alpha_holdout_payload_uses_default_split(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_report(**kwargs):  # noqa: ANN003, ANN202
        calls.update(kwargs)
        return _alpha_holdout_payload()

    monkeypatch.setattr(
        alpha_gate, "vmec_boozer_aggregate_line_search_holdout_report", fake_report
    )

    payload = alpha_gate.build_vmec_boozer_aggregate_alpha_holdout_payload(
        ntheta=4, mboz=21, nboz=21
    )

    assert payload["artifact_kind"] == "vmec_boozer_aggregate_alpha_holdout_gate"
    assert payload["passed"] is True
    assert payload["holdout_split"]["training_alphas"] == [0.0]
    assert payload["holdout_split"]["holdout_alphas"] == [0.5]
    assert calls["training_selected_ky_indices"] == (1, 2)
    assert calls["holdout_selected_ky_indices"] == (1, 2)
    assert calls["mboz"] == 21
    assert calls["nboz"] == 21


def test_alpha_holdout_main_uses_report(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    def fake_report(**kwargs):  # noqa: ANN003, ANN202
        calls.update(kwargs)
        return _alpha_holdout_payload()

    monkeypatch.setattr(
        alpha_gate, "vmec_boozer_aggregate_line_search_holdout_report", fake_report
    )

    result = alpha_gate.main(
        [
            "--out",
            str(tmp_path / "alpha_holdout.png"),
            "--holdout-alphas",
            "0.25",
            "--training-selected-ky-indices",
            "1",
            "2",
            "--json-only",
        ]
    )

    assert result == 0
    assert calls["holdout_alphas"] == (0.25,)
    assert calls["training_selected_ky_indices"] == (1, 2)


def test_surface_holdout_payload_contracts(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_report(**kwargs):  # noqa: ANN003, ANN202
        calls.update(kwargs)
        return _surface_holdout_payload()

    monkeypatch.setattr(
        surface_gate, "vmec_boozer_aggregate_line_search_holdout_report", fake_report
    )

    payload = surface_gate.build_vmec_boozer_aggregate_surface_holdout_payload(
        ntheta=4, mboz=21, nboz=21
    )

    assert payload["artifact_kind"] == "vmec_boozer_aggregate_surface_holdout_gate"
    assert payload["passed"] is True
    assert payload["blocked"] is False
    assert payload["blockers"] == []
    assert payload["holdout_split"]["training_surface_indices"] == [18]
    assert payload["holdout_split"]["holdout_surface_indices"] == [19]
    assert calls["training_surface_indices"] == (18,)
    assert calls["holdout_surface_indices"] == (19,)


def test_surface_holdout_payload_fails_closed_on_execution_error(monkeypatch) -> None:
    def fake_report(**_kwargs):  # noqa: ANN003, ANN202
        raise ValueError("surface_index is outside the VMEC metric radial grid")

    monkeypatch.setattr(
        surface_gate, "vmec_boozer_aggregate_line_search_holdout_report", fake_report
    )

    payload = surface_gate.build_vmec_boozer_aggregate_surface_holdout_payload(
        training_surface_indices=(18,), holdout_surface_indices=(99,)
    )

    assert payload["passed"] is False
    assert payload["blocked"] is True
    assert payload["blockers"] == ["surface_split_execution_failed"]
    assert payload["exception_type"] == "ValueError"
    assert "surface_index" in payload["exception_message"]


def test_surface_holdout_payload_rejects_non_holdout_surface_split(monkeypatch) -> None:
    calls: list[object] = []

    def fake_report(**kwargs):  # noqa: ANN003, ANN202
        calls.append(kwargs)
        return _surface_holdout_payload()

    monkeypatch.setattr(
        surface_gate, "vmec_boozer_aggregate_line_search_holdout_report", fake_report
    )

    payload = surface_gate.build_vmec_boozer_aggregate_surface_holdout_payload(
        training_surface_indices=(18,), holdout_surface_indices=(18,)
    )

    assert payload["passed"] is False
    assert payload["blocked"] is True
    assert payload["blockers"] == ["surface_split_not_held_out"]
    assert calls == []


def test_surface_holdout_main_uses_report(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    def fake_report(**kwargs):  # noqa: ANN003, ANN202
        calls.update(kwargs)
        return _surface_holdout_payload()

    monkeypatch.setattr(
        surface_gate, "vmec_boozer_aggregate_line_search_holdout_report", fake_report
    )

    result = surface_gate.main(
        [
            "--out",
            str(tmp_path / "surface_holdout.png"),
            "--training-surface-indices",
            "18",
            "--holdout-surface-indices",
            "19",
            "--json-only",
        ]
    )

    assert result == 0
    assert calls["training_surface_indices"] == (18,)
    assert calls["holdout_surface_indices"] == (19,)


def test_line_search_gate_main_uses_report(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    def fake_report(**kwargs):  # noqa: ANN003, ANN202
        calls.update(kwargs)
        return _line_search_payload()

    monkeypatch.setattr(
        line_gate,
        "vmec_boozer_aggregate_scalar_objective_line_search_report",
        fake_report,
    )

    result = line_gate.main(
        [
            "--out",
            str(tmp_path / "line_search.png"),
            "--selected-ky-indices",
            "1",
            "2",
            "--max-steps",
            "2",
            "--json-only",
        ]
    )

    assert result == 0
    assert calls["selected_ky_indices"] == (1, 2)
    assert calls["max_steps"] == 2
    assert calls["mboz"] == 21


def test_objective_gate_main_uses_report(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    def fake_report(**kwargs):  # noqa: ANN003, ANN202
        calls.update(kwargs)
        return _objective_payload()

    monkeypatch.setattr(
        objective_gate,
        "vmec_boozer_aggregate_scalar_objective_finite_difference_report",
        fake_report,
    )

    result = objective_gate.main(
        [
            "--out",
            str(tmp_path / "gate.png"),
            "--selected-ky-indices",
            "1",
            "2",
            "--surface-indices",
            "3",
            "5",
            "--json-only",
        ]
    )

    assert result == 0
    assert calls["surface_indices"] == (3, 5)
    assert calls["selected_ky_indices"] == (1, 2)
    assert calls["mboz"] == 21


def test_objective_gate_maps_physical_ky_and_torflux(
    monkeypatch, tmp_path: Path
) -> None:
    calls: dict[str, object] = {}

    def fake_report(**kwargs):  # noqa: ANN003, ANN202
        calls.update(kwargs)
        return _objective_payload()

    monkeypatch.setattr(
        objective_gate,
        "vmec_boozer_aggregate_scalar_objective_finite_difference_report",
        fake_report,
    )

    result = objective_gate.main(
        [
            "--out",
            str(tmp_path / "gate.png"),
            "--ky-values",
            "0.1",
            "0.3",
            "0.5",
            "--json-only",
        ]
    )
    assert result == 0
    assert calls["selected_ky_indices"] == (1, 3, 5)
    assert calls["ny"] == 12
    assert calls["ly"] == pytest.approx(2.0 * np.pi / 0.1)

    result = objective_gate.main(
        [
            "--out",
            str(tmp_path / "gate2.png"),
            "--torflux-values",
            "0.5",
            "0.7",
            "--json-only",
        ]
    )
    assert result == 0
    assert calls["surface_indices"] == (None,)
    assert calls["torflux_values"] == (0.5, 0.7)

    with pytest.raises(ValueError, match="torflux-values or --surface-indices"):
        objective_gate.main(
            ["--surface-indices", "3", "--torflux-values", "0.5", "--json-only"]
        )


def test_physical_ky_annotation_adds_resolved_metadata() -> None:
    payload = _objective_payload()
    objective_gate._annotate_physical_ky_samples(
        payload,
        requested_ky_values=[0.1, 0.2],
        solver_grid_options={
            "selected_ky_indices": (1, 2),
            "resolved_ky_values": (0.100000001, 0.200000003),
        },
    )

    assert payload["samples"][0]["ky"] == pytest.approx(0.1)
    assert payload["samples"][0]["selected_ky"] == pytest.approx(0.100000001)
    assert payload["samples"][0]["ky_abs_error"] == pytest.approx(1.0e-9)
    assert payload["samples"][1]["ky"] == pytest.approx(0.2)


def test_comparison_report_uses_same_sample_set(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_report(**kwargs):  # noqa: ANN003, ANN202
        calls.append(dict(kwargs))
        objective = str(kwargs["objective"])
        if objective == "growth":
            return _line_search_payload(objective, 2.0, 0.30, 0.29)
        return _line_search_payload(objective, 5.0, 0.90, 0.88)

    monkeypatch.setattr(
        comparison_gate,
        "vmec_boozer_aggregate_scalar_objective_line_search_report",
        fake_report,
    )

    payload = comparison_gate.build_vmec_boozer_aggregate_line_search_comparison_report(
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


def test_comparison_main_uses_report(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def fake_report(**kwargs):  # noqa: ANN003, ANN202
        calls.append(dict(kwargs))
        return _line_search_payload(str(kwargs["objective"]), 1.0, 1.0, 0.9)

    monkeypatch.setattr(
        comparison_gate,
        "vmec_boozer_aggregate_scalar_objective_line_search_report",
        fake_report,
    )

    result = comparison_gate.main(
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


def test_multi_point_payload_contracts(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_report(**kwargs):  # noqa: ANN003, ANN202
        calls.update(kwargs)
        return _multi_point_payload()

    monkeypatch.setattr(
        multi_point_gate,
        "vmec_boozer_aggregate_scalar_objective_finite_difference_report",
        fake_report,
    )

    payload = multi_point_gate.build_vmec_boozer_multi_point_objective_payload(
        max_wall_seconds=0.0
    )

    assert payload["artifact_kind"] == "vmec_boozer_multi_point_objective_gate"
    assert payload["passed"] is True
    assert payload["claim_scope"].startswith("bounded finite-difference")
    assert payload["multi_point_coverage"]["multi_alpha_or_surface"] is True
    assert payload["multi_point_coverage"]["n_samples_requested"] == 4
    assert payload["bounded_runtime"]["max_samples"] == 8
    assert calls["surface_indices"] == (None,)
    assert calls["alphas"] == (0.0, 0.5)
    assert calls["selected_ky_indices"] == (1, 2)
    assert calls["mboz"] == 21
    assert calls["nboz"] == 21


def test_multi_point_payload_accepts_two_surfaces(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_report(**kwargs):  # noqa: ANN003, ANN202
        calls.update(kwargs)
        return _multi_point_payload()

    monkeypatch.setattr(
        multi_point_gate,
        "vmec_boozer_aggregate_scalar_objective_finite_difference_report",
        fake_report,
    )

    payload = multi_point_gate.build_vmec_boozer_multi_point_objective_payload(
        surface_indices=(3, 5),
        alphas=(0.0,),
        selected_ky_indices=(1,),
        max_wall_seconds=0.0,
    )

    assert payload["multi_point_coverage"]["surface_indices"] == [3, 5]
    assert payload["multi_point_coverage"]["n_samples_requested"] == 2
    assert calls["surface_indices"] == (3, 5)
    assert calls["alphas"] == (0.0,)
    assert calls["selected_ky_indices"] == (1,)


def test_multi_point_main_uses_report_and_bounds(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    def fake_report(**kwargs):  # noqa: ANN003, ANN202
        calls.update(kwargs)
        return _multi_point_payload()

    monkeypatch.setattr(
        multi_point_gate,
        "vmec_boozer_aggregate_scalar_objective_finite_difference_report",
        fake_report,
    )

    result = multi_point_gate.main(
        [
            "--out",
            str(tmp_path / "gate.png"),
            "--surface-indices",
            "3",
            "5",
            "--alphas",
            "0.0",
            "--selected-ky-indices",
            "1",
            "--json-only",
            "--max-wall-seconds",
            "0",
        ]
    )

    assert result == 0
    assert calls["surface_indices"] == (3, 5)
    assert calls["alphas"] == (0.0,)
    assert calls["selected_ky_indices"] == (1,)


@pytest.mark.parametrize(
    "argv",
    [
        ["--alphas", "0.0", "--selected-ky-indices", "1", "--json-only"],
        [
            "--surface-indices",
            "3",
            "5",
            "--alphas",
            "0.0",
            "0.5",
            "--selected-ky-indices",
            "1",
            "2",
            "3",
            "--max-samples",
            "8",
            "--json-only",
        ],
    ],
    ids=["single_point", "over_sample_bound"],
)
def test_multi_point_main_rejects_invalid_coverage(
    monkeypatch, argv: list[str]
) -> None:
    def fake_report(**_kwargs):  # noqa: ANN003, ANN202
        raise AssertionError("report should not run for invalid coverage")

    monkeypatch.setattr(
        multi_point_gate,
        "vmec_boozer_aggregate_scalar_objective_finite_difference_report",
        fake_report,
    )

    with pytest.raises(SystemExit):
        multi_point_gate.main(argv)


def test_second_equilibrium_payload_passes_with_mode21_defaults(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_fd(**kwargs):  # noqa: ANN003, ANN202
        calls["fd"] = kwargs
        return _second_fd_payload()

    def fake_line(**kwargs):  # noqa: ANN003, ANN202
        calls["line"] = kwargs
        return _second_line_payload()

    monkeypatch.setattr(
        second_gate,
        "vmec_boozer_aggregate_scalar_objective_finite_difference_report",
        fake_fd,
    )
    monkeypatch.setattr(
        second_gate,
        "vmec_boozer_aggregate_scalar_objective_line_search_report",
        fake_line,
    )

    payload = second_gate.build_vmec_boozer_second_equilibrium_aggregate_payload(
        max_wall_seconds=0.0
    )

    assert payload["passed"] is True
    assert payload["feasible"] is True
    assert payload["case_name"] == "li383_low_res"
    assert payload["mode_bound"] == {
        "mboz": 21,
        "nboz": 21,
        "minimum_required": 21,
        "passed": True,
    }
    assert payload["coverage"]["selected_ky_indices"] == [1, 2]
    assert payload["finite_difference_summary"]["central_derivative"] == -1.0e5
    assert payload["line_search_summary"]["accepted_steps"] == 1
    assert calls["fd"]["mboz"] == 21
    assert calls["fd"]["nboz"] == 21
    assert calls["line"]["case_name"] == "li383_low_res"


def test_second_equilibrium_payload_fails_closed_on_backend_error(monkeypatch) -> None:
    def fake_fd(**_kwargs):  # noqa: ANN003, ANN202
        raise RuntimeError("vmec_jax example fixture missing")

    monkeypatch.setattr(
        second_gate,
        "vmec_boozer_aggregate_scalar_objective_finite_difference_report",
        fake_fd,
    )

    payload = second_gate.build_vmec_boozer_second_equilibrium_aggregate_payload(
        max_wall_seconds=0.0
    )

    assert payload["passed"] is False
    assert payload["feasible"] is False
    assert payload["blocker_type"] == "RuntimeError"
    assert payload["blocker_message"] == "vmec_jax example fixture missing"
    assert payload["mode_bound"]["passed"] is True


def test_second_equilibrium_json_only_uses_reports(monkeypatch, capsys) -> None:
    calls: dict[str, object] = {}

    def fake_fd(**kwargs):  # noqa: ANN003, ANN202
        calls["fd"] = kwargs
        return _second_fd_payload()

    def fake_line(**kwargs):  # noqa: ANN003, ANN202
        calls["line"] = kwargs
        return _second_line_payload()

    monkeypatch.setattr(
        second_gate,
        "vmec_boozer_aggregate_scalar_objective_finite_difference_report",
        fake_fd,
    )
    monkeypatch.setattr(
        second_gate,
        "vmec_boozer_aggregate_scalar_objective_line_search_report",
        fake_line,
    )

    result = second_gate.main(
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


@pytest.mark.parametrize(
    ("gate", "writer", "payload", "out_name", "json_key", "csv_token"),
    [
        (
            alpha_gate,
            "write_vmec_boozer_aggregate_alpha_holdout_artifacts",
            {
                **_alpha_holdout_payload(),
                "artifact_kind": "vmec_boozer_aggregate_alpha_holdout_gate",
                "wall_seconds": 1.25,
            },
            "alpha_holdout.png",
            "artifact_kind",
            "heldout",
        ),
        (
            surface_gate,
            "write_vmec_boozer_aggregate_surface_holdout_artifacts",
            {
                **_surface_holdout_payload(),
                "kind": "vmec_boozer_aggregate_surface_holdout_gate",
                "artifact_kind": "vmec_boozer_aggregate_surface_holdout_gate",
                "blocked": False,
                "blockers": [],
                "wall_seconds": 1.25,
                "holdout_split": {
                    "training_surface_indices": [18],
                    "training_alphas": [0.0],
                    "training_selected_ky_indices": [1, 2],
                    "holdout_surface_indices": [19],
                    "holdout_alphas": [0.0],
                    "holdout_selected_ky_indices": [1, 2],
                },
            },
            "surface_holdout.png",
            "artifact_kind",
            "heldout_surface",
        ),
        (
            line_gate,
            "write_vmec_boozer_aggregate_line_search_artifacts",
            _line_search_payload(),
            "line_search.png",
            "passed",
            "candidate_objective",
        ),
        (
            comparison_gate,
            "write_vmec_boozer_aggregate_line_search_comparison_artifacts",
            _comparison_payload(),
            "comparison.png",
            "same_initial_update_direction",
            "initial_update_direction",
        ),
        (
            objective_gate,
            "write_vmec_boozer_aggregate_objective_artifacts",
            {
                **_objective_payload(),
                "samples": [
                    {
                        **_objective_payload()["samples"][0],
                        "torflux": 0.64,
                        "surface": 0.64,
                        "ky": 0.1,
                        "selected_ky": 0.1,
                        "ky_abs_error": 0.0,
                    },
                    _objective_payload()["samples"][1],
                ],
            },
            "aggregate_gate.png",
            "passed",
            "ky_abs_error",
        ),
        (
            multi_point_gate,
            "write_vmec_boozer_multi_point_objective_artifacts",
            multi_point_gate._annotate_payload(
                _multi_point_payload(),
                surfaces=(None,),
                alphas=(0.0, 0.5),
                selected_ky_indices=(1, 2),
                max_samples=8,
                max_wall_seconds=300.0,
                elapsed_wall_seconds=1.25,
            ),
            "multi_point_gate.png",
            "artifact_kind",
            "alpha",
        ),
        (
            second_gate,
            "write_vmec_boozer_second_equilibrium_aggregate_artifacts",
            {
                "kind": "vmec_boozer_second_equilibrium_aggregate_gate",
                "passed": True,
                "feasible": True,
                "case_name": "li383_low_res",
                "objective": "quasilinear_flux",
                "mode_bound": {
                    "mboz": 21,
                    "nboz": 21,
                    "minimum_required": 21,
                    "passed": True,
                },
                "sample_bound": {
                    "n_samples_requested": 2,
                    "max_samples": 4,
                    "passed": True,
                },
                "bounded_runtime": {
                    "max_wall_seconds": 300.0,
                    "elapsed_wall_seconds": 41.2,
                    "passed": True,
                },
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
            },
            "second_gate.png",
            "kind",
            "fd_central_derivative",
        ),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_vmec_boozer_aggregate_writers(
    tmp_path: Path,
    gate,
    writer: str,
    payload: dict[str, object],
    out_name: str,
    json_key: str,
    csv_token: str,
) -> None:
    paths = getattr(gate, writer)(payload, out=tmp_path / out_name)
    _assert_artifacts(paths, json_key=json_key, csv_token=csv_token)
