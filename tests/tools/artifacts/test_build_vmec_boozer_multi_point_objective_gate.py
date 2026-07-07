from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "tools" / "build_vmec_boozer_multi_point_objective_gate.py"
spec = importlib.util.spec_from_file_location(
    "build_vmec_boozer_multi_point_objective_gate",
    SCRIPT,
)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def _payload() -> dict[str, object]:
    samples = [
        {"surface_index": None, "alpha": 0.0, "selected_ky_index": 1, "weight": 0.25},
        {"surface_index": None, "alpha": 0.0, "selected_ky_index": 2, "weight": 0.25},
        {"surface_index": None, "alpha": 0.5, "selected_ky_index": 1, "weight": 0.25},
        {"surface_index": None, "alpha": 0.5, "selected_ky_index": 2, "weight": 0.25},
    ]
    return {
        "kind": "vmec_boozer_aggregate_scalar_objective_finite_difference_report",
        "passed": True,
        "objective": "quasilinear_flux",
        "reduction": "mean",
        "n_samples": len(samples),
        "base_value": 0.9,
        "minus_value": 0.85,
        "plus_value": 0.95,
        "central_derivative": 5.0,
        "response_abs": 0.1,
        "curvature_ratio": 0.02,
        "samples": samples,
        "minus_sample_values": [0.70, 0.95, 0.75, 1.00],
        "base_sample_values": [0.80, 1.00, 0.85, 0.95],
        "plus_sample_values": [0.90, 1.05, 0.95, 0.90],
    }


def test_build_multi_point_payload_uses_multi_alpha_defaults(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_report(**kwargs):  # noqa: ANN003, ANN202
        calls.update(kwargs)
        return _payload()

    monkeypatch.setattr(
        mod,
        "vmec_boozer_aggregate_scalar_objective_finite_difference_report",
        fake_report,
    )

    payload = mod.build_vmec_boozer_multi_point_objective_payload(
        max_wall_seconds=0.0,
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


def test_build_multi_point_payload_accepts_two_surfaces(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_report(**kwargs):  # noqa: ANN003, ANN202
        calls.update(kwargs)
        return _payload()

    monkeypatch.setattr(
        mod,
        "vmec_boozer_aggregate_scalar_objective_finite_difference_report",
        fake_report,
    )

    payload = mod.build_vmec_boozer_multi_point_objective_payload(
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


def test_write_vmec_boozer_multi_point_objective_artifacts(tmp_path: Path) -> None:
    payload = mod._annotate_payload(
        _payload(),
        surfaces=(None,),
        alphas=(0.0, 0.5),
        selected_ky_indices=(1, 2),
        max_samples=8,
        max_wall_seconds=300.0,
        elapsed_wall_seconds=1.25,
    )
    paths = mod.write_vmec_boozer_multi_point_objective_artifacts(
        payload,
        out=tmp_path / "multi_point_gate.png",
    )

    for suffix in ("png", "pdf", "json", "csv"):
        assert Path(paths[suffix]).exists()
    data = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert data["artifact_kind"] == "vmec_boozer_multi_point_objective_gate"
    assert data["multi_point_coverage"]["alphas"] == [0.0, 0.5]
    assert "alpha" in Path(paths["csv"]).read_text(encoding="utf-8")


def test_main_json_only_uses_report_and_bounds(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    def fake_report(**kwargs):  # noqa: ANN003, ANN202
        calls.update(kwargs)
        return _payload()

    monkeypatch.setattr(
        mod,
        "vmec_boozer_aggregate_scalar_objective_finite_difference_report",
        fake_report,
    )

    result = mod.main(
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


def test_main_rejects_single_alpha_single_surface(monkeypatch) -> None:
    def fake_report(**_kwargs):  # noqa: ANN003, ANN202
        raise AssertionError("report should not run for invalid coverage")

    monkeypatch.setattr(
        mod,
        "vmec_boozer_aggregate_scalar_objective_finite_difference_report",
        fake_report,
    )

    with pytest.raises(SystemExit):
        mod.main(
            [
                "--alphas",
                "0.0",
                "--selected-ky-indices",
                "1",
                "--json-only",
            ]
        )


def test_main_rejects_over_sample_bound(monkeypatch) -> None:
    def fake_report(**_kwargs):  # noqa: ANN003, ANN202
        raise AssertionError("report should not run when sample cap is exceeded")

    monkeypatch.setattr(
        mod,
        "vmec_boozer_aggregate_scalar_objective_finite_difference_report",
        fake_report,
    )

    with pytest.raises(SystemExit):
        mod.main(
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
            ]
        )
