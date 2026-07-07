from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "tools" / "build_vmec_boozer_aggregate_objective_gate.py"
spec = importlib.util.spec_from_file_location("build_vmec_boozer_aggregate_objective_gate", SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def _payload() -> dict[str, object]:
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
            {"surface_index": None, "alpha": 0.0, "selected_ky_index": 1, "weight": 0.5},
            {"surface_index": None, "alpha": 0.0, "selected_ky_index": 2, "weight": 0.5},
        ],
        "minus_sample_values": [0.7, 1.0],
        "base_sample_values": [0.8, 1.0],
        "plus_sample_values": [0.9, 1.0],
    }


def test_write_vmec_boozer_aggregate_objective_artifacts(tmp_path: Path) -> None:
    payload = _payload()
    payload["samples"][0]["torflux"] = 0.64
    payload["samples"][0]["surface"] = 0.64
    payload["samples"][0]["ky"] = 0.1
    payload["samples"][0]["selected_ky"] = 0.1
    payload["samples"][0]["ky_abs_error"] = 0.0
    paths = mod.write_vmec_boozer_aggregate_objective_artifacts(
        payload,
        out=tmp_path / "aggregate_gate.png",
    )

    for suffix in ("png", "pdf", "json", "csv"):
        assert Path(paths[suffix]).exists()
    data = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert data["passed"] is True
    csv_text = Path(paths["csv"]).read_text(encoding="utf-8")
    assert "selected_ky_index" in csv_text
    assert "torflux" in csv_text
    assert "ky_abs_error" in csv_text


def test_vmec_boozer_aggregate_objective_gate_main_uses_report(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    def fake_report(**kwargs):  # noqa: ANN003, ANN202
        calls.update(kwargs)
        return _payload()

    monkeypatch.setattr(mod, "vmec_boozer_aggregate_scalar_objective_finite_difference_report", fake_report)

    result = mod.main(
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


def test_vmec_boozer_aggregate_objective_gate_main_maps_physical_ky_values(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls: dict[str, object] = {}

    def fake_report(**kwargs):  # noqa: ANN003, ANN202
        calls.update(kwargs)
        return _payload()

    monkeypatch.setattr(mod, "vmec_boozer_aggregate_scalar_objective_finite_difference_report", fake_report)

    result = mod.main(
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


def test_vmec_boozer_aggregate_objective_gate_main_forwards_physical_torflux(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls: dict[str, object] = {}

    def fake_report(**kwargs):  # noqa: ANN003, ANN202
        calls.update(kwargs)
        return _payload()

    monkeypatch.setattr(mod, "vmec_boozer_aggregate_scalar_objective_finite_difference_report", fake_report)

    result = mod.main(
        [
            "--out",
            str(tmp_path / "gate.png"),
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
        mod.main(
            [
                "--surface-indices",
                "3",
                "--torflux-values",
                "0.5",
                "--json-only",
            ]
        )


def test_physical_ky_annotation_adds_resolved_metadata() -> None:
    payload = _payload()
    mod._annotate_physical_ky_samples(
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
