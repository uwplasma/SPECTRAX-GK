from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "build_vmec_boozer_aggregate_surface_holdout_gate.py"
spec = importlib.util.spec_from_file_location("build_vmec_boozer_aggregate_surface_holdout_gate", SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


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


def test_build_surface_holdout_payload_uses_default_true_surface_split(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_report(**kwargs):  # noqa: ANN003, ANN202
        calls.update(kwargs)
        return _surface_holdout_payload()

    monkeypatch.setattr(mod, "vmec_boozer_aggregate_line_search_holdout_report", fake_report)

    payload = mod.build_vmec_boozer_aggregate_surface_holdout_payload(
        ntheta=4,
        mboz=21,
        nboz=21,
    )

    assert payload["artifact_kind"] == "vmec_boozer_aggregate_surface_holdout_gate"
    assert payload["passed"] is True
    assert payload["blocked"] is False
    assert payload["blockers"] == []
    assert payload["holdout_split"]["training_surface_indices"] == [18]
    assert payload["holdout_split"]["holdout_surface_indices"] == [19]
    assert calls["training_surface_indices"] == (18,)
    assert calls["holdout_surface_indices"] == (19,)
    assert calls["training_selected_ky_indices"] == (1, 2)
    assert calls["holdout_selected_ky_indices"] == (1, 2)


def test_build_surface_holdout_payload_fails_closed_on_execution_error(monkeypatch) -> None:
    def fake_report(**_kwargs):  # noqa: ANN003, ANN202
        raise ValueError("surface_index is outside the VMEC metric radial grid")

    monkeypatch.setattr(mod, "vmec_boozer_aggregate_line_search_holdout_report", fake_report)

    payload = mod.build_vmec_boozer_aggregate_surface_holdout_payload(
        training_surface_indices=(18,),
        holdout_surface_indices=(99,),
    )

    assert payload["passed"] is False
    assert payload["blocked"] is True
    assert payload["blockers"] == ["surface_split_execution_failed"]
    assert payload["exception_type"] == "ValueError"
    assert "surface_index" in payload["exception_message"]


def test_build_surface_holdout_payload_rejects_non_holdout_surface_split(monkeypatch) -> None:
    calls: list[object] = []

    def fake_report(**kwargs):  # noqa: ANN003, ANN202
        calls.append(kwargs)
        return _surface_holdout_payload()

    monkeypatch.setattr(mod, "vmec_boozer_aggregate_line_search_holdout_report", fake_report)

    payload = mod.build_vmec_boozer_aggregate_surface_holdout_payload(
        training_surface_indices=(18,),
        holdout_surface_indices=(18,),
    )

    assert payload["passed"] is False
    assert payload["blocked"] is True
    assert payload["blockers"] == ["surface_split_not_held_out"]
    assert calls == []


def test_write_surface_holdout_artifacts(tmp_path: Path) -> None:
    payload = {
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
    }
    paths = mod.write_vmec_boozer_aggregate_surface_holdout_artifacts(
        payload,
        out=tmp_path / "surface_holdout.png",
    )

    for suffix in ("png", "pdf", "json", "csv"):
        assert Path(paths[suffix]).exists()
    data = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert data["artifact_kind"] == "vmec_boozer_aggregate_surface_holdout_gate"
    assert "heldout_surface" in Path(paths["csv"]).read_text(encoding="utf-8")


def test_surface_holdout_main_uses_report(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    def fake_report(**kwargs):  # noqa: ANN003, ANN202
        calls.update(kwargs)
        return _surface_holdout_payload()

    monkeypatch.setattr(mod, "vmec_boozer_aggregate_line_search_holdout_report", fake_report)

    result = mod.main(
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
