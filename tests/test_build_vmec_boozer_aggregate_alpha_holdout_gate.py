from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "build_vmec_boozer_aggregate_alpha_holdout_gate.py"
spec = importlib.util.spec_from_file_location("build_vmec_boozer_aggregate_alpha_holdout_gate", SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def _holdout_payload() -> dict[str, object]:
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


def test_build_alpha_holdout_payload_uses_default_split(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_report(**kwargs):  # noqa: ANN003, ANN202
        calls.update(kwargs)
        return _holdout_payload()

    monkeypatch.setattr(mod, "vmec_boozer_aggregate_line_search_holdout_report", fake_report)

    payload = mod.build_vmec_boozer_aggregate_alpha_holdout_payload(
        ntheta=4,
        mboz=21,
        nboz=21,
    )

    assert payload["artifact_kind"] == "vmec_boozer_aggregate_alpha_holdout_gate"
    assert payload["passed"] is True
    assert payload["holdout_split"]["training_alphas"] == [0.0]
    assert payload["holdout_split"]["holdout_alphas"] == [0.5]
    assert calls["training_selected_ky_indices"] == (1, 2)
    assert calls["holdout_selected_ky_indices"] == (1, 2)
    assert calls["mboz"] == 21
    assert calls["nboz"] == 21


def test_write_alpha_holdout_artifacts(tmp_path: Path) -> None:
    payload = {
        **_holdout_payload(),
        "artifact_kind": "vmec_boozer_aggregate_alpha_holdout_gate",
        "wall_seconds": 1.25,
    }
    paths = mod.write_vmec_boozer_aggregate_alpha_holdout_artifacts(
        payload,
        out=tmp_path / "alpha_holdout.png",
    )

    for suffix in ("png", "pdf", "json", "csv"):
        assert Path(paths[suffix]).exists()
    data = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert data["artifact_kind"] == "vmec_boozer_aggregate_alpha_holdout_gate"
    assert "heldout" in Path(paths["csv"]).read_text(encoding="utf-8")


def test_alpha_holdout_main_uses_report(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    def fake_report(**kwargs):  # noqa: ANN003, ANN202
        calls.update(kwargs)
        return _holdout_payload()

    monkeypatch.setattr(mod, "vmec_boozer_aggregate_line_search_holdout_report", fake_report)

    result = mod.main(
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
