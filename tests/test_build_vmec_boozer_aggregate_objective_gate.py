from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
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
    paths = mod.write_vmec_boozer_aggregate_objective_artifacts(
        _payload(),
        out=tmp_path / "aggregate_gate.png",
    )

    for suffix in ("png", "pdf", "json", "csv"):
        assert Path(paths[suffix]).exists()
    data = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert data["passed"] is True
    assert "selected_ky_index" in Path(paths["csv"]).read_text(encoding="utf-8")


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
