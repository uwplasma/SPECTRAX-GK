from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "build_vmec_boozer_aggregate_line_search_gate.py"
spec = importlib.util.spec_from_file_location("build_vmec_boozer_aggregate_line_search_gate", SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def _payload() -> dict[str, object]:
    return {
        "kind": "vmec_boozer_aggregate_scalar_objective_line_search_report",
        "passed": True,
        "objective": "quasilinear_flux",
        "reduction": "mean",
        "n_samples": 2,
        "accepted_steps": 1,
        "max_steps": 1,
        "initial_objective": 0.9,
        "final_objective": 0.88,
        "relative_reduction": 0.022,
        "stop_reason": "max_steps",
        "history": [
            {
                "step": 0,
                "delta": 0.0,
                "objective": 0.9,
                "central_derivative": 5.0,
                "curvature_ratio": 0.02,
                "accepted": True,
                "candidate_delta": -1.0e-8,
                "candidate_objective": 0.88,
            }
        ],
    }


def test_write_vmec_boozer_aggregate_line_search_artifacts(tmp_path: Path) -> None:
    paths = mod.write_vmec_boozer_aggregate_line_search_artifacts(
        _payload(),
        out=tmp_path / "line_search.png",
    )

    for suffix in ("png", "pdf", "json", "csv"):
        assert Path(paths[suffix]).exists()
    data = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert data["passed"] is True
    assert "candidate_objective" in Path(paths["csv"]).read_text(encoding="utf-8")


def test_vmec_boozer_aggregate_line_search_gate_main_uses_report(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    def fake_report(**kwargs):  # noqa: ANN003, ANN202
        calls.update(kwargs)
        return _payload()

    monkeypatch.setattr(mod, "vmec_boozer_aggregate_scalar_objective_line_search_report", fake_report)

    result = mod.main(
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
