"""Tests for the VMEC/Boozer reduced nonlinear-window gradient artifact builder."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "build_vmec_boozer_nonlinear_window_gradient_gate.py"
spec = importlib.util.spec_from_file_location("build_vmec_boozer_nonlinear_window_gradient_gate", SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def _fake_payload() -> dict[str, object]:
    objective_names = [
        "gamma",
        "nonlinear_window_heat_flux_mean",
        "nonlinear_window_heat_flux_cv",
        "nonlinear_window_heat_flux_trend",
    ]
    return {
        "kind": "mode21_vmec_boozer_nonlinear_window_gradient_gate",
        "passed": True,
        "source_scope": "mode21_vmec_boozer_state",
        "parameter_names": ["Rcos_mid_surface_m1"],
        "objective_names": objective_names,
        "objective_gates": [
            {
                "objective": objective,
                "parameter": "Rcos_mid_surface_m1",
                "implicit": float(index + 1),
                "finite_difference": float(index + 1),
                "abs_error": 0.0,
                "rel_error": 0.0,
                "passed": True,
            }
            for index, objective in enumerate(objective_names)
        ],
        "eigenpair_gate": {
            "atol": 1.0e-6,
            "jacobian_implicit": [[1.0], [2.0], [3.0], [4.0]],
            "jacobian_fd": [[1.0], [2.0], [3.0], [4.0]],
        },
    }


def test_nonlinear_window_builder_writes_artifacts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(mod, "mode21_vmec_boozer_nonlinear_window_gradient_report", lambda **_kwargs: _fake_payload())

    out = tmp_path / "vmec_boozer_nonlinear_window_gradient_gate.png"
    assert mod.main(["--out", str(out), "--surface-stencil-width", "3"]) == 0

    for suffix in (".png", ".pdf", ".json", ".csv"):
        assert out.with_suffix(suffix).exists()
    saved = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert saved["kind"] == "mode21_vmec_boozer_nonlinear_window_gradient_gate"


def test_nonlinear_window_builder_json_only(capsys, monkeypatch) -> None:
    monkeypatch.setattr(mod, "mode21_vmec_boozer_nonlinear_window_gradient_report", lambda **_kwargs: _fake_payload())

    assert mod.main(["--json-only", "--nonlinear-steps", "12"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is True
    assert payload["objective_names"][-1] == "nonlinear_window_heat_flux_trend"
