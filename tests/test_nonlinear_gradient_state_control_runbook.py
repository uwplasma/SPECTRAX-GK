from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

from spectraxgk.nonlinear_gradient_followup import (
    NonlinearGradientStateControlRunbookConfig,
    nonlinear_gradient_state_control_runbook_report,
)


ROOT = Path(__file__).resolve().parents[1]


def _ql_screen() -> dict[str, object]:
    return {
        "kind": "nonlinear_turbulence_gradient_ql_seed_screen",
        "passed": True,
        "admitted_controls": [
            {
                "state_parameter": "Rsin_mid_surface_m1",
                "state_control_argument": "Rsin_mid_surface_m1:1",
                "descent_direction_sign": 1.0,
            },
            {
                "state_parameter": "Zcos_mid_surface_m1",
                "state_control_argument": "Zcos_mid_surface_m1:1",
                "descent_direction_sign": 1.0,
            },
        ],
    }


def _mapping() -> dict[str, object]:
    return {
        "kind": "vmec_state_to_input_control_mapping",
        "passed": True,
        "controls": [
            {
                "state_parameter": "Rsin_mid_surface_m1",
                "input_control_argument": "RBC(1,1):0.7",
                "passed": True,
                "condition_number": 12.0,
                "relative_residual": 0.01,
            },
            {
                "state_parameter": "Zcos_mid_surface_m1",
                "input_control_argument": "ZBS(1,1):-0.5",
                "passed": True,
                "condition_number": 15.0,
                "relative_residual": 0.02,
            },
        ],
    }


def test_state_control_runbook_fails_closed_without_mapping() -> None:
    report = nonlinear_gradient_state_control_runbook_report(_ql_screen())

    assert report["passed"] is False
    assert report["summary"]["admitted_state_control_count"] == 2
    assert report["summary"]["mapped_control_count"] == 0
    assert {row["blockers"][0] for row in report["controls"]} == {"missing_state_to_input_mapping"}
    assert "state-to-input mapping" in report["next_action"]
    assert "upstream evidence only" in report["scope_note"]


def test_state_control_runbook_admits_conditioned_mapping() -> None:
    report = nonlinear_gradient_state_control_runbook_report(_ql_screen(), mapping_artifacts=[_mapping()])

    assert report["passed"] is True
    assert report["summary"]["mapped_control_count"] == 2
    commands = {row["short_bracket_command_fragment"] for row in report["mapped_controls"]}
    assert commands == {
        "--control RBC(1,1):0.7 --relative-delta 0.02",
        "--control ZBS(1,1):-0.5 --relative-delta 0.02",
    }


def test_state_control_runbook_rejects_bad_mapping_and_validates_config() -> None:
    mapping = _mapping()
    mapping["controls"][0]["condition_number"] = 2.0e6  # type: ignore[index]
    mapping["controls"][1]["relative_residual"] = 0.2  # type: ignore[index]
    report = nonlinear_gradient_state_control_runbook_report(_ql_screen(), mapping_artifacts=[mapping])

    assert report["passed"] is False
    blockers = {row["state_parameter"]: row["blockers"] for row in report["controls"]}
    assert blockers["Rsin_mid_surface_m1"] == ["mapping_condition_number_too_large"]
    assert blockers["Zcos_mid_surface_m1"] == ["mapping_relative_residual_too_large"]

    missing_input = _mapping()
    del missing_input["controls"][0]["input_control_argument"]  # type: ignore[index]
    missing_report = nonlinear_gradient_state_control_runbook_report(_ql_screen(), mapping_artifacts=[missing_input])
    assert missing_report["controls"][0]["blockers"] == ["missing_input_control_argument"]

    validation_cases = [
        ("min_mapped_controls", NonlinearGradientStateControlRunbookConfig(min_mapped_controls=0)),
        (
            "max_mapping_condition_number",
            NonlinearGradientStateControlRunbookConfig(max_mapping_condition_number=0.0),
        ),
        (
            "max_mapping_relative_residual",
            NonlinearGradientStateControlRunbookConfig(max_mapping_relative_residual=-1.0),
        ),
        ("default_relative_delta", NonlinearGradientStateControlRunbookConfig(default_relative_delta=0.0)),
    ]
    for message, config in validation_cases:
        with pytest.raises(ValueError, match=message):
            nonlinear_gradient_state_control_runbook_report(_ql_screen(), config=config)


def test_design_nonlinear_gradient_state_control_runbook_tool_writes_artifacts(tmp_path: Path) -> None:
    path = ROOT / "tools" / "design_nonlinear_gradient_state_control_runbook.py"
    spec = importlib.util.spec_from_file_location("design_nonlinear_gradient_state_control_runbook", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    ql_path = tmp_path / "ql.json"
    mapping_path = tmp_path / "mapping.json"
    out_prefix = tmp_path / "runbook"
    ql_path.write_text(json.dumps(_ql_screen()), encoding="utf-8")
    mapping_path.write_text(json.dumps(_mapping()), encoding="utf-8")

    assert module.main([str(ql_path), "--mapping-artifact", str(mapping_path), "--out-prefix", str(out_prefix)]) == 0
    payload = json.loads(out_prefix.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["kind"] == "nonlinear_gradient_state_control_runbook"
    assert payload["passed"] is True
    assert out_prefix.with_suffix(".csv").exists()
    assert out_prefix.with_suffix(".png").exists()
    assert out_prefix.with_suffix(".pdf").exists()
