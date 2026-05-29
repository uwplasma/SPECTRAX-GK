from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "write_vmec_state_control_short_bracket_launch.py"


def _load_tool_module():
    spec = importlib.util.spec_from_file_location("write_vmec_state_control_short_bracket_launch", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _runbook() -> dict[str, object]:
    return {
        "kind": "nonlinear_gradient_state_control_runbook",
        "passed": True,
        "mapped_controls": [
            {
                "state_parameter": "Rsin_mid_surface_m1",
                "state_control_argument": "Rsin_mid_surface_m1:1",
                "mapping_ready": True,
                "condition_number": 1.2,
                "relative_residual": 1.0e-13,
                "input_control_argument": "RBS(1,1):1.5 ZBC(1,1):-0.25",
                "input_direction": {
                    "type": "least_squares_boundary_coefficient_direction",
                    "terms": [
                        {"coefficient": "RBS(1,1)", "weight": 1.5},
                        {"coefficient": "ZBC(1,1)", "weight": -0.25},
                    ],
                },
            }
        ],
    }


def _input_text() -> str:
    return """&INDATA
  LASYM = F
  RBC(0,0) = 1.0000000000000000E+00
  RBC(1,1) = 1.0000000000000000E-01,   ZBS(1,1) = 2.0000000000000000E-02
/
"""


def test_state_control_short_bracket_writer_inserts_weighted_lasym_inputs(tmp_path: Path) -> None:
    pytest.importorskip("vmec_jax")
    mod = _load_tool_module()
    runbook_path = tmp_path / "runbook.json"
    baseline = tmp_path / "input.final"
    out_prefix = tmp_path / "state_control_launch"
    runbook_path.write_text(json.dumps(_runbook()), encoding="utf-8")
    baseline.write_text(_input_text(), encoding="utf-8")

    rc = mod.main(
        [
            str(runbook_path),
            "--baseline-input",
            str(baseline),
            "--out-dir",
            str(tmp_path / "launch_inputs"),
            "--out-prefix",
            str(out_prefix),
            "--case",
            "qa_state_control",
            "--alpha-delta",
            "0.001",
            "--vmec-extra-args",
            "--outdir . --fast",
            "--horizons",
            "20",
            "--window-tmin",
            "10",
            "--window-tmax",
            "20",
        ]
    )

    payload = json.loads(out_prefix.with_suffix(".json").read_text(encoding="utf-8"))
    plus = (
        tmp_path
        / "launch_inputs"
        / "rsin_mid_surface_m1"
        / "input.qa_state_control_rsin_mid_surface_m1_plus_delta"
    ).read_text(encoding="utf-8")
    minus = (
        tmp_path
        / "launch_inputs"
        / "rsin_mid_surface_m1"
        / "input.qa_state_control_rsin_mid_surface_m1_minus_delta"
    ).read_text(encoding="utf-8")

    assert rc == 0
    assert payload["kind"] == "vmec_state_control_short_bracket_launch_plan"
    assert payload["claim_level"].endswith("not_simulation_claim")
    assert payload["launches"][0]["state_parameter"] == "Rsin_mid_surface_m1"
    assert payload["launches"][0]["generated_lasym"] is True
    assert payload["launches"][0]["alpha_delta"] == pytest.approx(0.001)
    assert "LASYM = .TRUE." in plus
    assert "RBS(1,1) = 1.5000000000000000E-03" in plus
    assert "ZBC(1,1) = -2.5000000000000001E-04" in plus
    assert "RBS(1,1) = -1.5000000000000000E-03" in minus
    assert "--outdir . --fast" in payload["vmec_run_commands"][0]
    assert "write_nonlinear_turbulence_gradient_campaign.py" in payload["campaign_commands_after_vmec_runs"][0]
    assert out_prefix.with_suffix(".csv").exists()
    assert out_prefix.with_suffix(".png").exists()
    assert out_prefix.with_suffix(".pdf").exists()


def test_state_control_short_bracket_writer_fails_closed(tmp_path: Path) -> None:
    mod = _load_tool_module()
    baseline = tmp_path / "input.final"
    baseline.write_text(_input_text(), encoding="utf-8")

    with pytest.raises(ValueError, match="runbook must pass"):
        mod.write_state_control_short_bracket_launch(
            runbook={"passed": False, "mapped_controls": []},
            runbook_path=None,
            baseline_input=baseline,
            out_dir=tmp_path / "out",
            out_prefix=tmp_path / "launch",
            case="bad",
        )

    bad = _runbook()
    bad["mapped_controls"][0]["input_direction"]["terms"][0]["weight"] = 0.0  # type: ignore[index]
    with pytest.raises(ValueError, match="nonfinite or zero"):
        mod.write_state_control_short_bracket_launch(
            runbook=bad,
            runbook_path=None,
            baseline_input=baseline,
            out_dir=tmp_path / "out",
            out_prefix=tmp_path / "launch",
            case="bad",
        )
