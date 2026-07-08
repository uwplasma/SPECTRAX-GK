from __future__ import annotations

from support.paths import REPO_ROOT, load_campaign_tool
import json
from pathlib import Path

import pytest


ROOT = REPO_ROOT
SCRIPT = ROOT / "tools" / "campaigns" / "write_vmec_state_to_input_mapping_campaign.py"


def _load_tool_module():
    return load_campaign_tool("write_vmec_state_to_input_mapping_campaign")


def _ql_seed_screen() -> dict[str, object]:
    return {
        "kind": "nonlinear_turbulence_gradient_ql_seed_screen",
        "passed": True,
        "admitted_controls": [
            {
                "state_parameter": "Rsin_mid_surface_m1",
                "state_control_argument": "Rsin_mid_surface_m1:1",
                "state_control_family": "Rsin",
                "descent_direction_sign": 1.0,
                "mean_abs_sensitivity": 12.0,
                "sign_consistency_fraction": 1.0,
                "n_cases": 2,
            },
            {
                "state_parameter": "Zcos_mid_surface_m1",
                "state_control_argument": "Zcos_mid_surface_m1:1",
                "state_control_family": "Zcos",
                "descent_direction_sign": 1.0,
                "mean_abs_sensitivity": 3.0,
                "sign_consistency_fraction": 1.0,
                "n_cases": 2,
            },
        ],
    }


def _input_text() -> str:
    return """&INDATA
  RBC(1,1) = 1.0000000000000000E-01,   ZBS(1,0) = -2.0000000000000000E-02
  ZBS(1,1) = 5.0000000000000000E-02
/
"""


def test_state_to_input_mapping_campaign_writes_fail_closed_launch_artifacts(
    tmp_path: Path,
) -> None:
    mod = _load_tool_module()
    ql_path = tmp_path / "ql_seed_screen.json"
    baseline = tmp_path / "input.final"
    ql_path.write_text(json.dumps(_ql_seed_screen()), encoding="utf-8")
    baseline.write_text(_input_text(), encoding="utf-8")
    out_prefix = tmp_path / "state_to_input_mapping"

    rc = mod.main(
        [
            str(ql_path),
            "--baseline-input",
            str(baseline),
            "--out-dir",
            str(tmp_path / "campaign_inputs"),
            "--out-prefix",
            str(out_prefix),
            "--case",
            "qa_state_map",
            "--coefficient",
            "RBC(1,1)",
            "--coefficient",
            "ZBS(1,0)",
            "--relative-delta",
            "0.10",
        ]
    )

    payload = json.loads(out_prefix.with_suffix(".json").read_text(encoding="utf-8"))
    zbs_plus = (
        tmp_path
        / "campaign_inputs"
        / "zbs_1_0"
        / "input.qa_state_map_zbs_1_0_plus_delta"
    ).read_text(encoding="utf-8")

    assert rc == 0
    assert payload["kind"] == "vmec_state_to_input_mapping_campaign"
    assert payload["claim_level"].endswith("not_mapping_evidence")
    assert payload["passed"] is False
    assert payload["ready_for_nonlinear_launch"] is False
    assert payload["rank_feasibility_precheck_passed"] is True
    assert payload["planned_response_matrix_shape"] == [2, 2]
    assert [row["state_parameter"] for row in payload["admitted_state_controls"]] == [
        "Rsin_mid_surface_m1",
        "Zcos_mid_surface_m1",
    ]
    assert [row["coefficient"] for row in payload["input_directions"]] == [
        "RBC(1,1)",
        "ZBS(1,0)",
    ]
    assert "vmec_response_artifact_missing" in payload["blockers"]
    assert "state_to_input_jacobian_not_extracted" in payload["blockers"]
    assert "RBC(1,1) = 1.0000000000000000E-01" in zbs_plus
    assert "ZBS(1,0) = -1.8000000000000002E-02" in zbs_plus
    assert out_prefix.with_suffix(".csv").exists()
    assert out_prefix.with_suffix(".png").exists()
    assert out_prefix.with_suffix(".pdf").exists()


def test_state_to_input_mapping_campaign_validates_controls(tmp_path: Path) -> None:
    mod = _load_tool_module()
    baseline = tmp_path / "input.final"
    baseline.write_text(_input_text(), encoding="utf-8")

    with pytest.raises(ValueError, match="admitted VMEC-state controls"):
        mod.write_state_to_input_mapping_campaign(
            ql_seed_screen={"admitted_controls": []},
            ql_seed_screen_path=None,
            baseline_input=baseline,
            out_dir=tmp_path / "out",
            out_prefix=tmp_path / "campaign",
            case="bad",
            coefficients=(mod._parse_coefficient_spec("RBC(1,1)"),),
        )

    with pytest.raises(ValueError, match="duplicate coefficient"):
        mod.write_state_to_input_mapping_campaign(
            ql_seed_screen=_ql_seed_screen(),
            ql_seed_screen_path=None,
            baseline_input=baseline,
            out_dir=tmp_path / "out",
            out_prefix=tmp_path / "campaign",
            case="bad",
            coefficients=(
                mod._parse_coefficient_spec("RBC(1,1)"),
                mod._parse_coefficient_spec("RBC(1,1)"),
            ),
        )

    with pytest.raises(ValueError, match="finite and positive"):
        mod.write_state_to_input_mapping_campaign(
            ql_seed_screen=_ql_seed_screen(),
            ql_seed_screen_path=None,
            baseline_input=baseline,
            out_dir=tmp_path / "out",
            out_prefix=tmp_path / "campaign",
            case="bad",
            coefficients=(mod._parse_coefficient_spec("RBC(1,1)"),),
            relative_delta=float("nan"),
        )
