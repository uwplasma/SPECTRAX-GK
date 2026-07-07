from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT
    / "tools"
    / "campaigns"
    / "write_vmec_asymmetric_state_to_input_mapping_campaign.py"
)


def _load_tool_module():
    spec = importlib.util.spec_from_file_location(
        "write_vmec_asymmetric_state_to_input_mapping_campaign", SCRIPT
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
  LASYM = F
  RBC(0,0) = 1.0000000000000000E+00
  RBC(1,1) = 1.0000000000000000E-01,   ZBS(1,1) = 2.0000000000000000E-02
/
"""


def test_asymmetric_campaign_writes_lasym_true_inserted_coefficients(
    tmp_path: Path,
) -> None:
    pytest.importorskip("vmec_jax")
    mod = _load_tool_module()
    ql_path = tmp_path / "ql_seed_screen.json"
    baseline = tmp_path / "input.final"
    ql_path.write_text(json.dumps(_ql_seed_screen()), encoding="utf-8")
    baseline.write_text(_input_text(), encoding="utf-8")
    out_prefix = tmp_path / "asymmetric_state_to_input_mapping"

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
            "qa_asym_state_map",
            "--coefficient",
            "RBS(1,1)",
            "--coefficient",
            "ZBC(1,1)",
            "--delta",
            "0.001",
            "--vmec-extra-args",
            "--outdir . --fast --max-iter 4200 --no-use-input-niter",
        ]
    )

    payload = json.loads(out_prefix.with_suffix(".json").read_text(encoding="utf-8"))
    rbs_baseline = (
        tmp_path
        / "campaign_inputs"
        / "rbs_1_1"
        / "input.qa_asym_state_map_rbs_1_1_baseline"
    ).read_text(encoding="utf-8")
    zbc_plus = (
        tmp_path
        / "campaign_inputs"
        / "zbc_1_1"
        / "input.qa_asym_state_map_zbc_1_1_plus_delta"
    ).read_text(encoding="utf-8")

    assert rc == 0
    assert payload["kind"] == "vmec_asymmetric_state_to_input_mapping_campaign"
    assert payload["claim_level"].endswith("not_mapping_evidence")
    assert payload["passed"] is False
    assert payload["ready_for_nonlinear_launch"] is False
    assert payload["rank_feasibility_precheck_passed"] is True
    assert payload["planned_response_matrix_shape"] == [2, 2]
    assert [row["coefficient"] for row in payload["input_directions"]] == [
        "RBS(1,1)",
        "ZBC(1,1)",
    ]
    assert all(
        row["inserted_missing_coefficient"] for row in payload["input_directions"]
    )
    assert "LASYM = .TRUE." in rbs_baseline
    assert "RBS(1,1) = 0.0000000000000000E+00" in rbs_baseline
    assert "ZBC(1,1) = 1.0000000000000000E-03" in zbc_plus
    assert "--max-iter 4200" in payload["vmec_run_commands"][0]
    assert "vmec_response_artifact_missing" in payload["blockers"]
    assert out_prefix.with_suffix(".csv").exists()
    assert out_prefix.with_suffix(".png").exists()
    assert out_prefix.with_suffix(".pdf").exists()


def test_asymmetric_campaign_validates_candidate_coefficients(tmp_path: Path) -> None:
    mod = _load_tool_module()
    baseline = tmp_path / "input.final"
    baseline.write_text(_input_text(), encoding="utf-8")

    with pytest.raises(ValueError, match="RBS or ZBC"):
        mod.write_asymmetric_state_to_input_mapping_campaign(
            ql_seed_screen=_ql_seed_screen(),
            ql_seed_screen_path=None,
            baseline_input=baseline,
            out_dir=tmp_path / "out",
            out_prefix=tmp_path / "campaign",
            case="bad",
            coefficients=(mod._parse_coefficient_spec("RBC(1,1)"),),
        )

    with pytest.raises(ValueError, match="duplicate coefficient"):
        mod.write_asymmetric_state_to_input_mapping_campaign(
            ql_seed_screen=_ql_seed_screen(),
            ql_seed_screen_path=None,
            baseline_input=baseline,
            out_dir=tmp_path / "out",
            out_prefix=tmp_path / "campaign",
            case="bad",
            coefficients=(
                mod._parse_coefficient_spec("RBS(1,1)"),
                mod._parse_coefficient_spec("RBS(1,1)"),
            ),
        )

    with pytest.raises(ValueError, match="finite and positive"):
        mod.write_asymmetric_state_to_input_mapping_campaign(
            ql_seed_screen=_ql_seed_screen(),
            ql_seed_screen_path=None,
            baseline_input=baseline,
            out_dir=tmp_path / "out",
            out_prefix=tmp_path / "campaign",
            case="bad",
            coefficients=(mod._parse_coefficient_spec("RBS(1,1)"),),
            delta=0.0,
        )
