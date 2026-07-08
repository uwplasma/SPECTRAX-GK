from __future__ import annotations

from support.paths import REPO_ROOT, load_campaign_tool
import json
from pathlib import Path

import pytest


ROOT = REPO_ROOT
SCRIPT = (
    ROOT / "tools" / "campaigns" / "write_vmec_boundary_profile_perturbation_inputs.py"
)


def _load_tool_module():
    return load_campaign_tool("write_vmec_boundary_profile_perturbation_inputs")


def _input_text() -> str:
    return """&INDATA
  RBC(1,1) = 1.0000000000000000E-01,   ZBS(1,0) = -2.0000000000000000E-02
  ZBS(1,1) = 5.0000000000000000E-02
/
"""


def test_profile_direction_writer_creates_normalized_launch_manifest(
    tmp_path: Path,
) -> None:
    mod = _load_tool_module()
    baseline = tmp_path / "input.final"
    baseline.write_text(_input_text(), encoding="utf-8")

    rc = mod.main(
        [
            "--baseline-input",
            str(baseline),
            "--out-dir",
            str(tmp_path / "profile"),
            "--case",
            "qa_profile_direction",
            "--control",
            "ZBS(1,1):1.0",
            "--control",
            "ZBS(1,0):-0.5",
            "--control",
            "RBC(1,1):0.25",
            "--relative-delta",
            "0.1",
            "--horizons",
            "5,10",
            "--grid",
            "n8:8:4:4:4",
            "--window-tmin",
            "5",
            "--window-tmax",
            "10",
        ]
    )

    manifest_path = (
        tmp_path / "profile" / "vmec_boundary_profile_direction_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    plus_text = (
        tmp_path / "profile" / "input.qa_profile_direction_plus_delta"
    ).read_text(encoding="utf-8")
    minus_text = (
        tmp_path / "profile" / "input.qa_profile_direction_minus_delta"
    ).read_text(encoding="utf-8")

    assert rc == 0
    assert manifest["kind"] == "vmec_boundary_profile_direction_perturbation_manifest"
    assert manifest["claim_level"].endswith("not_simulation_claim")
    assert manifest["parameter_name"] == "profile_direction_zbs_1_1_zbs_1_0_rbc_1_1"
    assert manifest["run_contract"]["analysis_window"] == [5.0, 10.0]
    assert [row["coefficient_slug"] for row in manifest["controls"]] == [
        "zbs_1_1",
        "zbs_1_0",
        "rbc_1_1",
    ]
    assert manifest["controls"][0]["coefficient_delta_per_unit_alpha"] == pytest.approx(
        0.005
    )
    assert manifest["controls"][1]["coefficient_delta_per_unit_alpha"] == pytest.approx(
        -0.001
    )
    assert manifest["controls"][2]["coefficient_delta_per_unit_alpha"] == pytest.approx(
        0.0025
    )
    assert manifest["delta_parameter"] == pytest.approx(
        (0.005**2 + 0.001**2 + 0.0025**2) ** 0.5
    )
    assert mod._coefficient_value(
        plus_text, mod._parse_coefficient_spec("ZBS(1,1)")
    ) == pytest.approx(0.055)
    assert mod._coefficient_value(
        plus_text, mod._parse_coefficient_spec("ZBS(1,0)")
    ) == pytest.approx(-0.021)
    assert mod._coefficient_value(
        plus_text, mod._parse_coefficient_spec("RBC(1,1)")
    ) == pytest.approx(0.1025)
    assert mod._coefficient_value(
        minus_text, mod._parse_coefficient_spec("ZBS(1,1)")
    ) == pytest.approx(0.045)
    assert (
        "write_nonlinear_turbulence_gradient_campaign.py"
        in manifest["campaign_command_after_vmec_runs"]
    )
    assert "--delta-parameter" in manifest["campaign_command_after_vmec_runs"]


def test_profile_direction_writer_rejects_ambiguous_or_unusable_controls(
    tmp_path: Path,
) -> None:
    mod = _load_tool_module()
    baseline = tmp_path / "input.final"
    baseline.write_text(_input_text(), encoding="utf-8")

    with pytest.raises(ValueError, match="at least two controls"):
        mod.write_profile_direction_inputs(
            baseline_input=baseline,
            out_dir=tmp_path / "profile",
            case="bad",
            controls=(mod._parse_weighted_coefficient("ZBS(1,1):1"),),
            relative_delta=0.03,
        )

    with pytest.raises(ValueError, match="duplicate coefficients"):
        mod.write_profile_direction_inputs(
            baseline_input=baseline,
            out_dir=tmp_path / "profile",
            case="bad",
            controls=(
                mod._parse_weighted_coefficient("ZBS(1,1):1"),
                mod._parse_weighted_coefficient("ZBS(1,1):2"),
            ),
            relative_delta=0.03,
        )

    with pytest.raises(ValueError, match="control weight"):
        mod._parse_weighted_coefficient("ZBS(1,1):0")

    with pytest.raises(ValueError, match="finite and positive"):
        mod.write_profile_direction_inputs(
            baseline_input=baseline,
            out_dir=tmp_path / "profile",
            case="bad",
            controls=(
                mod._parse_weighted_coefficient("ZBS(1,1):1"),
                mod._parse_weighted_coefficient("ZBS(1,0):1"),
            ),
            relative_delta=float("nan"),
        )
