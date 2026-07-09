from __future__ import annotations

import json
from pathlib import Path

import pytest

from support.paths import load_campaign_tool


def _boundary_input_text() -> str:
    return """&INDATA
  RBC(0,0) = 1.0000000000000000E+00
  RBC(1,1) = 1.0000000000000000E-01,   ZBS(1,0) = -2.0000000000000000E-02
/
"""


def _profile_input_text() -> str:
    return """&INDATA
  RBC(1,1) = 1.0000000000000000E-01,   ZBS(1,0) = -2.0000000000000000E-02
  ZBS(1,1) = 5.0000000000000000E-02
/
"""


def _symmetric_state_input_text() -> str:
    return """&INDATA
  RBC(1,1) = 1.0000000000000000E-01,   ZBS(1,0) = -2.0000000000000000E-02
  ZBS(1,1) = 5.0000000000000000E-02
/
"""


def _asymmetric_input_text() -> str:
    return """&INDATA
  LASYM = F
  RBC(0,0) = 1.0000000000000000E+00
  RBC(1,1) = 1.0000000000000000E-01,   ZBS(1,1) = 2.0000000000000000E-02
/
"""


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


def _state_control_runbook() -> dict[str, object]:
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


def test_boundary_writer_creates_baseline_plus_minus_inputs_and_manifest(
    tmp_path: Path,
) -> None:
    mod = load_campaign_tool("write_vmec_boundary_perturbation_inputs")
    baseline = tmp_path / "input.final"
    baseline.write_text(_boundary_input_text(), encoding="utf-8")
    out_dir = tmp_path / "vmec_delta"

    rc = mod.main(
        [
            "--baseline-input",
            str(baseline),
            "--out-dir",
            str(out_dir),
            "--case",
            "qa_ess_rbc11",
            "--coefficient",
            "RBC(1,1)",
            "--relative-delta",
            "0.02",
        ]
    )

    manifest = json.loads(
        (out_dir / "vmec_boundary_perturbation_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert rc == 0
    assert manifest["coefficient"] == "RBC(1,1)"
    assert manifest["coefficient_slug"] == "rbc_1_1"
    assert manifest["baseline_value"] == pytest.approx(0.1)
    assert manifest["delta_parameter"] == pytest.approx(0.002)
    assert manifest["state_values"]["plus_delta"] == pytest.approx(0.102)
    assert manifest["state_values"]["minus_delta"] == pytest.approx(0.098)
    assert (out_dir / "input.qa_ess_rbc11_baseline").exists()
    plus = (out_dir / "input.qa_ess_rbc11_plus_delta").read_text(encoding="utf-8")
    assert "RBC(1,1) = 1.0200000000000001E-01" in plus
    assert "ZBS(1,0) = -2.0000000000000000E-02" in plus
    assert (
        "write_nonlinear_turbulence_gradient_campaign.py"
        in manifest["campaign_command_after_vmec_runs"]
    )
    assert (
        "vmec_jax input.qa_ess_rbc11_plus_delta"
        in manifest["vmec_run_commands"]["plus_delta"]
    )


def test_boundary_writer_patches_second_coefficient_on_combined_vmec_line(
    tmp_path: Path,
) -> None:
    mod = load_campaign_tool("write_vmec_boundary_perturbation_inputs")
    baseline = tmp_path / "input.final"
    baseline.write_text(_boundary_input_text(), encoding="utf-8")
    out_dir = tmp_path / "vmec_delta"

    manifest = mod.write_perturbation_inputs(
        baseline_input=baseline,
        out_dir=out_dir,
        case="qa_ess_zbs10",
        coefficient=mod._parse_coefficient_spec("ZBS(1,0)"),
        relative_delta=0.10,
    )

    plus_text = (out_dir / "input.qa_ess_zbs10_plus_delta").read_text(encoding="utf-8")
    minus_text = (out_dir / "input.qa_ess_zbs10_minus_delta").read_text(
        encoding="utf-8"
    )
    assert manifest["baseline_value"] == pytest.approx(-0.02)
    assert manifest["delta_parameter"] == pytest.approx(0.002)
    assert "RBC(1,1) = 1.0000000000000000E-01" in plus_text
    assert "ZBS(1,0) = -1.8000000000000002E-02" in plus_text
    assert mod._coefficient_value(
        minus_text, mod._parse_coefficient_spec("ZBS(1,0)")
    ) == pytest.approx(-0.022)


def test_boundary_writer_ignores_comments_and_rejects_duplicate_coefficients(
    tmp_path: Path,
) -> None:
    mod = load_campaign_tool("write_vmec_boundary_perturbation_inputs")
    baseline = tmp_path / "input.comments"
    baseline.write_text(
        """&INDATA
  ! RBC(1,1) = 9.0000000000000000E-01 should not be patched
  RBC(1,1) = 1.0000000000000000E-01,   ZBS(1,0) = -2.0000000000000000D-02
/
""",
        encoding="utf-8",
    )
    out_dir = tmp_path / "vmec_delta"

    mod.write_perturbation_inputs(
        baseline_input=baseline,
        out_dir=out_dir,
        case="qa_ess_zbs10",
        coefficient=mod._parse_coefficient_spec("ZBS(1,0)"),
        relative_delta=0.10,
    )

    plus_text = (out_dir / "input.qa_ess_zbs10_plus_delta").read_text(encoding="utf-8")
    assert "! RBC(1,1) = 9.0000000000000000E-01 should not be patched" in plus_text
    assert "RBC(1,1) = 1.0000000000000000E-01" in plus_text
    assert "ZBS(1,0) = -1.8000000000000002E-02" in plus_text

    duplicate = tmp_path / "input.duplicate"
    duplicate.write_text(
        """&INDATA
  RBC(1,1) = 1.0000000000000000E-01, RBC(1,1) = 1.1000000000000000E-01
/
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="appears 2 times"):
        mod.write_perturbation_inputs(
            baseline_input=duplicate,
            out_dir=tmp_path / "duplicate_out",
            case="bad",
            coefficient=mod._parse_coefficient_spec("RBC(1,1)"),
            relative_delta=0.01,
        )


def test_boundary_writer_rejects_ambiguous_zero_or_nonfinite_perturbations(
    tmp_path: Path,
) -> None:
    mod = load_campaign_tool("write_vmec_boundary_perturbation_inputs")
    ambiguous = tmp_path / "input.ambiguous"
    ambiguous.write_text(
        """&INDATA
  RBC(1,1) = 1.0000000000000000E-01
  RBC(1,1) = 1.1000000000000000E-01
/
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="appears 2 times"):
        mod.write_perturbation_inputs(
            baseline_input=ambiguous,
            out_dir=tmp_path / "out",
            case="bad",
            coefficient=mod._parse_coefficient_spec("RBC(1,1)"),
            delta=0.01,
        )

    zero = tmp_path / "input.zero"
    zero.write_text(
        """&INDATA
  RBC(1,1) = 0.0000000000000000E+00
/
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="relative_delta"):
        mod.write_perturbation_inputs(
            baseline_input=zero,
            out_dir=tmp_path / "out",
            case="bad",
            coefficient=mod._parse_coefficient_spec("RBC(1,1)"),
            relative_delta=0.01,
        )

    baseline = tmp_path / "input.final"
    baseline.write_text(_boundary_input_text(), encoding="utf-8")
    for kwargs in (
        {"delta": float("nan")},
        {"delta": float("inf")},
        {"relative_delta": float("nan")},
        {"relative_delta": float("-inf")},
    ):
        with pytest.raises(ValueError, match="finite"):
            mod.write_perturbation_inputs(
                baseline_input=baseline,
                out_dir=tmp_path / f"out_{len(str(kwargs))}",
                case="bad",
                coefficient=mod._parse_coefficient_spec("RBC(1,1)"),
                **kwargs,
            )


def test_profile_direction_writer_creates_normalized_launch_manifest(
    tmp_path: Path,
) -> None:
    mod = load_campaign_tool("write_vmec_boundary_profile_perturbation_inputs")
    baseline = tmp_path / "input.final"
    baseline.write_text(_profile_input_text(), encoding="utf-8")

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

    manifest_path = tmp_path / "profile" / "vmec_boundary_profile_direction_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    plus_text = (tmp_path / "profile" / "input.qa_profile_direction_plus_delta").read_text(
        encoding="utf-8"
    )
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
    mod = load_campaign_tool("write_vmec_boundary_profile_perturbation_inputs")
    baseline = tmp_path / "input.final"
    baseline.write_text(_profile_input_text(), encoding="utf-8")

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


def test_state_to_input_mapping_campaign_writes_fail_closed_launch_artifacts(
    tmp_path: Path,
) -> None:
    mod = load_campaign_tool("write_vmec_state_to_input_mapping_campaign")
    ql_path = tmp_path / "ql_seed_screen.json"
    baseline = tmp_path / "input.final"
    ql_path.write_text(json.dumps(_ql_seed_screen()), encoding="utf-8")
    baseline.write_text(_symmetric_state_input_text(), encoding="utf-8")
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
    mod = load_campaign_tool("write_vmec_state_to_input_mapping_campaign")
    baseline = tmp_path / "input.final"
    baseline.write_text(_symmetric_state_input_text(), encoding="utf-8")

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


def test_asymmetric_campaign_writes_lasym_true_inserted_coefficients(
    tmp_path: Path,
) -> None:
    pytest.importorskip("vmec_jax")
    mod = load_campaign_tool("write_vmec_asymmetric_state_to_input_mapping_campaign")
    ql_path = tmp_path / "ql_seed_screen.json"
    baseline = tmp_path / "input.final"
    ql_path.write_text(json.dumps(_ql_seed_screen()), encoding="utf-8")
    baseline.write_text(_asymmetric_input_text(), encoding="utf-8")
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
    assert all(row["inserted_missing_coefficient"] for row in payload["input_directions"])
    assert "LASYM = .TRUE." in rbs_baseline
    assert "RBS(1,1) = 0.0000000000000000E+00" in rbs_baseline
    assert "ZBC(1,1) = 1.0000000000000000E-03" in zbc_plus
    assert "--max-iter 4200" in payload["vmec_run_commands"][0]
    assert "vmec_response_artifact_missing" in payload["blockers"]
    assert out_prefix.with_suffix(".csv").exists()
    assert out_prefix.with_suffix(".png").exists()
    assert out_prefix.with_suffix(".pdf").exists()


def test_asymmetric_campaign_validates_candidate_coefficients(tmp_path: Path) -> None:
    mod = load_campaign_tool("write_vmec_asymmetric_state_to_input_mapping_campaign")
    baseline = tmp_path / "input.final"
    baseline.write_text(_asymmetric_input_text(), encoding="utf-8")

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


def test_state_control_short_bracket_writer_inserts_weighted_lasym_inputs(
    tmp_path: Path,
) -> None:
    pytest.importorskip("vmec_jax")
    mod = load_campaign_tool("write_vmec_state_control_short_bracket_launch")
    runbook_path = tmp_path / "runbook.json"
    baseline = tmp_path / "input.final"
    out_prefix = tmp_path / "state_control_launch"
    runbook_path.write_text(json.dumps(_state_control_runbook()), encoding="utf-8")
    baseline.write_text(_asymmetric_input_text(), encoding="utf-8")

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
    assert (
        "write_nonlinear_turbulence_gradient_campaign.py"
        in payload["campaign_commands_after_vmec_runs"][0]
    )
    assert "--output-min-samples 60" in payload["campaign_commands_after_vmec_runs"][0]
    assert "--output-min-window-samples 30" in payload["campaign_commands_after_vmec_runs"][0]
    assert out_prefix.with_suffix(".csv").exists()
    assert out_prefix.with_suffix(".png").exists()
    assert out_prefix.with_suffix(".pdf").exists()


def test_state_control_short_bracket_writer_fails_closed(tmp_path: Path) -> None:
    mod = load_campaign_tool("write_vmec_state_control_short_bracket_launch")
    baseline = tmp_path / "input.final"
    baseline.write_text(_asymmetric_input_text(), encoding="utf-8")

    with pytest.raises(ValueError, match="runbook must pass"):
        mod.write_state_control_short_bracket_launch(
            runbook={"passed": False, "mapped_controls": []},
            runbook_path=None,
            baseline_input=baseline,
            out_dir=tmp_path / "out",
            out_prefix=tmp_path / "launch",
            case="bad",
        )

    bad = _state_control_runbook()
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
