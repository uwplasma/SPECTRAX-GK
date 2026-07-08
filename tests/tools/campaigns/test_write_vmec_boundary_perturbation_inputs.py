from __future__ import annotations

from support.paths import REPO_ROOT, load_campaign_tool
import json
from pathlib import Path

import pytest


ROOT = REPO_ROOT
SCRIPT = ROOT / "tools" / "campaigns" / "write_vmec_boundary_perturbation_inputs.py"


def _load_tool_module():
    return load_campaign_tool("write_vmec_boundary_perturbation_inputs")


def _input_text() -> str:
    return """&INDATA
  RBC(0,0) = 1.0000000000000000E+00
  RBC(1,1) = 1.0000000000000000E-01,   ZBS(1,0) = -2.0000000000000000E-02
/
"""


def test_writer_creates_baseline_plus_minus_inputs_and_manifest(tmp_path: Path) -> None:
    mod = _load_tool_module()
    baseline = tmp_path / "input.final"
    baseline.write_text(_input_text(), encoding="utf-8")
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
    assert "RBC(1,1) = 1.0200000000000001E-01" in (
        out_dir / "input.qa_ess_rbc11_plus_delta"
    ).read_text(encoding="utf-8")
    assert "RBC(1,1) = 9.8000000000000004E-02" in (
        out_dir / "input.qa_ess_rbc11_minus_delta"
    ).read_text(encoding="utf-8")
    assert "ZBS(1,0) = -2.0000000000000000E-02" in (
        out_dir / "input.qa_ess_rbc11_plus_delta"
    ).read_text(encoding="utf-8")
    assert (
        "write_nonlinear_turbulence_gradient_campaign.py"
        in manifest["campaign_command_after_vmec_runs"]
    )
    assert (
        "vmec_jax input.qa_ess_rbc11_plus_delta"
        in manifest["vmec_run_commands"]["plus_delta"]
    )


def test_writer_patches_second_coefficient_on_combined_vmec_line(
    tmp_path: Path,
) -> None:
    mod = _load_tool_module()
    baseline = tmp_path / "input.final"
    baseline.write_text(_input_text(), encoding="utf-8")
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


def test_writer_ignores_commented_coefficients_and_rejects_same_line_duplicates(
    tmp_path: Path,
) -> None:
    mod = _load_tool_module()
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


def test_writer_rejects_ambiguous_or_invalid_perturbations(tmp_path: Path) -> None:
    mod = _load_tool_module()
    baseline = tmp_path / "input.ambiguous"
    baseline.write_text(
        """&INDATA
  RBC(1,1) = 1.0000000000000000E-01
  RBC(1,1) = 1.1000000000000000E-01
/
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="appears 2 times"):
        mod.write_perturbation_inputs(
            baseline_input=baseline,
            out_dir=tmp_path / "out",
            case="bad",
            coefficient=mod._parse_coefficient_spec("RBC(1,1)"),
            delta=0.01,
        )


def test_writer_rejects_relative_delta_for_zero_coefficient(tmp_path: Path) -> None:
    mod = _load_tool_module()
    baseline = tmp_path / "input.zero"
    baseline.write_text(
        """&INDATA
  RBC(1,1) = 0.0000000000000000E+00
/
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="relative_delta"):
        mod.write_perturbation_inputs(
            baseline_input=baseline,
            out_dir=tmp_path / "out",
            case="bad",
            coefficient=mod._parse_coefficient_spec("RBC(1,1)"),
            relative_delta=0.01,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"delta": float("nan")}, "finite"),
        ({"delta": float("inf")}, "finite"),
        ({"relative_delta": float("nan")}, "finite"),
        ({"relative_delta": float("-inf")}, "finite"),
    ],
)
def test_writer_rejects_nonfinite_delta_values(
    tmp_path: Path,
    kwargs: dict[str, float],
    message: str,
) -> None:
    mod = _load_tool_module()
    baseline = tmp_path / "input.final"
    baseline.write_text(_input_text(), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        mod.write_perturbation_inputs(
            baseline_input=baseline,
            out_dir=tmp_path / "out",
            case="bad",
            coefficient=mod._parse_coefficient_spec("RBC(1,1)"),
            **kwargs,
        )
