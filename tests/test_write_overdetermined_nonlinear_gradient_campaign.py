from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "write_overdetermined_nonlinear_gradient_campaign.py"


def _load_tool_module():
    spec = importlib.util.spec_from_file_location("write_overdetermined_nonlinear_gradient_campaign", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _input_text() -> str:
    return """&INDATA
  RBC(1,1) = 1.0000000000000000E-01
  ZBS(1,0) = -2.0000000000000000E-02
  ZBS(1,1) = 5.0000000000000000E-02
/
"""


def test_overdetermined_writer_creates_multi_control_launch_manifest(tmp_path: Path) -> None:
    mod = _load_tool_module()
    baseline = tmp_path / "input.final"
    baseline.write_text(_input_text(), encoding="utf-8")
    previous = tmp_path / "ranking.json"
    previous.write_text(
        json.dumps(
            {
                "passed": False,
                "recommendation": "use an overdetermined least-squares/profile-gradient campaign",
                "best_candidate": {"label": "zbs_1_1"},
            }
        ),
        encoding="utf-8",
    )

    rc = mod.main(
        [
            "--baseline-input",
            str(baseline),
            "--out-dir",
            str(tmp_path / "campaign"),
            "--case",
            "qa_profile",
            "--coefficient",
            "ZBS(1,0)",
            "--coefficient",
            "ZBS(1,1)",
            "--relative-delta",
            "0.05",
            "--previous-ranking",
            str(previous),
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

    manifest_path = tmp_path / "campaign" / "overdetermined_nonlinear_gradient_campaign_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert manifest["kind"] == "overdetermined_nonlinear_turbulence_gradient_campaign_manifest"
    assert manifest["control_count"] == 2
    assert manifest["previous_ranking"]["passed"] is False
    assert "least-squares/profile-gradient" in manifest["previous_ranking"]["recommendation"]
    assert [row["coefficient_slug"] for row in manifest["controls"]] == ["zbs_1_0", "zbs_1_1"]
    assert all("vmec_jax input." in row["vmec_run_commands"]["plus_delta"] for row in manifest["controls"])
    assert all(
        "write_nonlinear_turbulence_gradient_campaign.py"
        in row["nonlinear_campaign_command_after_vmec_runs"]
        for row in manifest["controls"]
    )
    assert "rank_nonlinear_turbulence_gradient_candidates.py" in manifest["promotion_contract"][
        "candidate_ranking_command"
    ]
    assert len(manifest["promotion_contract"]["expected_fd_artifacts"]) == 2


def test_overdetermined_writer_requires_multiple_distinct_controls(tmp_path: Path) -> None:
    mod = _load_tool_module()
    baseline = tmp_path / "input.final"
    baseline.write_text(_input_text(), encoding="utf-8")

    with pytest.raises(ValueError, match="at least two controls"):
        mod.write_overdetermined_campaign(
            baseline_input=baseline,
            out_dir=tmp_path / "campaign",
            case="bad",
            coefficients=(mod._parse_coefficient_spec("ZBS(1,0)"),),
            relative_delta=0.05,
        )

    with pytest.raises(ValueError, match="duplicate controls"):
        mod.write_overdetermined_campaign(
            baseline_input=baseline,
            out_dir=tmp_path / "campaign",
            case="bad",
            coefficients=(
                mod._parse_coefficient_spec("ZBS(1,0)"),
                mod._parse_coefficient_spec("ZBS(1,0)"),
            ),
            relative_delta=0.05,
        )
