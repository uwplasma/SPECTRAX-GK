from __future__ import annotations

from support.paths import REPO_ROOT, load_campaign_tool
import json
from pathlib import Path

import pytest


ROOT = REPO_ROOT
SCRIPT = (
    ROOT
    / "tools"
    / "campaigns"
    / "postprocess_overdetermined_nonlinear_gradient_campaign.py"
)


def _load_module():
    return load_campaign_tool("postprocess_overdetermined_nonlinear_gradient_campaign")


def _manifest(tmp_path: Path) -> Path:
    path = tmp_path / "campaign_plan.json"
    path.write_text(
        json.dumps(
            {
                "kind": "overdetermined_nonlinear_turbulence_gradient_campaign_manifest",
                "controls": [
                    {
                        "coefficient_slug": "zbs_1_1",
                        "expected_nonlinear_campaign_manifest": (
                            "tools_out/overdetermined/nonlinear_campaigns/zbs_1_1/gradient_campaign_manifest.json"
                        ),
                    },
                    {
                        "coefficient_slug": "rbc_1_1",
                        "expected_nonlinear_campaign_manifest": (
                            "tools_out/overdetermined/nonlinear_campaigns/rbc_1_1/gradient_campaign_manifest.json"
                        ),
                    },
                ],
                "promotion_contract": {
                    "candidate_ranking_command": (
                        "python3 tools/campaigns/rank_nonlinear_turbulence_gradient_candidates.py "
                        "docs/_static/a.json docs/_static/b.json --json-out docs/_static/rank.json"
                    )
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_build_postprocess_commands_sequence(tmp_path: Path) -> None:
    module = _load_module()
    manifest_path = _manifest(tmp_path)
    manifest = module.load_manifest(manifest_path)

    commands = module.build_postprocess_commands(
        manifest,
        manifest_path=manifest_path,
        status_json=tmp_path / "status.json",
    )

    assert [command.step for command in commands] == [
        "per-control-postprocess",
        "per-control-postprocess",
        "candidate-ranking",
        "final-status",
    ]
    assert commands[0].label == "zbs_1_1"
    assert "--require-outputs" in commands[0].command
    assert "postprocess_summary.json" in commands[0].command
    assert "rank_nonlinear_turbulence_gradient_candidates.py" in commands[2].command
    assert "check_overdetermined_nonlinear_gradient_campaign.py" in commands[3].command
    assert "--fail-on-blocked" in commands[3].command


def test_dry_run_writes_summary(tmp_path: Path) -> None:
    module = _load_module()
    manifest_path = _manifest(tmp_path)
    summary = tmp_path / "summary.json"
    status = tmp_path / "status.json"

    assert (
        module.main(
            [
                str(manifest_path),
                "--status-json",
                str(status),
                "--summary-json",
                str(summary),
                "--dry-run",
            ]
        )
        == 0
    )

    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["kind"] == "overdetermined_nonlinear_gradient_postprocess_summary"
    assert payload["dry_run"] is True
    assert payload["passed"] is True
    assert len(payload["commands"]) == 4
    assert [row["returncode"] for row in payload["results"]] == [0, 0, 0, 0]


def test_wrong_manifest_kind_rejected(tmp_path: Path) -> None:
    module = _load_module()
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps({"kind": "wrong", "controls": [], "promotion_contract": {}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expected kind"):
        module.load_manifest(path)
