from __future__ import annotations

from support.paths import REPO_ROOT, load_campaign_tool
import json
from pathlib import Path

import pytest


ROOT = REPO_ROOT
SCRIPT = (
    ROOT / "tools" / "campaigns" / "run_overdetermined_nonlinear_gradient_campaign.py"
)


def _load_tool_module():
    return load_campaign_tool("run_overdetermined_nonlinear_gradient_campaign")


def _nested_manifest(tmp_path: Path, slug: str) -> Path:
    path = tmp_path / slug / "gradient_campaign_manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "kind": "nonlinear_turbulence_gradient_campaign_manifest",
                "state_ensemble_commands": {
                    "baseline": {
                        "direct_full_horizon_launch_commands": [
                            f"python3 -m spectraxgk.cli run-runtime-nonlinear --config tools_out/{slug}/baseline_seed31.toml --steps 2 --no-progress",
                        ],
                    },
                    "plus_delta": {
                        "direct_full_horizon_launch_commands": [
                            f"python3 -m spectraxgk.cli run-runtime-nonlinear --config tools_out/{slug}/plus_seed31.toml --steps 2 --no-progress",
                        ],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _overdetermined_manifest(tmp_path: Path) -> Path:
    zbs10 = _nested_manifest(tmp_path, "zbs_1_0")
    zbs11 = _nested_manifest(tmp_path, "zbs_1_1")
    path = tmp_path / "overdetermined.json"
    path.write_text(
        json.dumps(
            {
                "kind": "overdetermined_nonlinear_turbulence_gradient_campaign_manifest",
                "controls": [
                    {
                        "coefficient_slug": "zbs_1_0",
                        "expected_nonlinear_campaign_manifest": str(zbs10),
                    },
                    {
                        "coefficient_slug": "zbs_1_1",
                        "expected_nonlinear_campaign_manifest": str(zbs11),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_collect_overdetermined_tasks_preserves_control_and_state_order(
    tmp_path: Path,
) -> None:
    mod = _load_tool_module()
    manifest = mod.load_overdetermined_manifest(_overdetermined_manifest(tmp_path))

    tasks = mod.collect_overdetermined_direct_tasks(manifest)

    assert [(task.state, task.label) for task in tasks] == [
        ("zbs_1_0:baseline", "baseline_seed31.out.nc"),
        ("zbs_1_0:plus_delta", "plus_seed31.out.nc"),
        ("zbs_1_1:baseline", "baseline_seed31.out.nc"),
        ("zbs_1_1:plus_delta", "plus_seed31.out.nc"),
    ]


def test_overdetermined_runner_filters_and_dry_runs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    mod = _load_tool_module()
    manifest_path = _overdetermined_manifest(tmp_path)

    rc = mod.main(
        [
            str(manifest_path),
            "--control",
            "zbs_1_1",
            "--gpu",
            "0",
            "--gpu",
            "1",
            "--dry-run",
        ]
    )

    out = capsys.readouterr().out
    assert rc == 0
    assert "state=zbs_1_1:baseline" in out
    assert "state=zbs_1_0:baseline" not in out
    assert "gpu=0" in out
    assert "gpu=1" in out


def test_overdetermined_runner_rejects_wrong_kind(tmp_path: Path) -> None:
    mod = _load_tool_module()
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"kind": "wrong"}), encoding="utf-8")

    with pytest.raises(ValueError, match="overdetermined"):
        mod.load_overdetermined_manifest(path)
