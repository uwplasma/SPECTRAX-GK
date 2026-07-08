from __future__ import annotations

from support.paths import REPO_ROOT, load_campaign_tool
import json
from pathlib import Path

import pytest


ROOT = REPO_ROOT
SCRIPT = ROOT / "tools" / "campaigns" / "run_nonlinear_gradient_direct_campaign.py"


def _load_tool_module():
    return load_campaign_tool("run_nonlinear_gradient_direct_campaign")


def _manifest() -> dict[str, object]:
    return {
        "kind": "nonlinear_turbulence_gradient_campaign_manifest",
        "state_ensemble_commands": {
            "minus_delta": {
                "direct_full_horizon_launch_commands": [
                    "python3 -m spectraxgk.cli run-runtime-nonlinear --config tools_out/case/minus/m_t900_seed31.toml --steps 2 --no-progress",
                ],
            },
            "baseline": {
                "direct_full_horizon_launch_commands": [
                    "python3 -m spectraxgk.cli run-runtime-nonlinear --config tools_out/case/base/b_t900_seed31.toml --steps 2 --no-progress",
                    "python3 -m spectraxgk.cli run-runtime-nonlinear --config tools_out/case/base/b_t900_dt0p04.toml --steps 3 --no-progress",
                ],
            },
            "plus_delta": {
                "direct_full_horizon_launch_commands": [
                    "python3 -m spectraxgk.cli run-runtime-nonlinear --config tools_out/case/plus/p_t900_seed31.toml --steps 2 --no-progress",
                ],
            },
        },
    }


def _external_vmec_manifest() -> dict[str, object]:
    return {
        "kind": "external_vmec_holdout_config_manifest",
        "configs": [
            {
                "path": "tools_out/external/case_t350_n80.toml",
                "output_path": "tools_out/external/recorded_outputs/custom_t350_n80.out.nc",
            },
            {
                "path": "tools_out/external/case_t350_n64.toml",
                "output_path": "tools_out/external/recorded_outputs/custom_t350_n64.out.nc",
            },
        ],
        "direct_full_horizon_launch_commands": [
            "PYTHONPATH=src CUDA_VISIBLE_DEVICES=${DEVICE:-0} python3 -m spectraxgk.cli run-runtime-nonlinear --config tools_out/external/case_t350_n80.toml --steps 7000 --no-progress",
            "PYTHONPATH=src CUDA_VISIBLE_DEVICES=${DEVICE:-0} python3 -m spectraxgk.cli run-runtime-nonlinear --config tools_out/external/case_t350_n64.toml --steps 7000 --no-progress",
        ],
    }


def test_collect_direct_tasks_orders_states_and_derives_outputs() -> None:
    mod = _load_tool_module()

    tasks = mod.collect_direct_tasks(_manifest())

    assert [(task.state, task.label) for task in tasks] == [
        ("baseline", "b_t900_seed31.out.nc"),
        ("baseline", "b_t900_dt0p04.out.nc"),
        ("plus_delta", "p_t900_seed31.out.nc"),
        ("minus_delta", "m_t900_seed31.out.nc"),
    ]
    assert tasks[0].output == ROOT / "tools_out/case/base/b_t900_seed31.out.nc"


def test_collect_direct_tasks_accepts_external_vmec_manifest_and_env_prefixes() -> None:
    mod = _load_tool_module()

    tasks = mod.collect_direct_tasks(_external_vmec_manifest())

    assert [(task.state, task.label) for task in tasks] == [
        ("external_vmec", "custom_t350_n80.out.nc"),
        ("external_vmec", "custom_t350_n64.out.nc"),
    ]
    assert (
        tasks[0].output
        == ROOT / "tools_out/external/recorded_outputs/custom_t350_n80.out.nc"
    )
    argv, env = mod._split_command_env(tasks[0].command)
    assert argv[:3] == ["python3", "-m", "spectraxgk.cli"]
    assert env["PYTHONPATH"] == "src"
    assert env["CUDA_VISIBLE_DEVICES"] == "${DEVICE:-0}"


def test_collect_direct_tasks_filters_external_vmec_labels() -> None:
    mod = _load_tool_module()

    tasks = mod.collect_direct_tasks(
        _external_vmec_manifest(), labels={"case_t350_n64"}
    )

    assert len(tasks) == 1
    assert tasks[0].state == "external_vmec"
    assert tasks[0].label == "custom_t350_n64.out.nc"

    tasks = mod.collect_direct_tasks(
        _external_vmec_manifest(), labels={"custom_t350_n64"}
    )

    assert len(tasks) == 1
    assert tasks[0].state == "external_vmec"
    assert tasks[0].label == "custom_t350_n64.out.nc"


def test_collect_direct_tasks_filters_states_and_labels() -> None:
    mod = _load_tool_module()

    tasks = mod.collect_direct_tasks(
        _manifest(),
        states={"baseline"},
        labels={"b_t900_dt0p04"},
    )

    assert len(tasks) == 1
    assert tasks[0].label == "b_t900_dt0p04.out.nc"


def test_direct_campaign_dry_run_assigns_gpus(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    mod = _load_tool_module()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")

    rc = mod.main([str(manifest_path), "--gpu", "0", "--gpu", "1", "--dry-run"])

    out = capsys.readouterr().out
    assert rc == 0
    assert "[dry-run gpu=0 state=baseline]" in out
    assert "[dry-run gpu=1 state=baseline]" in out
    assert "[dry-run gpu=0 state=plus_delta]" in out
    assert "[dry-run gpu=1 state=minus_delta]" in out


def test_direct_campaign_dry_run_assigns_external_vmec_gpus(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    mod = _load_tool_module()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_external_vmec_manifest()), encoding="utf-8")

    rc = mod.main([str(manifest_path), "--gpu", "0", "--gpu", "1", "--dry-run"])

    out = capsys.readouterr().out
    assert rc == 0
    assert "[dry-run gpu=0 state=external_vmec]" in out
    assert "[dry-run gpu=1 state=external_vmec]" in out


def test_load_manifest_rejects_wrong_kind(tmp_path: Path) -> None:
    mod = _load_tool_module()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"kind": "wrong"}), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest kind"):
        mod.load_manifest(manifest_path)


def test_config_parser_rejects_commands_without_config() -> None:
    mod = _load_tool_module()

    with pytest.raises(ValueError, match="--config"):
        mod._config_from_command(
            "python3 -m spectraxgk.cli run-runtime-nonlinear --steps 3"
        )


def test_status_writer_records_initial_running_campaign(tmp_path: Path) -> None:
    mod = _load_tool_module()
    status = tmp_path / "status.json"

    mod._write_status(status, [], task_count=4, campaign_status="running")

    payload = json.loads(status.read_text(encoding="utf-8"))
    assert payload["kind"] == "nonlinear_gradient_direct_campaign_status"
    assert payload["status"] == "running"
    assert payload["task_count"] == 4
    assert payload["pending_count"] == 4
    assert payload["finished_count"] == 0
    assert payload["failed_count"] == 0


def test_skip_existing_requires_complete_runtime_bundle(tmp_path: Path) -> None:
    mod = _load_tool_module()
    output = tmp_path / "case.out.nc"
    output.write_text("partial", encoding="utf-8")

    assert mod._output_bundle_complete(output) is False
    assert [path.name for path in mod._output_bundle_paths(output)] == [
        "case.out.nc",
        "case.restart.nc",
        "case.big.nc",
    ]

    (tmp_path / "case.restart.nc").write_text("restart", encoding="utf-8")
    (tmp_path / "case.big.nc").write_text("big", encoding="utf-8")

    assert mod._output_bundle_complete(output) is True


def test_skip_existing_row_records_required_bundle(tmp_path: Path) -> None:
    mod = _load_tool_module()
    output = tmp_path / "case.out.nc"
    for suffix in ("out.nc", "restart.nc", "big.nc"):
        (tmp_path / f"case.{suffix}").write_text(suffix, encoding="utf-8")
    task = mod.DirectTask(
        state="external_vmec",
        label="case.out.nc",
        command="python3 -m spectraxgk.cli run-runtime-nonlinear --config tools_out/case.toml",
        config=ROOT / "tools_out/case.toml",
        output=output,
    )

    row = mod._run_one(
        task,
        gpu="0",
        log_dir=tmp_path / "logs",
        timeout_s=0.1,
        skip_existing=True,
    )

    assert row["status"] == "skipped"
    assert row["required_output_bundle"] == [
        str(output),
        str(tmp_path / "case.restart.nc"),
        str(tmp_path / "case.big.nc"),
    ]
