from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "run_nonlinear_gradient_direct_campaign.py"


def _load_tool_module():
    spec = importlib.util.spec_from_file_location("run_nonlinear_gradient_direct_campaign", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


def test_collect_direct_tasks_filters_states_and_labels() -> None:
    mod = _load_tool_module()

    tasks = mod.collect_direct_tasks(
        _manifest(),
        states={"baseline"},
        labels={"b_t900_dt0p04"},
    )

    assert len(tasks) == 1
    assert tasks[0].label == "b_t900_dt0p04.out.nc"


def test_direct_campaign_dry_run_assigns_gpus(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
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


def test_load_manifest_rejects_wrong_kind(tmp_path: Path) -> None:
    mod = _load_tool_module()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"kind": "wrong"}), encoding="utf-8")

    with pytest.raises(ValueError, match="nonlinear_turbulence_gradient_campaign_manifest"):
        mod.load_manifest(manifest_path)


def test_config_parser_rejects_commands_without_config() -> None:
    mod = _load_tool_module()

    with pytest.raises(ValueError, match="--config"):
        mod._config_from_command("python3 -m spectraxgk.cli run-runtime-nonlinear --steps 3")
