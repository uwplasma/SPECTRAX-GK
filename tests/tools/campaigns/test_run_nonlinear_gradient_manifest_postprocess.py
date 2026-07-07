from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "tools" / "run_nonlinear_gradient_manifest_postprocess.py"


def _load_tool_module():
    spec = importlib.util.spec_from_file_location(
        "run_nonlinear_gradient_manifest_postprocess",
        SCRIPT,
    )
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
                "expected_outputs": ["tmp/minus_seed31.out.nc"],
                "output_gate_command": "python3 tools/check_minus.py",
                "build_ensemble_command": "python3 tools/build_minus.py",
            },
            "baseline": {
                "expected_outputs": ["tmp/base_seed31.out.nc"],
                "output_gate_command": "python3 tools/check_base.py",
                "build_ensemble_command": "python3 tools/build_base.py",
            },
            "plus_delta": {
                "expected_outputs": ["tmp/plus_seed31.out.nc"],
                "output_gate_command": "python3 tools/check_plus.py",
                "build_ensemble_command": "python3 tools/build_plus.py",
            },
        },
        "promotion_contract": {
            "central_fd_command": "python3 tools/build_fd.py",
            "evidence_check_command": "python3 tools/check_evidence.py",
        },
    }


def test_collect_postprocess_commands_uses_dependency_order() -> None:
    mod = _load_tool_module()

    commands = mod.collect_postprocess_commands(_manifest())

    assert [(item.step, item.label) for item in commands] == [
        ("output-gates", "baseline"),
        ("output-gates", "plus_delta"),
        ("output-gates", "minus_delta"),
        ("ensembles", "baseline"),
        ("ensembles", "plus_delta"),
        ("ensembles", "minus_delta"),
        ("central-fd", "promotion"),
        ("evidence", "promotion"),
    ]


def test_collect_postprocess_commands_can_select_steps() -> None:
    mod = _load_tool_module()

    commands = mod.collect_postprocess_commands(_manifest(), steps={"ensembles"})

    assert [item.command for item in commands] == [
        "python3 tools/build_base.py",
        "python3 tools/build_plus.py",
        "python3 tools/build_minus.py",
    ]


def test_postprocess_dry_run_writes_summary(tmp_path: Path) -> None:
    mod = _load_tool_module()
    manifest_path = tmp_path / "manifest.json"
    summary_path = tmp_path / "summary.json"
    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")

    rc = mod.main(
        [
            str(manifest_path),
            "--step",
            "central-fd",
            "--dry-run",
            "--summary-json",
            str(summary_path),
        ]
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert summary["passed"] is True
    assert summary["dry_run"] is True
    assert summary["results"] == [
        {
            "step": "central-fd",
            "label": "promotion",
            "command": "python3 tools/build_fd.py",
            "returncode": 0,
        }
    ]


def test_postprocess_require_outputs_fails_before_running(tmp_path: Path) -> None:
    mod = _load_tool_module()
    manifest_path = tmp_path / "manifest.json"
    summary_path = tmp_path / "summary.json"
    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")

    rc = mod.main(
        [
            str(manifest_path),
            "--dry-run",
            "--require-outputs",
            "--summary-json",
            str(summary_path),
        ]
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert rc == 2
    assert summary["passed"] is False
    assert len(summary["missing_expected_outputs"]) == 3


def test_load_manifest_rejects_wrong_kind(tmp_path: Path) -> None:
    mod = _load_tool_module()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"kind": "other"}), encoding="utf-8")

    with pytest.raises(ValueError, match="nonlinear_turbulence_gradient_campaign_manifest"):
        mod.load_manifest(manifest_path)
