from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_tool_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "check_performance_optimization_manifest.py"
    spec = importlib.util.spec_from_file_location("check_performance_optimization_manifest", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest_text(*, tool: str, artifact: str, status: str = "active") -> str:
    return f"""
[metadata]
schema_version = 1

[[lanes]]
name = "lane"
owner = "owner"
status = "{status}"
priority = "high"
platforms = ["cpu"]
cases = ["case"]
profiling_tools = ["{tool}"]
metrics = ["runtime_s"]
artifact_paths = ["{artifact}"]
bottleneck_hypotheses = ["hypothesis"]
optimization_actions = ["action"]
gates = ["gate"]
"""


def test_repository_performance_manifest_is_well_formed() -> None:
    mod = _load_tool_module()
    summary = mod.validate_manifest(mod.load_manifest())

    assert summary["n_lanes"] >= 5
    active = set(summary["high_priority_active"])
    assert "cold_start_compile" in active
    assert "nonlinear_warm_throughput" in active
    rows = {row["name"]: row for row in summary["rows"]}
    assert rows["end_to_end_runtime_memory"]["n_tools"] >= 2
    assert rows["parallel_scaling"]["priority"] == "medium"


def test_performance_manifest_main_writes_summary_json(tmp_path: Path) -> None:
    mod = _load_tool_module()
    out_json = tmp_path / "summary.json"

    assert mod.main(["--out-json", str(out_json)]) == 0

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["n_lanes"] >= 5
    assert "memory_efficiency" in {row["name"] for row in payload["rows"]}


def test_performance_manifest_rejects_missing_tool(tmp_path: Path) -> None:
    mod = _load_tool_module()
    artifact = tmp_path / "docs" / "_static" / "runtime.png"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("artifact\n", encoding="utf-8")
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(_manifest_text(tool="tools/missing.py", artifact="docs/_static/runtime.png"), encoding="utf-8")
    old_root = mod.REPO_ROOT
    try:
        mod.REPO_ROOT = tmp_path
        with pytest.raises(ValueError, match="profiling tool does not exist"):
            mod.validate_manifest(mod.load_manifest(manifest))
    finally:
        mod.REPO_ROOT = old_root


def test_performance_manifest_rejects_invalid_status(tmp_path: Path) -> None:
    mod = _load_tool_module()
    tool = tmp_path / "tools" / "profile.py"
    tool.parent.mkdir(parents=True)
    tool.write_text("# tool\n", encoding="utf-8")
    artifact = tmp_path / "docs" / "_static" / "runtime.png"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("artifact\n", encoding="utf-8")
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        _manifest_text(tool="tools/profile.py", artifact="docs/_static/runtime.png", status="halfway"),
        encoding="utf-8",
    )
    old_root = mod.REPO_ROOT
    try:
        mod.REPO_ROOT = tmp_path
        with pytest.raises(ValueError, match="invalid status"):
            mod.validate_manifest(mod.load_manifest(manifest))
    finally:
        mod.REPO_ROOT = old_root
