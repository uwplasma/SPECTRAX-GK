from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from support.paths import REPO_ROOT
import sys

import numpy as np


def _load_tool_module():
    path = (
        REPO_ROOT / "tools" / "release" / "check_nonlinear_window_ensemble_readiness.py"
    )
    spec = importlib.util.spec_from_file_location(
        "check_nonlinear_window_ensemble_readiness", path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_trace(path: Path, offset: float = 0.0) -> None:
    t = np.linspace(0.0, 100.0, 101)
    heat = 5.0 + offset + 0.02 * np.sin(2.0 * np.pi * t / 10.0)
    lines = ["t,heat_flux"]
    lines.extend(f"{time:.12g},{value:.12g}" for time, value in zip(t, heat))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary(
    path: Path,
    trace: Path,
    *,
    case: str = "case_a",
    seed: int | None = None,
    timestep: float | None = None,
) -> Path:
    payload: dict[str, object] = {
        "kind": "nonlinear_window_summary",
        "case": case,
        "spectrax": trace.name,
        "tmin": 50.0,
        "tmax": 100.0,
        "promotion_gate": {"passed": True},
    }
    if seed is not None:
        payload["seed"] = seed
    if timestep is not None:
        payload["dt"] = timestep
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_readiness_tool_writes_convergence_reports_and_missing_manifest(
    tmp_path: Path,
) -> None:
    mod = _load_tool_module()
    trace = tmp_path / "trace.csv"
    _write_trace(trace)
    summary = _write_summary(tmp_path / "summary.json", trace)
    out_json = tmp_path / "manifest.json"
    reports_dir = tmp_path / "reports"

    rc = mod.main(
        [
            str(summary),
            "--out-json",
            str(out_json),
            "--reports-dir",
            str(reports_dir),
        ]
    )

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert rc == 1
    assert payload["passed"] is False
    assert (reports_dir / "summary.convergence.json").exists()
    assert payload["observed_artifacts"][0]["promotion_ready"] is True
    missing_axes = {item["variant_axis"] for item in payload["missing_artifacts"]}
    assert missing_axes == {"seed", "timestep"}
    assert all(item["missing_count"] == 2 for item in payload["missing_artifacts"])


def test_readiness_tool_passes_with_seed_and_timestep_replicates(
    tmp_path: Path,
) -> None:
    mod = _load_tool_module()
    summaries = []
    for idx, (seed, timestep, offset) in enumerate(
        ((11, 0.02, -0.01), (22, 0.01, 0.01))
    ):
        trace = tmp_path / f"trace_{idx}.csv"
        _write_trace(trace, offset=offset)
        summaries.append(
            _write_summary(
                tmp_path / f"summary_{idx}.json",
                trace,
                seed=seed,
                timestep=timestep,
            )
        )
    out_json = tmp_path / "manifest.json"

    rc = mod.main([*[str(path) for path in summaries], "--out-json", str(out_json)])

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert rc == 0
    assert payload["passed"] is True
    assert payload["missing_artifacts"] == []
    assert payload["cases"][0]["variant_axes"]["seed"]["observed_distinct_count"] == 2
    assert (
        payload["cases"][0]["variant_axes"]["timestep"]["observed_distinct_count"] == 2
    )
