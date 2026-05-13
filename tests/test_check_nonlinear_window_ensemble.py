from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

from spectraxgk.quasilinear_window import (
    NonlinearWindowConvergenceConfig,
    nonlinear_window_convergence_report,
)


def _load_tool_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "check_nonlinear_window_ensemble.py"
    spec = importlib.util.spec_from_file_location("check_nonlinear_window_ensemble", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _window_report(offset: float, *, case: str) -> dict[str, object]:
    t = np.linspace(0.0, 200.0, 201)
    heat = 4.0 + offset + 0.04 * np.sin(2.0 * np.pi * t / 10.0)
    return nonlinear_window_convergence_report(
        t,
        heat,
        case=case,
        source_artifact=f"{case}.csv",
        config=NonlinearWindowConvergenceConfig(
            transient_fraction=0.5,
            min_samples=64,
            min_blocks=4,
            max_running_mean_rel_drift=0.02,
            max_sem_rel=0.02,
        ),
    )


def test_nonlinear_window_ensemble_tool_writes_json_and_png(tmp_path: Path) -> None:
    mod = _load_tool_module()
    reports = []
    for idx, offset in enumerate((-0.02, 0.0, 0.02)):
        path = tmp_path / f"seed_{idx}.json"
        path.write_text(json.dumps(_window_report(offset, case=f"seed_{idx}")), encoding="utf-8")
        reports.append(path)

    out_json = tmp_path / "ensemble.json"
    out_png = tmp_path / "ensemble.png"
    rc = mod.main(
        [
            *[str(path) for path in reports],
            "--out-json",
            str(out_json),
            "--out-png",
            str(out_png),
            "--case",
            "seed_replicates",
            "--comparison",
            "random_seed_replicates",
            "--min-reports",
            "3",
            "--max-mean-rel-spread",
            "0.02",
            "--max-combined-sem-rel",
            "0.02",
        ]
    )

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert rc == 0
    assert out_png.exists()
    assert payload["passed"] is True
    assert payload["comparison"] == "random_seed_replicates"
    assert payload["statistics"]["n_reports"] == 3


def test_nonlinear_window_ensemble_tool_fails_closed_on_spread(tmp_path: Path) -> None:
    mod = _load_tool_module()
    paths = []
    for idx, offset in enumerate((0.0, 2.0)):
        path = tmp_path / f"dt_{idx}.json"
        path.write_text(json.dumps(_window_report(offset, case=f"dt_{idx}")), encoding="utf-8")
        paths.append(path)

    out_json = tmp_path / "ensemble.json"
    rc = mod.main(
        [
            *[str(path) for path in paths],
            "--out-json",
            str(out_json),
            "--max-mean-rel-spread",
            "0.05",
        ]
    )

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    failed = {gate["metric"] for gate in payload["gates"] if not gate["passed"]}
    assert rc == 1
    assert "mean_relative_spread" in failed
