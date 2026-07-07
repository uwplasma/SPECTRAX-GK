"""Tests for QL candidate regularization sensitivity artifacts."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys


def _load_tool_module():
    tools_dir = Path(__file__).resolve().parents[3] / "tools"
    sys.path.insert(0, str(tools_dir))
    path = tools_dir / "plot_quasilinear_candidate_regularization_sweep.py"
    spec = importlib.util.spec_from_file_location(
        "plot_quasilinear_candidate_regularization_sweep", path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_regularization_sweep_locks_tracked_near_miss() -> None:
    mod = _load_tool_module()

    report = mod.score_regularization_sweep(lambdas=(0.1, 0.2, 0.3, 0.5, 0.7, 1.0))

    assert report["kind"] == "quasilinear_candidate_regularization_sweep"
    assert report["claim_level"] == "spectral_envelope_regularization_audit_not_runtime_flux_predictor"
    assert report["case_count"] == 12
    assert report["holdout_count"] == 10
    assert report["best_lambda"] == 0.5
    assert 0.68 < report["best_mean_abs_relative_error"] < 0.70
    assert 0.76 < report["best_holdout_mean_abs_relative_error"] < 0.77
    assert report["best_mean_abs_relative_error"] > report["transport_gate"]
    assert report["promotion_gate"]["passed"] is False
    assert report["promotion_gate"]["accepted_lambdas"] == []
    assert report["promotion_gate"]["blockers"] == [
        "best_regularization_transport_error_above_gate"
    ]
    assert len(report["rows"]) == 6


def test_regularization_sweep_writes_sidecars_and_cli_fails_closed(tmp_path: Path) -> None:
    mod = _load_tool_module()
    report = mod.score_regularization_sweep(lambdas=(0.2, 0.3))
    paths = mod.write_regularization_sweep_figure(
        report,
        out=tmp_path / "regularization.png",
        title="regularization",
        dpi=80,
        write_pdf=False,
    )

    assert Path(paths["png"]).exists()
    assert Path(paths["json"]).exists()
    assert Path(paths["csv"]).exists()
    assert "pdf" not in paths
    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert payload["promotion_gate"]["passed"] is False
    assert Path(paths["csv"]).read_text(encoding="utf-8").startswith("lambda,")

    root = Path(__file__).resolve().parents[3]
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "tools" / "plot_quasilinear_candidate_regularization_sweep.py"),
            "--out",
            str(tmp_path / "cli_regularization.png"),
            "--lambdas",
            "0.2,0.3",
            "--no-pdf",
        ],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 2
    assert "promotion_passed=False" in completed.stdout
    assert (tmp_path / "cli_regularization.json").exists()
