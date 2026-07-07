from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


def _load_tool_module():
    path = Path(__file__).resolve().parents[3] / "tools" / "build_qi_branch_refinement_gate.py"
    spec = importlib.util.spec_from_file_location("build_qi_branch_refinement_gate", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_qi_branch_refinement_gate_blocks_marginal_branch(tmp_path: Path) -> None:
    mod = _load_tool_module()
    spectrum = {
        "ky": np.asarray([0.05, 0.07, 0.095, 0.119, 0.143]),
        "gamma": np.asarray([-1e-4, -1e-4, 1.7e-3, 1.7e-3, 3.8e-3]),
        "omega": np.asarray([-0.03, -0.03, -0.06, -0.07, -0.09]),
    }

    report = mod.build_qi_branch_refinement_report(
        spectrum,
        source=tmp_path / "qi.csv",
        krylov={"ky": 0.095, "gamma": 1.9e-3, "omega": -0.054},
    )

    assert report["passed"] is False
    assert report["nonlinear_launch_ready"] is False
    assert report["max_gamma"] == 3.8e-3
    assert report["positive_run_length"] == 3
    assert report["subgates"]["finite_rows"]["passed"] is True
    assert report["subgates"]["positive_run"]["passed"] is True
    assert report["subgates"]["krylov_consistency"]["passed"] is True
    assert report["subgates"]["nonlinear_launch_growth"]["passed"] is False


def test_qi_branch_refinement_gate_passes_strong_consistent_branch(tmp_path: Path) -> None:
    mod = _load_tool_module()
    spectrum = {
        "ky": np.asarray([0.05, 0.07, 0.095, 0.119, 0.143]),
        "gamma": np.asarray([-1e-4, 0.010, 0.022, 0.024, 0.021]),
        "omega": np.asarray([-0.03, -0.04, -0.06, -0.07, -0.09]),
    }

    report = mod.build_qi_branch_refinement_report(
        spectrum,
        source=tmp_path / "qi.csv",
        krylov={"ky": 0.095, "gamma": 0.0215, "omega": -0.061},
    )

    assert report["passed"] is True
    assert report["nonlinear_launch_ready"] is True
    assert report["max_gamma"] == 0.024


def test_qi_branch_refinement_tool_writes_fail_closed_artifacts(tmp_path: Path) -> None:
    mod = _load_tool_module()
    spectrum = tmp_path / "spectrum.csv"
    spectrum.write_text(
        "\n".join(
            [
                "ky,gamma,omega",
                "0.05,-0.0001,-0.03",
                "0.07,-0.0001,-0.03",
                "0.095,0.0017,-0.06",
                "0.119,0.0017,-0.07",
                "0.143,0.0038,-0.09",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    krylov = tmp_path / "krylov.json"
    krylov.write_text(json.dumps({"ky": 0.095, "gamma": 0.0019, "omega": -0.054}), encoding="utf-8")
    out = tmp_path / "gate.png"

    assert (
        mod.main(
            [
                "--spectrum",
                str(spectrum),
                "--krylov-summary",
                str(krylov),
                "--out",
                str(out),
                "--no-pdf",
                "--dpi",
                "80",
            ]
        )
        == 2
    )
    payload = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert payload["png"].endswith("gate.png")
    assert out.exists()
