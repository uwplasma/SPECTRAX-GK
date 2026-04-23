from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


def _load_tool_module():
    path = Path("/Users/rogeriojorge/local/SPECTRAX-GK/tools/generate_kbm_branch_gate_summary.py")
    spec = importlib.util.spec_from_file_location("generate_kbm_branch_gate_summary", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selected_candidate_rows_uses_only_selected_branch(tmp_path: Path) -> None:
    mod = _load_tool_module()
    path = tmp_path / "candidates.csv"
    pd.DataFrame(
        {
            "ky": [0.3, 0.1, 0.2],
            "gamma": [5.0, 0.10, 0.11],
            "omega": [5.0, 1.00, 1.02],
            "eig_overlap_prev": [0.1, float("nan"), 0.99],
            "selected": ["false", "true", "yes"],
        }
    ).to_csv(path, index=False)

    rows = mod.selected_candidate_rows(path)

    assert [row["ky"] for row in rows] == [0.1, 0.2]
    assert [row["gamma"] for row in rows] == [0.10, 0.11]
    assert rows[0]["eig_overlap_prev"] is None


def test_generate_kbm_branch_gate_summary_main_writes_strict_json(tmp_path: Path) -> None:
    mod = _load_tool_module()
    candidates = tmp_path / "candidates.csv"
    out = tmp_path / "summary.json"
    pd.DataFrame(
        {
            "ky": [0.1, 0.2, 0.3],
            "gamma": [0.10, 0.11, 0.12],
            "omega": [1.00, 1.02, 1.04],
            "eig_overlap_prev": [float("nan"), 0.99, 0.98],
            "selected": [True, True, True],
        }
    ).to_csv(candidates, index=False)

    assert mod.main(["--candidates", str(candidates), "--out", str(out)]) == 0

    payload = json.loads(out.read_text())
    assert payload["case"] == "kbm_linear_branch_continuity"
    assert payload["selected_count"] == 3
    assert payload["gate_passed"] is True
    assert payload["rows"][0]["eig_overlap_prev"] is None
    assert {gate["metric"] for gate in payload["gate_report"]["gates"]} == {
        "max_rel_gamma_jump",
        "max_rel_omega_jump",
        "successive_overlap_deficit",
    }
