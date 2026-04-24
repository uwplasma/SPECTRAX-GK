from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_tool_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "plot_nonlinear_window_statistics.py"
    spec = importlib.util.spec_from_file_location("plot_nonlinear_window_statistics", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_summary(path: Path, *, case: str, include: bool = True, heat_flux: float = 0.03) -> None:
    payload = {
        "case": case,
        "source": "synthetic GX diagnostics",
        "gate_mean_rel": 0.10,
        "gate_passed": True,
        "summary": [
            {"metric": "Wg", "mean_rel_abs": 0.02, "max_rel_abs": 0.04, "final_rel": 0.01},
            {"metric": "Wphi", "mean_rel_abs": 0.025, "max_rel_abs": 0.05, "final_rel": -0.01},
            {"metric": "HeatFlux", "mean_rel_abs": heat_flux, "max_rel_abs": 0.09, "final_rel": 0.02},
        ],
    }
    if not include:
        payload["gate_index_include"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_window_rows_excludes_exploratory_and_uses_repo_relative_paths(tmp_path: Path) -> None:
    mod = _load_tool_module()
    old_root = mod.ROOT
    mod.ROOT = tmp_path
    try:
        _write_summary(tmp_path / "nonlinear_cyclone_gate_summary.json", case="cyclone_nonlinear_long_window")
        _write_summary(
            tmp_path / "nonlinear_cyclone_short_gate_summary.json",
            case="cyclone_short_nonlinear_window",
            include=False,
        )

        rows = mod.load_window_rows(list(tmp_path.glob("*.json")))
    finally:
        mod.ROOT = old_root

    assert {row["case"] for row in rows} == {"cyclone_nonlinear_long_window"}
    assert {row["metric"] for row in rows} == {"Wg", "Wphi", "HeatFlux"}
    assert {row["artifact"] for row in rows} == {"nonlinear_cyclone_gate_summary.json"}


def test_plot_nonlinear_window_statistics_main_writes_artifacts(tmp_path: Path) -> None:
    mod = _load_tool_module()
    _write_summary(tmp_path / "nonlinear_cyclone_gate_summary.json", case="cyclone_nonlinear_long_window")
    _write_summary(tmp_path / "nonlinear_kbm_gate_summary.json", case="kbm_nonlinear_window", heat_flux=0.04)
    _write_summary(
        tmp_path / "nonlinear_cyclone_short_gate_summary.json",
        case="cyclone_short_nonlinear_window",
        include=False,
        heat_flux=0.5,
    )
    out = tmp_path / "panel.png"

    assert mod.main(["--glob", str(tmp_path / "*.json"), "--out", str(out)]) == 0

    assert out.exists()
    assert out.with_suffix(".pdf").exists()
    assert out.with_suffix(".csv").exists()
    assert out.with_suffix(".json").exists()
    meta = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["n_cases"] == 2
    assert meta["n_rows"] == 6
    assert meta["all_cases_pass_gate"] is True
    assert "cyclone_short_nonlinear_window" not in set(meta["cases"])
