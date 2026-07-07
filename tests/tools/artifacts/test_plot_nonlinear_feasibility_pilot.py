"""Tests for nonlinear feasibility pilot plotting."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


def _load_tool_module():
    path = (
        Path(__file__).resolve().parents[3]
        / "tools"
        / "artifacts"
        / "plot_nonlinear_feasibility_pilot.py"
    )
    spec = importlib.util.spec_from_file_location(
        "plot_nonlinear_feasibility_pilot", path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_window_summaries_track_late_slope() -> None:
    mod = _load_tool_module()
    t = np.linspace(0.0, 10.0, 11)
    heat = 2.0 + 0.1 * t
    wphi = 1.0 + 0.05 * t

    summaries = mod.window_summaries(t, heat, wphi, start_fractions=(0.5,))

    assert len(summaries) == 1
    assert summaries[0]["start_index"] == 5
    assert summaries[0]["n_samples"] == 6
    assert summaries[0]["heat_flux_slope"] == pytest.approx(0.1)
    assert summaries[0]["heat_flux_last"] == pytest.approx(3.0)


def test_write_pilot_panel_writes_replayable_artifacts(tmp_path: Path) -> None:
    mod = _load_tool_module()
    t = np.linspace(0.0, 20.0, 21)
    trace = {
        "t": t,
        "heat_flux": 1.0 + np.sin(t / 3.0) * 0.1,
        "wphi": 0.5 + np.cos(t / 4.0) * 0.05,
        "wg": 2.0 + 0.01 * t,
    }

    paths = mod.write_pilot_panel(
        trace,
        out=tmp_path / "pilot.png",
        source="synthetic.out.nc",
        title="Synthetic pilot",
        label="synthetic",
        start_fractions=(0.5, 0.75),
    )

    assert Path(paths["png"]).exists()
    assert Path(paths["pdf"]).exists()
    assert Path(paths["csv"]).exists()
    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert payload["kind"] == "nonlinear_feasibility_pilot"
    assert payload["promotion_gate"]["passed"] is False
    assert len(payload["window_summaries"]) == 2


def test_window_summaries_validate_inputs() -> None:
    mod = _load_tool_module()
    with pytest.raises(ValueError, match="same length"):
        mod.window_summaries([0, 1, 2], [1, 2], [1, 2, 3])
    with pytest.raises(ValueError, match="at least three samples"):
        mod.window_summaries([0, 1], [1, 2], [1, 2])
