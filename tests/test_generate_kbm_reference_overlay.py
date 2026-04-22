from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


def _load_tool_module():
    path = Path("/Users/rogeriojorge/local/SPECTRAX-GK/tools/generate_kbm_reference_overlay.py")
    spec = importlib.util.spec_from_file_location("generate_kbm_reference_overlay", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selected_candidate_row_requires_selected_match(tmp_path) -> None:
    mod = _load_tool_module()
    path = tmp_path / "candidates.csv"
    pd.DataFrame(
        {
            "ky": [0.3, 0.3],
            "selected": [False, True],
            "fit_window_tmin": [1.0, 2.0],
            "fit_window_tmax": [3.0, 4.0],
        }
    ).to_csv(path, index=False)

    row = mod._selected_candidate_row(path, 0.3)

    assert float(row["fit_window_tmin"]) == pytest.approx(2.0)
    assert float(row["fit_window_tmax"]) == pytest.approx(4.0)


def test_selected_candidate_row_rejects_missing_ky(tmp_path) -> None:
    mod = _load_tool_module()
    path = tmp_path / "candidates.csv"
    pd.DataFrame(
        {
            "ky": [0.2],
            "selected": [True],
            "fit_window_tmin": [1.0],
            "fit_window_tmax": [3.0],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError):
        mod._selected_candidate_row(path, 0.3)


def test_steps_for_fit_window_respects_stride_alignment() -> None:
    mod = _load_tool_module()

    steps = mod._steps_for_fit_window(fit_tmax=9.69, dt=0.01, fit_padding=0.5, sample_stride=2)

    assert steps == 1020
    assert steps % 2 == 0
