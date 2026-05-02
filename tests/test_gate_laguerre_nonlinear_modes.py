from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "gate_laguerre_nonlinear_modes.py"
spec = importlib.util.spec_from_file_location("gate_laguerre_nonlinear_modes", SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules["gate_laguerre_nonlinear_modes"] = mod
assert spec.loader is not None
spec.loader.exec_module(mod)


def test_write_laguerre_gate_csv_uses_lf_line_endings(tmp_path: Path) -> None:
    grid = {key: 1.0 for key in mod.DIAGNOSTIC_KEYS}
    spectral = {key: 1.0 for key in mod.DIAGNOSTIC_KEYS}
    grid["run_s"] = 2.0
    spectral["run_s"] = 1.0
    comparison = {f"{key}_rel_diff": 0.0 for key in mod.DIAGNOSTIC_KEYS}
    comparison["max_rel_diff"] = 0.0
    comparison["speedup_grid_over_spectral"] = 2.0

    out = tmp_path / "laguerre_gate.csv"
    mod._write_csv(
        out,
        [
            {
                "case": "cyclone",
                "status": "pass",
                "steps": 2,
                "dt": 0.05,
                "grid": grid,
                "spectral": spectral,
                "comparison": comparison,
            }
        ],
    )

    raw = out.read_bytes()
    assert b"\r" not in raw
    assert raw.count(b"\n") == 2
    assert b"speedup_grid_over_spectral" in raw
