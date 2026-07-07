from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


def _load_tool_module():
    path = Path(__file__).resolve().parents[3] / "tools" / "summarize_nonlinear_replicate_spread.py"
    spec = importlib.util.spec_from_file_location("summarize_nonlinear_replicate_spread", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_summary_artifacts(root: Path, label: str, *, axis: str, mean: float) -> str:
    summary = root / f"state_{label}_transport_window.json"
    summary.write_text(
        json.dumps(
            {
                "kind": "nonlinear_window_summary",
                "variant_label": label,
                "variant_axis": axis,
            }
        ),
        encoding="utf-8",
    )
    reports = root / "nonlinear_window_convergence_reports"
    reports.mkdir(exist_ok=True)
    (reports / f"{summary.stem}.convergence.json").write_text(
        json.dumps(
            {
                "statistics": {
                    "late_mean": mean,
                    "running_mean_rel_drift": 0.03,
                    "terminal_mean_rel_delta": 0.02,
                    "sem_rel": 0.04,
                    "n_blocks": 8,
                }
            }
        ),
        encoding="utf-8",
    )
    return summary.name


def test_summarize_nonlinear_replicate_spread_writes_artifacts(tmp_path: Path) -> None:
    mod = _load_tool_module()
    seed31 = _write_summary_artifacts(tmp_path, "seed31", axis="seed", mean=10.0)
    seed32 = _write_summary_artifacts(tmp_path, "seed32", axis="seed", mean=11.5)
    dt0p04 = _write_summary_artifacts(tmp_path, "dt0p04", axis="timestep", mean=8.5)
    ensemble = {
        "case": "qa_ess_nonlinear_gradient_plus_delta_t900_ensemble",
        "passed": False,
        "statistics": {
            "ensemble_mean": 10.0,
            "mean_rel_spread": 0.30,
            "combined_sem_rel": 0.04,
        },
        "config": {"max_mean_rel_spread": 0.15},
        "rows": [
            {
                "index": 0,
                "late_mean": 10.0,
                "sem": 0.2,
                "summary_artifact": seed31,
                "source_artifact": "state_seed31_heat_flux_trace.csv",
                "passed": True,
                "promotion_ready": True,
            },
            {
                "index": 1,
                "late_mean": 11.5,
                "sem": 0.2,
                "summary_artifact": seed32,
                "source_artifact": "state_seed32_heat_flux_trace.csv",
                "passed": True,
                "promotion_ready": True,
            },
            {
                "index": 2,
                "late_mean": 8.5,
                "sem": 0.2,
                "summary_artifact": dt0p04,
                "source_artifact": "state_dt0p04_heat_flux_trace.csv",
                "passed": True,
                "promotion_ready": True,
            },
        ],
    }
    ensemble_path = tmp_path / "ensemble.json"
    ensemble_path.write_text(json.dumps(ensemble), encoding="utf-8")
    out_prefix = tmp_path / "spread"

    rc = mod.main([str(ensemble_path), "--out-prefix", str(out_prefix)])

    payload = json.loads(out_prefix.with_suffix(".json").read_text(encoding="utf-8"))
    assert rc == 0
    assert out_prefix.with_suffix(".csv").exists()
    assert out_prefix.with_suffix(".png").exists()
    assert payload["state_rows"][0]["classification"] == "mixed_seed_timestep_spread"
    assert payload["replicate_rows"][0]["running_mean_rel_drift"] == 0.03
