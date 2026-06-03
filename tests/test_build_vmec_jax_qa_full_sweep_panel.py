from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "build_vmec_jax_qa_full_sweep_panel.py"
spec = importlib.util.spec_from_file_location("build_vmec_jax_qa_full_sweep_panel", SCRIPT)
assert spec is not None
assert spec.loader is not None
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def _write_case(
    root: Path,
    *,
    objective_final: float,
    transport_metric: float | None = None,
    gate_passed: bool = True,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    history = {
        "objective_initial": 4.0,
        "objective_final": objective_final,
        "aspect_initial": 8.8,
        "aspect_final": 5.1,
        "iota_initial": 0.1,
        "iota_final": 0.39,
        "qs_initial": 0.2,
        "qs_final": 0.07,
        "history": [{"objective": 4.0}, {"objective": 2.0}, {"objective": objective_final}],
        "total_wall_time_s": 12.5,
        "success": True,
    }
    if transport_metric is not None:
        history["transport_metric_final"] = transport_metric
        history["transport_metric_kind"] = "growth"
    (root / "history.json").write_text(json.dumps(history), encoding="utf-8")
    (root / "setup_summary.json").write_text(
        json.dumps({"transport_kind": "growth", "optimizer": {"method": "scalar_trust"}}),
        encoding="utf-8",
    )
    (root / "solved_wout_gate.json").write_text(
        json.dumps(
            {
                "passed": gate_passed,
                "checks": {
                    "aspect": {"passed": True},
                    "mean_iota": {"passed": gate_passed},
                    "quasisymmetry": {"passed": gate_passed},
                },
            }
        ),
        encoding="utf-8",
    )


def test_build_payload_discovers_completed_cases_without_faking_q_traces(tmp_path: Path) -> None:
    run_root = tmp_path / "campaign"
    _write_case(run_root / "runs" / "qa_baseline_scipy", objective_final=1.5)
    _write_case(
        run_root / "runs" / "growth_scalar_trust",
        objective_final=0.8,
        transport_metric=0.25,
        gate_passed=False,
    )
    (run_root / "logs").mkdir()
    (run_root / "logs" / "growth_scalar_trust.status").write_text(
        "[now] START growth_scalar_trust gpu=1\n",
        encoding="utf-8",
    )

    payload = mod.build_payload(run_root)

    assert payload["summary"]["n_cases"] == 2
    assert payload["summary"]["n_cases_with_nonlinear_q_traces"] == 0
    assert payload["summary"]["nonlinear_transport_audit_status"] == "pending_for_this_sweep"
    cases = {case["case_id"]: case for case in payload["cases"]}
    assert cases["growth_scalar_trust"]["history"]["transport_metric_final"] == 0.25
    assert "mean_iota" in cases["growth_scalar_trust"]["gate_blockers"]
    assert cases["growth_scalar_trust"]["q_traces"] == []


def test_q_trace_csv_is_used_only_when_present(tmp_path: Path) -> None:
    case = tmp_path / "campaign" / "runs" / "quasilinear_scalar_trust"
    _write_case(case, objective_final=0.6, transport_metric=0.1)
    (case / "audit_heat_flux_trace.csv").write_text(
        "t,heat_flux\n0.0,1.0\n1.0,2.0\n2.0,3.0\n",
        encoding="utf-8",
    )

    payload = mod.build_payload(tmp_path / "campaign")
    [row] = payload["cases"]

    assert payload["summary"]["n_cases_with_nonlinear_q_traces"] == 1
    assert row["q_traces"][0]["late_window_mean"] == 3.0
    assert row["q_traces"][0]["late_window_tmin"] == 2.0


def test_plot_payload_handles_missing_wouts_and_writes_panel(tmp_path: Path) -> None:
    run_root = tmp_path / "campaign"
    _write_case(run_root / "runs" / "qa_baseline_scipy", objective_final=1.2)
    _write_case(run_root / "runs" / "nonlinear_window_scalar_trust", objective_final=0.4, transport_metric=0.08)
    payload = mod.build_payload(run_root)
    out = tmp_path / "panel.png"

    mod.plot_payload(payload, out)
    cleaned = mod._json_ready(payload)

    assert out.exists()
    assert out.stat().st_size > 0
    assert cleaned["cases"][0]["history"]["transport_metric_final"] is None
    json.dumps(cleaned, allow_nan=False)
