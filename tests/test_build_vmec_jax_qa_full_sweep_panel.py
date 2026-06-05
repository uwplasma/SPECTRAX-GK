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
    completed: bool = True,
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
    campaign = root.parent.parent if root.parent.name in {"runs", "runs_onepoint"} else root.parent
    log_dir = campaign / ("logs_onepoint" if root.parent.name == "runs_onepoint" else "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"[now] START {root.name} gpu=0"]
    if completed:
        lines.append(f"[now] END {root.name} rc=0")
    (log_dir / f"{root.name}.status").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_failed_wout_reproducibility_gate(root: Path) -> None:
    (root / "wout_reproducibility_gate.json").write_text(
        json.dumps(
            {
                "passed": False,
                "checks": {
                    "mean_iota_reproducibility": {
                        "passed": False,
                        "absolute_error": 1.5e-3,
                        "absolute_tolerance": 5.0e-4,
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def _write_passed_rerun_wout_admission_gate(root: Path) -> None:
    (root / "rerun_wout_admission_gate.json").write_text(
        json.dumps(
            {
                "passed": True,
                "checks": {
                    "aspect": {"passed": True},
                    "mean_iota": {"passed": True},
                    "iota_profile": {"passed": True},
                    "quasisymmetry": {"passed": True},
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
    (run_root / "logs").mkdir(exist_ok=True)
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
    assert cases["growth_scalar_trust"]["diagnostic_gate_passed"] is False
    assert "quasisymmetry" in cases["growth_scalar_trust"]["diagnostic_gate_blockers"]
    assert cases["growth_scalar_trust"]["q_traces"] == []


def test_diagnostic_gate_accepts_only_iota_shortfall_above_floor() -> None:
    status, blockers = mod._diagnostic_gate_status(
        gate_passed=False,
        gate_blockers=["mean_iota"],
        iota_final=0.395,
    )

    assert status is True
    assert blockers == []

    status, blockers = mod._diagnostic_gate_status(
        gate_passed=False,
        gate_blockers=["mean_iota"],
        iota_final=0.37,
    )

    assert status is False
    assert blockers == ["mean_iota"]

    status, blockers = mod._diagnostic_gate_status(
        gate_passed=False,
        gate_blockers=["quasisymmetry"],
        iota_final=0.41,
    )

    assert status is False
    assert blockers == ["quasisymmetry"]


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


def test_compact_json_payload_keeps_q_trace_stats_without_dense_arrays(tmp_path: Path) -> None:
    case = tmp_path / "campaign" / "runs" / "qa_baseline_scipy"
    _write_case(case, objective_final=1.2)
    (case / "audit_heat_flux_trace.csv").write_text(
        "t,heat_flux\n0.0,1.0\n1.0,2.0\n2.0,3.0\n",
        encoding="utf-8",
    )

    payload = mod.build_payload(tmp_path / "campaign")
    compact = mod._compact_payload_for_json(payload)
    [trace] = compact["cases"][0]["q_traces"]

    assert trace["late_window_mean"] == 3.0
    assert trace["late_window_tmax"] == 2.0
    assert "t" not in trace
    assert "heat_flux" not in trace


def test_completed_wout_rows_include_reproducible_nonlinear_audit_command(tmp_path: Path) -> None:
    case = tmp_path / "campaign" / "runs" / "nonlinear_window_scalar_trust"
    _write_case(case, objective_final=0.4, transport_metric=0.08)
    (case / "wout_final.nc").write_bytes(b"not-a-real-netcdf-needed-for-command-test")

    payload = mod.build_payload(tmp_path / "campaign")
    [row] = payload["cases"]

    command = row["recommended_nonlinear_audit_command"]
    assert "write_optimized_equilibrium_transport_configs.py" in command
    assert "vmec_qa_full_sweep_nonlinear_window_scalar_trust" in command
    assert "--window-tmin 350 --window-tmax 700" in command


def test_iota_only_diagnostic_rows_are_audit_command_eligible(tmp_path: Path) -> None:
    case = tmp_path / "campaign" / "runs" / "growth_scalar_trust"
    _write_case(case, objective_final=0.5, transport_metric=0.12, gate_passed=False)
    (case / "wout_final.nc").write_bytes(b"diagnostic-wout")

    payload = mod.build_payload(tmp_path / "campaign")
    [row] = payload["cases"]

    assert row["gate_passed"] is False
    assert row["gate_blockers"] == ["mean_iota", "quasisymmetry"]
    assert row["diagnostic_gate_passed"] is False
    assert row["recommended_nonlinear_audit_command"] is None

    gate = json.loads((case / "solved_wout_gate.json").read_text(encoding="utf-8"))
    gate["checks"]["quasisymmetry"]["passed"] = True
    (case / "solved_wout_gate.json").write_text(json.dumps(gate), encoding="utf-8")

    payload = mod.build_payload(tmp_path / "campaign")
    [row] = payload["cases"]

    assert row["gate_passed"] is False
    assert row["gate_blockers"] == ["mean_iota"]
    assert row["diagnostic_gate_passed"] is True
    assert row["recommended_nonlinear_audit_command"] is not None


def test_failed_wout_reproducibility_gate_blocks_nonlinear_audit_promotion(tmp_path: Path) -> None:
    case = tmp_path / "campaign" / "runs" / "nonlinear_window_scalar_trust"
    _write_case(case, objective_final=0.4, transport_metric=0.08)
    _write_failed_wout_reproducibility_gate(case)
    (case / "wout_final.nc").write_bytes(b"not-a-real-netcdf-needed-for-command-test")

    payload = mod.build_payload(tmp_path / "campaign")
    [row] = payload["cases"]

    assert row["solved_wout_gate_passed"] is True
    assert row["wout_reproducibility_gate_passed"] is False
    assert row["gate_passed"] is False
    assert row["recommended_nonlinear_audit_command"] is None
    assert "wout_reproducibility:mean_iota_reproducibility" in row["gate_blockers"]


def test_authoritative_rerun_wout_gate_selects_rerun_wout_for_audit(tmp_path: Path) -> None:
    case = tmp_path / "campaign" / "runs" / "qa_baseline_scipy"
    _write_case(case, objective_final=0.4, transport_metric=0.08)
    _write_failed_wout_reproducibility_gate(case)
    _write_passed_rerun_wout_admission_gate(case)
    (case / "wout_final.nc").write_bytes(b"optimizer-state-wout")
    (case / "wout_final_rerun.nc").write_bytes(b"authoritative-rerun-wout")

    payload = mod.build_payload(tmp_path / "campaign")
    [row] = payload["cases"]

    assert row["gate_passed"] is True
    assert row["uses_authoritative_rerun_wout"] is True
    assert row["authoritative_wout"].endswith("wout_final_rerun.nc")
    assert row["authoritative_wout_source"] == "wout_final_rerun.nc"
    assert row["recommended_nonlinear_audit_command"] is not None
    assert "wout_final_rerun.nc" in row["recommended_nonlinear_audit_command"]
    assert row["gate_blockers"] == []
    assert row["gate_warnings"] == ["optimizer_state_wout_not_reproduced_authoritative_rerun_wout_used"]


def test_in_progress_wout_is_not_promoted_to_completed_or_audit_ready(tmp_path: Path) -> None:
    case = tmp_path / "campaign" / "runs" / "quasilinear_scalar_trust"
    _write_case(case, objective_final=0.9, transport_metric=0.2, completed=False)
    (case / "wout_final.nc").write_bytes(b"partial-output")

    payload = mod.build_payload(tmp_path / "campaign")
    [row] = payload["cases"]

    assert payload["summary"]["n_completed_wouts"] == 0
    assert row["run_completed"] is False
    assert row["recommended_nonlinear_audit_command"] is None


def test_runs_onepoint_root_uses_parent_status_directory(tmp_path: Path) -> None:
    case = tmp_path / "campaign" / "runs_onepoint" / "qa_baseline_scipy"
    _write_case(case, objective_final=1.1)
    (case / "wout_final.nc").write_bytes(b"fake-wout")

    payload = mod.build_payload(tmp_path / "campaign" / "runs_onepoint")
    [row] = payload["cases"]

    assert row["run_completed"] is True
    assert payload["summary"]["completed_case_ids"] == ["qa_baseline_scipy"]


def test_projected_child_without_status_is_complete_when_gate_and_wout_exist(tmp_path: Path) -> None:
    case = tmp_path / "campaign" / "runs_onepoint" / "projected_guarded_ladder" / "transport_weight_0p0005"
    _write_case(case, objective_final=0.9)
    (case / "wout_final.nc").write_bytes(b"fake-projected-wout")
    # Projected ladder children are tracked by the parent ladder status/log, not
    # by one status file per transport weight.
    for status in (tmp_path / "campaign" / "logs").glob("transport_weight_0p0005.status"):
        status.unlink()

    payload = mod.build_payload(tmp_path / "campaign" / "runs_onepoint")
    [row] = payload["cases"]

    assert row["case_id"] == "projected_guarded_ladder/transport_weight_0p0005"
    assert row["run_completed"] is True
    assert row["recommended_nonlinear_audit_command"] is not None


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
