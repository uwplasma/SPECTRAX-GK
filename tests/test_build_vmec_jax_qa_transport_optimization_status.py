from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "build_vmec_jax_qa_transport_optimization_status.py"
spec = importlib.util.spec_from_file_location("build_vmec_jax_qa_transport_optimization_status", SCRIPT)
assert spec is not None
assert spec.loader is not None
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def _candidate(
    root: Path,
    *,
    passed: bool,
    metric: float | None,
    qs: float = 0.02,
    wout_reproducibility_passed: bool | None = None,
    rerun_wout_admission_passed: bool | None = None,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "objective_initial": 2.0,
        "objective_final": 1.0,
        "aspect_final": 6.0,
        "iota_final": 0.427,
        "qs_final": qs,
        "total_wall_time_s": 1.0,
    }
    if metric is not None:
        payload["transport_metric_final"] = metric
        payload["transport_metric_kind"] = "nonlinear_window_heat_flux"
    (root / "history.json").write_text(json.dumps(payload), encoding="utf-8")
    (root / "solved_wout_gate.json").write_text(
        json.dumps(
            {
                "passed": passed,
                "checks": {
                    "aspect": {
                        "passed": True,
                        "absolute_tolerance": 0.05,
                        "absolute_error": 0.0,
                    },
                    "mean_iota": {"passed": True, "margin": 0.017},
                    "iota_profile": {
                        "passed": passed,
                        "minimum_iotas_excluding_axis": 0.414 if passed else 0.40,
                        "floor": 0.41,
                    },
                    "quasisymmetry": {
                        "passed": passed,
                        "margin": 0.05 - qs,
                    },
                },
                "next_action": "candidate may proceed" if passed else "do not promote",
            }
        ),
        encoding="utf-8",
    )
    if wout_reproducibility_passed is not None:
        (root / "wout_reproducibility_gate.json").write_text(
            json.dumps(
                {
                    "passed": wout_reproducibility_passed,
                    "checks": {
                        "mean_iota_reproducibility": {
                            "passed": wout_reproducibility_passed,
                            "absolute_error": 0.0 if wout_reproducibility_passed else 1.5e-3,
                            "absolute_tolerance": 5.0e-4,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
    if rerun_wout_admission_passed is not None:
        (root / "rerun_wout_admission_gate.json").write_text(
            json.dumps(
                {
                    "passed": rerun_wout_admission_passed,
                    "checks": {
                        "aspect": {"passed": rerun_wout_admission_passed},
                        "mean_iota": {"passed": rerun_wout_admission_passed},
                        "iota_profile": {"passed": rerun_wout_admission_passed},
                        "quasisymmetry": {"passed": rerun_wout_admission_passed},
                    },
                }
            ),
            encoding="utf-8",
        )
        if rerun_wout_admission_passed:
            (root / "wout_final_rerun.nc").write_bytes(b"authoritative-rerun-wout")


def _supporting_artifacts(tmp_path: Path) -> dict[str, Path]:
    line_search = tmp_path / "line_search.json"
    line_search.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "objective": "growth",
                        "passed": True,
                        "initial_objective": 2.0,
                        "final_objective": 1.8,
                        "relative_reduction": 0.1,
                        "initial_update_direction": "negative_delta",
                    },
                    {
                        "objective": "quasilinear_flux",
                        "passed": True,
                        "initial_objective": 3.0,
                        "final_objective": 2.4,
                        "relative_reduction": 0.2,
                        "initial_update_direction": "negative_delta",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    ql_rule = tmp_path / "ql_rule.json"
    ql_rule.write_text(
        json.dumps(
            {
                "rules": {
                    "linear_weight": {
                        "label": "linear weight",
                        "holdout_mean_abs_relative_error": 0.8,
                        "holdout_gate_passed": False,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    ql_model = tmp_path / "ql_model.json"
    ql_model.write_text(
        json.dumps(
            {
                "passed": True,
                "required_candidate": "spectral_envelope_ridge",
                "metrics": {"candidate_mean_abs_relative_error": 0.2},
            }
        ),
        encoding="utf-8",
    )
    nonlinear = tmp_path / "nonlinear.json"
    nonlinear.write_text(
        json.dumps(
            {
                "passed": True,
                "claim_level": "matched_baseline_to_optimized_replicated_nonlinear_transport_audit",
                "comparison": {
                    "baseline_mean": 12.0,
                    "optimized_mean": 9.0,
                    "relative_reduction": 0.25,
                    "uncertainty_separation_sigma": 4.0,
                },
            }
        ),
        encoding="utf-8",
    )
    return {
        "line_search_json": line_search,
        "ql_rule_json": ql_rule,
        "ql_model_json": ql_model,
        "nonlinear_audit_json": nonlinear,
    }


def test_build_payload_separates_gate_failures_from_transport_metrics(tmp_path: Path) -> None:
    constraints = tmp_path / "constraints"
    direct = tmp_path / "direct"
    projected_base = tmp_path / "projected_base"
    projected_step = tmp_path / "projected_step"
    _candidate(constraints, passed=True, metric=None)
    _candidate(direct, passed=False, metric=None, qs=1.0)
    _candidate(projected_base, passed=True, metric=0.1)
    _candidate(projected_step, passed=True, metric=0.11)

    payload = mod.build_payload(
        constraints_dir=constraints,
        direct_transport_dir=direct,
        projected_baseline_dir=projected_base,
        projected_step_dir=projected_step,
        **_supporting_artifacts(tmp_path),
    )

    assert payload["summary"]["qa_baseline_gate_passed"] is True
    assert payload["summary"]["direct_scalar_transport_blocked"] is True
    assert payload["summary"]["projected_transport_gate_passed"] is True
    assert payload["summary"]["projected_transport_improved"] is False
    assert payload["summary"]["quasilinear_model_selection_passed"] is True
    assert payload["summary"]["simple_quasilinear_absolute_flux_promoted"] is False
    assert payload["summary"]["long_window_nonlinear_audit_passed"] is True
    assert "direct scalar transport" in payload["summary"]["blocked_candidates"]


def test_status_plot_and_json_ready_handle_missing_transport_metric(tmp_path: Path) -> None:
    constraints = tmp_path / "constraints"
    direct = tmp_path / "direct"
    projected_base = tmp_path / "projected_base"
    projected_step = tmp_path / "projected_step"
    _candidate(constraints, passed=True, metric=None)
    _candidate(direct, passed=False, metric=None, qs=1.0)
    _candidate(projected_base, passed=True, metric=0.1)
    _candidate(projected_step, passed=True, metric=0.09)
    payload = mod.build_payload(
        constraints_dir=constraints,
        direct_transport_dir=direct,
        projected_baseline_dir=projected_base,
        projected_step_dir=projected_step,
        **_supporting_artifacts(tmp_path),
    )

    out = tmp_path / "status.png"
    mod.plot_payload(payload, out)
    cleaned = mod._json_ready(payload)

    assert out.exists()
    assert out.stat().st_size > 0
    assert cleaned["candidates"][0]["transport_metric_final"] is None
    json.dumps(cleaned, allow_nan=False)


def test_failed_wout_reproducibility_gate_blocks_status_admission(tmp_path: Path) -> None:
    constraints = tmp_path / "constraints"
    direct = tmp_path / "direct"
    projected_base = tmp_path / "projected_base"
    projected_step = tmp_path / "projected_step"
    _candidate(
        constraints,
        passed=True,
        metric=None,
        wout_reproducibility_passed=False,
    )
    _candidate(direct, passed=False, metric=None, qs=1.0)
    _candidate(projected_base, passed=True, metric=0.1)
    _candidate(projected_step, passed=True, metric=0.09)

    payload = mod.build_payload(
        constraints_dir=constraints,
        direct_transport_dir=direct,
        projected_baseline_dir=projected_base,
        projected_step_dir=projected_step,
        **_supporting_artifacts(tmp_path),
    )
    candidates = {candidate["label"]: candidate for candidate in payload["candidates"]}
    baseline = candidates["QA max_mode=5 baseline"]

    assert baseline["solved_wout_gate_passed"] is True
    assert baseline["wout_reproducibility_gate_passed"] is False
    assert baseline["passed_solved_wout_gate"] is False
    assert payload["summary"]["qa_baseline_gate_passed"] is False


def test_status_admits_explicit_authoritative_rerun_wout(tmp_path: Path) -> None:
    constraints = tmp_path / "constraints"
    direct = tmp_path / "direct"
    projected_base = tmp_path / "projected_base"
    projected_step = tmp_path / "projected_step"
    _candidate(
        constraints,
        passed=True,
        metric=None,
        wout_reproducibility_passed=False,
        rerun_wout_admission_passed=True,
    )
    _candidate(direct, passed=False, metric=None, qs=1.0)
    _candidate(projected_base, passed=True, metric=0.1)
    _candidate(projected_step, passed=True, metric=0.09)

    payload = mod.build_payload(
        constraints_dir=constraints,
        direct_transport_dir=direct,
        projected_baseline_dir=projected_base,
        projected_step_dir=projected_step,
        **_supporting_artifacts(tmp_path),
    )
    candidates = {candidate["label"]: candidate for candidate in payload["candidates"]}
    baseline = candidates["QA max_mode=5 baseline"]

    assert baseline["wout_reproducibility_gate_passed"] is False
    assert baseline["rerun_wout_admission_gate_passed"] is True
    assert baseline["uses_authoritative_rerun_wout"] is True
    assert baseline["passed_solved_wout_gate"] is True
    assert baseline["authoritative_wout"].endswith("wout_final_rerun.nc")
    assert payload["summary"]["qa_baseline_gate_passed"] is True
