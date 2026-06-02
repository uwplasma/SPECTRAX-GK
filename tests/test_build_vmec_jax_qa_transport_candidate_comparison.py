from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "build_vmec_jax_qa_transport_candidate_comparison.py"
spec = importlib.util.spec_from_file_location("build_vmec_jax_qa_transport_candidate_comparison", SCRIPT)
assert spec is not None
assert spec.loader is not None
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def _history(root: Path, *, qs: float = 0.02) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "history.json").write_text(
        json.dumps(
            {
                "aspect_initial": 6.1,
                "aspect_final": 6.0,
                "iota_initial": 0.42,
                "iota_final": 0.427,
                "qs_initial": 0.03,
                "qs_final": qs,
                "objective_initial": 2.0,
                "objective_final": 0.5,
                "history": [{"objective": 2.0}, {"objective": 0.5}],
                "nfev": 3,
                "success": True,
                "message": "ok",
                "total_wall_time_s": 1.0,
            }
        ),
        encoding="utf-8",
    )


def _solved_gate(root: Path, *, passed: bool = True, qs: float = 0.02) -> None:
    checks = {
        "aspect": {
            "value": 6.0,
            "target": 6.0,
            "absolute_error": 0.0,
            "absolute_tolerance": 0.05,
            "passed": True,
        },
        "mean_iota": {
            "value": 0.427,
            "minimum_abs": 0.41,
            "margin": 0.017,
            "passed": True,
        },
        "quasisymmetry": {
            "value": qs,
            "maximum": 0.05,
            "margin": 0.05 - qs,
            "source": "vmec_jax_state",
            "passed": qs <= 0.05,
        },
        "iota_profile": {
            "minimum_iotas_excluding_axis": 0.414,
            "minimum_iotaf": 0.412,
            "floor": 0.41,
            "source": "vmec_jax_state",
            "passed": True,
        },
    }
    if not passed:
        checks["quasisymmetry"]["passed"] = False
    (root / "solved_wout_gate.json").write_text(
        json.dumps(
            {
                "kind": "vmec_jax_solved_wout_candidate_gate",
                "passed": passed,
                "checks": checks,
                "next_action": "candidate may proceed" if passed else "do not promote",
            }
        ),
        encoding="utf-8",
    )


def test_payload_admits_only_authoritative_solved_wout_gates(tmp_path: Path, monkeypatch) -> None:
    constraints = tmp_path / "constraints"
    transport = tmp_path / "transport"
    _history(constraints)
    _history(transport)
    _solved_gate(constraints, passed=True)
    monkeypatch.setattr(
        mod,
        "_load_iota_profiles",
        lambda _root: (np.asarray([0.0, 0.414, 0.427]), np.asarray([0.412, 0.421])),
    )

    payload = mod.build_payload(constraints, transport)
    branches = {branch["label"]: branch for branch in payload["branches"]}

    assert branches["QA constraints"]["admitted_for_long_window_nonlinear_audit"] is True
    assert branches["QA constraints"]["gate_source"] == "solved_wout_gate.json"
    assert branches["QA + SPECTRAX-GK transport"]["gate_reported_passed"] is True
    assert branches["QA + SPECTRAX-GK transport"]["admitted_for_long_window_nonlinear_audit"] is False
    assert branches["QA + SPECTRAX-GK transport"]["admission_blockers"] == [
        "non_authoritative_reconstructed_gate"
    ]
    assert payload["summary"]["transport_candidate_admitted"] is False
    assert payload["summary"]["transport_optimization_status"] == "blocked_before_transport_claim"
    assert payload["summary"]["all_branches_passed_solved_wout_gate"] is False


def test_payload_admits_transport_candidate_with_authoritative_gate(tmp_path: Path, monkeypatch) -> None:
    constraints = tmp_path / "constraints"
    transport = tmp_path / "transport"
    _history(constraints)
    _history(transport)
    _solved_gate(constraints, passed=True)
    _solved_gate(transport, passed=True)
    monkeypatch.setattr(
        mod,
        "_load_iota_profiles",
        lambda _root: (np.asarray([0.0, 0.414, 0.427]), np.asarray([0.412, 0.421])),
    )

    payload = mod.build_payload(constraints, transport)

    assert payload["summary"]["all_branches_passed_solved_wout_gate"] is True
    assert payload["summary"]["all_branches_have_authoritative_gate"] is True
    assert payload["summary"]["ready_for_long_window_nonlinear_audit"] == [
        "QA constraints",
        "QA + SPECTRAX-GK transport",
    ]
    assert payload["summary"]["transport_candidate_admitted"] is True


def test_candidate_comparison_plot_handles_normalized_gate_metrics(tmp_path: Path, monkeypatch) -> None:
    constraints = tmp_path / "constraints"
    transport = tmp_path / "transport"
    _history(constraints)
    _history(transport, qs=0.04)
    _solved_gate(constraints, passed=True)
    _solved_gate(transport, passed=False, qs=0.08)
    monkeypatch.setattr(
        mod,
        "_load_iota_profiles",
        lambda _root: (np.asarray([0.0, 0.414, 0.427]), np.asarray([0.412, 0.421])),
    )
    payload = mod.build_payload(constraints, transport)
    out = tmp_path / "panel.png"

    mod.plot_payload(payload, out)

    assert out.exists()
    assert out.stat().st_size > 0
