from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "build_nonlinear_transport_horizon_audit.py"
spec = importlib.util.spec_from_file_location("build_nonlinear_transport_horizon_audit", SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def _write_json(root: Path, relative: str, payload: dict[str, object]) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_classify_record_separates_transport_startup_and_reduced() -> None:
    assert (
        mod.classify_record({"gate_passed": True, "effective_tmax": 100.0})
        == "release_transport_gate_passed"
    )
    assert (
        mod.classify_record(
            {
                "claim_level": "startup_transient_nonlinear_plumbing_fd_audit_not_transport_average",
                "transport_average_gate": False,
                "effective_tmax": 0.6,
            }
        )
        == "short_or_startup_not_transport_average"
    )
    assert (
        mod.classify_record(
            {
                "kind": "stellarator_optimization_model",
                "claim_level": "reduced nonlinear estimator optimization",
                "effective_tmax": 90.0,
            }
        )
        == "reduced_estimator_not_transport_average"
    )
    assert (
        mod.classify_record(
            {
                "claim_level": "negative_grid_convergence_result_not_transport_validation",
                "effective_tmax": 150.0,
                "convergence_gate_passed": False,
            }
        )
        == "long_but_failed_convergence"
    )


def test_build_payload_marks_short_fd_audit_outside_transport_scope(tmp_path: Path) -> None:
    _write_json(
        tmp_path,
        "docs/_static/nonlinear_cyclone_gate_summary.json",
        {
            "case": "cyclone_nonlinear_long_window",
            "gate_passed": True,
            "tmax": 400.0,
            "spectrax": "missing.csv",
        },
    )
    _write_json(
        tmp_path,
        "docs/_static/nonlinear_window_fd_audit.json",
        {
            "kind": "nonlinear_startup_window_finite_difference_audit",
            "claim_level": "startup_transient_nonlinear_plumbing_fd_audit_not_transport_average",
            "transport_average_gate": False,
            "passed": True,
            "metrics": {"max_tmax": 0.64},
        },
    )
    _write_json(
        tmp_path,
        "docs/_static/external_vmec_qh_nonlinear_t150_pilot.json",
        {
            "kind": "nonlinear_feasibility_pilot",
            "claim_level": "finite_reduced_grid_long_nonlinear_feasibility_not_grid_converged_transport_validation",
            "label": "QH pilot",
            "tmax": 150.0,
            "promotion_gate": {"passed": False, "reason": "not grid converged"},
        },
    )

    payload = mod.build_payload(tmp_path)
    rows = {row["case"]: row for row in payload["records"]}

    assert rows["cyclone_nonlinear_long_window"]["status"] == "release_transport_gate_passed"
    assert rows["Compact nonlinear FD startup audit"]["status"] == "short_or_startup_not_transport_average"
    assert rows["QH pilot"]["status"] == "long_feasibility_pending_convergence"
    assert payload["summary"]["release_transport_gate_passed"] == 1
    assert payload["summary"]["short_or_reduced_not_transport"] == 1
