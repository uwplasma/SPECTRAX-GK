from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "tools" / "artifacts" / "build_nonlinear_transport_horizon_audit.py"
spec = importlib.util.spec_from_file_location(
    "build_nonlinear_transport_horizon_audit", SCRIPT
)
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


def test_production_optimization_blockers_keep_transport_gates_as_prerequisites() -> (
    None
):
    transport_record = {
        "gate_passed": True,
        "effective_tmax": 100.0,
    }

    blockers = mod.production_optimization_blockers(transport_record)

    assert "missing grid-convergence gate for optimized nonlinear objective" in blockers
    assert (
        "missing timestep-convergence gate for optimized nonlinear objective"
        in blockers
    )
    assert "missing seed/initial-condition uncertainty gate" in blockers
    assert "missing optimized-equilibrium nonlinear audit" in blockers

    ready_record = {
        **transport_record,
        "grid_convergence_gate_passed": True,
        "timestep_convergence_gate_passed": True,
        "seed_ensemble_gate_passed": True,
        "optimized_equilibrium_audit_passed": True,
    }
    assert mod.production_optimization_blockers(ready_record) == []

    reduced_blockers = mod.production_optimization_blockers(
        {
            "kind": "stellarator_optimization_model",
            "claim_level": "reduced nonlinear estimator optimization",
            "effective_tmax": 90.0,
        }
    )
    assert (
        "reduced estimator output is not an actual nonlinear transport average"
        in reduced_blockers
    )


def test_build_payload_marks_short_fd_audit_outside_transport_scope(
    tmp_path: Path,
) -> None:
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
    _write_json(
        tmp_path,
        "docs/_static/external_vmec_dshape_t250_high_grid_convergence_gate.json",
        {
            "kind": "external_vmec_nonlinear_grid_convergence_gate",
            "case": "D-shaped grid gate",
            "claim_level": "passed_grid_convergence_candidate_for_transport_holdout",
            "gate_report": {"passed": True},
            "runs": [{"tmax": 250.0}, {"tmax": 250.0}],
        },
    )

    payload = mod.build_payload(tmp_path)
    rows = {row["case"]: row for row in payload["records"]}

    assert (
        rows["cyclone_nonlinear_long_window"]["status"]
        == "release_transport_gate_passed"
    )
    assert (
        rows["cyclone_nonlinear_long_window"]["production_nonlinear_optimization_ready"]
        is False
    )
    assert (
        rows["Compact nonlinear FD startup audit"]["status"]
        == "short_or_startup_not_transport_average"
    )
    assert (
        "missing long post-transient nonlinear transport average"
        in rows["Compact nonlinear FD startup audit"][
            "production_nonlinear_optimization_blockers"
        ]
    )
    assert rows["QH pilot"]["status"] == "long_feasibility_pending_convergence"
    assert rows["D-shaped grid gate"]["status"] == "release_transport_gate_passed"
    assert rows["D-shaped grid gate"]["grid_convergence_gate_passed"] is True
    assert (
        "missing grid-convergence gate for optimized nonlinear objective"
        not in rows["D-shaped grid gate"]["production_nonlinear_optimization_blockers"]
    )
    assert (
        "missing timestep-convergence gate for optimized nonlinear objective"
        in rows["D-shaped grid gate"]["production_nonlinear_optimization_blockers"]
    )
    assert (
        "missing seed/initial-condition uncertainty gate"
        in rows["D-shaped grid gate"]["production_nonlinear_optimization_blockers"]
    )
    assert (
        "missing optimized-equilibrium nonlinear audit"
        in rows["D-shaped grid gate"]["production_nonlinear_optimization_blockers"]
    )
    assert payload["summary"]["release_transport_gate_passed"] == 2
    assert payload["summary"]["short_or_reduced_not_transport"] == 1
    assert payload["summary"]["production_nonlinear_optimization_ready"] == 0
