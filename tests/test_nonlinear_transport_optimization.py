from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from spectraxgk.nonlinear_transport_optimization import (
    production_nonlinear_optimization_guard_report,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "check_production_nonlinear_optimization_guard.py"


def _tool_module():
    spec = importlib.util.spec_from_file_location("check_production_nonlinear_optimization_guard", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _optimization_payload(
    *,
    production_claim: bool = False,
    top_level_production_claim: bool = False,
) -> dict[str, object]:
    row: dict[str, object] = {"objective_kind": "nonlinear_heat_flux"}
    if production_claim:
        row["claim_level"] = "production nonlinear turbulent transport optimization"
    payload: dict[str, object] = {
        "parameter_names": ["a"],
        "observable_names": ["nonlinear_heat_flux_mean"],
        "results": [{"objective_kind": "growth"}, row],
    }
    if top_level_production_claim:
        payload["production_nonlinear_optimization_claim"] = True
    return payload


def _startup_payload() -> dict[str, object]:
    return {
        "kind": "nonlinear_startup_window_finite_difference_audit",
        "passed": True,
        "claim_level": "startup_transient_nonlinear_plumbing_fd_audit_not_transport_average",
        "transport_average_gate": False,
        "production_nonlinear_window_gradient_gate": False,
    }


def _ensemble_payload(*, case: str = "holdout", mean: float = 4.0) -> dict[str, object]:
    return {
        "kind": "nonlinear_window_ensemble_report",
        "case": case,
        "claim_level": "replicated_nonlinear_window_uncertainty_gate_not_simulation_claim",
        "passed": True,
        "gate_report": {"passed": True},
        "statistics": {
            "n_reports": 3,
            "ensemble_mean": mean,
            "mean_rel_spread": 0.04,
            "combined_sem_rel": 0.03,
        },
    }


def test_production_nonlinear_guard_is_release_safe_but_blocks_optimization_promotion() -> None:
    report = production_nonlinear_optimization_guard_report(
        optimization_artifact=_optimization_payload(),
        optimization_artifact_path="optimization.json",
        reduced_artifacts={"startup.json": _startup_payload()},
        replicated_ensemble_artifacts={
            "dshape.json": _ensemble_payload(case="dshape"),
            "circular.json": _ensemble_payload(case="circular", mean=3.8),
        },
    )

    assert report["passed"] is True
    assert report["safe_to_release"] is True
    assert report["production_nonlinear_optimization_promoted"] is False
    assert report["promotion_gate"]["blockers"] == [
        "optimized_equilibrium_replicated_transport_window"
    ]
    assert report["summary"]["qualifying_replicated_holdout_ensembles"] == 2


def test_production_nonlinear_guard_rejects_reduced_optimizer_overclaim() -> None:
    for payload in (
        _optimization_payload(production_claim=True),
        _optimization_payload(top_level_production_claim=True),
    ):
        report = production_nonlinear_optimization_guard_report(
            optimization_artifact=payload,
            optimization_artifact_path="optimization.json",
            reduced_artifacts={"startup.json": _startup_payload()},
            replicated_ensemble_artifacts={
                "dshape.json": _ensemble_payload(case="dshape"),
                "circular.json": _ensemble_payload(case="circular", mean=3.8),
            },
        )

        assert report["safe_to_release"] is False
        assert "reduced_optimizer_not_promoted" in report["safety_gate"]["blockers"]


def test_production_nonlinear_guard_promotes_only_optimized_equilibrium_replicates() -> None:
    report = production_nonlinear_optimization_guard_report(
        optimization_artifact=_optimization_payload(),
        optimization_artifact_path="optimization.json",
        reduced_artifacts={"startup.json": _startup_payload()},
        replicated_ensemble_artifacts={
            "dshape.json": _ensemble_payload(case="dshape"),
            "circular.json": _ensemble_payload(case="circular", mean=3.8),
        },
        optimized_equilibrium_artifacts={
            "optimized_equilibrium_final.json": _ensemble_payload(
                case="optimized_equilibrium_final", mean=2.6
            )
        },
    )

    assert report["safe_to_release"] is True
    assert report["production_nonlinear_optimization_promoted"] is True
    assert report["promotion_gate"]["blockers"] == []


def test_production_nonlinear_guard_tool_writes_artifacts(tmp_path: Path) -> None:
    mod = _tool_module()
    optimization = tmp_path / "optimization.json"
    startup = tmp_path / "startup.json"
    dshape = tmp_path / "dshape.json"
    circular = tmp_path / "circular.json"
    optimization.write_text(json.dumps(_optimization_payload()), encoding="utf-8")
    startup.write_text(json.dumps(_startup_payload()), encoding="utf-8")
    dshape.write_text(json.dumps(_ensemble_payload(case="dshape")), encoding="utf-8")
    circular.write_text(json.dumps(_ensemble_payload(case="circular", mean=3.8)), encoding="utf-8")
    out_json = tmp_path / "guard.json"
    out_png = tmp_path / "guard.png"

    rc = mod.main(
        [
            "--optimization-artifact",
            str(optimization),
            "--reduced-artifact",
            str(startup),
            "--replicated-ensemble",
            str(dshape),
            "--replicated-ensemble",
            str(circular),
            "--out-json",
            str(out_json),
            "--out-png",
            str(out_png),
            "--fail-on-unsafe",
        ]
    )

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert rc == 0
    assert out_png.exists()
    assert payload["safe_to_release"] is True
    assert payload["production_nonlinear_optimization_promoted"] is False
