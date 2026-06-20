from __future__ import annotations

import json
from pathlib import Path

from tools.build_nonlinear_campaign_admission_report import build_report, main

from spectraxgk.validation.stellarator.transport_policies import VMECJAXNonlinearCampaignPolicy


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _prelaunch(*, passed: bool = True, cross_sample: bool = True) -> dict:
    return {
        "kind": "vmec_jax_reduced_nonlinear_audit_prelaunch_report",
        "passed": passed,
        "blockers": [] if passed else ["insufficient_reduced_margin_for_nonlinear_audit"],
        "objective_sample_summary": {
            "passed": True,
            "sample_count": 18,
            "blockers": [],
        },
        "reduced_cross_sample_statistics": {
            "available": cross_sample,
            "passed": True if cross_sample else None,
            "rows": [],
        },
    }


def _landscape(*, passed: bool = True, reduction: float = 0.266, z_score: float = 18.0) -> dict:
    selected = {
        "label": "+3% RBC(0,1)",
        "relative_reduction": reduction,
        "uncertainty_z_score": z_score,
        "combined_sem_rel": 0.0067,
        "n_reports": 3,
    }
    return {
        "kind": "nonlinear_landscape_admission_report",
        "passed": passed,
        "selected_candidate": selected if passed else None,
        "next_action": "use selected direction",
    }


def test_campaign_builder_loads_json_and_admits_rbc_like_evidence(tmp_path: Path) -> None:
    prelaunch = _write(tmp_path / "prelaunch.json", _prelaunch())
    landscape = _write(tmp_path / "landscape.json", _landscape())

    report = build_report(
        prelaunch_report=prelaunch,
        landscape_admission=landscape,
        policy=VMECJAXNonlinearCampaignPolicy(),
    )

    assert report["passed"] is True
    assert report["campaign_admitted"] is True
    assert report["artifacts"]["reduced_prelaunch_report"].endswith("prelaunch.json")
    assert report["selected_landscape_candidate"]["label"] == "+3% RBC(0,1)"


def test_campaign_cli_writes_blocked_report_and_returns_nonzero(tmp_path: Path) -> None:
    prelaunch = _write(tmp_path / "prelaunch.json", _prelaunch(cross_sample=False))
    landscape = _write(tmp_path / "landscape.json", _landscape(reduction=0.01, z_score=0.2))
    out = tmp_path / "campaign.json"

    rc = main(
        [
            "--prelaunch-report",
            str(prelaunch),
            "--landscape-admission",
            str(landscape),
            "--out-json",
            str(out),
            "--fail-on-blocked",
        ]
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert rc == 1
    assert payload["campaign_admitted"] is False
    assert "reduced_cross_sample_statistics_missing" in payload["blockers"]
    assert "selected_landscape_reduction_too_small" in payload["blockers"]
