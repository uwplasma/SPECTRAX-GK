from __future__ import annotations

import json
from pathlib import Path

from tools.build_nonlinear_landscape_admission_report import build_report, main

from spectraxgk.validation.stellarator.transport_admission import VMECJAXNonlinearAuditPolicy


def _ensemble(mean: float, sem: float, *, passed: bool = True, n_reports: int = 3) -> dict:
    return {
        "kind": "nonlinear_window_ensemble_report",
        "case": f"ensemble_mean_{mean}",
        "passed": passed,
        "statistics": {
            "ensemble_mean": mean,
            "combined_sem": sem,
            "combined_sem_rel": sem / abs(mean),
            "n_reports": n_reports,
        },
    }


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_build_report_selects_best_uncertainty_resolved_candidate(tmp_path: Path) -> None:
    baseline = _write(tmp_path / "baseline.json", _ensemble(8.55, 0.12))
    p3 = _write(tmp_path / "p3.json", _ensemble(6.27, 0.04))
    p6 = _write(tmp_path / "p6.json", _ensemble(6.43, 0.04))

    report = build_report(
        baseline_ensemble=baseline,
        candidate_ensembles=[("+3%", p3), ("+6%", p6)],
        policy=VMECJAXNonlinearAuditPolicy(minimum_uncertainty_z_score=2.0),
    )

    assert report["passed"] is True
    assert report["selected_candidate"]["label"] == "+3%"
    assert report["artifacts"]["baseline_ensemble"].endswith("baseline.json")
    assert report["artifacts"]["candidate_ensembles"][0]["label"] == "+3%"


def test_cli_writes_report_and_fails_closed_when_requested(tmp_path: Path) -> None:
    baseline = _write(tmp_path / "baseline.json", _ensemble(8.0, 0.5))
    noisy = _write(tmp_path / "noisy.json", _ensemble(7.95, 0.5))
    out_json = tmp_path / "admission.json"

    assert (
        main(
            [
                "--baseline-ensemble",
                str(baseline),
                "--candidate-ensemble",
                "noisy",
                str(noisy),
                "--min-relative-reduction",
                "0.02",
                "--min-uncertainty-z-score",
                "2.0",
                "--out-json",
                str(out_json),
                "--fail-on-no-admission",
            ]
        )
        == 1
    )
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert payload["selected_candidate"] is None
    assert "insufficient_relative_reduction" in payload["candidates"][0]["admission_blockers"]
