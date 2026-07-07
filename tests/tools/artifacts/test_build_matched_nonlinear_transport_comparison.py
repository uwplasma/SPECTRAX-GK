from __future__ import annotations

import json
from pathlib import Path

from tools.build_matched_nonlinear_transport_comparison import build_comparison, main


def _ensemble(mean: float, sem: float, *, passed: bool = True) -> dict:
    return {
        "kind": "nonlinear_window_ensemble_report",
        "passed": passed,
        "statistics": {
            "ensemble_mean": mean,
            "combined_sem": sem,
            "mean_rel_spread": 0.03,
            "combined_sem_rel": sem / max(abs(mean), 1.0e-30),
        },
    }


def test_build_comparison_reports_relative_transport_reduction(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    report = build_comparison(
        baseline=_ensemble(12.0, 0.3),
        candidate=_ensemble(10.5, 0.4),
        baseline_artifact=baseline_path,
        candidate_artifact=candidate_path,
        case="qa_projected",
        min_relative_reduction=0.05,
    )

    assert report["passed"] is True
    assert report["statistics"]["absolute_reduction"] == 1.5
    assert report["statistics"]["relative_reduction"] == 0.125
    assert report["statistics"]["combined_uncertainty"] == 0.5
    assert all(gate["passed"] for gate in report["gates"])


def test_build_comparison_fails_when_candidate_not_lower_enough(tmp_path: Path) -> None:
    report = build_comparison(
        baseline=_ensemble(12.0, 0.3),
        candidate=_ensemble(11.8, 0.4),
        baseline_artifact=tmp_path / "baseline.json",
        candidate_artifact=tmp_path / "candidate.json",
        case="qa_projected",
        min_relative_reduction=0.05,
    )

    assert report["passed"] is False
    assert report["gates"][-1]["metric"] == "relative_transport_reduction"
    assert report["gates"][-1]["passed"] is False


def test_build_comparison_fails_closed_for_missing_ensemble_mean(tmp_path: Path) -> None:
    report = build_comparison(
        baseline={
            "kind": "nonlinear_window_ensemble_report",
            "passed": False,
            "statistics": {"ensemble_mean": None, "combined_sem": None},
        },
        candidate=_ensemble(10.5, 0.4),
        baseline_artifact=tmp_path / "baseline.json",
        candidate_artifact=tmp_path / "candidate.json",
        case="qa_outside_window",
        min_relative_reduction=0.0,
    )

    assert report["passed"] is False
    assert report["baseline"]["finite_mean"] is False
    assert report["statistics"]["relative_reduction"] is None
    assert "ensemble_mean is not finite" in report["gates"][0]["detail"]


def test_cli_writes_json_and_figure(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    out_json = tmp_path / "comparison.json"
    out_svg = tmp_path / "comparison.svg"
    baseline.write_text(json.dumps(_ensemble(12.0, 0.2)), encoding="utf-8")
    candidate.write_text(json.dumps(_ensemble(10.8, 0.25)), encoding="utf-8")

    assert (
        main(
            [
                "--baseline-ensemble",
                str(baseline),
                "--candidate-ensemble",
                str(candidate),
                "--case",
                "qa_projected",
                "--min-relative-reduction",
                "0.05",
                "--out-json",
                str(out_json),
                "--out-figure",
                str(out_svg),
                "--fail-on-unpromoted",
            ]
        )
        == 0
    )

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["claim_level"] == "matched_replicated_late_window_transport_comparison"
    assert out_svg.exists()


def test_cli_writes_failure_json_and_figure_for_empty_window_ensemble(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    out_json = tmp_path / "comparison.json"
    out_svg = tmp_path / "comparison.svg"
    baseline.write_text(
        json.dumps(
            {
                "kind": "nonlinear_window_ensemble_report",
                "passed": False,
                "statistics": {"ensemble_mean": None, "combined_sem": None},
            }
        ),
        encoding="utf-8",
    )
    candidate.write_text(json.dumps(_ensemble(10.8, 0.25)), encoding="utf-8")

    assert (
        main(
            [
                "--baseline-ensemble",
                str(baseline),
                "--candidate-ensemble",
                str(candidate),
                "--case",
                "qa_outside_window",
                "--min-relative-reduction",
                "0.0",
                "--out-json",
                str(out_json),
                "--out-figure",
                str(out_svg),
            ]
        )
        == 0
    )

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert payload["statistics"]["relative_reduction"] is None
    assert out_svg.exists()
