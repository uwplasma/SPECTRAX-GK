from __future__ import annotations

import json
from pathlib import Path

from support.paths import REPO_ROOT, load_release_tool


ROOT = REPO_ROOT
mod = load_release_tool("check_nonlinear_transport_matrix_portfolio")


def _matrix_report(
    path: Path,
    *,
    passed: bool,
    total: int = 18,
    completed: int = 18,
    passed_samples: int = 18,
    mean_reduction: float = 0.03,
) -> Path:
    pass_fraction = passed_samples / total if total else 0.0
    payload = {
        "kind": "matched_nonlinear_transport_matrix_report",
        "passed": passed,
        "summary": {
            "total_samples": total,
            "completed_samples": completed,
            "passed_samples": passed_samples,
            "pass_fraction": pass_fraction,
            "mean_relative_reduction": mean_reduction,
            "surfaces": [0.45, 0.64, 0.78],
            "alphas": [0.0, 0.7853981633974483],
            "ky_values": [0.1, 0.3, 0.5],
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _excluded_comparison(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "kind": "matched_nonlinear_transport_comparison",
                "passed": False,
                "statistics": {
                    "relative_reduction": -0.004,
                    "uncertainty_z_score": -0.2,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_portfolio_selects_best_passing_broad_matrix(tmp_path: Path) -> None:
    accepted = _matrix_report(
        tmp_path / "accepted.json", passed=True, mean_reduction=0.025
    )
    projected = _matrix_report(
        tmp_path / "projected.json", passed=True, mean_reduction=0.04
    )
    strict = _excluded_comparison(tmp_path / "strict_growth.json")

    report = mod.build_report(
        matrix_reports={"accepted_qa_ess": accepted, "projected_0p001": projected},
        excluded_comparisons={"strict_growth": strict},
    )

    assert report["passed"] is True
    assert report["selected_family"] == "projected_0p001"
    assert report["selected_report"]["summary"]["mean_relative_reduction"] == 0.04
    assert report["excluded_comparisons"][0]["label"] == "strict_growth"
    assert "excluded" in report["excluded_comparisons"][0]["note"]


def test_portfolio_blocks_missing_or_failed_broad_matrices(tmp_path: Path) -> None:
    failed = _matrix_report(
        tmp_path / "failed.json",
        passed=False,
        passed_samples=12,
        mean_reduction=0.01,
    )

    report = mod.build_report(
        matrix_reports={
            "accepted_qa_ess": failed,
            "projected_0p001": tmp_path / "missing.json",
        },
        excluded_comparisons={},
    )

    assert report["passed"] is False
    assert report["selected_family"] is None
    assert "no candidate family passed" in report["blockers"][0]
    rows = {row["label"]: row for row in report["matrix_reports"]}
    assert rows["accepted_qa_ess"]["qualifies_for_broad_promotion"] is False
    assert rows["projected_0p001"]["exists"] is False


def test_portfolio_cli_writes_report_and_figure(tmp_path: Path) -> None:
    accepted = _matrix_report(
        tmp_path / "accepted.json", passed=True, mean_reduction=0.025
    )
    out_json = tmp_path / "portfolio.json"
    out_png = tmp_path / "portfolio.png"

    rc = mod.main(
        [
            "--matrix-report",
            f"accepted_qa_ess={accepted}",
            "--out-json",
            str(out_json),
            "--out-figure",
            str(out_png),
            "--fail-on-blocked",
        ]
    )
    payload = json.loads(out_json.read_text(encoding="utf-8"))

    assert rc == 0
    assert payload["passed"] is True
    assert payload["selected_family"] == "accepted_qa_ess"
    assert out_png.exists()
