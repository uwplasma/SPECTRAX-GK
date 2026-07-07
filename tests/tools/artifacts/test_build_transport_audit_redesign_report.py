from __future__ import annotations

import json
from pathlib import Path

from tools.artifacts.build_transport_audit_redesign_report import main


def _comparison(path: Path, *, relative_reduction: float, passed: bool) -> None:
    path.write_text(
        json.dumps(
            {
                "kind": "matched_nonlinear_transport_comparison",
                "case": "qa_projected_transport_step1e3",
                "passed": passed,
                "baseline": {"passed": True},
                "candidate": {"passed": True},
                "statistics": {
                    "relative_reduction": relative_reduction,
                    "uncertainty_z_score": -0.2 if relative_reduction < 0.0 else 2.0,
                },
            }
        ),
        encoding="utf-8",
    )


def test_cli_writes_fail_closed_redesign_report(tmp_path: Path) -> None:
    comparison = tmp_path / "comparison.json"
    out = tmp_path / "redesign.json"
    _comparison(comparison, relative_reduction=-0.005, passed=False)

    assert (
        main(
            [
                "--matched-comparison",
                str(comparison),
                "--surface",
                "0.64",
                "--alpha",
                "0.0",
                "--ky",
                "0.3",
                "--out-json",
                str(out),
            ]
        )
        == 0
    )

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["requires_objective_redesign"] is True
    assert "insufficient_matched_reduction" in payload["blockers"]
    assert payload["recommended_sample_set"]["sample_count"] == 18


def test_cli_can_fail_on_required_redesign(tmp_path: Path) -> None:
    comparison = tmp_path / "comparison.json"
    out = tmp_path / "redesign.json"
    _comparison(comparison, relative_reduction=-0.005, passed=False)

    assert (
        main(
            [
                "--matched-comparison",
                str(comparison),
                "--out-json",
                str(out),
                "--fail-on-redesign",
            ]
        )
        == 1
    )
