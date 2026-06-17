from __future__ import annotations

import json
from pathlib import Path

from tools.build_reduced_nonlinear_audit_prelaunch_report import (
    build_metric_report,
    build_report,
    main,
)

from spectraxgk.validation.stellarator.transport_admission import VMECJAXReducedPrelaunchPolicy


def _landscape(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "kind": "vmec_boundary_transport_objective_landscape",
                "sample_set": {
                    "surfaces": [0.45, 0.64, 0.78],
                    "alphas": [0.0, 0.7853981633974483],
                    "ky_values": [0.1, 0.3, 0.5],
                },
                "rows": [
                    {
                        "label": "0",
                        "relative_fraction": 0.0,
                        "coefficient_value": 1.0,
                        "reduced_metrics": {"nonlinear_window_heat_flux": 0.06558065223919245},
                    },
                    {
                        "label": "p0p03",
                        "relative_fraction": 0.03,
                        "coefficient_value": 1.03,
                        "reduced_metrics": {"nonlinear_window_heat_flux": 0.06251277500404685},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_build_report_from_landscape_selects_candidate_row(tmp_path: Path) -> None:
    report = build_report(
        landscape_json=_landscape(tmp_path / "landscape.json"),
        baseline_selector="0",
        candidate_selector="p0p03",
        metric_key="nonlinear_window_heat_flux",
        failed_reference_relative_reduction=0.022876,
        policy=VMECJAXReducedPrelaunchPolicy(minimum_relative_reduction=0.04),
    )

    assert report["passed"] is True
    assert report["selected_rows"]["candidate"]["label"] == "p0p03"
    assert report["relative_reduced_reduction"] > 0.046


def test_cli_writes_blocked_report(tmp_path: Path) -> None:
    landscape = _landscape(tmp_path / "landscape.json")
    out_json = tmp_path / "prelaunch.json"

    assert (
        main(
            [
                "--landscape-json",
                str(landscape),
                "--candidate-row",
                "p0p03",
                "--min-relative-reduction",
                "0.10",
                "--out-json",
                str(out_json),
                "--fail-on-blocked",
            ]
        )
        == 1
    )
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert "insufficient_reduced_margin_for_nonlinear_audit" in payload["blockers"]


def test_metric_mode_builds_negative_prelaunch_reference() -> None:
    report = build_metric_report(
        baseline_metric=0.08010670290,
        candidate_metric=0.07827418221,
        sample_set={
            "surfaces": [0.45, 0.64, 0.78],
            "alphas": [0.0, 0.7853981633974483],
            "ky_values": [0.1, 0.3, 0.5],
        },
        metric_key="nonlinear_window_heat_flux",
        failed_reference_relative_reduction=0.022876,
        policy=VMECJAXReducedPrelaunchPolicy(
            minimum_relative_reduction=0.04,
            failed_reference_safety_factor=1.5,
        ),
    )

    assert report["passed"] is False
    assert report["relative_reduced_reduction"] < report["required_relative_reduced_reduction"]
