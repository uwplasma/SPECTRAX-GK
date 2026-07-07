from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

from spectraxgk.validation.nonlinear_gradient.followup_core import (
    NonlinearGradientQLSeedScreenConfig,
)
from spectraxgk.validation.nonlinear_gradient.followup_ql_seed import (
    nonlinear_gradient_ql_seed_screen_report,
)


ROOT = Path(__file__).resolve().parents[1]


def _ql_artifact(
    *,
    case: str = "qh",
    parameter: str = "Rcos_mid_surface_m1",
    primary_sensitivity: float = 4.0,
    gamma_sensitivity: float = 1.0,
    rel_error: float = 0.001,
    passed: bool = True,
) -> dict[str, object]:
    return {
        "kind": "mode21_vmec_boozer_quasilinear_gradient_gate",
        "passed": passed,
        "case_name": case,
        "parameter_names": [parameter],
        "parameter_indices": {"Rcos": [17, 1]},
        "objective_gates": [
            {
                "objective": "gamma",
                "parameter": parameter,
                "implicit": gamma_sensitivity,
                "finite_difference": gamma_sensitivity * (1.0 + rel_error),
                "rel_error": rel_error,
                "passed": True,
            },
            {
                "objective": "mixing_length_heat_flux_proxy",
                "parameter": parameter,
                "implicit": primary_sensitivity,
                "finite_difference": primary_sensitivity * (1.0 + rel_error),
                "rel_error": rel_error,
                "passed": True,
            },
        ],
    }


def test_ql_seed_screen_fails_closed_when_cross_case_sign_disagrees() -> None:
    report = nonlinear_gradient_ql_seed_screen_report(
        [
            _ql_artifact(case="qh", primary_sensitivity=10.0),
            _ql_artifact(case="li383", primary_sensitivity=-5.0),
        ]
    )

    assert report["passed"] is False
    assert report["summary"]["control_count"] == 1
    control = report["controls"][0]
    assert control["state_parameter"] == "Rcos_mid_surface_m1"
    assert control["sign_consistency_fraction"] == pytest.approx(0.5)
    assert control["blockers"] == ["cross_artifact_sign_not_consistent"]
    assert report["admitted_controls"] == []
    assert "vmec_jax state controls" in report["scope_note"]


def test_ql_seed_screen_promotes_two_distinct_state_controls_when_consistent() -> None:
    report = nonlinear_gradient_ql_seed_screen_report(
        [
            _ql_artifact(case="case_a", parameter="Rcos_mid_surface_m1", primary_sensitivity=3.0),
            _ql_artifact(case="case_b", parameter="Rcos_mid_surface_m1", primary_sensitivity=4.0),
            _ql_artifact(case="case_a", parameter="Zsin_mid_surface_m1", primary_sensitivity=-2.0),
            _ql_artifact(case="case_b", parameter="Zsin_mid_surface_m1", primary_sensitivity=-5.0),
        ]
    )

    assert report["passed"] is True
    assert report["summary"]["admitted_control_count"] == 2
    args = {row["state_control_argument"] for row in report["admitted_controls"]}
    assert args == {"Rcos_mid_surface_m1:-1", "Zsin_mid_surface_m1:1"}


def test_ql_seed_screen_tracks_gate_blockers_and_validation_errors() -> None:
    bad = _ql_artifact(rel_error=0.5, passed=False)
    bad["objective_gates"][0]["passed"] = False  # type: ignore[index]
    bad["objective_gates"][1]["implicit"] = 0.0  # type: ignore[index]

    report = nonlinear_gradient_ql_seed_screen_report(
        [bad],
        config=NonlinearGradientQLSeedScreenConfig(require_artifact_passed=True),
    )
    rows = report["objective_rows"]
    assert rows[0]["blockers"] == [
        "ad_fd_relative_error_too_large",
        "objective_gate_failed",
        "source_artifact_failed",
    ]
    assert rows[1]["blockers"] == [
        "unresolved_objective_sensitivity",
        "ad_fd_relative_error_too_large",
        "source_artifact_failed",
    ]
    assert rows[0]["source_artifact_passed"] is False

    primary_only_report = nonlinear_gradient_ql_seed_screen_report([_ql_artifact(passed=False)])
    assert primary_only_report["controls"][0]["source_rows"][0]["source_artifact_passed"] is False
    assert "source_artifact_failed" not in primary_only_report["objective_rows"][1]["blockers"]

    validation_cases = [
        ("target_objectives", NonlinearGradientQLSeedScreenConfig(target_objectives=())),
        (
            "primary_objective",
            NonlinearGradientQLSeedScreenConfig(
                target_objectives=("gamma",),
                primary_objective="omega",
            ),
        ),
        ("min_distinct_controls", NonlinearGradientQLSeedScreenConfig(min_distinct_controls=0)),
        ("min_cases_per_control", NonlinearGradientQLSeedScreenConfig(min_cases_per_control=0)),
        ("min_sign_consistency", NonlinearGradientQLSeedScreenConfig(min_sign_consistency=0.0)),
        ("min_sign_consistency", NonlinearGradientQLSeedScreenConfig(min_sign_consistency=1.1)),
        ("max_objective_rel_error", NonlinearGradientQLSeedScreenConfig(max_objective_rel_error=-1.0)),
        ("min_abs_sensitivity", NonlinearGradientQLSeedScreenConfig(min_abs_sensitivity=0.0)),
    ]
    for message, config in validation_cases:
        with pytest.raises(ValueError, match=message):
            nonlinear_gradient_ql_seed_screen_report([_ql_artifact()], config=config)

    with pytest.raises(ValueError, match="paths length"):
        nonlinear_gradient_ql_seed_screen_report([_ql_artifact()], paths=[None, None])
    with pytest.raises(ValueError, match="labels length"):
        nonlinear_gradient_ql_seed_screen_report([_ql_artifact()], labels=["one", "two"])


def test_design_nonlinear_gradient_ql_seed_screen_tool_writes_artifacts(tmp_path: Path) -> None:
    path = ROOT / "campaigns" / "design_nonlinear_gradient_ql_seed_screen.py"
    spec = importlib.util.spec_from_file_location("design_nonlinear_gradient_ql_seed_screen", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    artifact_a = tmp_path / "a.json"
    artifact_b = tmp_path / "b.json"
    out_prefix = tmp_path / "screen"
    artifact_a.write_text(json.dumps(_ql_artifact(case="a", primary_sensitivity=3.0)), encoding="utf-8")
    artifact_b.write_text(json.dumps(_ql_artifact(case="b", primary_sensitivity=4.0)), encoding="utf-8")

    assert module.main([str(artifact_a), str(artifact_b), "--out-prefix", str(out_prefix), "--min-distinct-controls", "1"]) == 0
    payload = json.loads(out_prefix.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["kind"] == "nonlinear_turbulence_gradient_ql_seed_screen"
    assert payload["passed"] is True
    assert payload["admitted_controls"][0]["state_control_argument"] == "Rcos_mid_surface_m1:-1"
    assert out_prefix.with_suffix(".csv").exists()
    assert out_prefix.with_suffix(".png").exists()
    assert out_prefix.with_suffix(".pdf").exists()
