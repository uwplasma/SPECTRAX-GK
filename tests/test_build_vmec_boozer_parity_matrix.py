"""Tests for the VMEC/Boozer parity-matrix artifact builder."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "build_vmec_boozer_parity_matrix.py"


def _load_tool_module():
    spec = importlib.util.spec_from_file_location(
        "build_vmec_boozer_parity_matrix", SCRIPT
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fake_report(**kwargs: object) -> dict[str, object]:
    assert int(kwargs["mboz"]) >= 21
    assert int(kwargs["nboz"]) >= 21
    return {
        "available": True,
        "case_name": kwargs["case_name"],
        "status": "diagnostic_open",
        "mboz": kwargs["mboz"],
        "nboz": kwargs["nboz"],
        "equal_arc_core_worst_normalized_max_abs": 4.0e-3,
        "equal_arc_core_worst_scalar_rel": 2.0e-3,
        "equal_arc_derivative_worst_normalized_max_abs": 2.0e-2,
        "equal_arc_metric_worst_normalized_max_abs": 3.0e-2,
        "equal_arc_drift_worst_normalized_max_abs": 7.0e-2,
        "equal_arc_core_tolerance": 1.0e-2,
        "equal_arc_derivative_tolerance": 3.0e-2,
        "equal_arc_metric_tolerance": 8.0e-2,
        "equal_arc_drift_tolerance": 8.0e-2,
        "equal_arc_core_passed": True,
        "equal_arc_derivative_passed": True,
        "equal_arc_metric_passed": True,
        "equal_arc_drift_passed": True,
        "production_parity_passed": False,
        "worst_core_normalized_max_abs": 2.0e-1,
        "worst_scalar_rel": 1.0e-3,
        "source_model": "vmec_jax:state->tensor-flux-tube vs imported-vmec-eik",
        "surface_index": 4,
        "torflux": 0.5,
        "alpha": 0.0,
    }


def _fake_artifact_resolver(case_name: str) -> tuple[str | None, str | None, str | None]:
    if case_name in {"nfp1_QI", "nfp2_QI", "nfp4_QI_finite_beta"}:
        return f"/tmp/input.{case_name}", None, None
    return f"/tmp/input.{case_name}", "/dev/null", None


def test_build_parity_matrix_uses_mode21_floor_and_summarizes_rows() -> None:
    mod = _load_tool_module()
    cases = (
        mod.ParityCase("nfp4_QH_warm_start", "QH", "stellarator", 16),
        mod.ParityCase("nfp3_QI_fixed_resolution_final", "QI", "stellarator", 8),
    )

    payload = mod.build_parity_matrix(
        cases=cases,
        reporter=_fake_report,
        artifact_resolver=_fake_artifact_resolver,
    )

    assert payload["kind"] == "vmec_boozer_parity_matrix"
    assert payload["minimum_boozer_mode_count"] == 21
    assert payload["summary"]["n_cases"] == 2
    assert payload["summary"]["n_equal_arc_passed"] == 2
    assert payload["summary"]["all_equal_arc_passed"] is True
    assert all(row["mode_floor_passed"] for row in payload["rows"])
    assert payload["rows"][0][
        "equal_arc_drift_worst_normalized_max_abs"
    ] == pytest.approx(7.0e-2)
    assert payload["claim_level"].endswith("not_full_transport_gradient_claim")
    assert payload["rows"][0]["production_parity_passed"] is False
    assert payload["rows"][0]["sample_set_id"] == (
        "nfp4_QH_warm_start:ntheta=16:mboz=21:nboz=21"
    )
    provenance = payload["sample_set_provenance"]
    assert provenance["bounded_run"] is True
    assert provenance["external_vmec_solves_launched"] is False
    assert provenance["summary"]["n_total_sample_sets"] == 7
    assert provenance["summary"]["n_unique_sample_sets"] == 6
    assert provenance["summary"]["all_modes_at_or_above_floor"] is True
    assert (
        provenance["matrix_cases"][0]["sample_set_id"]
        == payload["rows"][0]["sample_set_id"]
    )
    assert provenance["matrix_cases"][0]["field_line_alpha"] == pytest.approx(0.0)
    assert provenance["matrix_cases"][0]["surface_index"] == 4
    assert provenance["matrix_cases"][0]["torflux"] == pytest.approx(0.5)
    qi_summary = payload["qi_seed_robustness"]["summary"]
    assert qi_summary["n_variants"] == 5
    assert qi_summary["n_passed"] == 2
    assert qi_summary["n_rejected"] == 3
    assert qi_summary["seed_robust_gate_passed"] is True
    assert qi_summary["full_declared_seed_campaign_passed"] is False
    assert qi_summary["evaluated_reference_gate_passed"] is True
    assert qi_summary["robustness_status"] == "artifact_limited_passed"
    assert qi_summary["artifact_reason_counts"]["missing_bundled_wout_reference"] == 3


def test_build_parity_matrix_rejects_underresolved_boozer_modes() -> None:
    mod = _load_tool_module()
    cases = (
        mod.ParityCase(
            "nfp3_QI_fixed_resolution_final", "QI", "stellarator", 8, mboz=20, nboz=21
        ),
    )

    with pytest.raises(ValueError, match="mboz and nboz"):
        mod.build_parity_matrix(
            cases=cases,
            reporter=_fake_report,
            artifact_resolver=_fake_artifact_resolver,
        )


def test_qi_seed_robustness_records_failed_mode21_variant() -> None:
    mod = _load_tool_module()

    def failing_report(**kwargs: object) -> dict[str, object]:
        report = _fake_report(**kwargs)
        report["equal_arc_drift_worst_normalized_max_abs"] = 9.0e-2
        report["equal_arc_drift_passed"] = False
        return report

    payload = mod.build_parity_matrix(
        cases=(),
        qi_variants=(
            mod.ParityCase(
                "nfp3_QI_fixed_resolution_final",
                "QI",
                "quasi-isodynamic accepted reference",
                8,
            ),
        ),
        reporter=failing_report,
        artifact_resolver=_fake_artifact_resolver,
    )

    qi = payload["qi_seed_robustness"]
    assert qi["summary"]["n_failed"] == 1
    assert qi["summary"]["seed_robust_gate_passed"] is False
    row = qi["rows"][0]
    assert row["qi_gate_status"] == "fragile_open"
    assert row["artifact_reason"] == "mode21_qi_tolerance_exceeded"
    assert row["equal_arc_drift_worst_normalized_max_abs"] == pytest.approx(9.0e-2)


def test_qi_seed_robustness_rejects_input_only_variants() -> None:
    mod = _load_tool_module()
    payload = mod.build_parity_matrix(
        cases=(),
        qi_variants=(
            mod.ParityCase("nfp1_QI", "QI input nfp1", "quasi-isodynamic input variant", 8),
        ),
        reporter=_fake_report,
        artifact_resolver=_fake_artifact_resolver,
    )

    qi = payload["qi_seed_robustness"]
    assert qi["summary"]["n_rejected"] == 1
    assert qi["summary"]["seed_robust_gate_passed"] is False
    row = qi["rows"][0]
    assert row["sample_set_id"] == "nfp1_QI:ntheta=8:mboz=21:nboz=21"
    assert row["qi_gate_status"] == "artifact_rejected"
    assert row["artifact_reason"] == "missing_bundled_wout_reference"
    assert "does not launch VMEC solves" in row["rejection_reason"]


def test_write_parity_matrix_artifacts_writes_companions(tmp_path: Path) -> None:
    mod = _load_tool_module()
    payload = mod.build_parity_matrix(
        cases=(
            mod.ParityCase("shaped_tokamak_pressure", "tokamak", "axisymmetric", 8),
        ),
        reporter=_fake_report,
        artifact_resolver=_fake_artifact_resolver,
    )

    paths = mod.write_parity_matrix_artifacts(payload, out=tmp_path / "parity.png")

    for path in paths.values():
        assert Path(path).exists()
    saved = json.loads((tmp_path / "parity.json").read_text(encoding="utf-8"))
    assert saved["summary"]["n_equal_arc_passed"] == 1
    assert saved["sample_set_provenance"]["summary"]["n_total_sample_sets"] == 6
    csv_text = (tmp_path / "parity.csv").read_text(encoding="utf-8")
    assert "sample_set_id" in csv_text
    assert "shaped_tokamak_pressure:ntheta=8:mboz=21:nboz=21" in csv_text
    assert "missing_bundled_wout_reference" in csv_text
