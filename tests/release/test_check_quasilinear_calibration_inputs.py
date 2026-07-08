"""Tests for quasilinear calibration input validation gates."""

from __future__ import annotations

from support.paths import load_release_tool
import json
from pathlib import Path


def _load_tool_module():
    return load_release_tool("check_quasilinear_calibration_inputs")


def _write_report(path: Path, artifact: str, *, split: str = "holdout") -> None:
    payload = {
        "kind": "quasilinear_calibration_report",
        "points": [
            {
                "case": "synthetic",
                "split": split,
                "predicted_heat_flux": 1.0,
                "observed_heat_flux": 1.1,
                "saturation_rule": "linear_weight",
                "nonlinear_artifact": artifact,
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_audit_passes_when_required_point_matches_passed_gate(tmp_path: Path) -> None:
    mod = _load_tool_module()
    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps(
            {
                "case": "synthetic_nonlinear_window",
                "spectrax": "tools_out/synthetic.csv",
                "gate_report": {
                    "case": "synthetic_nonlinear_window",
                    "passed": True,
                    "gates": [],
                },
            }
        ),
        encoding="utf-8",
    )
    report = tmp_path / "report.json"
    _write_report(report, "tools_out/synthetic.csv")

    paths = mod.write_audit(
        [report],
        gate_patterns=[str(gate)],
        out_json=tmp_path / "audit.json",
        no_plot=True,
    )

    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert (
        payload["reports"][0]["points"][0]["reason"] == "matched passed nonlinear gate"
    )


def test_audit_passes_when_required_point_cites_passed_gate_sidecar(
    tmp_path: Path,
) -> None:
    mod = _load_tool_module()
    gate = tmp_path / "ensemble_gate.json"
    gate.write_text(
        json.dumps(
            {
                "case": "replicated_nonlinear_window",
                "kind": "nonlinear_window_ensemble_report",
                "promotion_gate": {"passed": True},
            }
        ),
        encoding="utf-8",
    )
    report = tmp_path / "report.json"
    _write_report(report, gate.as_posix())

    paths = mod.write_audit(
        [report],
        gate_patterns=[str(gate)],
        out_json=tmp_path / "audit.json",
        no_plot=True,
    )

    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    point = payload["reports"][0]["points"][0]
    assert payload["passed"] is True
    assert point["reason"] == "matched passed nonlinear gate"
    assert point["matched_gate"]["artifact"] == gate.as_posix()


def test_default_gate_glob_recurses_into_nested_holdout_artifacts(
    tmp_path: Path,
) -> None:
    mod = _load_tool_module()
    gate = tmp_path / "docs/_static/nested_holdouts/case/ensemble_gate.json"
    gate.parent.mkdir(parents=True)
    gate.write_text(
        json.dumps(
            {
                "case": "nested_replicated_ensemble",
                "kind": "nonlinear_window_ensemble_report",
                "claim_level": "replicated_nonlinear_window_uncertainty_gate_not_simulation_claim",
                "passed": True,
                "promotion_gate": {"passed": True},
            }
        ),
        encoding="utf-8",
    )
    report = tmp_path / "report.json"
    _write_report(report, gate.as_posix())
    old_default = mod.DEFAULT_GATE_GLOB
    mod.DEFAULT_GATE_GLOB = str(tmp_path / "docs/_static/**/*.json")
    try:
        paths = mod.write_audit(
            [report], out_json=tmp_path / "audit.json", no_plot=True
        )
    finally:
        mod.DEFAULT_GATE_GLOB = old_default

    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    point = payload["reports"][0]["points"][0]
    assert payload["passed"] is True
    assert point["matched_gate"]["artifact"].endswith(
        "nested_holdouts/case/ensemble_gate.json"
    )


def test_audit_normalizes_absolute_artifact_paths_from_other_checkouts(
    tmp_path: Path,
) -> None:
    mod = _load_tool_module()
    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps(
            {
                "case": "synthetic_nonlinear_window",
                "spectrax": "tools_out/synthetic.csv",
                "gate_report": {
                    "case": "synthetic_nonlinear_window",
                    "passed": True,
                    "gates": [],
                },
            }
        ),
        encoding="utf-8",
    )
    report = tmp_path / "report.json"
    _write_report(report, "/Users/example/local/SPECTRAX-GK/tools_out/synthetic.csv")

    paths = mod.write_audit(
        [report],
        gate_patterns=[str(gate)],
        out_json=tmp_path / "audit.json",
        no_plot=True,
    )

    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    point = payload["reports"][0]["points"][0]
    assert payload["passed"] is True
    assert point["nonlinear_artifact"] == "tools_out/synthetic.csv"
    assert point["reason"] == "matched passed nonlinear gate"


def test_audit_fails_when_required_point_uses_failed_gate(tmp_path: Path) -> None:
    mod = _load_tool_module()
    gate = tmp_path / "external_gate.json"
    gate.write_text(
        json.dumps(
            {
                "case": "external_cth_like",
                "promotion_gate": {"passed": False},
                "runs": [
                    {
                        "csv": "docs/_static/external_vmec_cth_like_nonlinear_t150_pilot.traces.csv"
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    report = tmp_path / "report.json"
    _write_report(
        report, "docs/_static/external_vmec_cth_like_nonlinear_t150_pilot.traces.csv"
    )

    paths = mod.write_audit(
        [report],
        gate_patterns=[str(gate)],
        out_json=tmp_path / "audit.json",
        no_plot=True,
    )

    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert (
        payload["reports"][0]["points"][0]["reason"]
        == "matching nonlinear gate is negative evidence for calibration admission"
    )
    assert payload["n_negative_evidence"] == 1


def test_audit_records_qh_gate_with_unacceptable_claim_as_negative_evidence(
    tmp_path: Path,
) -> None:
    mod = _load_tool_module()
    gate = tmp_path / "external_qh_gate.json"
    gate.write_text(
        json.dumps(
            {
                "case": "nfp4 QH external VMEC nonlinear high-grid convergence",
                "claim_level": "finite_high_grid_long_nonlinear_feasibility_not_yet_transport_validation",
                "gate_report": {
                    "case": "nfp4 QH external VMEC nonlinear high-grid convergence",
                    "passed": True,
                    "gates": [],
                },
                "kind": "external_vmec_nonlinear_grid_convergence_gate",
                "promotion_gate": {"passed": True},
                "runs": [
                    {
                        "csv": "docs/_static/external_vmec_qh_nonlinear_t150_n64_pilot.traces.csv"
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    report = tmp_path / "report.json"
    _write_report(
        report,
        "docs/_static/external_vmec_qh_nonlinear_t150_n64_pilot.traces.csv",
    )

    paths = mod.write_audit(
        [report],
        gate_patterns=[str(gate)],
        out_json=tmp_path / "audit.json",
        no_plot=True,
    )

    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    point = payload["reports"][0]["points"][0]
    assert payload["passed"] is False
    assert point["passed"] is False
    assert (
        point["reason"]
        == "matching nonlinear gate is negative evidence for calibration admission"
    )
    assert point["matched_gate"]["raw_gate_passed"] is True
    assert point["matched_gate"]["promotion_gate_passed"] is True
    assert point["matched_gate"]["claim_level_acceptable"] is False
    assert point["matched_gate"]["admission_blockers"] == ["claim_level_not_acceptable"]
    assert (
        payload["negative_evidence"][0]["case"]
        == "nfp4 QH external VMEC nonlinear high-grid convergence"
    )


def test_audit_fails_when_required_point_has_no_gate(tmp_path: Path) -> None:
    mod = _load_tool_module()
    report = tmp_path / "report.json"
    _write_report(report, "tools_out/missing.csv")

    paths = mod.write_audit(
        [report], gate_patterns=[], out_json=tmp_path / "audit.json", no_plot=True
    )

    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert (
        payload["reports"][0]["points"][0]["reason"]
        == "no matching nonlinear validation/convergence gate"
    )


def test_audit_accepts_nested_high_grid_admission_input_artifact(
    tmp_path: Path,
) -> None:
    mod = _load_tool_module()
    gate = tmp_path / "high_grid_admission.json"
    gate.write_text(
        json.dumps(
            {
                "kind": "external_vmec_high_grid_admission_gate",
                "case": "synthetic high-grid admission",
                "claim_level": "passed_high_grid_transport_holdout_admission_under_coarse_grid_exclusion",
                "inputs": {
                    "replicate_ensemble_gate": "docs/_static/replicate/ensemble_gate.json",
                },
                "promotion_gate": {"passed": True},
            }
        ),
        encoding="utf-8",
    )
    report = tmp_path / "report.json"
    _write_report(report, "docs/_static/replicate/ensemble_gate.json")

    paths = mod.write_audit(
        [report],
        gate_patterns=[str(gate)],
        out_json=tmp_path / "audit.json",
        no_plot=True,
    )

    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    point = payload["reports"][0]["points"][0]
    assert payload["passed"] is True
    assert point["matched_gate"]["case"] == "synthetic high-grid admission"


def test_audit_prefers_external_admission_gate_over_raw_nested_ensemble(
    tmp_path: Path,
) -> None:
    mod = _load_tool_module()
    raw = tmp_path / "docs/_static/external_vmec_holdouts/case/ensemble_gate.json"
    admission = tmp_path / "aa_admission.json"
    artifact = raw.as_posix()
    raw.parent.mkdir(parents=True)
    raw.write_text(
        json.dumps(
            {
                "case": "synthetic_external_vmec_ensemble",
                "kind": "nonlinear_window_ensemble_report",
                "claim_level": "replicated_nonlinear_window_uncertainty_gate_not_simulation_claim",
                "passed": True,
            }
        ),
        encoding="utf-8",
    )
    admission.write_text(
        json.dumps(
            {
                "case": "synthetic external admission",
                "kind": "external_vmec_replicate_admission_gate",
                "claim_level": "passed_replicated_external_vmec_transport_holdout_under_explicit_spread_gate",
                "inputs": {"replicate_ensemble_gate": artifact},
                "promotion_gate": {"passed": True},
            }
        ),
        encoding="utf-8",
    )
    report = tmp_path / "report.json"
    _write_report(report, artifact)

    paths = mod.write_audit(
        [report],
        gate_patterns=[str(raw), str(admission)],
        out_json=tmp_path / "audit.json",
        no_plot=True,
    )

    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    point = payload["reports"][0]["points"][0]
    assert payload["passed"] is True
    assert point["matched_gate"]["kind"] == "external_vmec_replicate_admission_gate"
    assert point["matched_gate"]["claim_level_acceptable"] is True


def test_audit_ignores_non_required_audit_split_without_gate(tmp_path: Path) -> None:
    mod = _load_tool_module()
    report = tmp_path / "report.json"
    _write_report(report, "tools_out/missing.csv", split="audit")

    paths = mod.write_audit(
        [report], gate_patterns=[], out_json=tmp_path / "audit.json", no_plot=True
    )

    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["reports"][0]["points"][0]["reason"] == "not required split"


def test_tracked_quasilinear_train_holdout_reports_use_passed_nonlinear_gates() -> None:
    mod = _load_tool_module()
    root = Path(__file__).resolve().parents[2]
    reports = [
        root / "docs/_static/quasilinear_cyclone_miller_train_holdout_report.json",
        root / "docs/_static/quasilinear_hsx_train_holdout_report.json",
        root / "docs/_static/quasilinear_w7x_train_holdout_report.json",
        root / "docs/_static/quasilinear_stellarator_train_holdout_report.json",
    ]

    payload = mod.audit_calibration_inputs(reports)

    assert payload["passed"] is True
    required_rows = [
        point
        for report in payload["reports"]
        for point in report["points"]
        if point["required"]
    ]
    assert len(required_rows) == 20
    assert all(point["matched_gate"] is not None for point in required_rows)
    matched_cases = {point["matched_gate"]["case"] for point in required_rows}
    assert matched_cases == {
        "cyclone_nonlinear_long_window",
        "cyclone_miller_nonlinear_window",
        "hsx_nonlinear_window",
        "w7x_nonlinear_window",
        "D-shaped tokamak external VMEC nonlinear t250 high-grid convergence",
        "ITERModel external VMEC nonlinear t350 high-grid convergence",
        "updown_asym_external_vmec_t450",
        "circular_external_vmec_t450",
        "CTH-like external VMEC modified-protocol high-grid admission",
        "Shaped tokamak pressure external VMEC dt=0.04 high-grid transport holdout admission",
        "qp_diag_nfp2_m4_final_t250_n64_seed_timestep_ensemble_gate",
        "solovev_reference_repair_dt002_amp1em5_n48_t250",
    }
    external_rows = [
        point
        for point in required_rows
        if "external_vmec" in str(point["matched_gate"]["artifact"])
    ]
    assert [point["case"] for point in external_rows] == [
        "dshape_external_vmec_t250_window",
        "itermodel_external_vmec_t350_window",
        "updown_asym_external_vmec_t450_window",
        "circular_external_vmec_t450_window",
        "cth_like_external_vmec_t700_high_grid_window",
        "shaped_tokamak_pressure_external_vmec_t650_high_grid_window",
        "solovev_reference_repair_dt002_amp1em5_n48_t250",
    ]
