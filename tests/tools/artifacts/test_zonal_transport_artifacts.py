from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import netCDF4 as nc
import numpy as np
import pytest

from support.paths import load_artifact_tool
from tools.artifacts.build_transport_audit_redesign_report import (
    main as transport_redesign_main,
)


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


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_transport_audit_redesign_writes_fail_closed_report(tmp_path: Path) -> None:
    comparison = tmp_path / "comparison.json"
    out = tmp_path / "redesign.json"
    _comparison(comparison, relative_reduction=-0.005, passed=False)

    assert (
        transport_redesign_main(
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


def test_transport_audit_redesign_can_fail_on_required_redesign(
    tmp_path: Path,
) -> None:
    comparison = tmp_path / "comparison.json"
    out = tmp_path / "redesign.json"
    _comparison(comparison, relative_reduction=-0.005, passed=False)

    assert (
        transport_redesign_main(
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


def test_plot_zonal_flow_response_output_subcommand(tmp_path: Path, monkeypatch) -> None:
    mod = load_artifact_tool("plot_zonal_flow_response")

    data_path = tmp_path / "diag.out.nc"
    with nc.Dataset(data_path, "w") as ds:
        ds.createDimension("time", 5)
        grids = ds.createGroup("Grids")
        diag = ds.createGroup("Diagnostics")
        grids.createVariable("time", "f8", ("time",))[:] = np.linspace(0.0, 4.0, 5)
        diag.createVariable("Phi2_zonal_t", "f8", ("time",))[:] = np.array(
            [1.0, 0.7, 0.55, 0.45, 0.4]
        )

    out = tmp_path / "zf_from_output.png"
    monkeypatch.setattr(
        sys, "argv", [str(mod.__file__), "output", str(data_path), "--out", str(out)]
    )

    assert mod.main() == 0
    assert out.exists()
    assert out.with_suffix(".pdf").exists()
    assert out.with_suffix(".csv").exists()
    meta = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["variable"] == "Phi2_zonal_t"
    assert meta["initial_policy"] == "window_abs_mean"
    assert meta["damping_method"] == "combined_envelope"
    assert meta["frequency_method"] == "peak_spacing"
    assert "peak_fit_count" in meta
    assert "zonal-energy proxy" in meta["notes"]


def test_plot_zonal_flow_response_output_subcommand_complex_mode_history(
    tmp_path: Path, monkeypatch
) -> None:
    mod = load_artifact_tool("plot_zonal_flow_response")

    data_path = tmp_path / "diag.out.nc"
    with nc.Dataset(data_path, "w") as ds:
        ds.createDimension("time", 5)
        ds.createDimension("kx", 2)
        ds.createDimension("ri", 2)
        grids = ds.createGroup("Grids")
        diag = ds.createGroup("Diagnostics")
        grids.createVariable("time", "f8", ("time",))[:] = np.linspace(0.0, 4.0, 5)
        raw = np.zeros((5, 2, 2), dtype=float)
        raw[:, 1, 0] = np.array([0.0, -0.4, -0.2, 0.1, 0.05])
        raw[:, 1, 1] = np.array([1.0, 0.6, 0.3, -0.2, -0.1])
        diag.createVariable("Phi_zonal_mode_kxt", "f8", ("time", "kx", "ri"))[:] = raw

    out = tmp_path / "zf_signed.png"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(mod.__file__),
            "output",
            str(data_path),
            "--var",
            "Phi_zonal_mode_kxt",
            "--kx-index",
            "1",
            "--align-phase",
            "--component",
            "real",
            "--out",
            str(out),
        ],
    )

    assert mod.main() == 0
    meta = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["variable"] == "Phi_zonal_mode_kxt"
    assert meta["initial_policy"] == "window_abs_mean"
    assert meta["damping_method"] == "combined_envelope"
    assert meta["frequency_method"] == "peak_spacing"
    assert "peak_fit_count" in meta


def test_plot_zonal_flow_response_csv_subcommand(tmp_path: Path, monkeypatch) -> None:
    mod = load_artifact_tool("plot_zonal_flow_response")

    csv_path = tmp_path / "response.csv"
    _write_csv(
        csv_path,
        [
            {"t": 0.0, "response": 1.0},
            {"t": 1.0, "response": 0.8},
            {"t": 2.0, "response": 0.7},
            {"t": 3.0, "response": 0.65},
            {"t": 4.0, "response": 0.6},
        ],
    )
    out = tmp_path / "zf_csv.png"
    monkeypatch.setattr(
        sys, "argv", [str(mod.__file__), "csv", str(csv_path), "--out", str(out)]
    )

    assert mod.main() == 0
    assert out.exists()
    meta = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["initial_policy"] == "window_abs_mean"
    assert "residual_level" in meta


def test_build_zonal_flow_objective_gate_writes_diagnostic_artifacts(
    tmp_path: Path,
) -> None:
    mod = load_artifact_tool("build_zonal_flow_objective_gate")
    summary = tmp_path / "summary.csv"
    comparison = tmp_path / "compare.csv"
    _write_csv(
        summary,
        [
            {
                "kx_target": 0.05,
                "residual_level": 0.20,
                "residual_std": 0.03,
                "gam_damping_rate": "",
            },
            {
                "kx_target": 0.10,
                "residual_level": 0.40,
                "residual_std": 0.02,
                "gam_damping_rate": 0.04,
            },
        ],
    )
    _write_csv(
        comparison,
        [
            {"kx": 0.05, "tail_std": 0.12, "reference_tail_std": 0.03},
            {"kx": 0.10, "tail_std": 0.05, "reference_tail_std": 0.05},
        ],
    )
    out_json = tmp_path / "gate.json"
    out_csv = tmp_path / "gate.csv"
    out_png = tmp_path / "gate.png"

    rc = mod.main(
        [
            "--summary-csv",
            str(summary),
            "--comparison-csv",
            str(comparison),
            "--out-json",
            str(out_json),
            "--out-csv",
            str(out_csv),
            "--out-png",
            str(out_png),
            "--recurrence-source",
            "tail_std_ratio",
            "--missing-damping-policy",
            "zero",
            "--recurrence-weight",
            "0.5",
        ]
    )

    assert rc == 0
    assert out_json.exists()
    assert out_csv.exists()
    assert out_png.exists()
    assert out_png.with_suffix(".pdf").exists()
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["validation_status"] == "diagnostic"
    assert payload["promotion_ready"] is False
    assert payload["missing_damping_count"] == 1
    assert payload["sample_count"] == 2
    assert payload["recurrence_source"] == "tail_std_ratio"
    assert payload["gate_index_include"] is False
    recurrences = [row["recurrence_amplitude"] for row in payload["row_table"]]
    np.testing.assert_allclose(recurrences, [4.0, 1.0])
    json.dumps(payload, allow_nan=False)


def test_build_zonal_flow_objective_gate_fail_policy_rejects_missing_damping(
    tmp_path: Path,
) -> None:
    mod = load_artifact_tool("build_zonal_flow_objective_gate")
    summary = tmp_path / "summary.csv"
    _write_csv(
        summary,
        [
            {
                "kx_target": 0.05,
                "residual_level": 0.20,
                "residual_std": 0.03,
                "gam_damping_rate": "",
            }
        ],
    )

    with pytest.raises(ValueError, match="missing finite damping_rate"):
        mod.main(
            [
                "--summary-csv",
                str(summary),
                "--comparison-csv",
                str(tmp_path / "missing.csv"),
                "--missing-damping-policy",
                "fail",
            ]
        )


def test_generate_miller_zonal_response_panel_main(
    tmp_path: Path, monkeypatch
) -> None:
    mod = load_artifact_tool("generate_miller_zonal_response_panel")

    config = tmp_path / "pilot.toml"
    config.write_text(
        """
[grid]
Nx = 4
Ny = 6
Nz = 8
Lx = 6.28
Ly = 6.28
boundary = "periodic"

[time]
t_max = 1.0
dt = 0.1
method = "rk2"
diagnostics = true
sample_stride = 1

[geometry]
model = "miller"
q = 1.4
s_hat = 0.8
epsilon = 0.18
R0 = 2.77778

[init]
init_field = "density"
init_amp = 1.0e-4
init_single = true

[physics]
adiabatic_electrons = true
nonlinear = false

[[species]]
name = "ion"
charge = 1.0
mass = 1.0
density = 1.0
temperature = 1.0
tprim = 2.49
fprim = 0.8
kinetic = true

[run]
ky = 0.0
kx = 0.1
Nl = 2
Nm = 2
dt = 0.1
steps = 10
sample_stride = 1
diagnostics = true
""".strip(),
        encoding="utf-8",
    )

    out_bundle = tmp_path / "pilot.out.nc"
    out_png = tmp_path / "pilot.png"

    def _fake_run(cfg, *, out, **kwargs):
        path = Path(out)
        with nc.Dataset(path, "w") as ds:
            ds.createDimension("time", 6)
            ds.createDimension("kx", 3)
            ds.createDimension("ri", 2)
            grids = ds.createGroup("Grids")
            diag = ds.createGroup("Diagnostics")
            grids.createVariable("time", "f8", ("time",))[:] = np.linspace(0.0, 5.0, 6)
            grids.createVariable("kx", "f8", ("kx",))[:] = np.array([-0.1, 0.0, 0.1])
            raw = np.zeros((6, 3, 2), dtype=float)
            raw[:, 2, 0] = np.array([1.0, 0.6, 0.35, 0.2, 0.12, 0.1])
            raw[:, 2, 1] = np.array([0.2, 0.12, 0.08, 0.03, 0.02, 0.01])
            diag.createVariable("Phi_zonal_mode_kxt", "f8", ("time", "kx", "ri"))[:] = raw
        return object(), {"out": str(path)}

    monkeypatch.setattr(mod, "run_runtime_nonlinear_with_artifacts", _fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(mod.__file__),
            "--config",
            str(config),
            "--out-bundle",
            str(out_bundle),
            "--out-png",
            str(out_png),
        ],
    )

    assert mod.main() == 0
    assert out_png.exists()
    assert out_png.with_suffix(".pdf").exists()
    assert out_png.with_suffix(".csv").exists()
    meta = json.loads(out_png.with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["variable"] == "Phi_zonal_mode_kxt"
    assert meta["kx_selected"] == 0.1
    assert meta["initial_policy"] == "first_abs"
    assert meta["literature_reference"]["case"] == "III"
    assert meta["literature_reference"]["residual_phi_over_phi0"] == 0.19
    assert meta["gate_tolerances"]["residual_atol"] == 0.015
    assert meta["gate_report"]["case"] == "merlo_case_iii_zonal_response"
    assert {gate["metric"] for gate in meta["gate_report"]["gates"]} == {
        "residual_level",
        "gam_frequency_R0_over_vi",
        "gam_growth_rate_R0_over_vi",
    }
    assert isinstance(meta["paper_scale_gate_passed"], bool)
    assert "gam_frequency_R0_over_vi" in meta
    assert "gam_growth_rate_R0_over_vi" in meta
    assert "omega_abs_error_vs_literature_R0_over_vi" in meta
    assert meta["damping_method"] == "branchwise_extrema"
    assert meta["frequency_method"] == "hilbert_phase"
    assert "peak_fit_count" in meta
    assert "fit_tmax" in meta
    assert meta["setup"] == "initial density perturbation"
    assert meta["validation_status"] == "open"
    assert "Merlo Case-III" in meta["notes"]
    assert "Rosenbluth-Hinton first-sample" in meta["notes"]
    assert "positive and negative extrema separately" in meta["notes"]
    assert "Hilbert-transform analytic signal" in meta["notes"]
    assert "initial density perturbation" in meta["notes"]
