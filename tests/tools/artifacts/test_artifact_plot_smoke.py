from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

from support.paths import load_artifact_tool
from spectraxgk.diagnostics.modes import save_eigenfunction_reference_bundle


def test_plot_independent_ky_scan_scaling_defaults_and_rows(tmp_path: Path) -> None:
    mod = load_artifact_tool("plot_independent_ky_scan_scaling")

    args = mod.build_parser().parse_args([])
    assert args.inputs == mod.DEFAULT_INPUTS
    assert args.out_prefix == mod.DEFAULT_PREFIX

    payload = {
        "backend": "cpu",
        "grid": {"Nx": 1, "Ny": 128, "Nz": 96, "Nl": 4, "Nm": 8},
        "time": {"steps": 240},
        "identity_passed": True,
        "rows": [
            {
                "requested_devices": 1,
                "actual_workers": 1,
                "timed_wall_s": 2.0,
                "strong_speedup_vs_1_device": 1.0,
                "parallel_efficiency": 1.0,
                "max_gamma_rel_error": 0.0,
                "max_omega_abs_error": 0.0,
                "identity_gate_pass": True,
                "error": None,
            }
        ],
    }
    path = tmp_path / "cpu.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    summary = mod.load_summary([path])

    assert summary["identity_passed"] is True
    assert summary["rows"][0]["backend"] == "cpu"
    assert summary["rows"][0]["grid_label"] == "Nx=1, Ny=128, Nz=96, Nl=4, Nm=8"


def test_eigenfunction_diagnostics_tool_writes_summary_and_overlay(
    tmp_path: Path,
) -> None:
    mod = load_artifact_tool("plot_eigenfunction_diagnostics")

    summary_csv = tmp_path / "overlap.csv"
    summary_csv.write_text(
        "ky,eig_overlap_gx,eig_rel_l2,selected\n"
        "0.2,0.95,0.08,true\n"
        "0.3,0.98,0.04,true\n"
        "0.4,0.10,0.90,false\n",
        encoding="utf-8",
    )
    summary_png = tmp_path / "overlap.png"
    assert (
        mod.main(["overlap-summary", "--csv", str(summary_csv), "--out", str(summary_png)])
        == 0
    )
    assert summary_png.exists()
    assert summary_png.with_suffix(".pdf").exists()

    theta = np.linspace(-1.0, 1.0, 5)
    mode = np.exp(1j * theta)
    reference = save_eigenfunction_reference_bundle(
        tmp_path / "reference.npz",
        theta=theta,
        mode=mode,
        source="unit-test",
        case="synthetic",
    )
    spectrax_csv = tmp_path / "spectrax.csv"
    spectrax_csv.write_text(
        "z,eigen_real,eigen_imag\n"
        + "\n".join(
            f"{z},{value.real},{value.imag}" for z, value in zip(theta, mode)
        )
        + "\n",
        encoding="utf-8",
    )
    overlay_png = tmp_path / "overlay.png"

    assert (
        mod.main(
            [
                "reference-overlay",
                str(reference),
                str(spectrax_csv),
                "--out",
                str(overlay_png),
            ]
        )
        == 0
    )
    assert overlay_png.exists()
    assert overlay_png.with_suffix(".pdf").exists()


def test_readme_panel_builder_treats_missing_optional_transport_artifacts_as_pending(
    tmp_path: Path,
) -> None:
    mod = load_artifact_tool("build_qa_itg_optimization_readme_panel")
    geometry_png = tmp_path / "geometry.png"
    fig, ax = plt.subplots(figsize=(1.0, 1.0))
    ax.imshow([[0.0, 1.0], [1.0, 0.0]])
    ax.set_axis_off()
    fig.savefig(geometry_png)
    plt.close(fig)

    sweep_json = tmp_path / "sweep.json"
    sweep_json.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "qa_baseline_scipy",
                        "iota_profile": {"s_iotaf": [0.2, 0.7], "iotaf": [0.41, 0.42]},
                        "q_traces": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    landscape_json = tmp_path / "landscape.json"
    landscape_json.write_text(
        json.dumps(
            {
                "coefficient": "RBC(1,1)",
                "rows": [
                    {
                        "label": "baseline",
                        "reduced_metric_reports": {
                            "growth": {
                                "payload": {
                                    "sample_statistics": {
                                        "weighted_standard_error": 0.01
                                    }
                                }
                            }
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    landscape_csv = tmp_path / "landscape.csv"
    landscape_csv.write_text(
        "label,relative_fraction,coefficient_value,growth,quasilinear_flux\n"
        "baseline,0.0,0.0,1.0,2.0\n",
        encoding="utf-8",
    )
    out = tmp_path / "panel.png"

    sidecar = mod.build_panel(
        geometry_png=geometry_png,
        sweep_json=sweep_json,
        landscape_json=landscape_json,
        landscape_csv=landscape_csv,
        admission_json=tmp_path / "missing_admission.json",
        matched_json=tmp_path / "missing_matched.json",
        out=out,
    )

    assert out.exists()
    assert sidecar["sources"]["landscape_admission_present"] is False
    assert sidecar["sources"]["matched_nonlinear_present"] is False
    assert sidecar["selected_landscape_candidate"] is None
    assert sidecar["matched_projected_candidate"] is None


def test_error_anatomy_report_and_cli_lock_fail_closed_residual_story(
    tmp_path: Path,
) -> None:
    mod = load_artifact_tool("build_quasilinear_error_anatomy")

    report = mod.build_error_anatomy_report()

    assert report["kind"] == "quasilinear_error_anatomy"
    assert (
        report["claim_level"]
        == "model_development_residual_anatomy_not_absolute_flux_promotion"
    )
    assert report["case_count"] == 12
    assert report["holdout_count"] == 10
    assert report["promotion_gate"]["passed"] is False
    assert "case_residuals_exceed_transport_gate" in report["promotion_gate"]["blockers"]
    assert 0.697 < report["candidate_mean_abs_relative_error"] < 0.698
    assert report["rows"][0]["case"] == "solovev_reference_repair_dt002_amp1em5_n48_t250"
    assert report["rows"][0]["above_transport_gate"] is True
    assert report["rows"][0]["overpredicts"] is True
    groups = {row["geometry_group"]: row for row in report["geometry_group_summary"]}
    assert groups["external axisymmetric VMEC"]["error_budget_fraction"] > 0.82
    assert groups["stellarator benchmark"]["mean_abs_relative_error"] < 0.35
    assert report["frozen_ledger_policy"]["additional_holdout_collection_active"] is False
    assert report["frozen_ledger_policy"]["ledger_case_count"] == 12
    assert report["dominant_residuals"][0]["case"] == report["rows"][0]["case"]
    core = report["core_portfolio_gate"]
    assert core["passed"] is True
    assert core["core_case_count"] == 10
    assert core["core_holdout_count"] == 8
    assert core["excluded_case_count"] == 2
    assert 0.27 < core["core_mean_abs_relative_error"] < 0.29
    assert 0.27 < core["core_holdout_mean_abs_relative_error"] < 0.29
    assert core["core_prediction_interval_coverage"] == 1.0
    assert core["screening_gate_passed"] is False

    root = Path(__file__).resolve().parents[3]
    out = tmp_path / "ql_error_anatomy.png"
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "tools" / "artifacts" / "build_quasilinear_error_anatomy.py"),
            "--out",
            str(out),
        ],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert completed.returncode == 2
    assert "promotion_passed=False" in completed.stdout
    assert out.exists()
    payload = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["promotion_gate"]["passed"] is False
    assert payload["core_portfolio_gate"]["passed"] is True
    assert payload["frozen_ledger_policy"]["additional_holdout_collection_active"] is False
    csv_text = out.with_suffix(".csv").read_text(encoding="utf-8")
    assert csv_text.startswith("case,label,split,geometry")


def _uq_result(kind: str, scale: float) -> dict[str, object]:
    params = [0.2, 0.1, 0.05, -0.03]
    covariance = [
        [scale * 4.0e-4, scale * 1.0e-4, 0.0, 0.0],
        [scale * 1.0e-4, scale * 3.0e-4, 0.0, 0.0],
        [0.0, 0.0, scale * 2.0e-4, 0.0],
        [0.0, 0.0, 0.0, scale * 1.5e-4],
    ]
    return {
        "objective_kind": kind,
        "parameter_names": [
            "minor_radius_log_shift",
            "vertical_elongation_shift",
            "helical_ripple_amplitude",
            "magnetic_shear_shift",
        ],
        "initial_params": [0.28, 0.46, 0.42, -0.32],
        "final_params": params,
        "initial_objective": 1.0,
        "final_objective": 0.4,
        "gradient_gate": {
            "passed": True,
            "max_abs_error": 1.0e-6,
            "max_rel_error": 1.0e-4,
            "tangent_max_abs_error": 2.0e-6,
            "jacobian_ad": [[0.1, -0.2, 0.3, -0.4]],
            "jacobian_fd": [[0.1000005, -0.199999, 0.300001, -0.400001]],
        },
        "covariance": {
            "covariance": covariance,
            "covariance_std": [0.02, 0.017, 0.014, 0.012],
            "covariance_correlation": [
                [1.0, 0.25, 0.0, 0.0],
                [0.25, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "covariance_eigenvalues": [1.0e-4, 2.0e-4, 3.0e-4, 4.0e-4],
            "uq_ellipse_area_1sigma": 2.0e-3,
            "jacobian_singular_values": [3.0, 1.0, 0.5, 0.1],
            "jacobian_condition_number": 30.0,
            "sensitivity_map_rank": 4,
        },
    }


def test_stellarator_optimization_uq_summary_artifacts_and_shape_gate(
    tmp_path: Path,
) -> None:
    mod = load_artifact_tool("plot_stellarator_optimization_uq")
    payload = {
        "parameter_names": [
            "minor_radius_log_shift",
            "vertical_elongation_shift",
            "helical_ripple_amplitude",
            "magnetic_shear_shift",
        ],
        "observable_names": ["growth_rate"],
        "parallel": {
            "requested_workers": 3,
            "effective_workers": 3,
            "executor": "thread",
            "finite_difference_workers": 2,
            "finite_difference_executor": "thread",
            "identity_contract": "parallel objective reports must preserve serial ordering and values",
        },
        "results": [
            _uq_result("growth", 1.0),
            _uq_result("quasilinear_flux", 1.2),
            _uq_result("nonlinear_heat_flux", 0.8),
        ],
    }

    summary = mod.build_uq_summary(payload)
    assert summary["kind"] == "stellarator_itg_optimization_uq"
    assert summary["all_gradient_gates_passed"] is True
    assert summary["all_sensitivity_maps_full_rank"] is True
    assert summary["parallel"]["requested_workers"] == 3
    assert summary["parallel"]["finite_difference_workers"] == 2
    assert len(summary["results"]) == 3
    assert summary["results"][0]["max_abs_error"] == 1.0e-6

    paths = mod.write_uq_figure(summary, out=tmp_path / "uq.png")

    assert Path(paths["png"]).exists()
    assert Path(paths["pdf"]).exists()
    written = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert (
        written["claim_level"]
        == "reduced_objective_uq_and_sensitivity_validation_not_full_vmec_gk_optimization"
    )

    try:
        mod.build_uq_summary(
            {"parameter_names": ["a", "b"], "results": [_uq_result("growth", 1.0)]}
        )
    except ValueError as exc:
        assert "gradient gate" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected malformed payload to be rejected")
