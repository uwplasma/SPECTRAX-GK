"""Tests for linear-validation artifact builders and gate reports."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import pytest

from spectraxgk.benchmarks import EigenfunctionComparisonMetrics
from support.paths import REPO_ROOT


def load_artifact_tool(script_name: str):
    tools_dir = REPO_ROOT / "tools" / "artifacts"
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    path = tools_dir / f"{script_name}.py"
    spec = importlib.util.spec_from_file_location(f"test_loaded_{script_name}", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_qi_branch_refinement_gate_blocks_marginal_branch(tmp_path: Path) -> None:
    mod = load_artifact_tool("build_qi_branch_refinement_gate")
    spectrum = {
        "ky": np.asarray([0.05, 0.07, 0.095, 0.119, 0.143]),
        "gamma": np.asarray([-1e-4, -1e-4, 1.7e-3, 1.7e-3, 3.8e-3]),
        "omega": np.asarray([-0.03, -0.03, -0.06, -0.07, -0.09]),
    }

    report = mod.build_qi_branch_refinement_report(
        spectrum,
        source=tmp_path / "qi.csv",
        krylov={"ky": 0.095, "gamma": 1.9e-3, "omega": -0.054},
    )

    assert report["passed"] is False
    assert report["nonlinear_launch_ready"] is False
    assert report["max_gamma"] == 3.8e-3
    assert report["positive_run_length"] == 3
    assert report["subgates"]["finite_rows"]["passed"] is True
    assert report["subgates"]["positive_run"]["passed"] is True
    assert report["subgates"]["krylov_consistency"]["passed"] is True
    assert report["subgates"]["nonlinear_launch_growth"]["passed"] is False


def test_qi_branch_refinement_gate_passes_strong_consistent_branch(
    tmp_path: Path,
) -> None:
    mod = load_artifact_tool("build_qi_branch_refinement_gate")
    spectrum = {
        "ky": np.asarray([0.05, 0.07, 0.095, 0.119, 0.143]),
        "gamma": np.asarray([-1e-4, 0.010, 0.022, 0.024, 0.021]),
        "omega": np.asarray([-0.03, -0.04, -0.06, -0.07, -0.09]),
    }

    report = mod.build_qi_branch_refinement_report(
        spectrum,
        source=tmp_path / "qi.csv",
        krylov={"ky": 0.095, "gamma": 0.0215, "omega": -0.061},
    )

    assert report["passed"] is True
    assert report["nonlinear_launch_ready"] is True
    assert report["max_gamma"] == 0.024


def test_qi_branch_refinement_tool_writes_fail_closed_artifacts(tmp_path: Path) -> None:
    mod = load_artifact_tool("build_qi_branch_refinement_gate")
    spectrum = tmp_path / "spectrum.csv"
    spectrum.write_text(
        "\n".join(
            [
                "ky,gamma,omega",
                "0.05,-0.0001,-0.03",
                "0.07,-0.0001,-0.03",
                "0.095,0.0017,-0.06",
                "0.119,0.0017,-0.07",
                "0.143,0.0038,-0.09",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    krylov = tmp_path / "krylov.json"
    krylov.write_text(
        json.dumps({"ky": 0.095, "gamma": 0.0019, "omega": -0.054}),
        encoding="utf-8",
    )
    out = tmp_path / "gate.png"

    assert (
        mod.main(
            [
                "--spectrum",
                str(spectrum),
                "--krylov-summary",
                str(krylov),
                "--out",
                str(out),
                "--no-pdf",
                "--dpi",
                "80",
            ]
        )
        == 2
    )
    payload = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert payload["png"].endswith("gate.png")
    assert out.exists()


def test_tem_branch_audit_tracks_sign_and_branch_mismatch(tmp_path: Path) -> None:
    mod = load_artifact_tool("build_tem_branch_parity_audit")
    table = tmp_path / "tem_mismatch.csv"
    table.write_text(
        "ky,gamma_ref,omega_ref,gamma_spectrax,omega_spectrax,rel_gamma,rel_omega\n"
        "0.2,1.0,2.0,1.2,-1.0,0.2,-1.5\n"
        "0.3,2.0,1.0,2.2,0.5,0.1,-0.5\n"
        "0.4,-1.0,-0.5,0.5,1.0,-1.5,-3.0\n",
        encoding="utf-8",
    )
    reference = tmp_path / "tem_reference.csv"
    reference.write_text("ky,omega,gamma\n0.2,2.0,1.0\n", encoding="utf-8")

    payload = mod.build_audit_payload(table=table, reference=reference)
    metrics = payload["metrics"]

    assert payload["status"] == "open"
    assert metrics["gamma_sign_mismatch_count"] == 1
    assert metrics["omega_sign_mismatch_count"] == 2
    assert metrics["omega_branch_inversion"] is True
    assert metrics["max_abs_rel_gamma"] == 1.5
    assert metrics["max_abs_rel_omega_ref_ge_0p2"] == 3.0


def test_tem_branch_audit_writes_artifacts(tmp_path: Path) -> None:
    mod = load_artifact_tool("build_tem_branch_parity_audit")
    table = tmp_path / "tem_mismatch.csv"
    table.write_text(
        "ky,gamma_ref,omega_ref,gamma_spectrax,omega_spectrax,rel_gamma,rel_omega\n"
        "0.2,1.0,2.0,1.1,1.8,0.1,-0.1\n"
        "0.3,1.5,1.0,1.4,0.8,-0.0666667,-0.2\n",
        encoding="utf-8",
    )

    payload = mod.build_audit_payload(table=table, reference=tmp_path / "missing.csv")
    paths = mod.write_artifacts(payload, out_png=tmp_path / "tem_audit.png")

    for path in paths.values():
        assert Path(path).exists()
    written = json.loads((tmp_path / "tem_audit.json").read_text(encoding="utf-8"))
    assert written["kind"] == "tem_branch_parity_audit"
    assert written["reference"]["available"] is False


def test_build_lastvalue_table_converts_scan_columns() -> None:
    mod = load_artifact_tool("derive_imported_linear_lastvalue_table")
    df = pd.DataFrame(
        {
            "ky": [0.1, 0.05],
            "gamma_last": [0.032, 0.012],
            "omega_last": [0.058, 0.029],
            "gamma_ref_last": [0.031, 0.011],
            "omega_ref_last": [0.059, 0.028],
        }
    )

    out = mod._build_lastvalue_table(df)

    assert list(out.columns) == [
        "ky",
        "gamma",
        "omega",
        "gamma_gx",
        "omega_gx",
        "rel_gamma",
        "rel_omega",
    ]
    assert list(out["ky"]) == [0.05, 0.1]
    row = out.iloc[0]
    assert row["gamma"] == pytest.approx(0.012)
    assert row["gamma_gx"] == pytest.approx(0.011)
    assert row["rel_gamma"] == pytest.approx((0.012 - 0.011) / 0.011)


def test_load_scan_requires_lastvalue_columns(tmp_path: Path) -> None:
    mod = load_artifact_tool("derive_imported_linear_lastvalue_table")
    path = tmp_path / "scan.csv"
    pd.DataFrame({"ky": [0.1], "gamma_last": [0.2]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        mod._load_scan(path)


def _w7x_pixel_y(
    value: float, box: tuple[int, int, int, int], y_range: tuple[float, float]
) -> int:
    _x0, _x1, y0, y1 = box
    return int(round(y0 + (y_range[1] - value) / (y_range[1] - y_range[0]) * (y1 - y0)))


def _synthetic_w7x_reference_image(mod) -> Image.Image:
    image = Image.new(
        "RGB", (mod.EXPECTED_IMAGE_SHAPE[1], mod.EXPECTED_IMAGE_SHAPE[0]), "white"
    )
    draw = ImageDraw.Draw(image)
    for panel in mod.PANEL_CALIBRATIONS:
        for _code, color, value in (
            ("stella", (255, 0, 0), 0.12),
            ("GENE", (0, 0, 255), 0.18),
        ):
            y_main = _w7x_pixel_y(value, panel.main_box, panel.y_range)
            y_inset = _w7x_pixel_y(value, panel.inset_box, panel.inset_y_range)
            draw.line(
                (panel.main_box[0], y_main, panel.main_box[1], y_main),
                fill=color,
                width=5,
            )
            draw.line(
                (panel.inset_box[0], y_inset, panel.inset_box[1], y_inset),
                fill=color,
                width=5,
            )
    return image


def test_w7x_zonal_digitizer_axis_mapping_round_trip() -> None:
    mod = load_artifact_tool("digitize_w7x_zonal_reference")
    panel = mod.PANEL_CALIBRATIONS[0]
    x, y = mod._pixel_to_data(
        np.array([panel.main_box[0], panel.main_box[1]]),
        np.array([panel.main_box[2], panel.main_box[3]]),
        box=panel.main_box,
        x_range=panel.t_range,
        y_range=panel.y_range,
    )

    assert np.allclose(x, np.array(panel.t_range))
    assert y[0] == pytest.approx(panel.y_range[1])
    assert y[1] == pytest.approx(panel.y_range[0])


def test_w7x_zonal_digitizer_extracts_synthetic_residuals() -> None:
    mod = load_artifact_tool("digitize_w7x_zonal_reference")
    image = np.asarray(_synthetic_w7x_reference_image(mod), dtype=np.uint8)

    trace_df, residual_df = mod.digitize_reference(image, samples_per_trace=11)

    assert set(trace_df["code"]) == {"stella", "GENE"}
    assert len(trace_df) == len(mod.PANEL_CALIBRATIONS) * 2 * 11
    medians = residual_df.set_index("code")["residual_median"].to_dict()
    assert medians["stella"] == pytest.approx(0.12, abs=5.0e-3)
    assert medians["GENE"] == pytest.approx(0.18, abs=5.0e-3)


def test_w7x_zonal_digitizer_main_writes_artifacts(tmp_path: Path) -> None:
    mod = load_artifact_tool("digitize_w7x_zonal_reference")
    figure = tmp_path / "synthetic_zf.png"
    _synthetic_w7x_reference_image(mod).save(figure)
    out_csv = tmp_path / "trace.csv"
    out_residuals = tmp_path / "residuals.csv"
    out_json = tmp_path / "meta.json"
    out_png = tmp_path / "qa.png"

    rc = mod.main(
        [
            "--figure",
            str(figure),
            "--out-csv",
            str(out_csv),
            "--out-residual-csv",
            str(out_residuals),
            "--out-json",
            str(out_json),
            "--out-png",
            str(out_png),
            "--samples-per-trace",
            "17",
        ]
    )

    assert rc == 0
    assert out_csv.exists()
    assert out_residuals.exists()
    assert out_png.exists()
    assert out_png.with_suffix(".pdf").exists()
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["validation_status"] == "reference"
    assert payload["figure"] == "Figure 11 / source figs/ZF.pdf"


def test_inject_z_wave_targets_resolved_hermite_mode() -> None:
    mod = load_artifact_tool("gate_linear_rhs_zero_norm_state_window")
    state = jnp.zeros((2, 4, 3, 2, 5), dtype=jnp.complex64)

    out = mod._inject_z_wave(state, ky_index=1, kx_index=0, amplitude=0.25, z_mode=1)

    assert jnp.linalg.norm(out[0, 3, 1, 0]) > 0.0
    assert jnp.linalg.norm(out.at[0, 3, 1, 0, :].set(0.0)) == 0.0


def test_build_summary_accepts_collision_skip_and_rejects_hypercollision_skip() -> None:
    mod = load_artifact_tool("gate_linear_rhs_zero_norm_state_window")
    rows = [
        {
            "state": "initial",
            "term_norms": {"collisions": 0.0, "hypercollisions": 0.0},
            "relative_skip_errors": {"collisions": 0.0, "hypercollisions": 0.0},
        },
        {
            "state": "z_wave",
            "term_norms": {"collisions": 0.0, "hypercollisions": 2.0e-4},
            "relative_skip_errors": {"collisions": 0.0, "hypercollisions": 3.0e-3},
        },
    ]

    payload = mod._build_summary(
        rows,
        config="examples/nonlinear/axisymmetric/runtime_cyclone_nonlinear.toml",
        ky=0.3,
        kx=None,
        nl=4,
        nm=8,
        identity_threshold=1.0e-10,
        activation_threshold=1.0e-8,
    )

    assert payload["kind"] == "linear_rhs_zero_norm_state_window_gate"
    assert payload["passed"] is True
    assert payload["candidates"]["collisions"]["safe_to_disable_over_window"] is True
    assert (
        payload["candidates"]["hypercollisions"]["safe_to_disable_over_window"] is False
    )
    assert payload["candidates"]["hypercollisions"]["active_states"] == ["z_wave"]


def test_selected_kbm_branch_candidate_rows_uses_only_selected_branch(
    tmp_path: Path,
) -> None:
    mod = load_artifact_tool("generate_kbm_branch_gate_summary")
    path = tmp_path / "candidates.csv"
    pd.DataFrame(
        {
            "ky": [0.3, 0.1, 0.2],
            "gamma": [5.0, 0.10, 0.11],
            "omega": [5.0, 1.00, 1.02],
            "eig_overlap_prev": [0.1, float("nan"), 0.99],
            "selected": ["false", "true", "yes"],
        }
    ).to_csv(path, index=False)

    rows = mod.selected_candidate_rows(path)

    assert [row["ky"] for row in rows] == [0.1, 0.2]
    assert [row["gamma"] for row in rows] == [0.10, 0.11]
    assert rows[0]["eig_overlap_prev"] is None


def test_generate_kbm_branch_gate_summary_main_writes_strict_json(
    tmp_path: Path,
) -> None:
    mod = load_artifact_tool("generate_kbm_branch_gate_summary")
    candidates = tmp_path / "candidates.csv"
    out = tmp_path / "summary.json"
    pd.DataFrame(
        {
            "ky": [0.1, 0.2, 0.3],
            "gamma": [0.10, 0.11, 0.12],
            "omega": [1.00, 1.02, 1.04],
            "eig_overlap_prev": [float("nan"), 0.99, 0.98],
            "selected": [True, True, True],
        }
    ).to_csv(candidates, index=False)

    assert mod.main(["--candidates", str(candidates), "--out", str(out)]) == 0

    payload = json.loads(out.read_text())
    assert payload["case"] == "kbm_linear_branch_continuity"
    assert payload["selected_count"] == 3
    assert payload["gate_passed"] is True
    assert payload["rows"][0]["eig_overlap_prev"] is None
    assert {gate["metric"] for gate in payload["gate_report"]["gates"]} == {
        "max_rel_gamma_jump",
        "max_rel_omega_jump",
        "successive_overlap_deficit",
    }


def test_selected_kbm_overlay_candidate_row_requires_selected_match(tmp_path) -> None:
    mod = load_artifact_tool("generate_kbm_reference_overlay")
    path = tmp_path / "candidates.csv"
    pd.DataFrame(
        {
            "ky": [0.3, 0.3],
            "selected": [False, True],
            "fit_window_tmin": [1.0, 2.0],
            "fit_window_tmax": [3.0, 4.0],
        }
    ).to_csv(path, index=False)

    row = mod._selected_candidate_row(path, 0.3)

    assert float(row["fit_window_tmin"]) == pytest.approx(2.0)
    assert float(row["fit_window_tmax"]) == pytest.approx(4.0)


def test_selected_kbm_overlay_candidate_row_rejects_missing_ky(tmp_path) -> None:
    mod = load_artifact_tool("generate_kbm_reference_overlay")
    path = tmp_path / "candidates.csv"
    pd.DataFrame(
        {
            "ky": [0.2],
            "selected": [True],
            "fit_window_tmin": [1.0],
            "fit_window_tmax": [3.0],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError):
        mod._selected_candidate_row(path, 0.3)


def test_steps_for_fit_window_respects_stride_alignment() -> None:
    mod = load_artifact_tool("generate_kbm_reference_overlay")

    steps = mod._steps_for_fit_window(
        fit_tmax=9.69, dt=0.01, fit_padding=0.5, sample_stride=2
    )

    assert steps == 1020
    assert steps % 2 == 0


def test_kbm_eigenfunction_gate_report_uses_strict_publication_thresholds() -> None:
    mod = load_artifact_tool("generate_kbm_reference_overlay")

    report = mod._kbm_eigenfunction_gate_report(
        EigenfunctionComparisonMetrics(overlap=0.63, relative_l2=0.79, phase_shift=0.0)
    )

    assert report.case == "kbm_linear_eigenfunction_ky0p3000"
    assert report.source == "GX raw eigenfunction bundle"
    assert report.passed is False
    assert mod.KBM_EIGENFUNCTION_GATE_TOLERANCES["min_overlap"] == 0.95
    assert mod.KBM_EIGENFUNCTION_GATE_TOLERANCES["max_relative_l2"] == 0.25


def test_load_convergence_series_from_resolution_column(tmp_path: Path) -> None:
    mod = load_artifact_tool("generate_observed_order_gate")
    path = tmp_path / "conv.csv"
    pd.DataFrame(
        {"N": [16, 4, 8], "error": [1.0 / 16**2, 1.0 / 4**2, 1.0 / 8**2]}
    ).to_csv(
        path,
        index=False,
    )

    h, err, rows = mod.load_convergence_series(
        path,
        step_column=None,
        resolution_column="N",
        error_column="error",
    )

    assert list(h) == pytest.approx([0.25, 0.125, 0.0625])
    assert list(err) == pytest.approx([1.0 / 4**2, 1.0 / 8**2, 1.0 / 16**2])
    assert rows[0]["step_source"] == "1/N"


def test_generate_observed_order_gate_main_writes_json_and_plot(tmp_path: Path) -> None:
    mod = load_artifact_tool("generate_observed_order_gate")
    csv_path = tmp_path / "conv.csv"
    out_json = tmp_path / "conv.json"
    out_png = tmp_path / "conv.png"
    pd.DataFrame({"h": [0.4, 0.2, 0.1], "err": [0.16, 0.04, 0.01]}).to_csv(
        csv_path,
        index=False,
    )

    assert (
        mod.main(
            [
                "--csv",
                str(csv_path),
                "--step-column",
                "h",
                "--error-column",
                "err",
                "--min-order",
                "1.9",
                "--max-final-error",
                "0.02",
                "--out-json",
                str(out_json),
                "--out-png",
                str(out_png),
            ]
        )
        == 0
    )

    payload = json.loads(out_json.read_text())
    assert payload["gate_passed"] is True
    assert payload["asymptotic_order"] == pytest.approx(2.0)
    assert out_png.exists()
    assert out_png.with_suffix(".pdf").exists()


def test_load_convergence_series_requires_one_step_source(tmp_path: Path) -> None:
    mod = load_artifact_tool("generate_observed_order_gate")
    path = tmp_path / "conv.csv"
    pd.DataFrame({"h": [0.1, 0.05], "N": [10, 20], "err": [0.01, 0.0025]}).to_csv(
        path,
        index=False,
    )

    with pytest.raises(ValueError, match="exactly one"):
        mod.load_convergence_series(
            path, step_column="h", resolution_column="N", error_column="err"
        )


def _write_validation_gate(path: Path, *, case: str, passed: bool) -> None:
    path.write_text(
        json.dumps(
            {
                "gate_report": {
                    "case": case,
                    "source": "synthetic",
                    "passed": passed,
                    "max_abs_error": 0.1,
                    "max_rel_error": 0.2,
                    "gates": [
                        {
                            "metric": "metric_a",
                            "passed": passed,
                        }
                    ],
                }
            }
        )
    )


def test_collect_gate_entries_reads_top_level_gate_report(tmp_path: Path) -> None:
    mod = load_artifact_tool("make_validation_gate_index")
    old_root = mod.REPO_ROOT
    mod.REPO_ROOT = tmp_path
    _write_validation_gate(tmp_path / "pass.json", case="passed_case", passed=True)
    nested = tmp_path / "nested"
    nested.mkdir()
    _write_validation_gate(
        nested / "nested_pass.json", case="nested_passed_case", passed=True
    )
    _write_validation_gate(tmp_path / "open.json", case="open_case", passed=False)
    (tmp_path / "ignored.json").write_text(json.dumps({"case": "no_gate"}))
    (tmp_path / "promotion.json").write_text(
        json.dumps(
            {
                "case": "promotion_case",
                "kind": "synthetic_promotion_gate",
                "gate_index_include": True,
                "promotion_gate": {
                    "passed": True,
                    "gates": [{"metric": "admission", "passed": True}],
                },
            }
        )
    )
    (tmp_path / "exploratory.json").write_text(
        json.dumps(
            {
                "gate_index_include": False,
                "gate_report": {
                    "case": "exploratory_case",
                    "source": "synthetic",
                    "passed": False,
                    "gates": [{"metric": "metric_a", "passed": False}],
                },
            }
        )
    )

    try:
        index = mod.build_index([str(tmp_path / "**" / "*.json")])
    finally:
        mod.REPO_ROOT = old_root

    assert index["n_reports"] == 4
    assert index["n_passed"] == 3
    assert index["n_open"] == 1
    assert index["patterns"] == ["**/*.json"]
    rows = {row["case"]: row for row in index["reports"]}
    assert rows["open_case"]["failed_metrics"] == "metric_a"
    assert rows["open_case"]["artifact"] == "open.json"
    assert rows["passed_case"]["n_failed"] == 0
    assert rows["passed_case"]["artifact"] == "pass.json"
    assert rows["promotion_case"]["n_failed"] == 0
    assert rows["promotion_case"]["artifact"] == "promotion.json"
    assert rows["nested_passed_case"]["n_failed"] == 0
    assert rows["nested_passed_case"]["artifact"] == "nested/nested_pass.json"
    assert "exploratory_case" not in rows


def test_make_validation_gate_index_main_writes_json_csv_and_plot(
    tmp_path: Path,
) -> None:
    mod = load_artifact_tool("make_validation_gate_index")
    _write_validation_gate(tmp_path / "gate.json", case="case_a", passed=True)
    out_json = tmp_path / "index.json"
    out_csv = tmp_path / "index.csv"
    out_png = tmp_path / "index.png"

    assert (
        mod.main(
            [
                "--glob",
                str(tmp_path / "*.json"),
                "--out-json",
                str(out_json),
                "--out-csv",
                str(out_csv),
                "--out-png",
                str(out_png),
            ]
        )
        == 0
    )

    payload = json.loads(out_json.read_text())
    assert payload["n_reports"] == 1
    assert out_csv.exists()
    assert out_png.exists()
    assert out_png.with_suffix(".pdf").exists()
