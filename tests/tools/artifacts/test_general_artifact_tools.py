"""Artifact maintainer tool contracts: general artifact tools."""

from __future__ import annotations


# ---- test_artifact_plot_smoke.py ----

import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import matplotlib
from PIL import Image

matplotlib.use("Agg")
import numpy as np

from support.paths import load_artifact_tool, load_comparison_tool, load_release_tool
from spectraxgk.diagnostics.modes import save_eigenfunction_reference_bundle
from spectraxgk.workflows.runtime.toml import load_runtime_from_toml
from tools.artifacts.make_benchmark_atlas import (
    _atlas_manifest_path,
    _build_convergence_gate_reports,
    _load_manifest,
    _resolve_asset_paths,
)


ROOT = Path(__file__).resolve().parents[3]


def _csv_columns(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        return set(next(reader))


def test_plot_scaling_panels_independent_ky_defaults_and_rows(tmp_path: Path) -> None:
    mod = load_artifact_tool("plot_scaling_panels")

    args = mod.build_independent_ky_parser().parse_args([])
    assert args.inputs == mod.DEFAULT_INDEPENDENT_KY_INPUTS
    assert args.out_prefix == mod.DEFAULT_INDEPENDENT_KY_PREFIX

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

    summary = mod.load_independent_ky_summary([path])

    assert summary["identity_passed"] is True
    assert summary["rows"][0]["backend"] == "cpu"
    assert summary["rows"][0]["grid_label"] == "Nx=1, Ny=128, Nz=96, Nl=4, Nm=8"


def test_eigenfunction_diagnostics_tool_writes_summary_and_overlay(
    tmp_path: Path,
) -> None:
    mod = load_artifact_tool("generate_linear_reference_overlays")

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
        mod.main(
            ["overlap-summary", "--csv", str(summary_csv), "--out", str(summary_png)]
        )
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
        + "\n".join(f"{z},{value.real},{value.imag}" for z, value in zip(theta, mode))
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


def test_compress_docs_previews_skips_release_manifest_paths(tmp_path: Path) -> None:
    mod = load_release_tool("check_repository_size_manifest")
    static = tmp_path / "docs" / "_static"
    static.mkdir(parents=True)
    keep = static / "keep.png"
    trim = static / "trim.png"
    Image.new("RGBA", (128, 64), (255, 255, 255, 255)).save(keep)
    Image.new("RGBA", (128, 64), (200, 220, 255, 255)).save(trim)
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        f'''
[[artifacts]]
path = "{keep.as_posix()}"
'''.strip()
        + "\n",
        encoding="utf-8",
    )

    reports = mod.compress_docs_previews(
        static_dir=static,
        manifest=manifest,
        min_bytes=1,
        max_width=32,
        colors=8,
    )

    by_name = {Path(row["path"]).name: row for row in reports}
    assert by_name["keep.png"]["skipped"] is True
    assert by_name["trim.png"]["skipped"] is False
    assert Image.open(keep).size == (128, 64)
    assert Image.open(trim).size == (32, 16)


def test_compress_png_preview_dry_run_does_not_modify_file(tmp_path: Path) -> None:
    mod = load_release_tool("check_repository_size_manifest")
    path = tmp_path / "panel.png"
    Image.new("RGBA", (64, 32), (255, 255, 255, 255)).save(path)
    before = path.read_bytes()

    report = mod.compress_png_preview(path, max_width=16, colors=8, dry_run=True)

    assert report["dry_run"] is True
    assert report["saved_bytes"] == 0
    assert path.read_bytes() == before


def test_release_preview_targets_and_compression_use_manifest(tmp_path: Path) -> None:
    mod = load_release_tool("check_repository_size_manifest")
    panel = tmp_path / "panel.png"
    ignored = tmp_path / "ignored.png"
    Image.new("RGBA", (64, 32), (255, 255, 255, 255)).save(panel)
    Image.new("RGBA", (64, 32), (0, 0, 0, 255)).save(ignored)
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        f'''
[[artifacts]]
path = "{panel.as_posix()}"
action = "keep_preview_in_repo"
preview_strategy = "compressed_preview"

[[artifacts]]
path = "{ignored.as_posix()}"
action = "keep_raw"
preview_strategy = "none"
'''.strip()
        + "\n",
        encoding="utf-8",
    )

    reports = mod.compress_release_previews(
        manifest=manifest,
        max_width=16,
        colors=8,
    )

    assert [Path(row["path"]).name for row in reports] == ["panel.png"]
    report = reports[0]
    assert report["original_dimensions"] == (64, 32)
    assert report["preview_dimensions"] == (16, 8)
    assert report["after_size_bytes"] > 0
    assert report["after_sha256"] != report["before_sha256"]
    assert Image.open(panel).size == (16, 8)
    assert Image.open(ignored).size == (64, 32)


def test_compress_previews_cli_supports_docs_and_release_modes(
    tmp_path: Path, capsys
) -> None:
    mod = load_release_tool("check_repository_size_manifest")
    static = tmp_path / "docs" / "_static"
    static.mkdir(parents=True)
    panel = static / "panel.png"
    Image.new("RGBA", (64, 32), (255, 255, 255, 255)).save(panel)
    manifest = tmp_path / "manifest.toml"
    manifest.write_text(
        f'''
[[artifacts]]
path = "{panel.as_posix()}"
action = "keep_preview_in_repo"
preview_strategy = "preview"
'''.strip()
        + "\n",
        encoding="utf-8",
    )

    assert (
        mod.main(
            [
                "compress-previews",
                "--mode",
                "docs",
                "--static-dir",
                str(static),
                "--manifest",
                str(manifest),
                "--min-bytes",
                "1",
                "--dry-run",
            ]
        )
        == 0
    )
    assert "total_saved=0" in capsys.readouterr().out
    assert (
        mod.main(
            [
                "compress-previews",
                "--mode",
                "release",
                "--manifest",
                str(manifest),
                "--max-width",
                "32",
                "--dry-run",
            ]
        )
        == 0
    )
    assert "panel.png" in capsys.readouterr().out


def test_benchmark_atlas_manifest_resolves_tracked_assets() -> None:
    manifest = _load_manifest(_atlas_manifest_path())
    assets = _resolve_asset_paths(manifest)

    assert set(assets) == {
        "imported_linear",
        "extended_linear",
        "core_linear",
        "core_nonlinear",
        "convergence",
    }
    assert (
        assets["imported_linear"]["w7x"]
        == ROOT / "docs" / "_static" / "w7x_linear_t2_scan.csv"
    )
    assert (
        assets["extended_linear"]["miller"]
        == ROOT / "docs" / "_static" / "kbm_miller_exact_growth_dump.csv"
    )
    assert assets["core_linear"]["cyclone"].name == "cyclone_mismatch_table.csv"
    assert assets["core_linear"]["kaw"].name == "kaw_exact_growth_dump.csv"
    assert assets["core_nonlinear"]["hsx"].name == "hsx_nonlinear_compare_t50_true.png"
    assert (
        assets["core_nonlinear"]["kbm"].name
        == "nonlinear_kbm_diag_compare_t400_stats.png"
    )
    assert set(assets["core_nonlinear"]) == {"cyclone", "kbm", "w7x", "hsx", "miller"}
    assert assets["convergence"]["cyclone_scan"].name == "cyclone_scan_convergence.csv"


def test_benchmark_atlas_manifest_uses_published_static_assets_and_plot_schemas() -> (
    None
):
    manifest = _load_manifest(_atlas_manifest_path())
    assets = _resolve_asset_paths(manifest)
    static_root = ROOT / "docs" / "_static"

    for group_assets in assets.values():
        for path in group_assets.values():
            assert static_root in path.parents
            assert ROOT / "docs" / "_build" not in path.parents

    schemas = {
        ("core_linear", "cyclone"): {
            "ky",
            "gamma_ref",
            "omega_ref",
            "gamma_spectrax",
            "omega_spectrax",
        },
        ("core_linear", "etg"): {
            "ky",
            "gamma_ref",
            "omega_ref",
            "gamma_spectrax",
            "omega_spectrax",
        },
        ("core_linear", "kbm"): {
            "ky",
            "gamma_ref",
            "omega_ref",
            "gamma_spectrax",
            "omega_spectrax",
        },
        ("core_linear", "w7x"): {
            "ky",
            "gamma_ref_last",
            "gamma_last",
            "omega_ref_last",
            "omega_last",
        },
        ("core_linear", "hsx"): {
            "ky",
            "gamma_ref_last",
            "gamma_last",
            "omega_ref_last",
            "omega_last",
        },
        ("core_linear", "miller"): {"ky", "gamma_gx", "gamma", "omega_gx", "omega"},
        (
            "core_linear",
            "kaw",
        ): {
            "ky",
            "gamma_ref",
            "gamma_spectrax",
            "omega_ref",
            "omega_spectrax",
            "rel_free_energy",
            "rel_electrostatic_energy",
            "rel_magnetic_energy",
        },
        (
            "core_linear",
            "kbm_miller",
        ): {
            "ky",
            "gamma_gx_dump",
            "gamma_sp_dump",
            "omega_gx_dump",
            "omega_sp_dump",
            "rel_gamma_sp_vs_gx_dump",
            "rel_omega_sp_vs_gx_dump",
        },
        ("extended_linear", "kinetic"): {
            "ky",
            "gamma_ref",
            "omega_ref",
            "gamma_spectrax",
            "omega_spectrax",
        },
        ("extended_linear", "tem"): {
            "ky",
            "gamma_ref",
            "omega_ref",
            "gamma_spectrax",
            "omega_spectrax",
        },
        ("convergence", "cyclone_scan"): {"ky", "rel_gamma_change", "rel_omega_change"},
        ("convergence", "cyclone_rhostar"): {
            "rho_star",
            "mean_gamma_ratio",
            "mean_omega_ratio",
        },
    }

    for (group, name), required in schemas.items():
        assert required <= _csv_columns(assets[group][name])


def test_benchmark_atlas_convergence_gate_report(tmp_path: Path) -> None:
    scan = tmp_path / "cyclone_scan_convergence.csv"
    rho = tmp_path / "cyclone_rhostar_convergence.csv"
    scan.write_text(
        "ky,gamma_low,gamma_high,omega_low,omega_high,rel_gamma_change,rel_omega_change\n"
        "0.3,1.0,1.01,2.0,2.02,0.01,0.01\n"
        "0.4,1.0,1.03,2.0,2.04,0.03,0.02\n",
        encoding="utf-8",
    )
    rho.write_text(
        "rho_star,mean_gamma_ratio,mean_omega_ratio\n1.0,1.0,1.0\n",
        encoding="utf-8",
    )

    reports = _build_convergence_gate_reports(
        {"cyclone_scan": scan, "cyclone_rhostar": rho},
        max_rel_change=0.05,
    )

    report = reports["cyclone_resolution_convergence"]
    assert report["passed"] is True
    assert report["case"] == "cyclone_resolution_convergence"
    assert report["source"] == "tracked high-vs-low production grid"
    assert {gate["metric"] for gate in report["gates"]} == {
        "max_rel_gamma_change",
        "max_rel_omega_change",
    }
    assert {gate["reference"] for gate in report["gates"]} == {0.0}
    assert {gate["rtol"] for gate in report["gates"]} == {0.0}
    assert {gate["atol"] for gate in report["gates"]} == {0.05}


def test_benchmark_atlas_convergence_gate_threshold_is_inclusive(
    tmp_path: Path,
) -> None:
    scan = tmp_path / "cyclone_scan_convergence.csv"
    rho = tmp_path / "cyclone_rhostar_convergence.csv"
    rho.write_text(
        "rho_star,mean_gamma_ratio,mean_omega_ratio\n1.0,1.0,1.0\n",
        encoding="utf-8",
    )

    scan.write_text(
        "ky,gamma_low,gamma_high,omega_low,omega_high,rel_gamma_change,rel_omega_change\n"
        "0.3,1.0,1.05,2.0,2.10,0.05,0.05\n",
        encoding="utf-8",
    )
    reports = _build_convergence_gate_reports(
        {"cyclone_scan": scan, "cyclone_rhostar": rho},
        max_rel_change=0.05,
    )
    assert reports["cyclone_resolution_convergence"]["passed"] is True

    scan.write_text(
        "ky,gamma_low,gamma_high,omega_low,omega_high,rel_gamma_change,rel_omega_change\n"
        "0.3,1.0,1.050001,2.0,2.10,0.050001,0.05\n",
        encoding="utf-8",
    )
    reports = _build_convergence_gate_reports(
        {"cyclone_scan": scan, "cyclone_rhostar": rho},
        max_rel_change=0.05,
    )
    assert reports["cyclone_resolution_convergence"]["passed"] is False


def test_error_anatomy_report_and_cli_lock_fail_closed_residual_story(
    tmp_path: Path,
) -> None:
    mod = load_artifact_tool("plot_quasilinear_model_development")

    report = mod.build_error_anatomy_report()

    assert report["kind"] == "quasilinear_error_anatomy"
    assert (
        report["claim_level"]
        == "model_development_residual_anatomy_not_absolute_flux_promotion"
    )
    assert report["case_count"] == 12
    assert report["holdout_count"] == 10
    assert report["promotion_gate"]["passed"] is False
    assert (
        "case_residuals_exceed_transport_gate" in report["promotion_gate"]["blockers"]
    )
    assert 0.697 < report["candidate_mean_abs_relative_error"] < 0.698
    assert (
        report["rows"][0]["case"] == "solovev_reference_repair_dt002_amp1em5_n48_t250"
    )
    assert report["rows"][0]["above_transport_gate"] is True
    assert report["rows"][0]["overpredicts"] is True
    groups = {row["geometry_group"]: row for row in report["geometry_group_summary"]}
    assert groups["external axisymmetric VMEC"]["error_budget_fraction"] > 0.82
    assert groups["stellarator benchmark"]["mean_abs_relative_error"] < 0.35
    assert (
        report["frozen_ledger_policy"]["additional_holdout_collection_active"] is False
    )
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
            str(root / "tools" / "artifacts" / "plot_quasilinear_model_development.py"),
            "error-anatomy",
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
    assert (
        payload["frozen_ledger_policy"]["additional_holdout_collection_active"] is False
    )
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


# ---- test_make_cyclone_assets.py ----

import jax.numpy as jnp


class _DummyFigure:
    def savefig(self, *_args, **_kwargs) -> None:
        return None


def test_make_tables_refresh_minimal_uses_reference_mismatch_scan(
    monkeypatch, tmp_path: Path
) -> None:
    import tools.artifacts.make_tables as make_tables

    ref = make_tables.LinearScanResult(
        ky=np.array([0.1, 0.2, 0.55]),
        gamma=np.array([0.3, 0.4, 0.5]),
        omega=np.array([0.5, 0.6, 0.7]),
    )
    called: dict[str, object] = {}

    def fake_reference_scan(scan_ref, cfg, *, verbose: bool, progress: bool):
        called["helper"] = "reference"
        called["Ny"] = cfg.grid.Ny
        assert np.allclose(scan_ref.ky, [0.1, 0.2])
        assert verbose is False
        assert progress is False
        return make_tables.LinearScanResult(
            ky=np.array([0.1, 0.2]),
            gamma=np.array([0.3, 0.4]),
            omega=np.array([0.5, 0.6]),
        )

    runtime_cfg, _ = make_tables.load_runtime_from_toml(
        make_tables.ROOT / "examples/linear/axisymmetric/cyclone.toml"
    )
    monkeypatch.setattr(make_tables, "ROOT", tmp_path)
    monkeypatch.setattr(
        make_tables,
        "load_runtime_from_toml",
        lambda _path: (runtime_cfg, {}),
    )
    monkeypatch.setattr(make_tables, "load_cyclone_reference", lambda: ref)
    monkeypatch.setattr(
        make_tables, "_cyclone_reference_mismatch_scan", fake_reference_scan
    )
    monkeypatch.setattr(
        make_tables,
        "_build_rows",
        lambda scan, ref_scan: [
            "ky,gamma,omega",
            f"{scan.ky[0]},{scan.gamma[0]},{scan.omega[0]}",
        ],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "make_tables.py",
            "--case",
            "cyclone",
            "--refresh-minimal",
            "--no-progress",
            "--quiet",
        ],
    )

    assert make_tables.main() == 0
    assert called == {"helper": "reference", "Ny": 24}
    assert (tmp_path / "docs" / "_static" / "cyclone_mismatch_table.csv").exists()


def test_linear_artifacts_cyclone_uses_reviewed_mismatch_table(
    monkeypatch, tmp_path: Path
) -> None:
    import tools.artifacts.build_linear_validation_artifacts as linear_artifacts
    import spectraxgk.artifacts.plotting as plotting
    import spectraxgk.benchmarking.shared as benchmark_data
    from spectraxgk.benchmarking.shared import LinearScanResult

    ref = LinearScanResult(
        ky=np.array([0.1, 0.2, 0.55]),
        gamma=np.array([0.3, 0.4, 0.5]),
        omega=np.array([0.5, 0.6, 0.7]),
    )
    outdir = tmp_path / "docs" / "_static"
    outdir.mkdir(parents=True)
    (outdir / "cyclone_mismatch_table.csv").write_text(
        "ky,gamma_spectrax,omega_spectrax\n0.1,0.31,0.51\n0.2,0.41,0.61\n",
        encoding="utf-8",
    )
    called: dict[str, np.ndarray] = {}
    monkeypatch.setattr(linear_artifacts, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(benchmark_data, "load_cyclone_reference", lambda: ref)
    monkeypatch.setattr(
        plotting,
        "cyclone_reference_figure",
        lambda _ref: (_DummyFigure(), None),
    )

    def fake_comparison(_ref, scan):
        called["gamma"] = np.asarray(scan.gamma)
        return _DummyFigure(), None

    monkeypatch.setattr(
        plotting,
        "cyclone_comparison_figure",
        fake_comparison,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_linear_validation_artifacts.py",
            "figures",
            "--case",
            "cyclone",
            "--no-progress",
        ],
    )

    assert linear_artifacts.main() == 0
    assert np.allclose(called["gamma"], [0.31, 0.41])


def test_make_tables_reference_mismatch_scan_uses_runtime_contract(
    monkeypatch,
) -> None:
    import tools.artifacts.make_tables as make_tables

    ref = make_tables.LinearScanResult(
        ky=np.array([0.1, 0.2]),
        gamma=np.array([0.3, 0.4]),
        omega=np.array([0.5, 0.6]),
    )
    called: dict[str, object] = {}

    def fake_runtime_scan(cfg, ky_values, *, Nl, Nm, progress=False):
        called["ky"] = np.asarray(ky_values).copy()
        called["Ny"] = cfg.grid.Ny
        called["Nl"] = Nl
        called["Nm"] = Nm
        called["progress"] = progress
        return make_tables.LinearScanResult(
            ky=np.asarray(ky_values),
            gamma=np.array([1.0, 2.0]),
            omega=np.array([3.0, 4.0]),
        )

    monkeypatch.setattr(make_tables, "_runtime_cyclone_scan", fake_runtime_scan)
    cfg, _ = make_tables.load_runtime_from_toml(
        make_tables.ROOT / "examples/linear/axisymmetric/cyclone.toml"
    )

    out = make_tables._cyclone_reference_mismatch_scan(
        ref,
        cfg,
        verbose=False,
        progress=False,
    )

    assert np.allclose(called["ky"], ref.ky)
    assert called["Ny"] == 24
    assert called["Nl"] == make_tables.CYCLONE_PUBLIC_NL
    assert called["Nm"] == make_tables.CYCLONE_PUBLIC_NM
    assert called["progress"] is False
    assert np.allclose(out.gamma, [1.0, 2.0])


def test_cyclone_runtime_scan_extends_only_slow_low_ky_modes(monkeypatch) -> None:
    import tools.artifacts.make_tables as make_tables

    calls: list[tuple[np.ndarray, dict[str, object]]] = []

    def fake_scan(_cfg, ky_values, **kwargs):
        ky = np.asarray(ky_values, dtype=float)
        calls.append((ky.copy(), kwargs))
        return make_tables.LinearScanResult(ky=ky, gamma=ky + 1.0, omega=ky + 2.0)

    monkeypatch.setattr(make_tables, "run_runtime_scan", fake_scan)
    cfg, _ = make_tables.load_runtime_from_toml(
        make_tables.ROOT / "examples/linear/axisymmetric/cyclone.toml"
    )
    out = make_tables._runtime_cyclone_scan(
        cfg,
        np.array([0.2, 0.05, 0.3, 0.1]),
        Nl=16,
        Nm=48,
    )

    assert np.allclose(calls[0][0], [0.05, 0.1])
    assert calls[0][1]["steps"] == 17160
    assert np.allclose(calls[1][0], [0.2, 0.3])
    assert calls[1][1]["steps"] == 8580
    assert all(call[1]["method"] == "rk4" for call in calls)
    assert np.allclose(out.gamma, [1.2, 1.05, 1.3, 1.1])


def test_cyclone_low_ky_gx_policy_extends_runtime_and_late_window() -> None:
    import tools.artifacts.make_tables as make_tables

    Nl, Nm, tmax = make_tables._gx_balanced_policy(0.05)
    assert (Nl, Nm, tmax) == (16, 8, 320.0)

    window = make_tables._gx_window_policy(0.05, make_tables.REFERENCE_CYCLONE_WINDOW)
    assert window["start_fraction"] == 0.65
    assert window["end_fraction"] == 0.95
    assert window["late_penalty"] == 0.0


def test_make_tables_etg_reference_mismatch_scan_uses_runtime_contract(
    monkeypatch,
) -> None:
    import tools.artifacts.make_tables as make_tables

    ref = make_tables.LinearScanResult(
        ky=np.array([10.0, 20.0]),
        gamma=np.array([1.0, 2.0]),
        omega=np.array([3.0, 4.0]),
    )
    called: dict[str, object] = {}

    def fake_run_runtime_scan(cfg, ky_values, **kwargs):
        called["cfg"] = cfg
        called["ky"] = np.asarray(ky_values)
        called.update(kwargs)
        return make_tables.LinearScanResult(
            ky=np.asarray(ky_values),
            gamma=np.array([15.0, 25.0]),
            omega=np.array([17.0, 27.0]),
        )

    monkeypatch.setattr(make_tables, "run_runtime_scan", fake_run_runtime_scan)
    cfg = make_tables._etg_runtime_case()

    out = make_tables._etg_reference_mismatch_scan(
        ref,
        cfg,
        verbose=False,
        progress=False,
    )

    assert np.allclose(called["ky"], ref.ky)
    assert called["dt"] == pytest.approx(1.6e-4)
    assert called["steps"] == 12500
    assert called["batch_ky"] is True
    assert called["method"] == "rk4"
    assert np.allclose(out.gamma, [15.0, 25.0])
    assert np.allclose(out.omega, [17.0, 27.0])


def test_run_etg_tables_uses_tracked_mismatch_helper(monkeypatch, tmp_path) -> None:
    import tools.artifacts.make_tables as make_tables

    called: dict[str, object] = {}

    def fake_run_runtime_linear(*args, **kwargs):
        return type("Res", (), {"gamma": 1.0, "omega": -2.0})()

    def fake_load_etg_reference():
        return make_tables.LinearScanResult(
            ky=np.array([10.0, 20.0]),
            gamma=np.array([1.0, 2.0]),
            omega=np.array([3.0, 4.0]),
        )

    def fake_etg_reference_mismatch_scan(ref, cfg, *, verbose, progress):
        called["ref"] = ref
        called["cfg"] = cfg
        called["verbose"] = verbose
        called["progress"] = progress
        return make_tables.LinearScanResult(
            ky=np.array([10.0, 20.0]),
            gamma=np.array([5.0, 6.0]),
            omega=np.array([7.0, 8.0]),
        )

    monkeypatch.setattr(make_tables, "run_runtime_linear", fake_run_runtime_linear)
    monkeypatch.setattr(make_tables, "load_etg_reference", fake_load_etg_reference)
    monkeypatch.setattr(
        make_tables,
        "_etg_reference_mismatch_scan",
        fake_etg_reference_mismatch_scan,
    )

    make_tables._run_etg_tables(outdir=tmp_path, verbose=False, progress=False)

    assert called["verbose"] is False
    assert called["progress"] is False
    cfg = called["cfg"]
    assert cfg.physics.adiabatic_ions is True
    assert cfg.physics.adiabatic_electrons is False
    assert len(cfg.species) == 1
    assert cfg.species[0].charge < 0.0
    assert (tmp_path / "etg_mismatch_table.csv").exists()


def test_run_etg_figures_uses_tracked_case(monkeypatch, tmp_path: Path) -> None:
    import tools.artifacts.build_linear_validation_artifacts as linear_artifacts
    import spectraxgk.artifacts.plotting as plotting
    import spectraxgk.benchmarking.shared as benchmark_data
    import spectraxgk.runtime as runtime_module
    from spectraxgk.benchmarking.shared import LinearScanResult

    called: dict[str, object] = {}

    def fake_run_runtime_scan(cfg, ky_values, **kwargs):
        called["ky_values"] = np.asarray(ky_values).copy()
        called["cfg"] = cfg
        called.update(kwargs)
        return LinearScanResult(
            ky=np.asarray(ky_values),
            gamma=np.array([1.0, 2.0, 3.0]),
            omega=np.array([-4.0, -5.0, -6.0]),
        )

    monkeypatch.setattr(
        benchmark_data,
        "load_etg_reference",
        lambda: LinearScanResult(
            ky=np.array([10.0, 20.0, 30.0]),
            gamma=np.array([1.0, 2.0, 3.0]),
            omega=np.array([-4.0, -5.0, -6.0]),
        ),
    )
    monkeypatch.setattr(runtime_module, "run_runtime_scan", fake_run_runtime_scan)
    monkeypatch.setattr(
        plotting,
        "scan_comparison_figure",
        lambda *args, **kwargs: (_DummyFigure(), None),
    )

    linear_artifacts._run_etg_figures(outdir=tmp_path, progress=False)

    cfg = called["cfg"]
    assert np.allclose(called["ky_values"], [10.0, 20.0, 30.0])
    assert called["batch_ky"] is True
    assert called["method"] == "rk4"
    assert cfg.physics.adiabatic_ions is True
    assert len(cfg.species) == 1


def test_run_etg_figures_prefers_existing_mismatch_csv(
    monkeypatch, tmp_path: Path
) -> None:
    import tools.artifacts.build_linear_validation_artifacts as linear_artifacts
    import spectraxgk.artifacts.plotting as plotting
    import spectraxgk.benchmarking.shared as benchmark_data
    import spectraxgk.runtime as runtime_module
    from spectraxgk.benchmarking.shared import LinearScanResult

    mismatch = tmp_path / "etg_mismatch_table.csv"
    mismatch.write_text(
        "ky,gamma_ref,omega_ref,gamma_spectrax,omega_spectrax,rel_gamma,rel_omega\n"
        "10,1,2,3,4,0,0\n"
        "20,5,6,7,8,0,0\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        benchmark_data,
        "load_etg_reference",
        lambda: LinearScanResult(
            ky=np.array([10.0, 20.0]),
            gamma=np.array([1.0, 5.0]),
            omega=np.array([2.0, 6.0]),
        ),
    )
    monkeypatch.setattr(
        runtime_module,
        "run_runtime_scan",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("should reuse mismatch csv")
        ),
    )
    monkeypatch.setattr(
        plotting,
        "scan_comparison_figure",
        lambda *args, **kwargs: (_DummyFigure(), None),
    )

    linear_artifacts._run_etg_figures(outdir=tmp_path, progress=False)


def test_make_tables_cyclone_reference_scan_falls_back_from_project_to_max(
    monkeypatch,
) -> None:
    import tools.artifacts.make_tables as make_tables

    cfg = make_tables.CycloneBaseCase()

    monkeypatch.setattr(
        make_tables,
        "SAlphaGeometry",
        type(
            "Geom",
            (),
            {
                "from_config": staticmethod(
                    lambda _cfg: type("G", (), {"gradpar": lambda self: 1.0})()
                )
            },
        ),
    )
    monkeypatch.setattr(
        make_tables,
        "build_spectral_grid",
        lambda _grid: type(
            "Grid", (), {"ky": np.array([0.4]), "z": np.array([0.0, 1.0, 2.0])}
        )(),
    )
    monkeypatch.setattr(make_tables, "select_ky_index", lambda _ky, _val: 0)
    monkeypatch.setattr(make_tables, "select_ky_grid", lambda grid, _idx: grid)
    monkeypatch.setattr(
        make_tables,
        "_build_initial_condition",
        lambda *args, **kwargs: jnp.zeros((2, 2), dtype=jnp.complex64),
    )
    monkeypatch.setattr(
        make_tables, "build_linear_cache", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        make_tables,
        "integrate_linear_explicit",
        lambda *args, **kwargs: (
            np.array([0.0, 1.0]),
            np.ones((2, 1, 1, 3), dtype=np.complex128),
            None,
            None,
        ),
    )
    methods: list[str] = []

    def fake_extract(_phi_t, _sel, *, method: str):
        methods.append(method)
        if method == "project":
            return np.array([1.0e-12 + 0.0j, 1.0e-12 + 0.0j])
        return np.array([1.0 + 0.0j, np.exp((0.1 - 0.4j))])

    monkeypatch.setattr(make_tables, "extract_mode_time_series", fake_extract)

    def fake_fit(t, signal, **_kwargs):
        if np.max(np.abs(signal)) < 1.0e-6:
            return 1.0e-13, 0.0, float(t[0]), float(t[-1])
        return 0.1, 0.4, float(t[0]), float(t[-1])

    monkeypatch.setattr(make_tables, "fit_growth_rate_auto", fake_fit)

    scan = make_tables._cyclone_reference_scan(
        np.array([0.4]),
        cfg,
        make_tables.WINDOWS["cyclone"],
        verbose=False,
        progress=False,
    )

    assert methods == ["project", "max"]
    assert np.allclose(scan.gamma, [0.1])
    assert np.allclose(scan.omega, [0.4])


def test_kbm_public_rows_from_gx_mismatch_uses_gx_reference_columns(
    tmp_path: Path,
) -> None:
    import tools.artifacts.make_tables as make_tables

    csv_path = tmp_path / "kbm_reference_mismatch.csv"
    csv_path.write_text(
        "\n".join(
            [
                "ky,solver,gamma_gx,gamma,rel_gamma,omega_gx,omega,rel_omega,eig_overlap_gx,eig_rel_l2,eig_overlap_prev,branch_score,fit_window_tmin,fit_window_tmax",
                "0.3,gx_time@project,0.22,0.20,-0.09,1.14,1.27,0.11,0.98,0.15,0.99,0.19,5.0,10.0",
                "0.1,gx_time@project_late,0.14,0.13,-0.07,0.66,0.67,0.01,0.98,0.16,,0.13,7.0,11.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = make_tables._kbm_public_rows_from_gx_mismatch(csv_path)

    assert (
        rows[0]
        == "ky,gamma_ref,omega_ref,gamma_spectrax,omega_spectrax,rel_gamma,rel_omega"
    )
    assert rows[1].startswith("0.100,0.140000,0.660000,0.130000,0.670000,")
    assert rows[2].startswith("0.300,0.220000,1.140000,0.200000,1.270000,")


def test_kbm_public_rows_use_only_selected_continuity_branch(tmp_path: Path) -> None:
    import tools.artifacts.make_tables as make_tables

    csv_path = tmp_path / "kbm_reference_candidates.csv"
    csv_path.write_text(
        "\n".join(
            [
                "ky,solver,gamma_gx,gamma,rel_gamma,omega_gx,omega,rel_omega,selected",
                "0.2,early,0.30,0.36,0.20,0.88,0.89,0.01,False",
                "0.2,continuous,0.30,0.27,0.10,0.88,0.92,0.05,True",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = make_tables._kbm_public_rows_from_gx_mismatch(csv_path)

    assert len(rows) == 2
    assert rows[1].startswith("0.200,0.300000,0.880000,0.270000,0.920000,")


def test_kbm_public_rows_from_gx_mismatch_prefers_better_lowky_checkpoint(
    tmp_path: Path,
) -> None:
    import tools.artifacts.make_tables as make_tables

    csv_path = tmp_path / "kbm_reference_mismatch.csv"
    csv_path.write_text(
        "\n".join(
            [
                "ky,solver,gamma_gx,gamma,rel_gamma,omega_gx,omega,rel_omega,eig_overlap_gx,eig_rel_l2,eig_overlap_prev,branch_score,fit_window_tmin,fit_window_tmax",
                "0.2,gx_time@max,0.30,0.36,0.20,0.88,0.89,0.01,0.99,0.13,1.0,0.2,20.0,40.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    ckpt_path = tmp_path / "kbm_probe_lowky_ckpt.csv"
    ckpt_path.write_text(
        "\n".join(
            [
                "ky,steps,horizon_t,solver,gamma_gx,gamma,rel_gamma,omega_gx,omega,rel_omega,eig_overlap_gx,eig_rel_l2",
                "0.2,800,8.0,gx_time@project,0.30,0.3006,0.002,0.88,0.9988,0.135,0.98,0.16",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = make_tables._kbm_public_rows_from_gx_mismatch(
        csv_path, lowky_ckpt_path=ckpt_path
    )

    assert rows[1].startswith("0.200,0.300000,0.880000,0.300600,0.998800,")


def test_write_kbm_public_mismatch_table_prefers_gx_mismatch_when_present(
    tmp_path: Path,
) -> None:
    import tools.artifacts.make_tables as make_tables

    (tmp_path / "kbm_reference_mismatch.csv").write_text(
        "\n".join(
            [
                "ky,solver,gamma_gx,gamma,rel_gamma,omega_gx,omega,rel_omega,eig_overlap_gx,eig_rel_l2,eig_overlap_prev,branch_score,fit_window_tmin,fit_window_tmax",
                "0.2,gx_time@max,0.30,0.31,0.03,0.88,0.89,0.01,0.99,0.13,1.0,0.2,20.0,40.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    make_tables._write_kbm_public_mismatch_table(
        tmp_path,
        verbose=False,
        progress=False,
        stiff_spot_check=False,
        stiff_spot_topk=0,
        stiff_spot_dt=0.01,
        stiff_spot_tmax=1.0,
        stiff_spot_replace=False,
    )

    table_text = (tmp_path / "kbm_mismatch_table.csv").read_text(encoding="utf-8")
    assert "0.200,0.300000,0.880000,0.310000,0.890000" in table_text


def test_write_kbm_public_mismatch_table_requires_reviewed_provenance(
    tmp_path: Path,
) -> None:
    import tools.artifacts.make_tables as make_tables

    with pytest.raises(FileNotFoundError, match="compare_gx_kbm.py"):
        make_tables._write_kbm_public_mismatch_table(
            tmp_path,
            verbose=False,
            progress=False,
            stiff_spot_check=False,
            stiff_spot_topk=0,
            stiff_spot_dt=0.01,
            stiff_spot_tmax=1.0,
            stiff_spot_replace=False,
        )


def test_run_tem_tables_restores_fixed_late_window_contract(
    monkeypatch, tmp_path: Path
) -> None:
    import tools.artifacts.make_tables as make_tables

    called: dict[str, object] = {}

    def fake_load_tem_reference():
        return make_tables.LinearScanResult(
            ky=np.array([0.3]),
            gamma=np.array([2.0]),
            omega=np.array([1.0]),
        )

    def fake_scan_linear_verbose(**kwargs):
        called["dt"] = kwargs["dt"]
        called["steps"] = kwargs["steps"]
        called["tmin"] = kwargs["tmin"]
        called["tmax"] = kwargs["tmax"]
        called["auto_window"] = kwargs["auto_window"]
        called["run_kwargs"] = dict(kwargs["run_kwargs"])
        return (np.array([0.3]), np.array([2.1]), np.array([1.1]))

    monkeypatch.setattr(make_tables, "load_tem_reference", fake_load_tem_reference)
    monkeypatch.setattr(make_tables, "_scan_linear_verbose", fake_scan_linear_verbose)

    make_tables._run_tem_tables(outdir=tmp_path, verbose=False, progress=False)

    assert called["dt"] == 0.001
    assert called["steps"] == 2000
    assert called["tmin"] == 0.8
    assert called["tmax"] == 1.7
    assert called["auto_window"] is False
    assert called["run_kwargs"] == {"mode_method": "z_index"}
    assert (tmp_path / "tem_mismatch_table.csv").exists()


def test_run_kinetic_tables_restores_fixed_krylov_contract(
    monkeypatch, tmp_path: Path
) -> None:
    import tools.artifacts.make_tables as make_tables

    called: dict[str, object] = {}
    ky = np.array([0.2, 0.3], dtype=float)
    steps = make_tables._scale_steps(ky, base_steps=20000, ky_ref=0.3, max_steps=30000)
    dt = make_tables._scale_dt(ky, base_dt=0.0005, ky_ref=0.3)

    def fake_load_cyclone_reference_kinetic():
        return make_tables.LinearScanResult(
            ky=ky,
            gamma=np.array([0.1, 0.2]),
            omega=np.array([0.3, 0.4]),
        )

    def fake_scan_linear_verbose(**kwargs):
        called["cfg"] = kwargs["cfg"]
        called["solver"] = kwargs["solver"]
        called["krylov_cfg"] = kwargs["krylov_cfg"]
        called["tmin"] = np.asarray(kwargs["tmin"], dtype=float)
        called["tmax"] = np.asarray(kwargs["tmax"], dtype=float)
        called["auto_window"] = kwargs["auto_window"]
        return ky, np.array([0.11, 0.22]), np.array([0.33, 0.44])

    monkeypatch.setattr(
        make_tables,
        "load_cyclone_reference_kinetic",
        fake_load_cyclone_reference_kinetic,
    )
    monkeypatch.setattr(make_tables, "_scan_linear_verbose", fake_scan_linear_verbose)

    make_tables._run_kinetic_tables(
        outdir=tmp_path,
        verbose=False,
        progress=False,
        stiff_spot_check=False,
        stiff_spot_topk=0,
        stiff_spot_min_ky=0.0,
        stiff_spot_dt=0.0,
        stiff_spot_tmax=0.0,
        stiff_spot_replace=False,
    )

    assert called["solver"] == "krylov"
    assert called["krylov_cfg"] == make_tables.KINETIC_KRYLOV
    assert called["auto_window"] is False
    assert called["cfg"].grid.Ny == 2 * ky.size + 1
    assert np.allclose(called["tmin"], 0.6 * dt * steps)
    assert np.allclose(called["tmax"], 0.95 * dt * steps)
    assert (tmp_path / "kinetic_mismatch_table.csv").exists()


# ---- test_linear_validation_artifact_reports.py ----

import pandas as pd
from PIL import ImageDraw
import pytest

from spectraxgk.diagnostics.validation_gates import EigenfunctionComparisonMetrics


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
    mod = load_artifact_tool("build_tem_validation_artifacts")
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

    payload = mod.build_branch_audit_payload(table=table, reference=reference)
    metrics = payload["metrics"]

    assert payload["status"] == "open"
    assert metrics["gamma_sign_mismatch_count"] == 1
    assert metrics["omega_sign_mismatch_count"] == 2
    assert metrics["omega_branch_inversion"] is True
    assert metrics["max_abs_rel_gamma"] == 1.5
    assert metrics["max_abs_rel_omega_ref_ge_0p2"] == 3.0


def test_tem_branch_audit_writes_artifacts(tmp_path: Path) -> None:
    mod = load_artifact_tool("build_tem_validation_artifacts")
    table = tmp_path / "tem_mismatch.csv"
    table.write_text(
        "ky,gamma_ref,omega_ref,gamma_spectrax,omega_spectrax,rel_gamma,rel_omega\n"
        "0.2,1.0,2.0,1.1,1.8,0.1,-0.1\n"
        "0.3,1.5,1.0,1.4,0.8,-0.0666667,-0.2\n",
        encoding="utf-8",
    )

    payload = mod.build_branch_audit_payload(
        table=table, reference=tmp_path / "missing.csv"
    )
    paths = mod.write_branch_artifacts(payload, out_png=tmp_path / "tem_audit.png")

    for path in paths.values():
        assert Path(path).exists()
    written = json.loads((tmp_path / "tem_audit.json").read_text(encoding="utf-8"))
    assert written["kind"] == "tem_branch_parity_audit"
    assert written["reference"]["available"] is False


def test_build_lastvalue_table_converts_scan_columns() -> None:
    mod = load_artifact_tool("make_tables")
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
    mod = load_artifact_tool("make_tables")
    path = tmp_path / "scan.csv"
    pd.DataFrame({"ky": [0.1], "gamma_last": [0.2]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        mod._load_lastvalue_scan(path)


def test_make_tables_lastvalue_subcommand_writes_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = load_artifact_tool("make_tables")
    scan = tmp_path / "scan.csv"
    out = tmp_path / "lastvalue.csv"
    pd.DataFrame(
        {
            "ky": [0.1],
            "gamma_last": [0.032],
            "omega_last": [0.058],
            "gamma_ref_last": [0.031],
            "omega_ref_last": [0.059],
        }
    ).to_csv(scan, index=False)

    monkeypatch.setattr(
        "sys.argv",
        [
            "make_tables.py",
            "--lastvalue-scan",
            str(scan),
            "--lastvalue-out",
            str(out),
        ],
    )

    assert mod.main() == 0
    written = pd.read_csv(out)
    assert list(written.columns) == [
        "ky",
        "gamma",
        "omega",
        "gamma_gx",
        "omega_gx",
        "rel_gamma",
        "rel_omega",
    ]
    assert written.loc[0, "rel_omega"] == pytest.approx((0.058 - 0.059) / 0.059)


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
    mod = load_artifact_tool("build_w7x_zonal_reference_artifacts")
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
    mod = load_artifact_tool("build_w7x_zonal_reference_artifacts")
    image = np.asarray(_synthetic_w7x_reference_image(mod), dtype=np.uint8)

    trace_df, residual_df = mod.digitize_reference(image, samples_per_trace=11)

    assert set(trace_df["code"]) == {"stella", "GENE"}
    assert len(trace_df) == len(mod.PANEL_CALIBRATIONS) * 2 * 11
    medians = residual_df.set_index("code")["residual_median"].to_dict()
    assert medians["stella"] == pytest.approx(0.12, abs=5.0e-3)
    assert medians["GENE"] == pytest.approx(0.18, abs=5.0e-3)


def test_w7x_zonal_digitizer_main_writes_artifacts(tmp_path: Path) -> None:
    mod = load_artifact_tool("build_w7x_zonal_reference_artifacts")
    figure = tmp_path / "synthetic_zf.png"
    _synthetic_w7x_reference_image(mod).save(figure)
    out_csv = tmp_path / "trace.csv"
    out_residuals = tmp_path / "residuals.csv"
    out_json = tmp_path / "meta.json"
    out_png = tmp_path / "qa.png"

    rc = mod.main(
        [
            "digitize",
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
    mod = load_artifact_tool("generate_linear_rhs_parallel_gates")
    state = jnp.zeros((2, 4, 3, 2, 5), dtype=jnp.complex64)

    out = mod._inject_z_wave(state, ky_index=1, kx_index=0, amplitude=0.25, z_mode=1)

    assert jnp.linalg.norm(out[0, 3, 1, 0]) > 0.0
    assert jnp.linalg.norm(out.at[0, 3, 1, 0, :].set(0.0)) == 0.0


def test_build_summary_accepts_collision_skip_and_rejects_hypercollision_skip() -> None:
    mod = load_artifact_tool("generate_linear_rhs_parallel_gates")
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
    mod = load_artifact_tool("build_linear_validation_artifacts")
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
    mod = load_artifact_tool("build_linear_validation_artifacts")
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

    assert (
        mod.main(["kbm-branch", "--candidates", str(candidates), "--out", str(out)])
        == 0
    )

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


def test_kbm_branch_command_runs_with_source_only_pythonpath(tmp_path: Path) -> None:
    output = tmp_path / "kbm-branch.json"
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"

    completed = subprocess.run(
        [
            sys.executable,
            "tools/artifacts/build_linear_validation_artifacts.py",
            "kbm-branch",
            "--out",
            str(output),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(output.read_text(encoding="utf-8"))["gate_passed"] is True


def test_generate_collision_table_is_reproducible_and_matches_tracked_data(
    tmp_path: Path,
) -> None:
    mod = load_artifact_tool("build_linear_validation_artifacts")
    out = tmp_path / "collision.npy"
    metadata_out = tmp_path / "collision.json"

    assert (
        mod.main(
            [
                "collision-table",
                "--out",
                str(out),
                "--metadata-out",
                str(metadata_out),
                "--digits",
                "80",
            ]
        )
        == 0
    )
    metadata = json.loads(metadata_out.read_text())
    tracked = np.load(mod.DEFAULT_COLLISION_TABLE, allow_pickle=False)
    generated = np.load(out, allow_pickle=False)
    np.testing.assert_array_equal(generated, tracked)
    assert metadata["models"] == ["sugama", "improved_sugama", "coulomb"]
    assert metadata["shape"] == [3, 8, 8]
    assert metadata["sha256"] == mod.hashlib.sha256(out.read_bytes()).hexdigest()
    assert metadata["precision_decimal_digits"] == 80


def test_collision_artifact_import_does_not_initialize_jax() -> None:
    """Offline multiprecision generation must not probe CPU/GPU devices."""
    code = """
import importlib.util
import sys
spec = importlib.util.spec_from_file_location(
    'linear_artifacts', 'tools/artifacts/build_linear_validation_artifacts.py'
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
raise SystemExit(int('jax' in sys.modules))
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=env, timeout=30, check=False
    )
    assert completed.returncode == 0


def test_coulomb_speed_integrals_match_independent_quadrature() -> None:
    """Frei et al. Eqs. (A8a)--(A8b) must match their defining integrals."""
    from scipy.integrate import quad
    from scipy.special import erf

    mod = load_artifact_tool("build_linear_validation_artifacts")
    for chi in (0.35, 1.0, 2.2):
        e_values, E_values = mod.coulomb_speed_integrals(5, chi, digits=80)
        for order in range(6):
            e_reference = quad(
                lambda speed: (
                    speed ** (2 * order)
                    * 2.0
                    / np.sqrt(np.pi)
                    * np.exp(-((chi * speed) ** 2))
                    * np.exp(-(speed**2))
                ),
                0.0,
                np.inf,
                epsabs=1.0e-13,
            )[0]
            E_reference = quad(
                lambda speed: (
                    speed ** (2 * order + 1) * erf(chi * speed) * np.exp(-(speed**2))
                ),
                0.0,
                np.inf,
                epsabs=1.0e-13,
            )[0]
            np.testing.assert_allclose(e_values[order], e_reference, rtol=2.0e-12)
            np.testing.assert_allclose(E_values[order], E_reference, rtol=2.0e-12)

    for args, message in (((-1, 1.0), "max_order"), ((1, 0.0), "chi")):
        with pytest.raises(ValueError, match=message):
            mod.coulomb_speed_integrals(*args)
    with pytest.raises(ValueError, match="digits"):
        mod.coulomb_speed_integrals(1, 1.0, digits=10)


def test_coulomb_speed_moments_match_direct_maxwellian_quadrature() -> None:
    """Frei et al. Eqs. (A5) and (A13) must integrate Eqs. (A2) and (A10)."""
    from scipy.integrate import quad
    from scipy.special import erf, gamma, gammainc, gammaincc

    mod = load_artifact_tool("build_linear_validation_artifacts")
    inverse_sqrt_pi = 1.0 / np.sqrt(np.pi)

    def incomplete_gamma(order: int, argument: float, *, upper: bool) -> float:
        shape = 0.5 * (order + 1)
        regularized = gammaincc(shape, argument) if upper else gammainc(shape, argument)
        return float(gamma(shape) * regularized * inverse_sqrt_pi)

    def direct_speed_functions(
        speed: float,
        p: int,
        j: int,
        sigma: float,
        tau: float,
    ) -> tuple[float, float]:
        chi = np.sqrt(tau / sigma)
        species_speed = chi * speed
        gaussian = 2.0 * inverse_sqrt_pi * np.exp(-(species_speed**2))
        erf_over_speed = erf(species_speed) / species_speed
        laguerre = mod.associated_laguerre_monomial_coefficients(j, p)
        test = 0.0
        field = 0.0
        for monomial_order, laguerre_coefficient in enumerate(laguerre):
            common = (
                4 * monomial_order**2 + 4 * monomial_order * (p - 1) + 1.5 * p * (p - 1)
            )
            test_erf = (
                -4 * sigma * (tau - 1) / tau,
                4 * sigma / tau * monomial_order * (tau - 2)
                - p * (p - 2 * sigma + 4 * sigma / tau + 1),
                sigma / tau * common,
            )
            test_gaussian = (
                4 * (tau - 1) * (sigma + tau) / tau,
                -2 * (p + 2 * monomial_order) * (sigma / tau * (tau - 2) - 1),
                -sigma / tau * common,
            )
            test += (
                laguerre_coefficient
                * chi
                * speed ** (2 * monomial_order + p)
                * sum(
                    (
                        erf_over_speed * test_erf[summation_order]
                        + gaussian * test_gaussian[summation_order]
                    )
                    / speed ** (2 * summation_order)
                    for summation_order in range(3)
                )
            )

            field_prime = (
                4 * tau**2 / sigma,
                -8 * (p * p + p - 1) / ((2 * p - 1) * (2 * p + 3)),
            )
            field_plus = (
                -8 * p * (p - 1) * (monomial_order + 1) / ((2 * p + 1) * (2 * p - 1))
                - 8 * tau * (1 + p - sigma * p) / (sigma * (2 * p + 1)),
                8 * (p + 1) * (p + 2) / ((2 * p + 1) * (2 * p + 3)),
            )
            field_minus = (
                4
                * (p + 1)
                * (p + 2)
                * (2 * p + 2 * monomial_order + 3)
                / ((2 * p + 1) * (2 * p + 3))
                + 8 * chi**2 * (p - sigma * p - sigma) / (2 * p + 1),
                -8 * p * (p - 1) / ((2 * p + 1) * (2 * p - 1)),
            )
            field += (
                laguerre_coefficient
                / chi
                * sum(
                    species_speed ** (p + 2 * (monomial_order + summation_order))
                    * field_prime[summation_order]
                    * gaussian
                    + incomplete_gamma(
                        2 * monomial_order + 1,
                        species_speed**2,
                        upper=True,
                    )
                    * species_speed ** (p + 2 * summation_order)
                    * field_plus[summation_order]
                    + field_minus[summation_order]
                    * incomplete_gamma(
                        2 * (p + monomial_order + 1),
                        species_speed**2,
                        upper=False,
                    )
                    / species_speed ** (p + 1 - 2 * summation_order)
                    for summation_order in range(2)
                )
            )
        return float(test), float(field)

    cases = (
        (0, 0, 0, 1.0, 1.0),
        (1, 0, 0, 1.0, 1.0),
        (2, 1, 0, 1.0, 1.0),
        (1, 2, 1, 0.25, 2.0),
        (3, 1, 2, 4.0, 0.5),
        (0, 2, 2, 0.5, 3.0),
    )
    for p, j, d, sigma, tau in cases:
        generated = mod.coulomb_speed_moments(p, j, d, sigma, tau, digits=80)

        def integrand(speed: float, component: int) -> float:
            maxwellian_measure = 4 * inverse_sqrt_pi * speed**2 * np.exp(-(speed**2))
            return (
                maxwellian_measure
                * speed ** (p + 2 * d)
                * direct_speed_functions(speed, p, j, sigma, tau)[component]
            )

        projected = tuple(
            quad(
                lambda speed, component=component: integrand(speed, component),
                1.0e-6,
                12.0,
                epsabs=2.0e-10,
                epsrel=2.0e-10,
                limit=300,
            )[0]
            for component in range(2)
        )
        np.testing.assert_allclose(generated, projected, rtol=3.0e-11, atol=3.0e-10)

    density = mod.coulomb_speed_moments(0, 0, 0, 1.0, 1.0)
    momentum = mod.coulomb_speed_moments(1, 0, 0, 1.0, 1.0)
    np.testing.assert_allclose(density, 0.0, atol=2.0e-14)
    np.testing.assert_allclose(sum(momentum), 0.0, atol=2.0e-14)

    for args, message in (
        ((-1, 0, 0, 1.0, 1.0), "basis orders"),
        ((0, 0, 0, 0.0, 1.0), "mass_ratio"),
        ((0, 0, 0, 1.0, 0.0), "temperature_ratio"),
    ):
        with pytest.raises(ValueError, match=message):
            mod.coulomb_speed_moments(*args)
    with pytest.raises(ValueError, match="collision_frequency"):
        mod.coulomb_speed_moments(0, 0, 0, 1.0, 1.0, collision_frequency=-1.0)
    with pytest.raises(ValueError, match="digits"):
        mod.coulomb_speed_moments(0, 0, 0, 1.0, 1.0, digits=10)


def test_associated_laguerre_monomial_coefficients_reconstruct_polynomials() -> None:
    """Frei et al. Eq. (3.10) must reconstruct independent polynomials."""
    from scipy.special import eval_genlaguerre

    mod = load_artifact_tool("build_linear_validation_artifacts")
    x = np.linspace(0.0, 12.0, 61)
    for tensor_order in (0, 1, 4):
        for polynomial_order in (0, 3, 8):
            coefficients = mod.associated_laguerre_monomial_coefficients(
                polynomial_order,
                tensor_order,
                digits=80,
            )
            reconstructed = np.polynomial.polynomial.polyval(x, coefficients)
            expected = eval_genlaguerre(
                polynomial_order,
                tensor_order + 0.5,
                x,
            )
            np.testing.assert_allclose(
                reconstructed, expected, rtol=2.0e-10, atol=2.0e-8
            )

    for args, message in (((-1, 0), "polynomial_order"), ((1, -1), "tensor_order")):
        with pytest.raises(ValueError, match=message):
            mod.associated_laguerre_monomial_coefficients(*args)
    with pytest.raises(ValueError, match="digits"):
        mod.associated_laguerre_monomial_coefficients(1, 1, digits=10)


def test_legendre_hermite_laguerre_transform_matches_velocity_projection() -> None:
    """Jorge et al. Eqs. (A3)--(A4) must match quadrature and invert."""
    from scipy.special import (
        eval_genlaguerre,
        eval_hermite,
        eval_laguerre,
        eval_legendre,
    )

    mod = load_artifact_tool("build_linear_validation_artifacts")
    parallel, parallel_weights = np.polynomial.hermite.hermgauss(80)
    perpendicular, perpendicular_weights = np.polynomial.laguerre.laggauss(80)
    x_parallel = parallel[:, None]
    x_perpendicular = perpendicular[None, :]
    speed_squared = x_parallel**2 + x_perpendicular
    speed = np.sqrt(speed_squared)
    pitch = x_parallel / speed
    cases = (
        (0, 0, 0, 0),
        (1, 0, 1, 0),
        (2, 0, 2, 0),
        (2, 0, 0, 1),
        (0, 1, 2, 0),
        (3, 1, 1, 2),
        (4, 2, 4, 2),
        (4, 2, 0, 4),
    )
    for legendre_order, radial_order, hermite_order, laguerre_order in cases:
        coefficient = mod.legendre_to_hermite_laguerre_coefficient(
            legendre_order,
            radial_order,
            hermite_order,
            laguerre_order,
            digits=80,
        )
        left_basis = (
            speed**legendre_order
            * eval_legendre(legendre_order, pitch)
            * eval_genlaguerre(
                radial_order,
                legendre_order + 0.5,
                speed_squared,
            )
        )
        right_basis = eval_hermite(hermite_order, x_parallel) * eval_laguerre(
            laguerre_order, x_perpendicular
        )
        projected = np.sum(
            parallel_weights[:, None]
            * perpendicular_weights[None, :]
            * left_basis
            * right_basis
        ) / (np.sqrt(np.pi) * 2**hermite_order * math.factorial(hermite_order))
        np.testing.assert_allclose(coefficient, projected, rtol=2.0e-11, atol=5.0e-12)

    # Each fixed-total-degree shell is a square transformation.
    highest_condition = 0.0
    for degree in range(13):
        orders = [(degree - 2 * index, index) for index in range(degree // 2 + 1)]
        forward = np.asarray(
            [
                [
                    mod.legendre_to_hermite_laguerre_coefficient(
                        legendre_order,
                        radial_order,
                        hermite_order,
                        laguerre_order,
                    )
                    for legendre_order, radial_order in orders
                ]
                for hermite_order, laguerre_order in orders
            ]
        )
        inverse = np.asarray(
            [
                [
                    mod.hermite_laguerre_to_legendre_coefficient(
                        hermite_order,
                        laguerre_order,
                        legendre_order,
                        radial_order,
                    )
                    for hermite_order, laguerre_order in orders
                ]
                for legendre_order, radial_order in orders
            ]
        )
        np.testing.assert_allclose(
            inverse @ forward,
            np.eye(len(orders)),
            rtol=3.0e-13,
            atol=1.0e-12,
        )
        highest_condition = float(np.linalg.cond(forward))

    # The high-order gate exercises the cancellation regime, not only easy shells.
    assert highest_condition > 1.0e8
    for orders in ((20, 0, 20, 0), (10, 5, 0, 10), (0, 10, 20, 0)):
        coefficient_40 = mod.legendre_to_hermite_laguerre_coefficient(
            *orders, digits=40
        )
        coefficient_100 = mod.legendre_to_hermite_laguerre_coefficient(
            *orders, digits=100
        )
        np.testing.assert_allclose(coefficient_40, coefficient_100, rtol=2.0e-15)

    assert mod.legendre_to_hermite_laguerre_coefficient(0, 0, 2, 0) == 0.0
    with pytest.raises(ValueError, match="basis orders"):
        mod.legendre_to_hermite_laguerre_coefficient(-1, 0, 0, 0)
    with pytest.raises(ValueError, match="digits"):
        mod.legendre_to_hermite_laguerre_coefficient(0, 0, 0, 0, digits=10)
    with pytest.raises(ValueError, match="basis orders"):
        mod.hermite_laguerre_to_legendre_coefficient(-1, 0, 0, 0)
    with pytest.raises(ValueError, match="digits"):
        mod.hermite_laguerre_to_legendre_coefficient(0, 0, 0, 0, digits=10)


def test_associated_basis_transform_matches_velocity_projection() -> None:
    """Jorge et al. finite-m coefficients must match direct projection."""
    from scipy.special import eval_genlaguerre, eval_hermite, eval_laguerre, lpmv

    mod = load_artifact_tool("build_linear_validation_artifacts")
    parallel, parallel_weights = np.polynomial.hermite.hermgauss(80)
    perpendicular, perpendicular_weights = np.polynomial.laguerre.laggauss(80)
    x_parallel = parallel[:, None]
    x_perpendicular = perpendicular[None, :]
    speed = np.sqrt(x_parallel**2 + x_perpendicular)
    pitch = x_parallel / speed
    cases = (
        (1, 0, 1, 0, 0),
        (2, 0, 1, 1, 0),
        (3, 0, 1, 0, 0),  # A genuine lower-shell coefficient.
        (3, 0, 1, 2, 0),
        (3, 0, 1, 0, 1),
        (4, 0, 2, 0, 0),
        (4, 0, 2, 2, 0),
        (5, 1, 3, 0, 0),
    )
    for (
        legendre_order,
        radial_order,
        bessel_order,
        hermite_order,
        laguerre_order,
    ) in cases:
        coefficient = mod.associated_legendre_to_hermite_laguerre_coefficient(
            legendre_order,
            radial_order,
            bessel_order,
            hermite_order,
            laguerre_order,
            digits=80,
        )
        left_basis = (
            speed**legendre_order
            * lpmv(bessel_order, legendre_order, pitch)
            * eval_genlaguerre(
                radial_order,
                legendre_order + 0.5,
                speed**2,
            )
            / x_perpendicular ** (bessel_order / 2)
        )
        right_basis = eval_hermite(hermite_order, x_parallel) * eval_laguerre(
            laguerre_order,
            x_perpendicular,
        )
        projected = np.sum(
            parallel_weights[:, None]
            * perpendicular_weights[None, :]
            * left_basis
            * right_basis
        ) / (np.sqrt(np.pi) * 2**hermite_order * math.factorial(hermite_order))
        np.testing.assert_allclose(coefficient, projected, rtol=3.0e-11, atol=2.0e-10)

    # The convention-corrected m=0 endpoint is exactly the isotropic map.
    for orders in ((0, 0, 0, 0), (2, 1, 2, 1), (4, 2, 0, 4)):
        legendre_order, radial_order, hermite_order, laguerre_order = orders
        finite_order = mod.associated_legendre_to_hermite_laguerre_coefficient(
            legendre_order,
            radial_order,
            0,
            hermite_order,
            laguerre_order,
        )
        isotropic = mod.legendre_to_hermite_laguerre_coefficient(*orders)
        np.testing.assert_allclose(finite_order, isotropic, rtol=2.0e-14, atol=2.0e-14)


def test_associated_basis_overlap_factorization_matches_direct_sum() -> None:
    """Separated radial/angular contractions must preserve equation (B5)."""

    mod = load_artifact_tool("build_linear_validation_artifacts")
    import mpmath as mp

    with mp.workdps(60):

        def associated_laguerre(order: int, alpha: object, power: int) -> object:
            return mod._associated_laguerre_monomial_coefficient_mp(
                order, alpha, power, mp
            )

        def legendre_monomial(order: int, power: int) -> object:
            return mod._legendre_monomial_coefficient_mp(order, power, mp)

        indices = (4, 2, 2, 3, 1)
        factorized = mod._associated_legendre_overlap_mp(
            *indices,
            mp,
            associated_laguerre=associated_laguerre,
            legendre_monomial=legendre_monomial,
        )
        direct = mp.mpf(0)
        p, j, m, auxiliary_p, auxiliary_j = indices
        for left_monomial in range(j + 1):
            for auxiliary_monomial in range(auxiliary_j + 1):
                for left_power in range(m, p + 1):
                    for auxiliary_power in range(auxiliary_p + 1):
                        parity = 1 + (-1) ** (left_power + auxiliary_power - m)
                        direct += (
                            associated_laguerre(j, p, left_monomial)
                            * associated_laguerre(
                                auxiliary_j, auxiliary_p, auxiliary_monomial
                            )
                            * legendre_monomial(p, left_power)
                            * legendre_monomial(auxiliary_p, auxiliary_power)
                            * mp.factorial(left_power)
                            / mp.factorial(left_power - m)
                            * parity
                            * mp.gamma(
                                left_monomial
                                + auxiliary_monomial
                                + mp.mpf(p + auxiliary_p - m + 3) / 2
                            )
                            / (2 * (left_power + auxiliary_power + 1 - m))
                        )
        assert mp.almosteq(factorized, direct)


def test_direct_associated_transform_matches_overlap_oracle() -> None:
    """Polynomial projection and equation-(B5) overlap must remain identical."""
    import mpmath as mp

    mod = load_artifact_tool("build_linear_validation_artifacts")
    cases = (
        (1, 0, 1, 0, 0),
        (3, 0, 1, 0, 1),
        (4, 2, 2, 2, 2),
        (7, 3, 3, 4, 3),
        (9, 4, 4, 3, 6),
    )
    with mp.workdps(60):
        for indices in cases:
            projected = mod._associated_legendre_to_hermite_laguerre_mp(
                *indices,
                mp,
            )
            overlap = mod._associated_legendre_to_hermite_laguerre_overlap_mp(
                *indices,
                mp,
            )
            assert abs(projected - overlap) <= mp.mpf("1e-50") * max(
                mp.mpf(1),
                abs(overlap),
            )


def test_associated_basis_transform_reconstructs_and_inverts_parity_blocks() -> None:
    """Finite-m lower-triangular blocks must reconstruct the physical basis."""
    from scipy.special import eval_genlaguerre, eval_hermite, eval_laguerre, lpmv

    mod = load_artifact_tool("build_linear_validation_artifacts")
    sample_points = ((0.2, 0.4), (-0.7, 1.3), (1.1, 0.15))
    largest_error = 0.0
    for bessel_order in range(4):
        for maximum_degree in (5, 6):
            right_orders, left_orders, forward, inverse = (
                mod.associated_basis_transform_matrices(
                    bessel_order,
                    maximum_degree,
                    digits=80,
                )
            )
            np.testing.assert_allclose(
                inverse @ forward,
                np.eye(len(left_orders)),
                rtol=3.0e-12,
                atol=2.0e-11,
            )
            if maximum_degree == 6:
                analytic_inverse = np.asarray(
                    [
                        [
                            mod.hermite_laguerre_to_associated_legendre_coefficient(
                                hermite_order,
                                laguerre_order,
                                spherical_order,
                                spherical_radial_order,
                                bessel_order,
                                digits=80,
                            )
                            for hermite_order, laguerre_order in right_orders
                        ]
                        for spherical_order, spherical_radial_order in left_orders
                    ]
                )
                np.testing.assert_allclose(
                    analytic_inverse,
                    inverse,
                    rtol=4.0e-12,
                    atol=2.0e-11,
                )
            for column, (legendre_order, radial_order) in enumerate(left_orders):
                for parallel, perpendicular_squared in sample_points:
                    speed = np.sqrt(parallel**2 + perpendicular_squared)
                    pitch = parallel / speed
                    expected = (
                        speed**legendre_order
                        * lpmv(bessel_order, legendre_order, pitch)
                        * eval_genlaguerre(
                            radial_order,
                            legendre_order + 0.5,
                            speed**2,
                        )
                    )
                    right_basis = np.asarray(
                        [
                            eval_hermite(hermite_order, parallel)
                            * eval_laguerre(laguerre_order, perpendicular_squared)
                            * perpendicular_squared ** (bessel_order / 2)
                            for hermite_order, laguerre_order in right_orders
                        ]
                    )
                    error = abs(expected - right_basis @ forward[:, column])
                    largest_error = max(largest_error, float(error))
    assert largest_error < 3.0e-10


def test_associated_basis_transform_precision_and_validation() -> None:
    """Finite-m generation must retain cancellation accuracy and reject bad orders."""
    mod = load_artifact_tool("build_linear_validation_artifacts")
    coefficient_40 = mod.associated_legendre_to_hermite_laguerre_coefficient(
        9, 2, 3, 4, 2, digits=40
    )
    coefficient_100 = mod.associated_legendre_to_hermite_laguerre_coefficient(
        9, 2, 3, 4, 2, digits=100
    )
    np.testing.assert_allclose(coefficient_40, coefficient_100, rtol=3.0e-14)

    assert mod.associated_legendre_to_hermite_laguerre_coefficient(3, 0, 1, 4, 0) == 0.0
    assert mod.hermite_laguerre_to_associated_legendre_coefficient(0, 0, 1, 0, 0) == 0.0
    with pytest.raises(ValueError, match="basis orders"):
        mod.associated_legendre_to_hermite_laguerre_coefficient(-1, 0, 0, 0, 0)
    with pytest.raises(ValueError, match="bessel_order"):
        mod.associated_legendre_to_hermite_laguerre_coefficient(1, 0, 2, 0, 0)
    with pytest.raises(ValueError, match="digits"):
        mod.associated_legendre_to_hermite_laguerre_coefficient(
            1, 0, 1, 0, 0, digits=10
        )
    with pytest.raises(ValueError, match="bessel_order"):
        mod.associated_basis_transform_matrices(-1, 2)
    with pytest.raises(ValueError, match="maximum_reduced_degree"):
        mod.associated_basis_transform_matrices(0, -1)
    with pytest.raises(ValueError, match="digits"):
        mod.associated_basis_transform_matrices(0, 2, digits=10)
    with pytest.raises(ValueError, match="basis orders"):
        mod.hermite_laguerre_to_associated_legendre_coefficient(-1, 0, 0, 0, 0)
    with pytest.raises(ValueError, match="bessel_order"):
        mod.hermite_laguerre_to_associated_legendre_coefficient(0, 0, 1, 0, 2)
    with pytest.raises(ValueError, match="digits"):
        mod.hermite_laguerre_to_associated_legendre_coefficient(
            0, 0, 0, 0, 0, digits=10
        )


def test_laguerre_product_coefficients_reconstruct_collision_products() -> None:
    """Frei et al. Eqs. (3.36) and (3.44) must hold pointwise."""
    from scipy.special import eval_genlaguerre, eval_laguerre

    mod = load_artifact_tool("build_linear_validation_artifacts")
    x = np.linspace(0.0, 8.0, 41)
    cases = (
        (0, 2, 3, 0),
        (1, 2, 1, 0),
        (1, 2, 1, 1),
        (2, 1, 2, 2),
    )
    for associated_order, first_order, second_order, radial_power in cases:
        maximum_order = first_order + second_order + radial_power
        coefficients = np.asarray(
            [
                mod.laguerre_product_expansion_coefficient(
                    associated_order,
                    first_order,
                    second_order,
                    output_order,
                    radial_power=radial_power,
                )
                for output_order in range(maximum_order + 1)
            ]
        )
        expected = (
            x**radial_power
            * eval_genlaguerre(first_order, associated_order, x)
            * eval_laguerre(second_order, x)
        )
        reconstructed = sum(
            coefficient * eval_laguerre(output_order, x)
            for output_order, coefficient in enumerate(coefficients)
        )
        np.testing.assert_allclose(
            reconstructed,
            expected,
            rtol=3.0e-12,
            atol=2.0e-9,
        )

    assert mod.laguerre_product_expansion_coefficient(1, 1, 1, 4) == 0.0
    with pytest.raises(ValueError, match="Laguerre orders"):
        mod.laguerre_product_expansion_coefficient(-1, 0, 0, 0)
    with pytest.raises(ValueError, match="digits"):
        mod.laguerre_product_expansion_coefficient(0, 0, 0, 0, digits=10)


def test_gyroaveraged_spherical_moment_matches_direct_velocity_projection() -> None:
    """Frei et al. Eq. (3.35) must match a Bessel-weighted projection."""
    from scipy.special import (
        eval_genlaguerre,
        eval_hermite,
        eval_laguerre,
        jv,
        lpmv,
    )

    mod = load_artifact_tool("build_linear_validation_artifacts")
    parallel, parallel_weights = np.polynomial.hermite.hermgauss(80)
    perpendicular, perpendicular_weights = np.polynomial.laguerre.laggauss(80)
    x_parallel = parallel[:, None]
    x_perpendicular = perpendicular[None, :]
    speed = np.sqrt(x_parallel**2 + x_perpendicular)
    pitch = x_parallel / speed
    cases = (
        (0, 0, 0, 0, 0, 0.7),
        (1, 0, 1, 0, 0, 0.7),
        (2, 0, 1, 1, 0, 0.7),
        (3, 1, 2, 1, 1, 1.1),
        (4, 0, 2, 0, 2, 1.1),
        (5, 0, 3, 2, 1, 1.3),
    )
    for (
        spherical_order,
        spherical_radial_order,
        bessel_order,
        hermite_order,
        laguerre_order,
        b,
    ) in cases:
        coefficient = mod.gyroaveraged_spherical_moment_coefficient(
            spherical_order,
            spherical_radial_order,
            bessel_order,
            hermite_order,
            laguerre_order,
            b,
            maximum_bessel_laguerre_order=32,
            digits=80,
        )
        observable = (
            jv(bessel_order, b * np.sqrt(x_perpendicular))
            * speed**spherical_order
            * lpmv(bessel_order, spherical_order, pitch)
            * eval_genlaguerre(
                spherical_radial_order,
                spherical_order + 0.5,
                speed**2,
            )
        )
        gyro_moment_basis = eval_hermite(hermite_order, x_parallel) * eval_laguerre(
            laguerre_order,
            x_perpendicular,
        )
        projected = np.sum(
            parallel_weights[:, None]
            * perpendicular_weights[None, :]
            * observable
            * gyro_moment_basis
        ) / (np.sqrt(np.pi) * np.sqrt(2**hermite_order * math.factorial(hermite_order)))
        np.testing.assert_allclose(
            coefficient,
            projected,
            rtol=2.0e-10,
            atol=8.0e-9,
        )

        coefficient_20 = mod.gyroaveraged_spherical_moment_coefficient(
            spherical_order,
            spherical_radial_order,
            bessel_order,
            hermite_order,
            laguerre_order,
            b,
            maximum_bessel_laguerre_order=20,
        )
        np.testing.assert_allclose(
            coefficient_20, coefficient, rtol=2.0e-12, atol=2.0e-11
        )

    assert mod.gyroaveraged_spherical_moment_coefficient(0, 0, 0, 0, 0, 0.0) == 1.0
    assert mod.gyroaveraged_spherical_moment_coefficient(1, 0, 1, 0, 0, 0.0) == 0.0

    with pytest.raises(ValueError, match="basis and truncation"):
        mod.gyroaveraged_spherical_moment_coefficient(-1, 0, 0, 0, 0, 1.0)
    with pytest.raises(ValueError, match="bessel_order"):
        mod.gyroaveraged_spherical_moment_coefficient(1, 0, 2, 0, 0, 1.0)
    with pytest.raises(ValueError, match="bessel_argument"):
        mod.gyroaveraged_spherical_moment_coefficient(1, 0, 1, 0, 0, -1.0)
    with pytest.raises(ValueError, match="digits"):
        mod.gyroaveraged_spherical_moment_coefficient(1, 0, 1, 0, 0, 1.0, digits=10)


def test_finite_wavelength_kernel_factorization_preserves_multiprecision() -> None:
    """Cached Poisson kernels must preserve Frei et al. Eqs. (3.35), (3.41)."""
    import mpmath as mp

    mod = load_artifact_tool("build_linear_validation_artifacts")
    with mp.workdps(70):
        bessel_argument = mp.mpf("0.7")
        half_argument = bessel_argument / 2
        radial_argument = half_argument**2
        exponential = mp.exp(-radial_argument)
        truncation = 6
        bessel_order = 1
        bessel_kernels = tuple(
            exponential
            * radial_argument**order
            * half_argument**bessel_order
            / mp.factorial(order + bessel_order)
            for order in range(truncation + 1)
        )
        weighted_products = mod._bessel_weighted_laguerre_products_mp(
            bessel_order,
            2,
            tuple(range(truncation + 1)),
            bessel_kernels,
            lambda m, n, k, output: mod._laguerre_product_expansion_coefficient_mp(
                m, n, k, output, 0, mp
            ),
            mp,
        )
        for product_order, weighted_product in enumerate(weighted_products):
            direct_product = sum(
                bessel_kernel
                * mod._laguerre_product_expansion_coefficient_mp(
                    bessel_order, order, 2, product_order, 0, mp
                )
                for order, bessel_kernel in enumerate(bessel_kernels)
            )
            assert mp.almosteq(weighted_product, direct_product)
        radial_kernels = tuple(
            exponential * radial_argument**order / mp.factorial(order)
            for order in range(16)
        )

        moment_args = (3, 1, bessel_order, 0, 1, bessel_argument, truncation, mp)
        direct_moment = mod._gyroaveraged_spherical_moment_coefficient_mp(*moment_args)
        cached_moment = mod._gyroaveraged_spherical_moment_coefficient_mp(
            *moment_args,
            bessel_kernels=bessel_kernels,
        )
        assert mp.almosteq(cached_moment, direct_moment)

        polarization_args = (3, 1, bessel_order, bessel_argument, truncation, mp)
        direct_polarization = mod._gyroaveraged_polarization_coefficient_mp(
            *polarization_args
        )
        cached_polarization = mod._gyroaveraged_polarization_coefficient_mp(
            *polarization_args,
            bessel_kernels=bessel_kernels,
            radial_kernels=radial_kernels,
        )
        assert mp.almosteq(cached_polarization, direct_polarization)

        assert (
            mod._gyroaveraged_spherical_moment_coefficient_mp(
                2, 0, 1, 0, 0, bessel_argument, truncation, mp
            )
            == 0
        )
        assert (
            mod._gyroaveraged_polarization_coefficient_mp(
                2, 0, 1, bessel_argument, truncation, mp
            )
            == 0
        )


def test_coulomb_nonpolarized_matrix_recovers_drift_kinetic_physics() -> None:
    """Frei et al. Eqs. (3.48)--(3.49) must recover the published DK block."""
    mod = load_artifact_tool("build_linear_validation_artifacts")
    test_matrix, field_matrix = mod.coulomb_nonpolarized_moment_matrices(
        3,
        1,
        0.0,
        1.0,
        1.0,
        maximum_spherical_order=5,
        maximum_spherical_radial_order=2,
        maximum_bessel_laguerre_order=0,
        digits=80,
    )
    high_bessel_test, high_bessel_field = mod.coulomb_nonpolarized_moment_matrices(
        3,
        1,
        0.0,
        1.0,
        1.0,
        maximum_spherical_order=5,
        maximum_spherical_radial_order=2,
        maximum_bessel_laguerre_order=6,
        digits=80,
    )
    np.testing.assert_array_equal(high_bessel_test, test_matrix)
    np.testing.assert_array_equal(high_bessel_field, field_matrix)
    laguerre_sign = np.asarray([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    code_convention = (
        laguerre_sign[:, None] * laguerre_sign[None, :] * (test_matrix + field_matrix)
    )
    published = mod.build_collision_table(digits=80)[2]
    published_entries = published != 0.0
    np.testing.assert_allclose(
        code_convention[published_entries],
        published[published_entries],
        rtol=3.0e-13,
        atol=3.0e-13,
    )

    np.testing.assert_allclose(code_convention, code_convention.T, atol=2.0e-13)
    assert np.max(np.linalg.eigvalsh(code_convention)) < 1.0e-12
    density = np.eye(8)[0]
    parallel_momentum = np.eye(8)[2]
    thermal_energy = np.asarray([0.0, 1.0, 0.0, 0.0, 1.0 / np.sqrt(2), 0.0, 0.0, 0.0])
    for invariant in (density, parallel_momentum, thermal_energy):
        np.testing.assert_allclose(code_convention @ invariant, 0.0, atol=5.0e-13)

    assert np.count_nonzero(np.abs(code_convention) > 1.0e-14) > np.count_nonzero(
        published_entries
    )

    same_test, same_field = mod.coulomb_nonpolarized_moment_matrices(
        1,
        0,
        0.4,
        2.0,
        0.8,
        source_bessel_argument=0.4,
        maximum_spherical_order=2,
        maximum_spherical_radial_order=1,
        maximum_bessel_laguerre_order=3,
        digits=60,
    )
    unequal_test, unequal_field = mod.coulomb_nonpolarized_moment_matrices(
        1,
        0,
        0.4,
        2.0,
        0.8,
        source_bessel_argument=0.9,
        maximum_spherical_order=2,
        maximum_spherical_radial_order=1,
        maximum_bessel_laguerre_order=3,
        digits=60,
    )
    np.testing.assert_allclose(unequal_test, same_test, rtol=0.0, atol=1.0e-14)
    assert np.linalg.norm(unequal_field - same_field) > 5.0e-2


@pytest.mark.parametrize(
    ("mass_ratio", "temperature_ratio"),
    ((1.0, 1.0), (0.25, 2.0)),
)
def test_direct_drift_kinetic_coulomb_matches_finite_wavelength_endpoint(
    mass_ratio: float,
    temperature_ratio: float,
) -> None:
    """Frei et al. Eqs. (3.53)--(3.56) must equal Eqs. (3.48)--(3.49) at b=0."""
    mod = load_artifact_tool("build_linear_validation_artifacts")
    expected = mod.coulomb_nonpolarized_moment_matrices(
        2,
        1,
        0.0,
        mass_ratio,
        temperature_ratio,
        maximum_spherical_order=3,
        maximum_spherical_radial_order=1,
        maximum_bessel_laguerre_order=0,
        digits=60,
    )
    actual = mod.coulomb_drift_kinetic_moment_matrices(
        2,
        1,
        mass_ratio,
        temperature_ratio,
        maximum_spherical_order=3,
        maximum_spherical_radial_order=1,
        digits=60,
    )
    for actual_component, expected_component in zip(actual, expected, strict=True):
        np.testing.assert_allclose(
            actual_component,
            expected_component,
            rtol=2.0e-15,
            atol=1.0e-50,
        )


def test_drift_kinetic_coulomb_parallel_speed_precompute_preserves_matrices() -> None:
    """Forked independent speed moments must preserve the serial equations."""
    mod = load_artifact_tool("build_linear_validation_artifacts")
    inputs = dict(
        maximum_hermite_order=4,
        maximum_laguerre_order=2,
        mass_ratio=1.0,
        temperature_ratio=1.0,
        maximum_spherical_order=8,
        maximum_spherical_radial_order=4,
        digits=60,
        float64_final_contraction=True,
    )
    serial = mod.coulomb_drift_kinetic_moment_matrices(**inputs, worker_count=1)
    parallel = mod.coulomb_drift_kinetic_moment_matrices(**inputs, worker_count=2)
    for serial_component, parallel_component in zip(serial, parallel, strict=True):
        np.testing.assert_array_equal(parallel_component, serial_component)


def test_drift_kinetic_float_contraction_is_roundoff_equivalent() -> None:
    """The optional final BLAS contraction must retain multiprecision coefficients."""
    mod = load_artifact_tool("build_linear_validation_artifacts")
    inputs = dict(
        maximum_hermite_order=4,
        maximum_laguerre_order=2,
        mass_ratio=1.0,
        temperature_ratio=1.0,
        maximum_spherical_order=8,
        maximum_spherical_radial_order=4,
        digits=60,
    )
    exact = mod.coulomb_drift_kinetic_moment_matrices(**inputs)
    contracted = mod.coulomb_drift_kinetic_moment_matrices(
        **inputs, float64_final_contraction=True
    )
    for exact_component, contracted_component in zip(exact, contracted, strict=True):
        np.testing.assert_allclose(
            contracted_component, exact_component, rtol=3.0e-16, atol=1.0e-15
        )


def test_original_sugama_reconstruction_matches_c6_and_invariants() -> None:
    """Equal-temperature rank restoration must reproduce published C6 entries."""
    mod = load_artifact_tool("build_linear_validation_artifacts")
    test, _field = mod.coulomb_drift_kinetic_moment_matrices(
        3,
        1,
        1.0,
        1.0,
        maximum_spherical_order=5,
        maximum_spherical_radial_order=2,
        digits=60,
    )
    original_test, original_field = mod.original_sugama_like_species_moment_matrices(
        test, 3, 1
    )
    convention_sign = np.asarray([1.0, -1.0] * 4)
    convention = convention_sign[:, None] * convention_sign[None, :]
    collision = convention * (original_test + original_field)
    published = np.load(ROOT / "src/spectraxgk/data/advanced_collision_six_moment.npy")[
        0
    ]
    mask = published != 0.0
    np.testing.assert_allclose(collision[mask], published[mask], rtol=3.0e-15)

    momentum = np.zeros(8)
    momentum[2] = 1.0
    energy = np.zeros(8)
    energy[1] = 1.0
    energy[4] = 1 / np.sqrt(2.0)
    np.testing.assert_allclose(collision @ momentum, 0.0, atol=8.0e-16)
    np.testing.assert_allclose(collision @ energy, 0.0, atol=8.0e-16)
    np.testing.assert_allclose(collision, collision.T, atol=8.0e-15)
    assert np.linalg.eigvalsh(collision).max() < 2.0e-15

    finite_test, finite_field = (
        mod.finite_wavelength_original_sugama_like_species_moment_matrices(
            (convention * test)[None, ...], np.asarray([0.0]), 3, 1
        )
    )
    np.testing.assert_allclose(finite_test[0], convention * original_test, atol=0.0)
    np.testing.assert_allclose(
        finite_field[0], convention * original_field, rtol=2.0e-15, atol=3.0e-16
    )


def test_finite_wavelength_original_sugama_restores_published_flow_channels(
    tmp_path: Path,
) -> None:
    """Frei et al. Eqs. (3.79), (3.94) must restore all finite-b flows."""

    mod = load_artifact_tool("build_linear_validation_artifacts")
    bessel_argument = 0.2
    coulomb_path = tmp_path / "coulomb.npz"
    mod.write_equal_species_finite_wavelength_coulomb_table(
        coulomb_path,
        bessel_arguments=(0.1, bessel_argument),
        maximum_hermite_order=2,
        maximum_laguerre_order=1,
        maximum_angular_bessel_order=2,
        maximum_bessel_laguerre_order=2,
        digits=32,
    )
    sugama_path = tmp_path / "original_sugama.npz"
    metadata = mod.write_equal_species_finite_wavelength_original_sugama_table(
        coulomb_path, sugama_path
    )
    assert metadata["claim_scope"].endswith("original_sugama_table")
    with np.load(sugama_path) as archive:
        test = np.asarray(archive["test_table"])[1]
        field = np.asarray(archive["field_table"])[1]
    signs = np.asarray([1.0, -1.0] * 3)
    radial = 0.25 * bessel_argument**2
    kernels = np.exp(-radial) * np.asarray([1.0, radial, radial**2 / 2.0])
    parallel = signs * np.asarray(
        [0.0, 0.0, kernels[0] / np.sqrt(2.0), kernels[1] / np.sqrt(2.0), 0.0, 0.0]
    )
    perpendicular = signs * np.asarray(
        [
            0.5 * bessel_argument * kernels[0],
            0.5 * bessel_argument * (kernels[1] - kernels[0]),
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )
    temperature = signs * np.asarray(
        [
            -2.0 * kernels[1] / 3.0,
            2.0 * (2.0 * kernels[1] - 2.0 * kernels[2] - kernels[0]) / 3.0,
            0.0,
            0.0,
            2.0 * kernels[0] / (3.0 * np.sqrt(2.0)),
            2.0 * kernels[1] / (3.0 * np.sqrt(2.0)),
        ]
    )
    collision = test + field
    for invariant in (parallel, perpendicular, temperature):
        np.testing.assert_allclose(collision @ invariant, 0.0, atol=2.0e-15)
        np.testing.assert_allclose(invariant @ collision, 0.0, atol=2.0e-15)
    assert np.linalg.matrix_rank(field, tol=1.0e-12) <= 3


@pytest.mark.parametrize(
    ("matrix", "maximum_hermite", "maximum_laguerre", "message"),
    (
        (np.zeros((4, 4)), 1, 1, "maximum_hermite_order"),
        (np.zeros((6, 6)), 2, 0, "maximum_laguerre_order"),
        (np.zeros((3, 3)), 2, 1, "shape"),
        (np.full((6, 6), np.nan), 2, 1, "finite"),
        (np.zeros((6, 6)), 2, 1, "dissipate"),
    ),
)
def test_original_sugama_reconstruction_rejects_invalid_input(
    matrix: np.ndarray,
    maximum_hermite: int,
    maximum_laguerre: int,
    message: str,
) -> None:
    mod = load_artifact_tool("build_linear_validation_artifacts")
    with pytest.raises(ValueError, match=message):
        mod.original_sugama_like_species_moment_matrices(
            matrix, maximum_hermite, maximum_laguerre
        )


def test_improved_sugama_reconstruction_matches_c103_and_invariants() -> None:
    """The arbitrary-order field correction must reproduce published C103."""
    mod = load_artifact_tool("build_linear_validation_artifacts")
    test, _field = mod.coulomb_drift_kinetic_moment_matrices(
        3,
        1,
        1.0,
        1.0,
        maximum_spherical_order=5,
        maximum_spherical_radial_order=2,
        digits=60,
    )
    improved_test, improved_field = (
        mod.improved_sugama_equal_temperature_moment_matrices(
            test, 3, 1, correction_order=1, digits=60
        )
    )
    convention_sign = np.asarray([1.0, -1.0] * 4)
    convention = convention_sign[:, None] * convention_sign[None, :]
    collision = convention * (improved_test + improved_field)
    published = np.load(ROOT / "src/spectraxgk/data/advanced_collision_six_moment.npy")[
        1
    ]
    mask = published != 0.0
    np.testing.assert_allclose(collision[mask], published[mask], rtol=3.0e-15)

    momentum = np.zeros(8)
    momentum[2] = 1.0
    np.testing.assert_allclose(collision @ momentum, 0.0, atol=8.0e-16)
    np.testing.assert_allclose(collision, collision.T, atol=8.0e-15)
    assert np.linalg.eigvalsh(collision).max() < 2.0e-15


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"correction_order": -1}, "correction_order"),
        ({"correction_order": 2}, "maximum_hermite_order"),
        ({"correction_order": 1, "digits": 10}, "digits"),
    ),
)
def test_improved_sugama_reconstruction_rejects_incomplete_shells(
    kwargs: dict[str, object], message: str
) -> None:
    mod = load_artifact_tool("build_linear_validation_artifacts")
    matrix = np.eye(8)
    with pytest.raises(ValueError, match=message):
        mod.improved_sugama_equal_temperature_moment_matrices(matrix, 3, 1, **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"maximum_hermite_order": -1}, "maximum_hermite_order"),
        ({"maximum_laguerre_order": -1}, "maximum_laguerre_order"),
        ({"mass_ratio": 0.0}, "mass_ratio"),
        ({"temperature_ratio": float("inf")}, "temperature_ratio"),
        ({"maximum_spherical_order": -1}, "maximum_spherical_order"),
        ({"maximum_spherical_radial_order": -1}, "maximum_spherical_radial_order"),
        ({"digits": 10}, "digits"),
        ({"worker_count": 0}, "worker_count"),
    ),
)
def test_direct_drift_kinetic_coulomb_rejects_invalid_domain(
    kwargs: dict[str, object], message: str
) -> None:
    mod = load_artifact_tool("build_linear_validation_artifacts")
    inputs: dict[str, object] = {
        "maximum_hermite_order": 1,
        "maximum_laguerre_order": 1,
        "mass_ratio": 1.0,
        "temperature_ratio": 1.0,
    }
    inputs.update(kwargs)
    with pytest.raises(ValueError, match=message):
        mod.coulomb_drift_kinetic_moment_matrices(**inputs)


def test_drift_kinetic_response_artifact_closes_nested_physics_gates(
    tmp_path: Path,
) -> None:
    """A nested Frei hierarchy must conserve invariants and converge its current."""
    mod = load_artifact_tool("build_linear_validation_artifacts")
    out_json = tmp_path / "json" / "response.json"
    out_csv = tmp_path / "csv" / "response.csv"
    out_png = tmp_path / "figure" / "response.png"
    summary = mod.write_drift_kinetic_response_convergence_artifacts(
        out_json,
        out_csv,
        out_png,
        resolutions=((3, 1), (5, 2)),
        required_resolution=(5, 2),
        nested_current_rtol=5.0e-2,
        improved_sugama_correction_order=2,
        digits=32,
    )

    assert summary["gate_passed"] is True
    assert all(summary["gates"].values())
    assert summary["schema_version"] == 4
    assert len(summary["rows"]) == 10
    final = summary["resolutions"][-1]
    assert final["maximum_relative_change"] < 5.0e-2
    assert max(final["invariant_residuals"].values()) < 2.0e-12
    assert max(final["original_sugama_invariant_residuals"].values()) < 2.0e-12
    assert max(final["improved_sugama_invariant_residuals"].values()) < 2.0e-12
    assert final["original_sugama_relative_gap"][0] > 8.0e-2
    assert final["original_sugama_relative_gap"][-1] < 2.0e-2
    assert final["improved_sugama_correction_order"] == 2
    assert final["improved_correction_order_maximum_change"] < 5.0e-2
    assert max(abs(value) for value in final["improved_sugama_relative_gap"]) < 1.0e-2
    assert final["spitzer_relative_error"][-1] < 8.0e-2
    saturation = summary["saturation"]
    assert saturation["paper_normalized_field"] == 1.0e-3
    assert saturation["maximum_saturation_relative_error"] < 1.0e-3
    assert saturation["maximum_field_linearity_relative_error"] < 2.0e-12
    assert (
        -9.5e-4
        < saturation["models"]["coulomb"]["stationary_current_over_vte"]
        < -8.5e-4
    )
    assert final["maximum_eigenvalue"] < 2.0e-12
    assert json.loads(out_json.read_text(encoding="utf-8"))["gate_passed"] is True
    assert len(pd.read_csv(out_csv)) == 10
    with Image.open(out_png) as image:
        assert image.width > 1_000
        assert image.height > 700
    assert out_png.with_suffix(".pdf").exists()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"resolutions": ()}, "at least one"),
        ({"resolutions": ((1, 1),)}, "P >= 2"),
        ({"required_resolution": (1, 1)}, "required_resolution"),
        ({"ion_charges": (0.0,)}, "ion_charges"),
        ({"ion_charges": (2.0, 1.0)}, "increase strictly"),
        ({"nested_current_rtol": 0.0}, "nested_current_rtol"),
        ({"algebra_atol": float("inf")}, "algebra_atol"),
        (
            {"original_sugama_low_charge_gap_min": 1.0},
            "low_charge_gap_min",
        ),
        (
            {"original_sugama_high_charge_gap_max": 0.0},
            "high_charge_gap_max",
        ),
        ({"improved_sugama_coulomb_gap_max": 1.0}, "coulomb_gap_max"),
        ({"improved_sugama_correction_order": 0}, "correction_order"),
        ({"paper_normalized_field": 0.0}, "paper_normalized_field"),
        ({"saturation_times": (1.0, 2.0)}, "saturation_times"),
        ({"saturation_times": (0.0, 2.0, 1.0)}, "saturation_times"),
        ({"saturation_charge": 3.0}, "saturation_charge"),
        ({"saturation_rtol": 0.0}, "saturation_rtol"),
        ({"field_linearity_rtol": float("inf")}, "field_linearity_rtol"),
        ({"spitzer_high_charge_minimum": 0.0}, "high_charge_minimum"),
        ({"spitzer_high_charge_rtol": 0.0}, "high_charge_rtol"),
    ),
)
def test_drift_kinetic_response_artifact_rejects_invalid_policy(
    kwargs: dict[str, object], message: str
) -> None:
    mod = load_artifact_tool("build_linear_validation_artifacts")
    with pytest.raises(ValueError, match=message):
        mod.build_drift_kinetic_response_convergence_summary(**kwargs)


def test_coulomb_nonpolarized_matrix_rejects_invalid_domain() -> None:
    mod = load_artifact_tool("build_linear_validation_artifacts")
    for args, message in (
        ((-1, 1, 0.0, 1.0, 1.0), "maximum_hermite_order"),
        ((1, -1, 0.0, 1.0, 1.0), "maximum_laguerre_order"),
        ((1, 1, -1.0, 1.0, 1.0), "bessel_argument"),
        ((1, 1, 0.0, 0.0, 1.0), "mass_ratio"),
        ((1, 1, 0.0, 1.0, 0.0), "temperature_ratio"),
    ):
        with pytest.raises(ValueError, match=message):
            mod.coulomb_nonpolarized_moment_matrices(*args)
    with pytest.raises(ValueError, match="maximum_bessel_laguerre_order"):
        mod.coulomb_nonpolarized_moment_matrices(
            1, 1, 0.0, 1.0, 1.0, maximum_bessel_laguerre_order=-1
        )
    with pytest.raises(ValueError, match="maximum_angular_bessel_order"):
        mod.coulomb_nonpolarized_moment_matrices(
            1, 1, 0.0, 1.0, 1.0, maximum_angular_bessel_order=-1
        )
    with pytest.raises(ValueError, match="maximum_spherical_order"):
        mod.coulomb_nonpolarized_moment_matrices(
            1, 1, 0.0, 1.0, 1.0, maximum_spherical_order=-1
        )
    with pytest.raises(ValueError, match="maximum_spherical_radial_order"):
        mod.coulomb_nonpolarized_moment_matrices(
            1, 1, 0.0, 1.0, 1.0, maximum_spherical_radial_order=-1
        )
    with pytest.raises(ValueError, match="source_bessel_argument"):
        mod.coulomb_nonpolarized_moment_matrices(
            1, 1, 0.0, 1.0, 1.0, source_bessel_argument=-1.0
        )
    with pytest.raises(ValueError, match="digits"):
        mod.coulomb_nonpolarized_moment_matrices(1, 1, 0.0, 1.0, 1.0, digits=10)
    with pytest.raises(ValueError, match="worker_count"):
        mod.coulomb_nonpolarized_moment_matrices(1, 1, 0.0, 1.0, 1.0, worker_count=0)


def test_finite_wavelength_pair_table_matches_equation_generators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shared table generation must preserve equations and runtime signs."""

    mod = load_artifact_tool("build_linear_validation_artifacts")
    calls = {"moment": 0, "polarization": 0}
    moment_coefficient = mod._gyroaveraged_spherical_moment_coefficient_mp
    polarization_coefficient = mod._gyroaveraged_polarization_coefficient_mp

    def counted_moment(*args: object, **kwargs: object) -> object:
        calls["moment"] += 1
        return moment_coefficient(*args, **kwargs)

    def counted_polarization(*args: object, **kwargs: object) -> object:
        calls["polarization"] += 1
        return polarization_coefficient(*args, **kwargs)

    monkeypatch.setattr(
        mod, "_gyroaveraged_spherical_moment_coefficient_mp", counted_moment
    )
    monkeypatch.setattr(
        mod,
        "_gyroaveraged_polarization_coefficient_mp",
        counted_polarization,
    )
    grid = (0.0, 0.3)
    tables = mod.build_finite_wavelength_coulomb_pair_tables(
        grid,
        1,
        1,
        1.0,
        1.0,
        maximum_spherical_order=2,
        maximum_spherical_radial_order=1,
        maximum_bessel_laguerre_order=2,
        digits=32,
    )
    assert calls == {"moment": 96, "polarization": 24}
    assert [table.shape for table in tables] == [
        (2, 2, 4, 4),
        (2, 2, 4, 4),
        (2, 2, 4),
        (2, 2, 4),
        (2, 2, 4),
        (2, 2, 4),
    ]
    raw_matrices = mod.coulomb_nonpolarized_moment_matrices(
        1,
        1,
        grid[1],
        1.0,
        1.0,
        source_bessel_argument=grid[0],
        maximum_spherical_order=2,
        maximum_spherical_radial_order=1,
        maximum_bessel_laguerre_order=2,
        digits=32,
    )
    raw_vectors = mod.coulomb_polarization_vectors(
        1,
        1,
        grid[1],
        grid[0],
        1.0,
        1.0,
        maximum_spherical_order=2,
        maximum_spherical_radial_order=1,
        maximum_bessel_laguerre_order=2,
        digits=32,
    )
    sign = np.asarray([1.0, -1.0, 1.0, -1.0])
    for table, raw in zip(tables[:2], raw_matrices, strict=True):
        np.testing.assert_allclose(table[1, 0], sign[:, None] * sign[None, :] * raw)
    for table, raw in zip(tables[2:], raw_vectors, strict=True):
        np.testing.assert_allclose(table[1, 0], sign * raw)

    for invalid, message in (
        ((0.0,), "at least two"),
        ((0.0, -0.1), "finite and >= 0"),
        ((0.0, 0.0), "strictly increasing"),
    ):
        with pytest.raises(ValueError, match=message):
            mod.build_finite_wavelength_coulomb_pair_tables(invalid, 0, 0, 1.0, 1.0)


def test_finite_wavelength_hermite_workers_preserve_exact_rows() -> None:
    """Forked Hermite rows must be bitwise identical to serial equations."""

    mod = load_artifact_tool("build_linear_validation_artifacts")
    kwargs = {
        "maximum_spherical_order": 3,
        "maximum_spherical_radial_order": 1,
        "maximum_bessel_laguerre_order": 2,
        "digits": 32,
    }
    serial = mod.coulomb_nonpolarized_moment_matrices(2, 1, 0.4, 1.0, 1.0, **kwargs)
    parallel = mod.coulomb_nonpolarized_moment_matrices(
        2, 1, 0.4, 1.0, 1.0, **kwargs, worker_count=2
    )
    for serial_matrix, parallel_matrix in zip(serial, parallel, strict=True):
        np.testing.assert_array_equal(parallel_matrix, serial_matrix)

    table_kwargs = {
        "maximum_spherical_order": 2,
        "maximum_spherical_radial_order": 1,
        "maximum_bessel_laguerre_order": 2,
        "digits": 32,
    }
    serial_tables = mod.build_finite_wavelength_coulomb_pair_tables(
        (0.0, 0.3), 1, 1, 1.0, 1.0, **table_kwargs
    )
    parallel_tables = mod.build_finite_wavelength_coulomb_pair_tables(
        (0.0, 0.3), 1, 1, 1.0, 1.0, **table_kwargs, worker_count=2
    )
    for serial_table, parallel_table in zip(
        serial_tables, parallel_tables, strict=True
    ):
        np.testing.assert_array_equal(parallel_table, serial_table)


def test_finite_wavelength_float_contraction_is_roundoff_equivalent() -> None:
    """The fast finite-b archive path must retain exact generated coefficients."""

    mod = load_artifact_tool("build_linear_validation_artifacts")
    inputs = dict(
        maximum_hermite_order=3,
        maximum_laguerre_order=1,
        target_bessel_argument=0.4,
        mass_ratio=1.0,
        temperature_ratio=1.0,
        maximum_spherical_order=5,
        maximum_spherical_radial_order=2,
        maximum_angular_bessel_order=3,
        maximum_bessel_laguerre_order=3,
        digits=50,
    )
    exact = mod.coulomb_nonpolarized_moment_matrices(**inputs)
    contracted = mod.coulomb_nonpolarized_moment_matrices(
        **inputs, float64_final_contraction=True
    )
    parallel = mod.coulomb_nonpolarized_moment_matrices(
        **inputs, float64_final_contraction=True, worker_count=4
    )
    for exact_component, contracted_component, parallel_component in zip(
        exact, contracted, parallel, strict=True
    ):
        np.testing.assert_allclose(
            contracted_component, exact_component, rtol=5.0e-15, atol=1.0e-14
        )
        np.testing.assert_array_equal(parallel_component, contracted_component)

    matrix_shards = [
        mod.coulomb_nonpolarized_moment_matrices(
            **inputs,
            included_angular_orders=(angular_order,),
            float64_final_contraction=True,
        )
        for angular_order in range(4)
    ]
    for component_index, contracted_component in enumerate(contracted):
        recombined = sum(shard[component_index] for shard in matrix_shards)
        np.testing.assert_allclose(
            recombined, contracted_component, rtol=5.0e-15, atol=1.0e-14
        )
    parallel_matrix_shard = mod.coulomb_nonpolarized_moment_matrices(
        **inputs,
        included_angular_orders=(0,),
        float64_final_contraction=True,
        worker_count=4,
    )
    for serial_shard, parallel_shard in zip(
        matrix_shards[0], parallel_matrix_shard, strict=True
    ):
        np.testing.assert_allclose(
            parallel_shard, serial_shard, rtol=5.0e-15, atol=1.0e-14
        )

    vector_inputs = dict(
        maximum_hermite_order=3,
        maximum_laguerre_order=1,
        target_bessel_argument=0.4,
        source_bessel_argument=0.4,
        mass_ratio=1.0,
        temperature_ratio=1.0,
        maximum_spherical_order=5,
        maximum_spherical_radial_order=2,
        maximum_angular_bessel_order=3,
        maximum_bessel_laguerre_order=3,
        digits=50,
    )
    exact_vectors = mod.coulomb_polarization_vectors(**vector_inputs)
    fast_vectors = mod.coulomb_polarization_vectors(
        **vector_inputs, float64_final_contraction=True
    )
    decomposed_vectors = mod.coulomb_polarization_vectors(
        **vector_inputs, float64_final_contraction=True, worker_count=2
    )
    for exact_vector, fast_vector, decomposed_vector in zip(
        exact_vectors, fast_vectors, decomposed_vectors, strict=True
    ):
        np.testing.assert_allclose(
            fast_vector, exact_vector, rtol=5.0e-15, atol=1.0e-14
        )
        np.testing.assert_allclose(
            decomposed_vector, exact_vector, rtol=5.0e-15, atol=1.0e-14
        )

    vector_shards = [
        mod.coulomb_polarization_vectors(
            **vector_inputs,
            included_angular_orders=(angular_order,),
            float64_final_contraction=True,
        )
        for angular_order in range(4)
    ]
    for component_index, fast_vector in enumerate(fast_vectors):
        recombined = sum(shard[component_index] for shard in vector_shards)
        np.testing.assert_allclose(recombined, fast_vector, rtol=5.0e-15, atol=1.0e-14)
    parallel_vector_shard = mod.coulomb_polarization_vectors(
        **vector_inputs,
        included_angular_orders=(0,),
        float64_final_contraction=True,
        worker_count=4,
    )
    for serial_shard, parallel_shard in zip(
        vector_shards[0], parallel_vector_shard, strict=True
    ):
        np.testing.assert_allclose(
            parallel_shard, serial_shard, rtol=5.0e-15, atol=1.0e-14
        )

    with pytest.raises(ValueError, match="unique, and sorted"):
        mod.coulomb_nonpolarized_moment_matrices(
            **inputs, included_angular_orders=(1, 0)
        )
    with pytest.raises(ValueError, match="exceed the angular truncation"):
        mod.coulomb_polarization_vectors(**vector_inputs, included_angular_orders=(4,))


def test_finite_wavelength_angular_cutoff_retains_complete_basis() -> None:
    """An angular cutoff at the spherical limit is exactly the full sum."""

    mod = load_artifact_tool("build_linear_validation_artifacts")
    kwargs = {
        "maximum_spherical_order": 3,
        "maximum_spherical_radial_order": 1,
        "maximum_bessel_laguerre_order": 3,
        "digits": 32,
    }
    full_matrices = mod.coulomb_nonpolarized_moment_matrices(
        2, 1, 0.3, 1.0, 1.0, **kwargs
    )
    bounded_matrices = mod.coulomb_nonpolarized_moment_matrices(
        2,
        1,
        0.3,
        1.0,
        1.0,
        **kwargs,
        maximum_angular_bessel_order=3,
    )
    full_vectors = mod.coulomb_polarization_vectors(2, 1, 0.3, 0.3, 1.0, 1.0, **kwargs)
    bounded_vectors = mod.coulomb_polarization_vectors(
        2,
        1,
        0.3,
        0.3,
        1.0,
        1.0,
        **kwargs,
        maximum_angular_bessel_order=3,
    )
    for bounded, full in zip(
        (*bounded_matrices, *bounded_vectors),
        (*full_matrices, *full_vectors),
        strict=True,
    ):
        np.testing.assert_array_equal(bounded, full)


def test_finite_wavelength_endpoint_archive_is_replayable(tmp_path: Path) -> None:
    """The fixed-wavelength generator records all numerical truncations."""

    mod = load_artifact_tool("build_linear_validation_artifacts")
    out = tmp_path / "endpoint.npz"
    metadata = mod.write_finite_wavelength_coulomb_endpoint(
        out,
        bessel_argument=0.2,
        maximum_hermite_order=2,
        maximum_laguerre_order=1,
        maximum_angular_bessel_order=2,
        maximum_bessel_laguerre_order=2,
        digits=32,
    )
    assert metadata["resolution"] == [2, 1]
    assert metadata["maximum_angular_bessel_order"] == 2
    with np.load(out) as archive:
        replayed = json.loads(str(archive["metadata"]))
        assert replayed["checksum"] == pytest.approx(metadata["checksum"])
        assert [archive[f"array_{index}"].shape for index in range(6)] == [
            (6, 6),
            (6, 6),
            (6,),
            (6,),
            (6,),
            (6,),
        ]
    parallel_out = tmp_path / "endpoint_parallel.npz"
    parallel_metadata = mod.write_finite_wavelength_coulomb_endpoint(
        parallel_out,
        bessel_argument=0.2,
        maximum_hermite_order=2,
        maximum_laguerre_order=1,
        maximum_angular_bessel_order=2,
        maximum_bessel_laguerre_order=2,
        digits=32,
        worker_count=2,
    )
    assert parallel_metadata["speed_precompute_seconds"] >= 0.0
    assert parallel_metadata["float64_final_contraction"] is True
    with np.load(out) as serial, np.load(parallel_out) as parallel:
        for index in range(6):
            np.testing.assert_allclose(
                parallel[f"array_{index}"],
                serial[f"array_{index}"],
                rtol=5.0e-15,
                atol=1.0e-14,
            )


def test_equal_species_finite_wavelength_table_is_runtime_ready(
    tmp_path: Path,
) -> None:
    """A shared-cache diagonal archive records a complete ordered B grid."""

    mod = load_artifact_tool("build_linear_validation_artifacts")
    out = tmp_path / "diagonal_table.npz"
    metadata = mod.write_equal_species_finite_wavelength_coulomb_table(
        out,
        bessel_arguments=(0.1, 0.2),
        maximum_hermite_order=1,
        maximum_laguerre_order=1,
        maximum_angular_bessel_order=1,
        maximum_bessel_laguerre_order=1,
        digits=24,
        worker_count=1,
    )
    assert metadata["bessel_argument_grid"] == [0.1, 0.2]
    assert metadata["laguerre_convention"] == "runtime_signed"
    assert metadata["float64_final_contraction"] is True
    assert len(metadata["wavelength_seconds"]) == 2
    with np.load(out) as archive:
        assert archive["test_table"].shape == (2, 4, 4)
        assert archive["field_table"].shape == (2, 4, 4)
        assert archive["test_phi1"].shape == (2, 4)
        assert np.all(np.isfinite(archive["field_phi2"]))

    parallel_out = tmp_path / "diagonal_table_parallel.npz"
    parallel_metadata = mod.write_equal_species_finite_wavelength_coulomb_table(
        parallel_out,
        bessel_arguments=(0.1, 0.2),
        maximum_hermite_order=1,
        maximum_laguerre_order=1,
        maximum_angular_bessel_order=1,
        maximum_bessel_laguerre_order=1,
        digits=24,
        worker_count=2,
        wavelength_worker_count=2,
    )
    assert parallel_metadata["wavelength_worker_count"] == 2
    assert parallel_metadata["workers_per_wavelength"] == 1
    with np.load(out) as serial, np.load(parallel_out) as parallel:
        for name in (
            "test_table",
            "field_table",
            "test_phi1",
            "field_phi1",
            "test_phi2",
            "field_phi2",
        ):
            np.testing.assert_allclose(
                parallel[name], serial[name], rtol=5.0e-15, atol=1.0e-14
            )

    shard_paths = []
    for angular_order in (0, 1):
        shard_path = tmp_path / f"diagonal_table_m{angular_order}.npz"
        mod.write_equal_species_finite_wavelength_coulomb_table(
            shard_path,
            bessel_arguments=(0.1, 0.2),
            maximum_hermite_order=1,
            maximum_laguerre_order=1,
            maximum_angular_bessel_order=1,
            maximum_bessel_laguerre_order=1,
            included_angular_orders=(angular_order,),
            digits=24,
        )
        shard_paths.append(shard_path)
    combined_path = tmp_path / "diagonal_table_combined.npz"
    combined_metadata = mod.combine_equal_species_finite_wavelength_angular_shards(
        tuple(shard_paths), combined_path
    )
    assert combined_metadata["included_angular_orders"] == [0, 1]
    with np.load(out) as serial, np.load(combined_path) as combined:
        for name in (
            "test_table",
            "field_table",
            "test_phi1",
            "field_phi1",
            "test_phi2",
            "field_phi2",
        ):
            np.testing.assert_allclose(
                combined[name], serial[name], rtol=5.0e-15, atol=1.0e-14
            )
    with pytest.raises(ValueError, match="complete contiguous coverage"):
        mod.combine_equal_species_finite_wavelength_angular_shards(
            (shard_paths[0],), tmp_path / "incomplete.npz"
        )

    wavelength_tables = []
    for wavelength_index, bessel_argument in enumerate((0.1, 0.2)):
        point_shards = []
        for angular_order in (0, 1):
            point_shard = tmp_path / f"point_{wavelength_index}_m{angular_order}.npz"
            mod.write_equal_species_finite_wavelength_coulomb_table(
                point_shard,
                bessel_arguments=(bessel_argument,),
                maximum_hermite_order=1,
                maximum_laguerre_order=1,
                maximum_angular_bessel_order=1,
                maximum_bessel_laguerre_order=1,
                included_angular_orders=(angular_order,),
                digits=24,
            )
            point_shards.append(point_shard)
        point_table = tmp_path / f"point_{wavelength_index}.npz"
        mod.combine_equal_species_finite_wavelength_angular_shards(
            tuple(point_shards), point_table
        )
        wavelength_tables.append(point_table)
    wavelength_combined = tmp_path / "wavelength_combined.npz"
    mod.combine_equal_species_finite_wavelength_tables(
        tuple(reversed(wavelength_tables)), wavelength_combined
    )
    with np.load(out) as serial, np.load(wavelength_combined) as combined:
        np.testing.assert_array_equal(
            combined["bessel_argument_grid"], serial["bessel_argument_grid"]
        )
        for name in (
            "test_table",
            "field_table",
            "test_phi1",
            "field_phi1",
            "test_phi2",
            "field_phi2",
        ):
            np.testing.assert_allclose(
                combined[name], serial[name], rtol=5.0e-15, atol=1.0e-14
            )
    with pytest.raises(ValueError, match="strictly increasing"):
        mod.combine_equal_species_finite_wavelength_tables(
            (wavelength_tables[0], wavelength_tables[0]),
            tmp_path / "duplicate_wavelength.npz",
        )

    shared_path = tmp_path / "diagonal_table_shared.npz"
    shared_metadata = mod.write_shared_precompute_angular_coulomb_table(
        shared_path,
        bessel_arguments=(0.1, 0.2),
        maximum_hermite_order=1,
        maximum_laguerre_order=1,
        maximum_angular_bessel_order=1,
        maximum_bessel_laguerre_order=1,
        digits=24,
        worker_count=2,
        wavelength_worker_count=1,
    )
    assert shared_metadata["shared_precompute_seconds"] >= 0.0
    assert shared_metadata["shard_worker_count"] == 1
    assert len(shared_metadata["shard_total_seconds"]) == 2
    resumed_metadata = mod.write_shared_precompute_angular_coulomb_table(
        shared_path,
        bessel_arguments=(0.1, 0.2),
        maximum_hermite_order=1,
        maximum_laguerre_order=1,
        maximum_angular_bessel_order=1,
        maximum_bessel_laguerre_order=1,
        digits=24,
        worker_count=2,
        wavelength_worker_count=1,
    )
    assert resumed_metadata["reused_angular_orders"] == [0, 1]
    assert resumed_metadata["shared_precompute_seconds"] == 0.0
    with np.load(out) as serial, np.load(shared_path) as shared:
        replayed = json.loads(str(shared["metadata"]))
        assert replayed["shard_worker_count"] == 1
        for name in (
            "test_table",
            "field_table",
            "test_phi1",
            "field_phi1",
            "test_phi2",
            "field_phi2",
        ):
            np.testing.assert_allclose(
                shared[name], serial[name], rtol=5.0e-15, atol=1.0e-14
            )

    with pytest.raises(ValueError, match="strictly increasing"):
        mod.write_equal_species_finite_wavelength_coulomb_table(
            out,
            bessel_arguments=(0.2, 0.1),
            maximum_hermite_order=1,
            maximum_laguerre_order=0,
            maximum_angular_bessel_order=1,
            maximum_bessel_laguerre_order=1,
            digits=24,
        )

    with pytest.raises(ValueError, match="worker counts"):
        mod.write_equal_species_finite_wavelength_coulomb_table(
            out,
            bessel_arguments=(0.1, 0.2),
            maximum_hermite_order=1,
            maximum_laguerre_order=0,
            maximum_angular_bessel_order=1,
            maximum_bessel_laguerre_order=1,
            digits=24,
            wavelength_worker_count=0,
        )


def test_collision_table_contraction_gate_checks_coefficients_and_speed(
    tmp_path: Path,
) -> None:
    """Fast archive promotion requires both roundoff parity and real speedup."""

    mod = load_artifact_tool("build_linear_validation_artifacts")
    exact_path = tmp_path / "exact.npz"
    fast_path = tmp_path / "fast.npz"
    metadata = {
        "resolution": [3, 1],
        "bessel_argument_grid": [0.1, 0.2],
        "laguerre_convention": "runtime_signed",
        "total_seconds": 10.0,
    }
    arrays = {
        name: np.ones((2, 2))
        for name in (
            "test_table",
            "field_table",
            "test_phi1",
            "field_phi1",
            "test_phi2",
            "field_phi2",
        )
    }
    np.savez_compressed(exact_path, metadata=np.asarray(json.dumps(metadata)), **arrays)
    fast_metadata = {
        **metadata,
        "total_seconds": 5.0,
        "float64_final_contraction": True,
    }
    np.savez_compressed(
        fast_path, metadata=np.asarray(json.dumps(fast_metadata)), **arrays
    )
    report = mod.write_collision_table_contraction_gate(
        exact_path, fast_path, tmp_path / "gate.json"
    )
    assert report["gate_passed"] is True
    assert report["speedup"] == 2.0

    fast_metadata["resolution"] = [4, 1]
    np.savez_compressed(
        fast_path, metadata=np.asarray(json.dumps(fast_metadata)), **arrays
    )
    with pytest.raises(ValueError, match="metadata mismatch"):
        mod.write_collision_table_contraction_gate(
            exact_path, fast_path, tmp_path / "mismatch.json"
        )


def test_tracked_collision_table_contraction_gate_passes() -> None:
    """The retained P12/J5 profile supports the bounded fast archive path."""

    payload = json.loads(
        (
            ROOT
            / "docs/_static/collision_finite_wavelength_table_contraction_gate.json"
        ).read_text()
    )
    assert payload["gate_passed"] is True
    assert payload["resolution"] == [12, 5]
    assert payload["speedup"] > 1.5
    assert max(row["relative_l2"] for row in payload["arrays"].values()) < 7.0e-16


def test_tracked_collision_angular_shard_gate_passes() -> None:
    """The production shard topology reproduces a monolithic P12 table."""

    payload = json.loads(
        (
            ROOT / "docs/_static/collision_finite_wavelength_angular_shard_gate.json"
        ).read_text()
    )
    assert payload["gate_passed"] is True
    assert payload["resolution"] == [12, 5]
    assert payload["maximum_relative_l2"] < 6.0e-16
    assert payload["maximum_absolute_error"] < 8.0e-15


def test_coulomb_polarization_coefficients_match_projection_and_cancel() -> None:
    """Frei et al. Eqs. (3.41) and (3.50) must satisfy direct and null gates."""
    from scipy.special import eval_genlaguerre, jv, lpmv
    from spectraxgk.operators.linear import (
        apply_finite_wavelength_coulomb_moment_operator,
    )

    mod = load_artifact_tool("build_linear_validation_artifacts")
    parallel, parallel_weights = np.polynomial.hermite.hermgauss(80)
    perpendicular, perpendicular_weights = np.polynomial.laguerre.laggauss(80)
    x_parallel = parallel[:, None]
    x_perpendicular = perpendicular[None, :]
    speed = np.sqrt(x_parallel**2 + x_perpendicular)
    pitch = x_parallel / speed
    cases = (
        (0, 0, 0, 0.7),
        (1, 0, 1, 0.7),
        (2, 0, 0, 0.7),
        (2, 0, 2, 0.7),
        (3, 1, 1, 1.2),
    )
    for spherical_order, spherical_radial_order, bessel_order, b in cases:
        coefficient = mod.gyroaveraged_polarization_coefficient(
            spherical_order,
            spherical_radial_order,
            bessel_order,
            b,
            maximum_bessel_laguerre_order=14,
            digits=80,
        )
        spherical_basis = (
            speed**spherical_order
            * lpmv(bessel_order, spherical_order, pitch)
            * eval_genlaguerre(
                spherical_radial_order,
                spherical_order + 0.5,
                speed**2,
            )
        )
        projected = np.sum(
            parallel_weights[:, None]
            * perpendicular_weights[None, :]
            * spherical_basis
            * jv(0, b * np.sqrt(x_perpendicular))
            * jv(bessel_order, b * np.sqrt(x_perpendicular))
        ) / np.sqrt(np.pi)
        np.testing.assert_allclose(
            coefficient,
            projected,
            rtol=3.0e-10,
            atol=3.0e-10,
        )

    assert mod.gyroaveraged_polarization_coefficient(0, 0, 0, 0.0) == 1.0
    low = mod.coulomb_polarization_vectors(
        1,
        0,
        0.7,
        0.7,
        1.0,
        1.0,
        maximum_spherical_order=1,
        maximum_spherical_radial_order=0,
        maximum_bessel_laguerre_order=3,
        digits=60,
    )
    high = mod.coulomb_polarization_vectors(
        1,
        0,
        0.7,
        0.7,
        1.0,
        1.0,
        maximum_spherical_order=1,
        maximum_spherical_radial_order=0,
        maximum_bessel_laguerre_order=5,
        digits=60,
    )
    for low_vector, high_vector in zip(low, high):
        np.testing.assert_allclose(low_vector, high_vector, rtol=2.0e-6, atol=4.0e-8)
    np.testing.assert_allclose(sum(high), 0.0, atol=3.0e-13)

    test_matrix, field_matrix = mod.coulomb_nonpolarized_moment_matrices(
        1,
        0,
        0.7,
        1.0,
        1.0,
        source_bessel_argument=0.7,
        maximum_spherical_order=1,
        maximum_spherical_radial_order=0,
        maximum_bessel_laguerre_order=5,
        digits=60,
    )
    state = jnp.asarray([[[[[[0.3 + 0.1j]]], [[[0.2 - 0.05j]]]]]])
    runtime = apply_finite_wavelength_coulomb_moment_operator(
        state,
        jnp.asarray(test_matrix)[None, None],
        jnp.asarray(field_matrix)[None, None],
        *(jnp.asarray(vector)[None, None] for vector in high),
        phi=jnp.asarray([[[0.4]]]),
        pair_frequency=jnp.ones((1, 1)),
        charge_over_temperature=jnp.ones(1),
    )
    packed_state = np.asarray(state).reshape(2)
    expected = ((test_matrix + field_matrix) @ packed_state).reshape(state.shape)
    np.testing.assert_allclose(runtime, expected, rtol=3.0e-6, atol=3.0e-6)

    with pytest.raises(ValueError, match="basis and truncation"):
        mod.gyroaveraged_polarization_coefficient(-1, 0, 0, 1.0)
    with pytest.raises(ValueError, match="bessel_order"):
        mod.gyroaveraged_polarization_coefficient(1, 0, 2, 1.0)
    with pytest.raises(ValueError, match="bessel_argument"):
        mod.gyroaveraged_polarization_coefficient(1, 0, 1, -1.0)
    with pytest.raises(ValueError, match="digits"):
        mod.gyroaveraged_polarization_coefficient(1, 0, 1, 1.0, digits=10)
    for args, message in (
        ((-1, 0, 0.0, 0.0, 1.0, 1.0), "maximum_hermite_order"),
        ((0, -1, 0.0, 0.0, 1.0, 1.0), "maximum_laguerre_order"),
        ((0, 0, -1.0, 0.0, 1.0, 1.0), "target_bessel_argument"),
        ((0, 0, 0.0, -1.0, 1.0, 1.0), "source_bessel_argument"),
        ((0, 0, 0.0, 0.0, 0.0, 1.0), "mass_ratio"),
        ((0, 0, 0.0, 0.0, 1.0, 0.0), "temperature_ratio"),
    ):
        with pytest.raises(ValueError, match=message):
            mod.coulomb_polarization_vectors(*args)
    with pytest.raises(ValueError, match="maximum_bessel_laguerre_order"):
        mod.coulomb_polarization_vectors(
            0, 0, 0.0, 0.0, 1.0, 1.0, maximum_bessel_laguerre_order=-1
        )
    with pytest.raises(ValueError, match="digits"):
        mod.coulomb_polarization_vectors(0, 0, 0.0, 0.0, 1.0, 1.0, digits=10)


def test_tracked_coulomb_operator_verification_closes_release_gates() -> None:
    """The committed collision artifact must remain complete and admissible."""
    json_path = ROOT / "docs/_static/collision_operator_verification.json"
    png_path = json_path.with_suffix(".png")
    summary = json.loads(json_path.read_text(encoding="utf-8"))

    assert summary["gate_passed"] is True
    assert all(summary["gates"].values())
    assert summary["claim_scope"] == "offline_operator_algebra_not_runtime_transport"
    assert np.asarray(summary["matrix"]).shape == (8, 8)
    assert len(summary["eigenvalues"]) == 8
    assert sum(abs(value) < 5.0e-13 for value in summary["eigenvalues"]) == 3
    assert summary["metrics"]["maximum_projection_relative_error"] < 5.0e-10
    assert summary["metrics"]["maximum_invariant_residual"] < 5.0e-13
    assert summary["metrics"]["maximum_eigenvalue"] < 1.0e-12
    diffusion = summary["gyrocenter_diffusion"]
    assert diffusion["density_row_infinity_norm"][0] < 5.0e-12
    assert diffusion["density_row_infinity_norm"][-1] > 1.0e-4
    assert 1.7 < diffusion["test_small_b_observed_order"] < 2.3
    assert 1.7 < diffusion["field_small_b_observed_order"] < 2.3
    assert 1.7 < diffusion["small_b_observed_order"] < 2.3
    matrix_errors = summary["matrix_truncation"]["relative_errors"]
    assert matrix_errors[0] > 1.0e-4
    assert matrix_errors[1] < 5.0e-6
    spherical_errors = summary["spherical_truncation"]["relative_errors"]
    assert summary["matrix_truncation"]["spherical_cutoff"] == {
        "maximum_order": 8,
        "maximum_radial_order": 4,
    }
    assert spherical_errors[0] > 2.0e-1
    assert spherical_errors[1] < 2.0e-6
    with Image.open(png_path) as image:
        image.verify()
    assert png_path.stat().st_size > 100_000


def test_tracked_drift_kinetic_response_closes_convergence_gates() -> None:
    """The committed paper-scale response must retain its prospective gates."""
    json_path = ROOT / "docs/_static/collision_response_convergence.json"
    csv_path = json_path.with_suffix(".csv")
    png_path = json_path.with_suffix(".png")
    pdf_path = json_path.with_suffix(".pdf")
    summary = json.loads(json_path.read_text(encoding="utf-8"))

    assert summary["gate_passed"] is True
    assert all(summary["gates"].values())
    assert summary["schema_version"] == 4
    assert summary["required_resolution"] == [20, 5]
    assert len(summary["resolutions"]) == 7
    final = summary["resolutions"][-1]
    assert final["maximum_hermite_order"] == 20
    assert final["maximum_laguerre_order"] == 5
    assert final["maximum_relative_change"] < 5.0e-3
    assert max(final["invariant_residuals"].values()) < 2.0e-12
    assert max(final["original_sugama_invariant_residuals"].values()) < 2.0e-12
    assert max(final["improved_sugama_invariant_residuals"].values()) < 2.0e-12
    assert final["original_sugama_relative_gap"][0] > 8.0e-2
    assert final["original_sugama_relative_gap"][-1] < 2.0e-2
    assert final["improved_sugama_correction_order"] == 5
    assert final["improved_correction_order_maximum_change"] < 5.0e-3
    assert max(abs(value) for value in final["improved_sugama_relative_gap"]) < 1.0e-2
    assert final["spitzer_relative_error"][-1] < 8.0e-2
    assert summary["conductivity_normalization"]["high_charge_relative_error"] < 8.0e-2
    assert summary["saturation"]["maximum_saturation_relative_error"] < 1.0e-3
    assert summary["saturation"]["maximum_field_linearity_relative_error"] < 2.0e-12
    assert final["symmetry_max_abs"] < 2.0e-12
    assert final["maximum_eigenvalue"] < 2.0e-12
    assert final["solve_residual_max"] < 2.0e-12
    assert len(pd.read_csv(csv_path)) == 35
    with Image.open(png_path) as image:
        image.verify()
    assert png_path.stat().st_size > 100_000
    assert pdf_path.stat().st_size > 10_000


def test_tracked_finite_wavelength_generation_hierarchy_stays_fail_closed() -> None:
    """A faster intermediate table must not masquerade as ITG acceptance."""
    path = ROOT / "docs/_static/collision_finite_wavelength_generation_hierarchy.json"
    summary = json.loads(path.read_text(encoding="utf-8"))

    assert summary["schema_version"] == 4
    assert summary["claim_scope"] == (
        "offline_generator_hierarchy_not_transport_acceptance"
    )
    assert summary["literature_required_resolution"] == [18, 6]
    assert summary["gates"] == {
        "all_generated_coefficients_finite": True,
        "b_0p5_p12_completed_within_600_seconds": True,
        "b_0p5_p7_p9_common_block_converged": True,
        "b_0p5_p9_p12_common_block_converged": True,
        "literature_resolution_reached": False,
        "paper_required_wavelength_generated": True,
        "paper_wavelength_factorized_arrays_bitwise_identical": True,
        "paper_wavelength_p12_completed_within_600_seconds": True,
        "paper_wavelength_p7_p9_common_block_converged": False,
        "paper_wavelength_p9_p12_common_block_converged": True,
    }
    factorized = summary["factorized_generator"]
    assert factorized["source_commit"] == "84a96912"
    assert factorized["array_identity"].startswith("bitwise_all_six_arrays")
    assert [row["maximum_hermite_order"] for row in factorized["resolutions"]] == [
        7,
        9,
        12,
    ]
    assert factorized["resolutions"][-1]["total_seconds"] < 150.0
    parallel_p12 = factorized["four_worker_p12"]
    assert parallel_p12["array_identity"].startswith("bitwise_all_six_arrays")
    assert (
        parallel_p12["total_seconds"] < factorized["resolutions"][-1]["total_seconds"]
    )
    assert parallel_p12["worker_policy"].startswith("polarization_serial")
    assert summary["normalization"] == {
        "generator_coordinate": "B=k_perp*v_thermal/Omega=sqrt(2*tau)*k_perp",
        "paper_kperp_for_B_0p5_at_tau_1": pytest.approx(0.5 / np.sqrt(2.0)),
        "paper_required_B_at_tau_1": pytest.approx(1.0 / np.sqrt(2.0)),
        "paper_required_kperp": 0.5,
        "runtime_coordinate": "B=sqrt(2*cache.b)",
    }
    p9 = summary["p9_j4_B_0p5"]
    assert p9["resolution"] == [9, 4]
    assert p9["spherical_order"] == 17
    assert p9["radial_order"] == 8
    assert p9["bessel_laguerre_order"] == 10
    assert p9["checksum"] == pytest.approx(-152.93360627939981, abs=1.0e-13)
    assert (
        p9["optimized"]["total_seconds"]
        < 0.5 * p9["pre_radial_factorization"]["total_seconds"]
    )

    paper = summary["paper_wavelength_hierarchy"]
    assert paper["paper_kperp"] == 0.5
    assert paper["bessel_argument"] == pytest.approx(1.0 / np.sqrt(2.0))
    assert [
        (row["maximum_hermite_order"], row["maximum_laguerre_order"])
        for row in paper["resolutions"]
    ] == [(7, 3), (9, 4), (12, 5)]
    assert paper["resolutions"][-1]["total_seconds"] < 600.0
    changes = paper["common_block_relative_l2"]
    assert changes["p7_j3_to_p9_j4"]["test_matrix"] > 0.05
    assert changes["p9_j4_to_p12_j5"]["test_matrix"] < 0.05
    assert changes["p9_j4_to_p12_j5"]["field_matrix"] < 0.02
    assert changes["p9_j4_to_p12_j5"]["test_phi2"] < 3.0e-9
    assert max(summary["p7_p9_common_low_order_relative_l2"].values()) < 0.05
    p12 = summary["p12_j5_B_0p5"]
    assert p12["resolution"] == [12, 5]
    assert p12["spherical_order"] == 22
    assert p12["radial_order"] == 11
    assert p12["backend"] == "gmpy2"
    assert p12["checksum"] == pytest.approx(-266.19961889691695, abs=1.0e-13)
    assert p12["total_seconds"] < 600.0
    assert p12["pure_python_total_completed"] is False
    assert max(p12["p9_p12_common_low_order_relative_l2"].values()) < 0.03


def test_finite_wavelength_itg_summary_separates_resolved_collision_range(
    tmp_path: Path,
) -> None:
    """A converged collisional interval must not hide an unresolved low-nu limit."""

    mod = load_artifact_tool("build_linear_validation_artifacts")
    collision_frequency = [0.0, 0.01, 0.03, 0.1]
    curves = []
    for p, j, growth in (
        (7, 3, [0.090, 0.085, 0.0800, 0.0680]),
        (9, 4, [0.087, 0.083, 0.0797, 0.0685]),
        (12, 5, [0.081, 0.082, 0.0796, 0.0686]),
    ):
        curves.append(
            {
                "maximum_hermite_order": p,
                "maximum_laguerre_order": j,
                "mode_count": (p + 1) * (j + 1),
                "bessel_argument": 1.0 / np.sqrt(2.0),
                "paper_kperp": 0.5,
                "collision_frequency": collision_frequency,
                "growth": growth,
                "frequency": [-0.1] * len(collision_frequency),
            }
        )

    collisionless = [
        {
            "maximum_hermite_order": 15,
            "maximum_laguerre_order": 6,
            "mode_count": 112,
            "growth": 0.0832,
            "frequency": -0.103,
        },
        {
            "maximum_hermite_order": 18,
            "maximum_laguerre_order": 6,
            "mode_count": 133,
            "growth": 0.0837,
            "frequency": -0.105,
        },
    ]
    summary = mod.summarize_finite_wavelength_itg_curves(
        curves, collisionless_hierarchy=collisionless
    )
    assert summary["gate_passed"] is False
    assert summary["gates"] == {
        "collisionless_p15_p18_converged": True,
        "equivalent_growth_convergence_reached": False,
        "intermediate_collision_range_converged": True,
        "literature_resolution_reached": False,
        "low_collisionality_growth_converged": False,
        "paper_wavelength_reproduced": True,
    }
    assert summary["comparisons"][-1]["maximum_all_frequency_relative_change"] > 0.05
    assert (
        summary["comparisons"][-1]["maximum_resolved_unstable_relative_change"] < 0.05
    )

    output = tmp_path / "finite_wavelength_itg.png"
    mod.write_finite_wavelength_itg_figure(summary, output)
    with Image.open(output) as image:
        image.verify()
    assert output.stat().st_size > 20_000


def test_tracked_finite_wavelength_itg_convergence_closes_equivalent_gate() -> None:
    """The paper panel must retain both finite- and zero-collision checks."""

    path = ROOT / "docs/_static/collision_finite_wavelength_itg_convergence.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    assert summary["schema_version"] == 2
    assert summary["claim_scope"] == (
        "paper_protocol_slab_itg_equivalent_growth_convergence"
    )
    assert summary["literature_required_resolution"] == [18, 6]
    assert summary["gate_passed"] is True
    assert summary["gates"] == {
        "collisionless_p15_p18_converged": True,
        "equivalent_growth_convergence_reached": True,
        "intermediate_collision_range_converged": True,
        "literature_resolution_reached": False,
        "low_collisionality_growth_converged": True,
        "paper_wavelength_reproduced": True,
    }
    assert [
        (curve["maximum_hermite_order"], curve["maximum_laguerre_order"])
        for curve in summary["curves"]
    ] == [(7, 3), (9, 4), (12, 5), (15, 6)]
    comparison = summary["comparisons"][-1]
    assert comparison["maximum_all_frequency_relative_change"] == pytest.approx(
        0.019864931248835725
    )
    assert comparison["maximum_resolved_unstable_relative_change"] < 4.0e-4
    assert summary["collisionless_endpoint_relative_change"] < 0.006


@pytest.mark.slow
def test_coulomb_operator_verification_artifact_closes_physical_gates(
    tmp_path: Path,
) -> None:
    """Nightly multiprecision regeneration must reproduce every tracked gate."""
    mod = load_artifact_tool("build_linear_validation_artifacts")
    out_json = tmp_path / "collision_verification.json"
    out_png = tmp_path / "collision_verification.png"
    summary = mod.write_coulomb_operator_verification_artifacts(
        out_json,
        out_png,
        digits=60,
    )

    assert summary["gate_passed"] is True
    assert all(summary["gates"].values())
    assert summary["claim_scope"] == "offline_operator_algebra_not_runtime_transport"
    assert np.asarray(summary["matrix"]).shape == (8, 8)
    assert len(summary["eigenvalues"]) == 8
    assert sum(abs(value) < 5.0e-13 for value in summary["eigenvalues"]) == 3
    assert summary["metrics"]["maximum_projection_relative_error"] < 5.0e-10
    assert summary["metrics"]["maximum_invariant_residual"] < 5.0e-13
    assert summary["metrics"]["maximum_eigenvalue"] < 1.0e-12
    diffusion = summary["gyrocenter_diffusion"]
    assert diffusion["density_row_infinity_norm"][0] < 5.0e-12
    assert diffusion["density_row_infinity_norm"][-1] > 1.0e-4
    assert 1.7 < diffusion["test_small_b_observed_order"] < 2.3
    assert 1.7 < diffusion["field_small_b_observed_order"] < 2.3
    assert 1.7 < diffusion["small_b_observed_order"] < 2.3
    matrix_errors = summary["matrix_truncation"]["relative_errors"]
    assert matrix_errors[0] > 1.0e-4
    assert matrix_errors[1] < 5.0e-6
    spherical_errors = summary["spherical_truncation"]["relative_errors"]
    assert summary["matrix_truncation"]["spherical_cutoff"] == {
        "maximum_order": 8,
        "maximum_radial_order": 4,
    }
    assert spherical_errors[0] > 2.0e-1
    assert spherical_errors[1] < 2.0e-6

    assert json.loads(out_json.read_text())["gate_passed"] is True
    assert out_png.stat().st_size > 100_000


def test_selected_kbm_overlay_candidate_row_requires_selected_match(tmp_path) -> None:
    mod = load_artifact_tool("generate_linear_reference_overlays")
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
    mod = load_artifact_tool("generate_linear_reference_overlays")
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
    mod = load_artifact_tool("generate_linear_reference_overlays")

    steps = mod._steps_for_fit_window(
        fit_tmax=9.69, dt=0.01, fit_padding=0.5, sample_stride=2
    )

    assert steps == 1020
    assert steps % 2 == 0


def test_kbm_eigenfunction_gate_report_uses_strict_publication_thresholds() -> None:
    mod = load_artifact_tool("generate_linear_reference_overlays")

    report = mod._kbm_eigenfunction_gate_report(
        EigenfunctionComparisonMetrics(overlap=0.63, relative_l2=0.79, phase_shift=0.0)
    )

    assert report.case == "kbm_linear_eigenfunction_ky0p3000"
    assert report.source == "GX raw eigenfunction bundle"
    assert report.passed is False
    assert mod.KBM_EIGENFUNCTION_GATE_TOLERANCES["min_overlap"] == 0.95
    assert mod.KBM_EIGENFUNCTION_GATE_TOLERANCES["max_relative_l2"] == 0.25


def test_load_convergence_series_from_resolution_column(tmp_path: Path) -> None:
    mod = load_artifact_tool("build_linear_validation_artifacts")
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
    mod = load_artifact_tool("build_linear_validation_artifacts")
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
                "observed-order",
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
    mod = load_artifact_tool("build_linear_validation_artifacts")
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
    mod = load_release_tool("check_validation_coverage_manifest")
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
        index = mod.build_gate_index([str(tmp_path / "**" / "*.json")])
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
    mod = load_release_tool("check_validation_coverage_manifest")
    _write_validation_gate(tmp_path / "gate.json", case="case_a", passed=True)
    out_json = tmp_path / "index.json"
    out_csv = tmp_path / "index.csv"
    out_png = tmp_path / "index.png"

    assert (
        mod.main(
            [
                "gate-index",
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


# ---- test_w7x_artifact_panels.py ----

import netCDF4 as nc

from spectraxgk.config import (
    GeometryConfig,
    GridConfig,
    InitializationConfig,
    TimeConfig,
)
from spectraxgk.workflows.runtime.config import (
    RuntimeConfig,
    RuntimePhysicsConfig,
    RuntimeSpeciesConfig,
)


# Zonal and nonlinear-transport artifact assertions
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
    mod = load_artifact_tool("build_nonlinear_transport_admission")
    comparison = tmp_path / "comparison.json"
    out = tmp_path / "redesign.json"
    _comparison(comparison, relative_reduction=-0.005, passed=False)

    assert (
        mod.main(
            [
                "redesign",
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
    mod = load_artifact_tool("build_nonlinear_transport_admission")
    comparison = tmp_path / "comparison.json"
    out = tmp_path / "redesign.json"
    _comparison(comparison, relative_reduction=-0.005, passed=False)

    assert (
        mod.main(
            [
                "redesign",
                "--matched-comparison",
                str(comparison),
                "--out-json",
                str(out),
                "--fail-on-redesign",
            ]
        )
        == 1
    )


def _complete_collisional_zonal_records():
    trace_records = []
    tail = {
        "coulomb": {0.05: 0.0053, 0.1: 0.0022, 0.2: 1.4e-4},
        "original_sugama": {0.05: 0.0048, 0.1: 0.0015, 0.2: 8.0e-5},
        "improved_sugama": {0.05: 0.0051, 0.1: 0.0018, 0.2: 1.0e-4},
    }
    rate = {"coulomb": 0.35, "original_sugama": 0.44, "improved_sugama": 0.38}
    for model, model_tail in tail.items():
        for kx, asymptote in model_tail.items():
            for time in np.linspace(0.0, 30.0, 61):
                trace_records.append(
                    {
                        "model": model,
                        "kx": kx,
                        "t_nu": time,
                        "response": asymptote
                        + (1.0 - asymptote) * np.exp(-rate[model] * time),
                        "p_max": 24,
                        "j_max": 10,
                    }
                )
    section_records = []
    for model in tail:
        for coordinate in ("parallel", "perpendicular"):
            abscissa = (
                np.linspace(-3.0, 3.0, 51)
                if coordinate == "parallel"
                else np.linspace(0.0, 4.0, 51)
            )
            values = (
                np.exp(-(((np.abs(abscissa) - 0.65) / 0.7) ** 2))
                if coordinate == "parallel"
                else np.exp(-1.2 * abscissa)
            )
            values /= np.max(values)
            for location, value in zip(abscissa, values, strict=True):
                section_records.append(
                    {
                        "model": model,
                        "coordinate": coordinate,
                        "kx": 0.2,
                        "t_nu": 5.0,
                        "abscissa": location,
                        "normalized_distribution": value,
                        "p_max": 24,
                        "j_max": 10,
                    }
                )
    return trace_records, section_records


def test_collisional_zonal_campaign_requires_complete_paper_protocol(
    tmp_path: Path,
) -> None:
    mod = load_artifact_tool("build_zonal_flow_artifacts")
    traces, sections = _complete_collisional_zonal_records()

    summary = mod.write_collisional_zonal_artifacts(
        traces,
        sections,
        out_json=tmp_path / "collisional_zonal.json",
        out_png=tmp_path / "collisional_zonal.png",
    )

    assert summary["gate_passed"] is True
    assert all(summary["gates"].values())
    assert summary["protocol"]["maximum_hermite_order"] == 24
    assert summary["protocol"]["maximum_laguerre_order"] == 10
    with Image.open(tmp_path / "collisional_zonal.png") as image:
        image.verify()


def test_drift_kinetic_collisional_zonal_subset_writes_paper_panel(
    tmp_path: Path,
) -> None:
    mod = load_artifact_tool("build_zonal_flow_artifacts")
    traces, _sections = _complete_collisional_zonal_records()

    summary = mod.write_drift_kinetic_collisional_zonal_artifacts(
        traces,
        out_json=tmp_path / "drift_kinetic_zonal.json",
        out_png=tmp_path / "drift_kinetic_zonal.png",
    )

    assert summary["gate_passed"] is True
    assert summary["gates"]["original_sugama_damps_most_strongly"] is True
    assert (
        summary["early_window_rms_error_vs_coulomb"]["improved_sugama"]
        < summary["early_window_rms_error_vs_coulomb"]["original_sugama"]
    )
    with Image.open(tmp_path / "drift_kinetic_zonal.png") as image:
        image.verify()


def test_drift_kinetic_collisional_zonal_subset_fails_closed() -> None:
    mod = load_artifact_tool("build_zonal_flow_artifacts")
    traces, _sections = _complete_collisional_zonal_records()
    incomplete = [row for row in traces if row["model"] != "improved_sugama"]

    summary = mod.summarize_drift_kinetic_collisional_zonal_campaign(incomplete)

    assert summary["gate_passed"] is False
    assert summary["gates"]["all_drift_kinetic_models_present"] is False


def test_tracked_drift_kinetic_zonal_artifact_is_replayable_and_passed() -> None:
    static = ROOT / "docs" / "_static"
    payload = json.loads(
        (static / "collision_drift_kinetic_zonal_response.json").read_text(
            encoding="utf-8"
        )
    )
    with (static / "collision_drift_kinetic_zonal_response.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))

    assert payload["gate_passed"] is True
    assert all(payload["gates"].values())
    assert {row["model"] for row in rows} == {
        "coulomb",
        "original_sugama",
        "improved_sugama",
    }
    assert max(float(row["t_nu"]) for row in rows) >= 30.0


def test_collisional_zonal_campaign_fails_when_a_velocity_section_is_missing() -> None:
    mod = load_artifact_tool("build_zonal_flow_artifacts")
    traces, sections = _complete_collisional_zonal_records()
    sections = [
        row
        for row in sections
        if not (
            row["model"] == "improved_sugama" and row["coordinate"] == "perpendicular"
        )
    ]

    summary = mod.summarize_collisional_zonal_campaign(traces, sections)

    assert summary["gate_passed"] is False
    assert summary["gates"]["velocity_sections_present_at_tnu5"] is False


def test_collisional_zonal_frequency_matches_paper_normalization() -> None:
    mod = load_artifact_tool("build_zonal_flow_artifacts")

    frequency = mod.collisional_zonal_frequency(
        normalized_collisionality=3.13,
        q=1.4,
        epsilon=0.1,
    )

    assert frequency == pytest.approx(0.04999209121124531)
    with pytest.raises(ValueError, match="finite and > 0"):
        mod.collisional_zonal_frequency(
            normalized_collisionality=3.13,
            q=0.0,
            epsilon=0.1,
        )


def test_collisional_zonal_miller_surface_has_paper_inverse_aspect_ratio() -> None:
    cfg, _raw = load_runtime_from_toml(
        ROOT / "benchmarks" / "collisional_zonal_response.toml"
    )

    assert cfg.geometry.rhoc / cfg.geometry.R0 == pytest.approx(0.1)


def test_collisional_zonal_finite_wavelength_spans_fieldline_b() -> None:
    """Finite-B tables must cover the Miller field line, not one mean point."""

    mod = load_artifact_tool("build_zonal_flow_artifacts")
    ranges = []
    for kx in (0.1, 0.2):
        problem = mod._build_collisional_zonal_problem(
            config=ROOT / "benchmarks" / "collisional_zonal_response.toml",
            kx=kx,
            nz=32,
            n_laguerre=2,
            n_hermite=4,
        )
        bessel_argument = np.sqrt(2.0 * np.asarray(problem.cache.b))[
            0, 0, problem.kx_index
        ]
        ranges.append((float(np.min(bessel_argument)), float(np.max(bessel_argument))))
    np.testing.assert_allclose(ranges[1], 2.0 * np.asarray(ranges[0]), rtol=2.0e-6)
    assert ranges[0][0] < np.sqrt(2.0) * 0.1 < ranges[0][1]
    assert ranges[1][0] < np.sqrt(2.0) * 0.2 < ranges[1][1]


def test_collisional_zonal_requested_mode_must_survive_dealiasing() -> None:
    mod = load_artifact_tool("build_zonal_flow_artifacts")
    grid = type(
        "Grid",
        (),
        {
            "ky": np.asarray([0.0, 0.1]),
            "dealias_mask": np.asarray([[True, False, True], [True, True, True]]),
        },
    )()

    mod.require_active_zonal_mode(grid, kx_index=0)
    with pytest.raises(ValueError, match="outside the active dealiased spectrum"):
        mod.require_active_zonal_mode(grid, kx_index=1)
    with pytest.raises(ValueError, match="outside the spectral grid"):
        mod.require_active_zonal_mode(grid, kx_index=5)


def test_collisional_zonal_runtime_archive_fails_closed_before_simulation(
    tmp_path: Path,
) -> None:
    mod = load_artifact_tool("build_zonal_flow_artifacts")
    archive = tmp_path / "incomplete_models.npz"
    np.savez(archive, coulomb=np.eye(2))

    with pytest.raises(ValueError, match="archive is missing"):
        mod.run_drift_kinetic_collisional_zonal_trace(
            config=ROOT / "benchmarks" / "collisional_zonal_response.toml",
            model_archive=archive,
            model="coulomb",
            out_csv=tmp_path / "trace.csv",
        )

    with pytest.raises(ValueError, match="archive is missing"):
        mod.run_finite_wavelength_collisional_zonal_trace(
            config=ROOT / "benchmarks" / "collisional_zonal_response.toml",
            table_archive=archive,
            kx=0.1,
            out_csv=tmp_path / "finite_b_trace.csv",
        )


def test_finite_wavelength_collisional_zonal_table_runs_through_integrator(
    tmp_path: Path,
) -> None:
    """The exact table route reaches a physical finite-B zonal time step."""

    linear_tool = load_artifact_tool("build_linear_validation_artifacts")
    zonal_tool = load_artifact_tool("build_zonal_flow_artifacts")
    table = tmp_path / "finite_b_table.npz"
    linear_tool.write_equal_species_finite_wavelength_coulomb_table(
        table,
        bessel_arguments=(0.10, 0.20),
        maximum_hermite_order=1,
        maximum_laguerre_order=0,
        maximum_angular_bessel_order=1,
        maximum_bessel_laguerre_order=1,
        digits=24,
    )
    out_csv = tmp_path / "finite_b_trace.csv"
    report = zonal_tool.run_finite_wavelength_collisional_zonal_trace(
        config=ROOT / "benchmarks" / "collisional_zonal_response.toml",
        table_archive=table,
        kx=0.1,
        out_csv=out_csv,
        dt=0.005,
        maximum_normalized_time=0.001,
        sample_stride=1,
        nz=8,
    )
    assert report["claim_scope"] == "finite_wavelength_collisional_zonal_trace"
    assert report["steps"] >= 1
    assert report["finite"] is True
    assert out_csv.exists()
    assert len(list(csv.DictReader(out_csv.open(encoding="utf-8")))) >= 2
    gate = zonal_tool.write_finite_wavelength_zonal_grid_gate(
        coarse_traces={0.1: out_csv, 0.2: out_csv},
        fine_traces={0.1: out_csv, 0.2: out_csv},
        out_json=tmp_path / "grid_gate.json",
    )
    assert gate["gate_passed"] is True


def test_tracked_finite_wavelength_zonal_b_grid_pilot_passes() -> None:
    """The tracked pilot closes interpolation, not moment resolution."""

    payload = json.loads(
        (
            ROOT / "docs/_static/collision_finite_wavelength_zonal_b_grid_pilot.json"
        ).read_text()
    )
    assert payload["gate_passed"] is True
    assert payload["resolution"] == [7, 3]
    assert max(row["relative_l2"] for row in payload["traces"].values()) < 4.0e-4
    assert "does not close" in payload["notes"]


def test_finite_wavelength_zonal_moment_gate_passes_and_fails_closed(
    tmp_path: Path,
) -> None:
    """The hierarchy gate uses declared metadata and normalized observables."""

    zonal_tool = load_artifact_tool("build_zonal_flow_artifacts")

    def write_trace(
        path: Path, resolution: tuple[int, int], response: list[float]
    ) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "model",
                    "kx",
                    "t_nu",
                    "response",
                    "response_imag",
                    "p_max",
                    "j_max",
                ),
            )
            writer.writeheader()
            for index, value in enumerate(response):
                writer.writerow(
                    {
                        "model": "coulomb",
                        "kx": 0.1,
                        "t_nu": index,
                        "response": value,
                        "response_imag": 0.0,
                        "p_max": resolution[0],
                        "j_max": resolution[1],
                    }
                )

    low_paths = {kx: tmp_path / f"low_{kx}.csv" for kx in (0.1, 0.2)}
    high_paths = {kx: tmp_path / f"high_{kx}.csv" for kx in (0.1, 0.2)}
    for path in low_paths.values():
        write_trace(path, (3, 1), [2.0, 1.0, 0.5])
    for path in high_paths.values():
        write_trace(path, (5, 2), [4.0, 2.0, 1.0])
    passed = zonal_tool.write_finite_wavelength_zonal_moment_gate(
        hierarchy=[((3, 1), low_paths), ((5, 2), high_paths)],
        out_json=tmp_path / "passed.json",
    )
    assert passed["gate_passed"] is True

    write_trace(high_paths[0.2], (5, 2), [4.0, 3.0, 2.0])
    failed = zonal_tool.write_finite_wavelength_zonal_moment_gate(
        hierarchy=[((3, 1), low_paths), ((5, 2), high_paths)],
        out_json=tmp_path / "failed.json",
    )
    assert failed["gate_passed"] is False
    assert failed["comparisons"][0]["traces"]["0.2"]["passed"] is False
    with pytest.raises(ValueError, match="declared resolution"):
        zonal_tool.write_finite_wavelength_zonal_moment_gate(
            hierarchy=[((2, 1), low_paths), ((5, 2), high_paths)],
            out_json=tmp_path / "metadata.json",
        )


def test_tracked_finite_wavelength_zonal_moment_hierarchy_remains_open() -> None:
    """The tracked hierarchy records negative evidence without promotion."""

    payload = json.loads(
        (
            ROOT
            / "docs/_static/collision_finite_wavelength_zonal_moment_hierarchy.json"
        ).read_text()
    )
    assert payload["gate_passed"] is False
    assert payload["resolutions"] == [
        [7, 3],
        [12, 5],
        [15, 6],
        [18, 7],
        [21, 8],
    ]
    latest = payload["comparisons"][-1]
    assert latest["passed"] is False
    assert latest["traces"]["0.1"]["passed"] is True
    assert latest["traces"]["0.2"]["passed"] is False
    assert 0.05 < latest["traces"]["0.2"]["relative_l2"] < 0.06


def test_plot_zonal_flow_response_output_subcommand(
    tmp_path: Path, monkeypatch
) -> None:
    mod = load_artifact_tool("build_zonal_flow_artifacts")

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
        sys,
        "argv",
        [str(mod.__file__), "response-output", str(data_path), "--out", str(out)],
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
    mod = load_artifact_tool("build_zonal_flow_artifacts")

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
            "response-output",
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
    mod = load_artifact_tool("build_zonal_flow_artifacts")

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
        sys,
        "argv",
        [str(mod.__file__), "response-csv", str(csv_path), "--out", str(out)],
    )

    assert mod.main() == 0
    assert out.exists()
    meta = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["initial_policy"] == "window_abs_mean"
    assert "residual_level" in meta


def test_build_zonal_flow_objective_gate_writes_diagnostic_artifacts(
    tmp_path: Path,
) -> None:
    mod = load_artifact_tool("build_zonal_flow_artifacts")
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
            "objective-gate",
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
    mod = load_artifact_tool("build_zonal_flow_artifacts")
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
                "objective-gate",
                "--summary-csv",
                str(summary),
                "--comparison-csv",
                str(tmp_path / "missing.csv"),
                "--missing-damping-policy",
                "fail",
            ]
        )


def test_build_zonal_flow_miller_panel(tmp_path: Path, monkeypatch) -> None:
    mod = load_artifact_tool("build_zonal_flow_artifacts")

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
            diag.createVariable("Phi_zonal_mode_kxt", "f8", ("time", "kx", "ri"))[:] = (
                raw
            )
        return object(), {"out": str(path)}

    monkeypatch.setattr(mod, "run_runtime_nonlinear_with_artifacts", _fake_run)
    assert (
        mod.main(
            [
                "miller-panel",
                "--config",
                str(config),
                "--out-bundle",
                str(out_bundle),
                "--out-png",
                str(out_png),
            ]
        )
        == 0
    )
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


# W7-X TEM extension status assertions
def test_w7x_tem_extension_status_tracks_open_tem_and_multiflux(tmp_path: Path) -> None:
    spectrum = tmp_path / "w7x_spectrum.json"
    spectrum.write_text(
        json.dumps(
            {
                "source_gate_passed": True,
                "time_samples": 12,
                "dominant_phi_ky": 0.19,
                "dominant_heat_flux_ky": 1.28,
            }
        ),
        encoding="utf-8",
    )
    tem = tmp_path / "tem.csv"
    tem.write_text(
        "ky,gamma_ref,omega_ref,gamma_spectrax,omega_spectrax,rel_gamma,rel_omega\n"
        "0.2,2.0,1.0,3.0,0.5,0.5,-0.5\n"
        "0.3,2.0,1.0,2.1,1.1,0.05,0.1\n",
        encoding="utf-8",
    )

    payload = load_artifact_tool(
        "build_tem_validation_artifacts"
    ).build_w7x_status_payload(
        w7x_spectrum=spectrum, tem_table=tem, tem_audit=tmp_path / "missing.json"
    )
    rows = {row["lane"]: row for row in payload["rows"]}

    assert payload["summary"] == {
        "n_rows": 4,
        "n_closed": 1,
        "n_partial": 0,
        "n_open": 3,
    }
    assert rows["W7-X nonlinear fluctuation spectrum"]["status"] == "closed"
    assert rows["TEM / kinetic-electron linear parity"]["status"] == "open"
    assert (
        rows["TEM / kinetic-electron linear parity"]["key_metrics"]["max_abs_rel_gamma"]
        == 0.5
    )
    assert rows["W7-X multi-flux-tube and multi-surface scan"]["status"] == "open"
    assert rows["W7-X kinetic-electron/TEM nonlinear window"]["status"] == "open"


def test_w7x_tem_extension_status_prefers_tem_audit_when_available(
    tmp_path: Path,
) -> None:
    spectrum = tmp_path / "w7x_spectrum.json"
    spectrum.write_text(json.dumps({"source_gate_passed": True}), encoding="utf-8")
    tem = tmp_path / "tem.csv"
    tem.write_text(
        "ky,gamma_ref,omega_ref,gamma_spectrax,omega_spectrax,rel_gamma,rel_omega\n"
        "0.2,2.0,1.0,2.0,1.0,0.0,0.0\n",
        encoding="utf-8",
    )
    audit = tmp_path / "tem_audit.json"
    audit.write_text(
        json.dumps(
            {
                "status": "open",
                "claim_level": "provisional",
                "metrics": {
                    "n_ky": 1,
                    "max_abs_rel_gamma": 4.0,
                    "max_abs_rel_omega_ref_ge_0p2": 2.0,
                    "omega_branch_inversion": True,
                },
            }
        ),
        encoding="utf-8",
    )

    payload = load_artifact_tool(
        "build_tem_validation_artifacts"
    ).build_w7x_status_payload(w7x_spectrum=spectrum, tem_table=tem, tem_audit=audit)
    row = {row["lane"]: row for row in payload["rows"]}[
        "TEM / kinetic-electron linear parity"
    ]

    assert row["status"] == "open"
    assert row["primary_artifact"] == "docs/_static/tem_branch_parity_audit.json"
    assert row["key_metrics"]["audit_available"] is True
    assert row["key_metrics"]["max_abs_rel_gamma"] == 4.0


def test_w7x_tem_extension_status_writes_artifacts(tmp_path: Path) -> None:
    payload = {
        "kind": "w7x_tem_extension_status",
        "rows": [
            {
                "lane": "W7-X nonlinear fluctuation spectrum",
                "status": "closed",
                "claim_level": "validated",
                "primary_artifact": "w7x.json",
                "key_metrics": {"time_samples": 4, "dominant_phi_ky": 0.2},
                "next_action": "Keep scoped.",
            },
            {
                "lane": "TEM / kinetic-electron linear parity",
                "status": "open",
                "claim_level": "open",
                "primary_artifact": "tem.csv",
                "key_metrics": {"max_abs_rel_gamma": 0.75},
                "next_action": "Fix mismatch.",
            },
        ],
        "summary": {"n_rows": 2, "n_closed": 1, "n_partial": 0, "n_open": 1},
    }

    paths = load_artifact_tool(
        "build_tem_validation_artifacts"
    ).write_w7x_status_artifacts(payload, out_png=tmp_path / "status.png")

    for path in paths.values():
        assert Path(path).exists()
    assert (
        json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))["summary"][
            "n_open"
        ]
        == 1
    )


# W7-X reference overlay assertions
def test_w7x_reference_loader_rejects_nonfinite_bundle(tmp_path: Path) -> None:
    mod = load_artifact_tool("generate_linear_reference_overlays")
    bundle = tmp_path / "bad_w7x_ref.npz"
    save_eigenfunction_reference_bundle(
        bundle,
        theta=np.array([-1.0, 0.0, 1.0]),
        mode=np.array([1.0 + 0.0j, np.nan + 0.0j, 0.5 + 0.0j]),
        source="GX",
        case="w7x_linear",
    )

    with pytest.raises(ValueError, match="non-finite reference mode"):
        mod._load_finite_reference(bundle)


def test_w7x_eigenfunction_gate_report_uses_strict_publication_thresholds() -> None:
    mod = load_artifact_tool("generate_linear_reference_overlays")

    report = mod._w7x_eigenfunction_gate_report(
        EigenfunctionComparisonMetrics(overlap=0.50, relative_l2=0.80, phase_shift=0.0)
    )

    assert report.case == "w7x_linear_eigenfunction_ky0p3000"
    assert report.source == "GX raw eigenfunction bundle"
    assert report.passed is False
    assert mod.W7X_EIGENFUNCTION_GATE_TOLERANCES["min_overlap"] == 0.95
    assert mod.W7X_EIGENFUNCTION_GATE_TOLERANCES["max_relative_l2"] == 0.25


def test_w7x_overlay_main_writes_gate_artifacts(tmp_path: Path, monkeypatch) -> None:
    mod = load_artifact_tool("generate_linear_reference_overlays")
    theta = np.linspace(-np.pi, np.pi, 32)
    reference = np.cos(theta) + 0.25j * np.sin(theta)
    bundle = tmp_path / "w7x_ref.npz"
    spectrax_csv = tmp_path / "w7x_spectrax.csv"
    out_png = tmp_path / "w7x_overlay.png"
    out_json = tmp_path / "w7x_overlay.json"
    save_eigenfunction_reference_bundle(
        bundle,
        theta=theta,
        mode=reference,
        source="GX",
        case="w7x_linear",
        metadata={"ky": 0.3},
    )

    def fake_gx_reference(_path):
        time = np.array([0.0, 1.0])
        ky = np.array([0.0, 0.3])
        kx = np.array([0.0])
        zero = np.zeros((time.size, ky.size, kx.size), dtype=float)
        return time, ky, kx, zero, zero, zero, zero, zero

    def fake_run(_args, *, reference_times, output_steps):
        assert np.array_equal(reference_times, np.array([0.0, 1.0]))
        assert np.array_equal(output_steps, np.array([0, 1]))
        return {
            "theta": theta,
            "mode": reference * np.exp(0.37j),
            "gamma_last": 0.0093,
            "omega_last": -0.2319,
            "Wg_last": 1.0,
            "Wphi_last": 2.0,
            "Wapar_last": 0.0,
            "Phi2_last": 3.0,
            "t_final": 1.0,
            "nl": 8,
            "nm": 16,
            "ny": 82,
            "kx_local": 0.0,
            "kx_ref": 0.0,
        }

    monkeypatch.setattr(mod, "_load_gx_reference", fake_gx_reference)
    monkeypatch.setattr(mod, "_run_w7x_spectrax_mode", fake_run)

    mod.main(
        [
            "w7x",
            "--gx",
            str(tmp_path / "dummy.out.nc"),
            "--gx-input",
            str(tmp_path / "dummy.in"),
            "--geometry-file",
            str(tmp_path / "dummy.eik.nc"),
            "--bundle-out",
            str(bundle),
            "--out-csv",
            str(spectrax_csv),
            "--out-png",
            str(out_png),
            "--out-json",
            str(out_json),
        ]
    )

    assert spectrax_csv.exists()
    assert out_png.exists()
    assert out_png.with_suffix(".pdf").exists()
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["eigenfunction_gate_passed"] is True
    assert data["validation_status"] == "closed"
    assert data["gate_report"]["case"] == "w7x_linear_eigenfunction_ky0p3000"


# W7-X zonal response panel assertions
def test_w7x_zonal_response_panel_main(tmp_path, monkeypatch) -> None:
    mod = load_artifact_tool("build_w7x_zonal_validation_artifacts")

    config = tmp_path / "w7x_test4.toml"
    config.write_text(
        """
[[species]]
name = "ion"
charge = 1.0
mass = 1.0
density = 1.0
temperature = 1.0
tprim = 0.0
fprim = 0.0
kinetic = true

[grid]
Nx = 6
Ny = 4
Nz = 32
Lx = 125.66370614359172
Ly = 62.8
boundary = "linked"
nperiod = 4

[time]
t_max = 10.0
dt = 0.1
method = "rk4"
sample_stride = 1
diagnostics = true
fixed_dt = true

[geometry]
model = "vmec"
vmec_file = "$W7X_VMEC_FILE"
torflux = 0.64
alpha = 0.0
R0 = 5.485

[init]
init_field = "phi"
init_amp = 1.0e-6
gaussian_init = true
gaussian_width = 0.5
init_single = true

[physics]
adiabatic_electrons = true
nonlinear = false
collisions = false
hypercollisions = false

[run]
ky = 0.0
kx = 0.05
Nl = 4
Nm = 8
dt = 0.1
steps = 100
sample_stride = 1
diagnostics = true
""".strip()
    )

    out_dir = tmp_path / "w7x_out"
    out_png = tmp_path / "w7x_panel.png"
    run_calls = []

    def _fake_run(cfg, *, out, kx_target, **kwargs):
        run_calls.append(
            (
                float(kx_target),
                cfg.grid,
                cfg.time.nstep_restart,
                cfg.output,
                cfg.init,
                cfg.physics,
                cfg.terms,
                cfg.collisions,
                dict(kwargs),
            )
        )
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        t = np.linspace(0.0, 10.0, 41)
        signal = np.exp(-0.18 * t) * np.cos(1.35 * t) + 0.12
        with nc.Dataset(path, "w") as ds:
            ds.createDimension("time", t.size)
            ds.createDimension("kx", 3)
            ds.createDimension("ri", 2)
            grids = ds.createGroup("Grids")
            diag = ds.createGroup("Diagnostics")
            grids.createVariable("time", "f8", ("time",))[:] = t
            grids.createVariable("kx", "f8", ("kx",))[:] = np.array(
                [-float(kx_target), 0.0, float(kx_target)]
            )
            raw = np.zeros((t.size, 3, 2), dtype=float)
            raw[:, 2, 0] = signal
            diag.createVariable("Phi_zonal_line_kxt", "f8", ("time", "kx", "ri"))[:] = (
                raw
            )
        return object(), {"out": str(path)}

    monkeypatch.setattr(mod, "run_runtime_nonlinear_with_artifacts", _fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(mod.__file__),
            "--config",
            str(config),
            "--out-dir",
            str(out_dir),
            "--out-png",
            str(out_png),
            "--dt",
            "0.2",
            "--steps",
            "80",
            "--sample-stride",
            "2",
            "--time-scale",
            "3",
            "--checkpoint-steps",
            "20",
            "--Nl",
            "6",
            "--Nm",
            "10",
            "--gaussian-width",
            "1.25",
            "--nu-hyper-m",
            "0.01",
            "--p-hyper-m",
            "4",
            "--show-progress",
        ],
    )

    assert mod.main(["response-panel", *sys.argv[1:]]) == 0
    assert out_png.exists()
    assert out_png.with_suffix(".pdf").exists()
    assert out_png.with_suffix(".csv").exists()
    assert out_png.with_suffix(".traces.csv").exists()
    meta = json.loads(out_png.with_suffix(".json").read_text())
    assert meta["summary_csv"].endswith("w7x_panel.csv")
    assert meta["traces_csv"].endswith("w7x_panel.traces.csv")
    assert meta["initial_policy"] == "first_abs"
    assert meta["initial_normalization"] == "line_first"
    assert meta["initial_level_override"] is None
    assert meta["damping_method"] == "branchwise_extrema"
    assert meta["frequency_method"] == "hilbert_phase"
    assert meta["validation_status"] == "open"
    assert len(meta["cases"]) == 4
    assert meta["literature_reference"]["test"] == 4
    assert meta["literature_reference"]["flux_tube"] == "bean"
    assert (
        meta["literature_reference"]["observable"]
        == "unweighted line-averaged electrostatic potential"
    )
    assert "t=0 line-average" in meta["literature_reference"]["normalization"]
    assert "slower stellarator-specific oscillation" in meta["notes"]
    assert "default --initial-normalization=line_first" in meta["notes"]
    assert "clipped initial portion of Fig. 11" in meta["notes"]
    assert "manuscript-policy inference" in meta["notes"]
    assert "digitized-reference gate" in meta["notes"]
    assert meta["audit_overrides"]["enable_hypercollisions"] is True
    assert meta["audit_overrides"]["gaussian_width"] == 1.25
    assert meta["audit_overrides"]["nu_hyper"] is None
    assert meta["audit_overrides"]["nu_hyper_m"] == 0.01
    assert meta["audit_overrides"]["p_hyper_m"] == 4.0
    assert meta["audit_overrides"]["hypercollisions_const"] == 1.0
    assert meta["audit_overrides"]["hypercollisions_kz"] is None
    assert meta["runtime"] == {
        "dt": 0.2,
        "steps": 80,
        "sample_stride": 2,
        "checkpoint_steps": 20,
        "resume_output": False,
        "time_scale": 3.0,
        "diagnostics": True,
        "show_progress": True,
        "expected_tmax": 16.0,
        "Nl": 6,
        "Nm": 10,
    }
    assert len(run_calls) == 4
    trace = np.loadtxt(out_dir / "w7x_test4_kx050.csv", delimiter=",", skiprows=1)
    assert np.isclose(trace[-1, 0], 30.0)
    combined = np.genfromtxt(
        out_png.with_suffix(".traces.csv"), delimiter=",", names=True
    )
    assert combined.size == 4 * 41
    assert np.isclose(np.max(combined["t_reference"]), 30.0)
    assert "response_normalized" in combined.dtype.names
    for (
        kx_target,
        grid,
        nstep_restart,
        output,
        init,
        physics,
        terms,
        collisions,
        kwargs,
    ) in run_calls:
        assert grid.boundary == "periodic"
        assert grid.non_twist is True
        assert grid.jtwist is None
        assert np.isclose(grid.Lx, 2.0 * np.pi / kx_target)
        assert nstep_restart == 20
        assert output.restart_if_exists is False
        assert output.append_on_restart is True
        assert output.save_for_restart is True
        assert init.gaussian_width == 1.25
        assert physics.hypercollisions is True
        assert terms.hypercollisions == 1.0
        assert collisions.nu_hyper_m == 0.01
        assert collisions.p_hyper_m == 4.0
        assert collisions.hypercollisions_const == 1.0
        assert collisions.hypercollisions_kz == 1.0
        assert kwargs["dt"] == 0.2
        assert kwargs["steps"] == 80
        assert kwargs["sample_stride"] == 2
        assert kwargs["show_progress"] is True
        assert kwargs["Nl"] == 6
        assert kwargs["Nm"] == 10


def test_w7x_zonal_response_panel_resume_output(tmp_path, monkeypatch) -> None:
    mod = load_artifact_tool("build_w7x_zonal_validation_artifacts")

    config = tmp_path / "w7x_test4.toml"
    config.write_text(
        """
[[species]]
name = "ion"
charge = 1.0
mass = 1.0
density = 1.0
temperature = 1.0

[grid]
Nx = 6
Ny = 4
Nz = 16
Lx = 125.66370614359172
Ly = 62.8
boundary = "linked"

[time]
t_max = 1.0
dt = 0.1
method = "rk4"
sample_stride = 1
diagnostics = true
fixed_dt = true

[geometry]
model = "s-alpha"
R0 = 5.485

[init]
init_field = "phi"
init_amp = 1.0e-6
gaussian_init = true
init_single = true

[physics]
adiabatic_electrons = true
nonlinear = false

[run]
ky = 0.0
Nl = 2
Nm = 4
dt = 0.1
steps = 4
sample_stride = 1
diagnostics = true
""".strip()
    )

    out_dir = tmp_path / "w7x_out"
    out_png = tmp_path / "w7x_panel.png"
    seen = []

    def _fake_run(cfg, *, out, kx_target, **_kwargs):
        seen.append(
            (
                cfg.output.restart_if_exists,
                cfg.output.append_on_restart,
                Path(cfg.output.path),
                Path(out),
            )
        )
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        t = np.linspace(0.0, 1.0, 8)
        with nc.Dataset(path, "w") as ds:
            ds.createDimension("time", t.size)
            ds.createDimension("kx", 3)
            ds.createDimension("ri", 2)
            grids = ds.createGroup("Grids")
            diag = ds.createGroup("Diagnostics")
            grids.createVariable("time", "f8", ("time",))[:] = t
            grids.createVariable("kx", "f8", ("kx",))[:] = np.array(
                [-float(kx_target), 0.0, float(kx_target)]
            )
            raw = np.zeros((t.size, 3, 2), dtype=float)
            raw[:, 2, 0] = 1.0 + 0.01 * t
            diag.createVariable("Phi_zonal_line_kxt", "f8", ("time", "kx", "ri"))[:] = (
                raw
            )
        return object(), {"out": str(path)}

    monkeypatch.setattr(mod, "run_runtime_nonlinear_with_artifacts", _fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(mod.__file__),
            "--config",
            str(config),
            "--out-dir",
            str(out_dir),
            "--out-png",
            str(out_png),
            "--kx-values",
            "0.07",
            "--resume-output",
        ],
    )

    assert mod.run_response_panel() == 0
    assert seen == [
        (
            True,
            True,
            out_dir / "w7x_test4_kx070.out.nc",
            out_dir / "w7x_test4_kx070.out.nc",
        )
    ]
    meta = json.loads(out_png.with_suffix(".json").read_text())
    assert meta["runtime"]["resume_output"] is True
    assert meta["runtime"]["time_scale"] == 1.0


def test_generate_w7x_zonal_response_formats_unresolved_damping() -> None:
    mod = load_artifact_tool("build_w7x_zonal_validation_artifacts")

    assert mod._finite_or_none(float("nan")) is None
    assert mod._format_metric(None) == "not fitted"
    assert mod._format_metric(float("nan")) == "not fitted"
    assert mod._format_metric(1.23456) == "1.235"


# W7-X exact-state audit assertions
def _write_w7x_exact_state_audit_fixture(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "startup.log").write_text(
        "\n".join(
            [
                "g_state      max|ref|=6.751e-04 max|test|=6.751e-04 max|diff|=6.000e-11 max|rel|=1.332e-07 rms_rel=7.303e-08 idx=(0,)",
                "phi          max|ref|=5.252e-04 max|test|=5.252e-04 max|diff|=1.567e-10 max|rel|=7.362e-07 rms_rel=1.218e-07 idx=(0,)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (path / "diag_state.log").write_text(
        "\n".join(
            [
                "kperp2       max|ref|=1.474e+01 max|test|=1.474e+01 max|diff|=3.338e-06 max|rel|=7.082e-07 rms_rel=9.822e-08 idx=(0,)",
                "fluxfac      max|ref|=1.950e-02 max|test|=1.950e-02 max|diff|=5.588e-09 max|rel|=3.055e-07 rms_rel=1.379e-07 idx=(0,)",
                "apar         max|ref|=0.000e+00 max|test|=0.000e+00 max|diff|=0.000e+00 max|rel|=nan rms_rel=nan idx=(0,)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "time_index": 10,
                "t": 32.4,
                "metric": "Wg",
                "gx_out": 11.0,
                "spectrax_dump": 11.0 * (1 + 1.0e-7),
                "rel_dump": 1.0e-7,
                "spectrax_solve": 11.0 * (1 + 2.0e-7),
                "rel_solve": 2.0e-7,
            },
            {
                "time_index": 10,
                "t": 32.4,
                "metric": "Wapar",
                "gx_out": 0.0,
                "spectrax_dump": 0.0,
                "rel_dump": 0.0,
                "spectrax_solve": 0.0,
                "rel_solve": 0.0,
            },
        ]
    ).to_csv(path / "diag_state.csv", index=False)


def test_w7x_exact_state_audit_parses_and_writes_outputs(tmp_path: Path) -> None:
    mod = load_comparison_tool("build_exact_state_audit")
    audit_dir = tmp_path / "audit" / "w7x_vmec"
    _write_w7x_exact_state_audit_fixture(audit_dir)

    rows = mod.build_rows(audit_dir)
    assert {row["phase"] for row in rows} == {
        "startup",
        "late arrays",
        "late diagnostics",
    }
    assert (
        max(float(row["value"]) for row in rows if row["value"] == row["value"])
        < 1.0e-6
    )

    out_png = tmp_path / "w7x_exact_state_audit.png"
    rc = mod.main(["report", "--audit-dir", str(audit_dir), "--out-png", str(out_png)])

    assert rc == 0
    assert out_png.exists()
    assert out_png.with_suffix(".pdf").exists()
    payload = json.loads(out_png.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["validation_status"] == "closed"
    assert payload["gate_index_include"] is False
    assert payload["max_finite_relative_error"] < 1.0e-6


# W7-X fluctuation spectrum assertions
def _plot_w7x_fluctuation_spectrum_panel_synthetic_output(path: Path) -> None:
    netcdf4 = pytest.importorskip("netCDF4")
    time = np.linspace(0.0, 30.0, 10)
    ky = np.array([0.0, 0.1, 0.2])
    kx = np.array([-0.2, 0.0, 0.2, 0.4])
    with netcdf4.Dataset(path, "w") as root:
        root.createDimension("time", time.size)
        root.createDimension("ky", ky.size)
        root.createDimension("kx", kx.size)
        root.createDimension("s", 1)
        root.createDimension("ri", 2)
        grids = root.createGroup("Grids")
        grids.createVariable("time", "f8", ("time",))[:] = time
        grids.createVariable("ky", "f8", ("ky",))[:] = ky
        grids.createVariable("kx", "f8", ("kx",))[:] = kx
        diag = root.createGroup("Diagnostics")
        envelope = 1.0 + 0.2 * np.sin(0.5 * time)
        phi2 = np.outer(envelope, np.array([0.02, 1.0, 0.4]))
        diag.createVariable("Phi2_kyt", "f8", ("time", "ky"))[:] = phi2
        phi2_map = np.zeros((time.size, ky.size, kx.size))
        for t_idx, scale in enumerate(envelope):
            phi2_map[t_idx] = scale * np.outer(
                np.array([0.02, 1.0, 0.4]), np.array([0.1, 0.2, 1.0, 0.3])
            )
        diag.createVariable("Phi2_kxkyt", "f8", ("time", "ky", "kx"))[:] = phi2_map
        wphi = np.zeros((time.size, 1, ky.size))
        wphi[:, 0, :] = np.outer(
            1.0 + 0.1 * np.cos(0.4 * time), np.array([0.01, 0.5, 1.5])
        )
        diag.createVariable("Wphi_kyst", "f8", ("time", "s", "ky"))[:] = wphi
        heat = np.zeros((time.size, 1, ky.size))
        heat[:, 0, :] = np.outer(
            1.0 + 0.05 * np.sin(0.3 * time), np.array([0.0, 2.0, 0.7])
        )
        diag.createVariable("HeatFlux_kyst", "f8", ("time", "s", "ky"))[:] = heat
        zonal = np.zeros((time.size, kx.size, 2))
        zonal[:, 2, 0] = np.cos(0.6 * time)
        zonal[:, 2, 1] = np.sin(0.6 * time)
        diag.createVariable("Phi_zonal_mode_kxt", "f8", ("time", "kx", "ri"))[:] = zonal


def test_w7x_fluctuation_spectrum_report_and_artifacts(tmp_path: Path) -> None:
    mod = load_artifact_tool("plot_w7x_fluctuation_spectrum_panel")
    out_nc = tmp_path / "w7x.out.nc"
    _plot_w7x_fluctuation_spectrum_panel_synthetic_output(out_nc)
    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps(
            {"gate_passed": True, "gate_report": {"passed": True, "case": "W7-X"}}
        )
    )

    report = mod.build_w7x_fluctuation_spectrum_report(
        nonlinear=out_nc,
        gate_summary=gate,
        time_min=3.0,
        time_max=28.0,
    )
    paths = mod.write_w7x_fluctuation_spectrum_artifacts(
        report, out=tmp_path / "panel.png"
    )

    assert (
        report["claim_level"]
        == "validated_nonlinear_simulation_spectrum_not_experimental_validation"
    )
    assert report["gate_index_include"] is False
    assert report["source_gate_passed"] is True
    assert report["dominant_phi_ky"] == pytest.approx(0.1)
    assert report["dominant_heat_flux_ky"] == pytest.approx(0.1)
    assert report["dominant_zonal_kx"] == pytest.approx(0.2)
    assert np.sum(report["phi2_ky_distribution"]) == pytest.approx(1.0)
    assert Path(paths["png"]).exists()
    assert Path(paths["pdf"]).exists()
    assert Path(paths["json"]).exists()
    assert Path(paths["csv"]).exists()


def test_w7x_fluctuation_spectrum_rejects_failed_gate(tmp_path: Path) -> None:
    mod = load_artifact_tool("plot_w7x_fluctuation_spectrum_panel")
    out_nc = tmp_path / "w7x.out.nc"
    _plot_w7x_fluctuation_spectrum_panel_synthetic_output(out_nc)
    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps(
            {"gate_passed": False, "gate_report": {"passed": False, "case": "W7-X"}}
        )
    )

    with pytest.raises(ValueError, match="did not pass"):
        mod.build_w7x_fluctuation_spectrum_report(nonlinear=out_nc, gate_summary=gate)


# W7-X zonal closure ladder assertions
def _plot_w7x_zonal_closure_ladder_reference(path: Path, *, kx: float = 0.07) -> None:
    t = np.linspace(0.0, 20.0, 21)
    rows = []
    for code, offset in (("stella", -0.01), ("GENE", 0.01)):
        for time_value in t:
            rows.append(
                {
                    "kx_rhoi": kx,
                    "code": code,
                    "t_vti_over_a": time_value,
                    "response": 0.2 + np.exp(-0.2 * time_value) + offset,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def _plot_w7x_zonal_closure_ladder_output(
    path: Path, *, kx: float = 0.07, offset: float = 0.02
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0.0, 20.0, 21)
    kx_grid = np.array([-kx, 0.0, kx])
    response = 0.2 + np.exp(-0.2 * t) + offset
    with nc.Dataset(path, "w") as ds:
        ds.createDimension("time", t.size)
        ds.createDimension("kx", kx_grid.size)
        ds.createDimension("ri", 2)
        ds.createDimension("s", 1)
        ds.createDimension("m", 8)
        ds.createDimension("l", 4)
        grids = ds.createGroup("Grids")
        diag = ds.createGroup("Diagnostics")
        grids.createVariable("time", "f8", ("time",))[:] = t
        grids.createVariable("kx", "f8", ("kx",))[:] = kx_grid
        phi = np.zeros((t.size, kx_grid.size, 2), dtype=float)
        phi[:, 2, 0] = response / response[0]
        diag.createVariable("Phi_zonal_line_kxt", "f8", ("time", "kx", "ri"))[:] = phi
        wg = np.ones((t.size, 1, 8, 4), dtype=float)
        wg[:, 0, -2:, :] *= 3.0
        diag.createVariable("Wg_lmst", "f8", ("time", "s", "m", "l"))[:] = wg


def test_w7x_zonal_closure_ladder_builds_rows_and_main(tmp_path: Path) -> None:
    mod = load_artifact_tool("build_w7x_zonal_recurrence_artifacts")
    reference = tmp_path / "reference.csv"
    out_nc = tmp_path / "run" / "w7x_test4_kx070.out.nc"
    _plot_w7x_zonal_closure_ladder_reference(reference)
    _plot_w7x_zonal_closure_ladder_output(out_nc)

    reference_t, reference_y = mod.load_reference_trace(reference, 0.07)
    rows, traces = mod.build_rows(
        [("synthetic", "paper", out_nc)],
        reference_t=reference_t,
        reference_y=reference_y,
        kx=0.07,
        t_compare=20.0,
        tail_fraction=0.3,
    )

    assert len(rows) == 1
    assert rows[0]["label"] == "synthetic"
    assert rows[0]["mean_abs_error"] < 1.0
    assert rows[0]["reference_tail_std"] > 0.0
    assert rows[0]["tail_std_ratio"] is not None
    assert rows[0]["hermite_tail_last"] > 0.0
    assert "synthetic" in traces

    out_png = tmp_path / "closure.png"
    rc = mod.main(
        [
            "closure-ladder",
            "--reference-traces",
            str(reference),
            "--run",
            "synthetic",
            "paper",
            str(out_nc),
            "--out-png",
            str(out_png),
        ]
    )

    assert rc == 0
    assert out_png.exists()
    assert out_png.with_suffix(".pdf").exists()
    payload = json.loads(out_png.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["validation_status"] == "open"
    assert payload["gate_index_include"] is False
    assert payload["rows"][0]["family"] == "paper"


# W7-X zonal contract audit assertions
def _w7x_zonal_contract_audit_inputs(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path, Path]:
    reference_trace_rows = []
    spectrax_trace_rows = []
    compare_rows = []
    residual_rows = []
    for kx in (0.05, 0.07, 0.10, 0.30):
        t = np.linspace(0.0, 20.0, 31)
        response = 0.1 + kx + 0.04 * np.exp(-0.12 * t) * np.cos(0.7 * t)
        for code, offset in (("stella", -0.002), ("GENE", 0.002)):
            for time_value, value in zip(t, response + offset, strict=True):
                reference_trace_rows.append(
                    {
                        "kx_rhoi": kx,
                        "code": code,
                        "t_vti_over_a": time_value,
                        "response": value,
                    }
                )
            residual_rows.append(
                {
                    "kx_rhoi": kx,
                    "code": code,
                    "residual_median": 0.1 + kx + offset,
                }
            )
        spectrax_response = response + (0.004 if kx < 0.1 else 0.02)
        for time_value, value in zip(t, spectrax_response, strict=True):
            spectrax_trace_rows.append(
                {
                    "kx_target": kx,
                    "kx_selected": kx,
                    "t_reference": time_value,
                    "phi_zonal_real": value,
                    "response_normalized": value,
                    "initial_level": 1.0,
                    "initial_normalization": "line_first",
                    "source_path": "synthetic.nc",
                }
            )
        compare_rows.append(
            {
                "kx": kx,
                "spectrax_residual": 0.1 + kx + (0.004 if kx < 0.1 else 0.02),
                "spectrax_residual_std": 0.01,
                "spectrax_tmax": 20.0,
                "reference_residual": 0.1 + kx,
                "reference_min": 0.1 + kx - 0.002,
                "reference_max": 0.1 + kx + 0.002,
                "reference_tmax": 20.0,
                "coverage_ratio": 1.0,
                "residual_abs_error": 0.004 if kx < 0.1 else 0.02,
                "residual_atol_effective": 0.02,
                "trace_available": 1,
                "tail_std": 0.01 + 0.1 * kx,
                "reference_tail_std": 0.01,
                "tail_mean_abs_error": 0.003,
                "tail_max_abs_error": 0.005,
            }
        )
    ref_traces = tmp_path / "reference_traces.csv"
    ref_residuals = tmp_path / "reference_residuals.csv"
    summary = tmp_path / "spectrax_summary.csv"
    traces = tmp_path / "spectrax_traces.csv"
    compare = tmp_path / "compare.csv"
    pd.DataFrame(reference_trace_rows).to_csv(ref_traces, index=False)
    pd.DataFrame(residual_rows).to_csv(ref_residuals, index=False)
    pd.DataFrame(
        {
            "kx_target": [0.05],
            "residual_level": [0.1],
            "residual_std": [0.01],
            "tmax": [20.0],
        }
    ).to_csv(
        summary,
        index=False,
    )
    pd.DataFrame(spectrax_trace_rows).to_csv(traces, index=False)
    pd.DataFrame(compare_rows).to_csv(compare, index=False)
    return ref_traces, ref_residuals, summary, traces, compare


def test_w7x_zonal_contract_audit_rows_and_main(tmp_path: Path) -> None:
    mod = load_artifact_tool("build_w7x_zonal_validation_artifacts")
    ref_traces, ref_residuals, summary, traces, compare = (
        _w7x_zonal_contract_audit_inputs(tmp_path)
    )
    rows = mod.load_audit_rows(compare)

    assert len(rows) == 4
    assert rows[0]["residual_gate_passed"] is True
    assert rows[-1]["tail_std_ratio"] > 1.0

    out_png = tmp_path / "audit.png"
    out_csv = tmp_path / "audit.csv"
    out_json = tmp_path / "audit.json"
    rc = mod.main(
        [
            "contract",
            "--reference-traces",
            str(ref_traces),
            "--reference-residuals",
            str(ref_residuals),
            "--spectrax-summary",
            str(summary),
            "--spectrax-traces",
            str(traces),
            "--compare-csv",
            str(compare),
            "--out-png",
            str(out_png),
            "--out-csv",
            str(out_csv),
            "--out-json",
            str(out_json),
        ]
    )

    assert rc == 0
    assert out_png.exists()
    assert out_png.with_suffix(".pdf").exists()
    assert out_csv.exists()
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["validation_status"] == "open"
    assert payload["gate_index_include"] is False
    assert payload["reference_contract"]["normalization"].startswith(
        "line-averaged potential"
    )


# W7-X zonal moment-tail audit assertions
def _w7x_zonal_moment_tail_output(
    path: Path, *, kx_target: float = 0.07, nm: int = 8, nl: int = 4
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0.0, 20.0, 11)
    kx = np.array([-kx_target, 0.0, kx_target])
    with nc.Dataset(path, "w") as ds:
        ds.createDimension("time", t.size)
        ds.createDimension("kx", kx.size)
        ds.createDimension("ri", 2)
        ds.createDimension("s", 1)
        ds.createDimension("m", nm)
        ds.createDimension("l", nl)
        grids = ds.createGroup("Grids")
        diag = ds.createGroup("Diagnostics")
        grids.createVariable("time", "f8", ("time",))[:] = t
        grids.createVariable("kx", "f8", ("kx",))[:] = kx
        wg = np.ones((t.size, 1, nm, nl), dtype=float)
        wg[:, 0, -2:, :] *= np.linspace(1.0, 5.0, t.size)[:, None, None]
        wg[:, 0, :, -1:] *= 2.0
        diag.createVariable("Wg_lmst", "f8", ("time", "s", "m", "l"))[:] = wg
        phi = np.zeros((t.size, kx.size, 2), dtype=float)
        phi[:, 2, 0] = 1.0 + 0.1 * np.sin(t)
        diag.createVariable("Phi_zonal_line_kxt", "f8", ("time", "kx", "ri"))[:] = phi


def test_w7x_zonal_moment_tail_loads_rows_and_main(tmp_path: Path) -> None:
    mod = load_artifact_tool("build_w7x_zonal_recurrence_artifacts")
    run_dir = tmp_path / "run"
    _w7x_zonal_moment_tail_output(run_dir / "w7x_test4_kx070.out.nc")

    rows, heatmap = mod.load_tail_rows(
        [("synthetic", run_dir)],
        kx_values=(0.07,),
        tail_fraction=0.3,
        hermite_tail_fraction=0.25,
        laguerre_tail_fraction=0.25,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["label"] == "synthetic"
    assert row["kx_index"] == 2
    assert row["hermite_tail_last"] > 0.0
    assert row["laguerre_tail_last"] > 0.0
    assert heatmap is None

    out_png = tmp_path / "audit.png"
    rc = mod.main(
        [
            "moment-tail",
            "--run",
            "synthetic",
            str(run_dir),
            "--kx-values",
            "0.07",
            "--out-png",
            str(out_png),
        ]
    )

    assert rc == 0
    assert out_png.exists()
    assert out_png.with_suffix(".pdf").exists()
    payload = json.loads(out_png.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["validation_status"] == "open"
    assert payload["gate_index_include"] is False
    assert payload["rows"][0]["Nm"] == 8


# W7-X zonal recurrence sweep assertions
def _w7x_zonal_recurrence_reference(path: Path, *, kx: float = 0.07) -> None:
    t = np.linspace(0.0, 20.0, 21)
    rows = []
    for code, offset in (("stella", -0.01), ("GENE", 0.01)):
        for time_value in t:
            rows.append(
                {
                    "kx_rhoi": kx,
                    "code": code,
                    "t_vti_over_a": time_value,
                    "response": 0.25 + np.exp(-0.15 * time_value) + offset,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def _w7x_zonal_recurrence_output(
    path: Path, *, kx: float = 0.07, nm: int = 8, nl: int = 4, offset: float = 0.0
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0.0, 20.0, 21)
    kx_grid = np.array([-kx, 0.0, kx])
    response = 0.25 + np.exp(-0.15 * t) + offset
    with nc.Dataset(path, "w") as ds:
        ds.createDimension("time", t.size)
        ds.createDimension("kx", kx_grid.size)
        ds.createDimension("ri", 2)
        ds.createDimension("s", 1)
        ds.createDimension("m", nm)
        ds.createDimension("l", nl)
        grids = ds.createGroup("Grids")
        diag = ds.createGroup("Diagnostics")
        grids.createVariable("time", "f8", ("time",))[:] = t
        grids.createVariable("kx", "f8", ("kx",))[:] = kx_grid
        phi = np.zeros((t.size, kx_grid.size, 2), dtype=float)
        phi[:, 2, 0] = response
        diag.createVariable("Phi_zonal_line_kxt", "f8", ("time", "kx", "ri"))[:] = phi
        wg = np.ones((t.size, 1, nm, nl), dtype=float)
        wg[:, 0, -2:, :] *= 2.0
        diag.createVariable("Wg_lmst", "f8", ("time", "s", "m", "l"))[:] = wg


def test_w7x_zonal_recurrence_sweep_builds_rows_and_main(tmp_path: Path) -> None:
    mod = load_artifact_tool("build_w7x_zonal_recurrence_artifacts")
    reference = tmp_path / "reference.csv"
    out_a = tmp_path / "moment" / "a.out.nc"
    out_b = tmp_path / "closure" / "b.out.nc"
    _w7x_zonal_recurrence_reference(reference)
    _w7x_zonal_recurrence_output(out_a, nm=8, nl=4, offset=0.02)
    _w7x_zonal_recurrence_output(out_b, nm=12, nl=6, offset=0.01)

    reference_t, reference_y = mod.load_reference_trace(reference, 0.07)
    rows, traces = mod.build_sweep(
        [
            ("moment", "moment_resolution", "none", out_a),
            ("closure", "closure_source", "kz", out_b),
        ],
        reference_t=reference_t,
        reference_y=reference_y,
        kx=0.07,
        analysis_tmax=20.0,
        tail_fraction=0.3,
    )

    assert len(rows) == 2
    assert rows[0]["sweep"] == "moment_resolution"
    assert rows[1]["closure_source"] == "kz"
    assert rows[0]["tail_std_ratio"] is not None
    assert rows[0]["tail_std_ratio"] > 0.0
    assert "moment" in traces

    out_png = tmp_path / "recurrence.png"
    rc = mod.main(
        [
            "sweep",
            "--reference-traces",
            str(reference),
            "--run",
            "moment",
            "moment_resolution",
            "none",
            str(out_a),
            "--run",
            "closure",
            "closure_source",
            "kz",
            str(out_b),
            "--out-png",
            str(out_png),
        ]
    )

    assert rc == 0
    assert out_png.exists()
    assert out_png.with_suffix(".pdf").exists()
    payload = json.loads(out_png.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["validation_status"] == "open"
    assert payload["gate_index_include"] is False


def test_w7x_zonal_recurrence_sweep_allows_one_sweep_family(tmp_path: Path) -> None:
    mod = load_artifact_tool("build_w7x_zonal_recurrence_artifacts")
    reference = tmp_path / "reference.csv"
    out_nc = tmp_path / "closure" / "b.out.nc"
    _w7x_zonal_recurrence_reference(reference)
    _w7x_zonal_recurrence_output(out_nc, nm=12, nl=6, offset=0.01)

    out_png = tmp_path / "closure_only.png"
    rc = mod.main(
        [
            "sweep",
            "--reference-traces",
            str(reference),
            "--run",
            "closure",
            "closure_source",
            "const",
            str(out_nc),
            "--out-png",
            str(out_png),
        ]
    )

    assert rc == 0
    assert out_png.exists()
    payload = json.loads(out_png.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["rows"][0]["sweep"] == "closure_source"


# W7-X zonal state-convention assertions
def _w7x_zonal_state_convention_audit_cfg() -> RuntimeConfig:
    return RuntimeConfig(
        grid=GridConfig(
            Nx=6, Ny=4, Nz=32, Lx=2.0 * np.pi / 0.07, Ly=62.8, boundary="periodic"
        ),
        time=TimeConfig(t_max=0.1, dt=0.01, method="rk4", use_diffrax=False),
        geometry=GeometryConfig(
            model="s-alpha", q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778
        ),
        species=(
            RuntimeSpeciesConfig(
                name="ion", charge=1.0, density=1.0, temperature=1.0, kinetic=True
            ),
        ),
        init=InitializationConfig(
            init_field="phi",
            init_amp=0.25,
            gaussian_init=True,
            gaussian_width=1.0,
            init_single=True,
        ),
        physics=RuntimePhysicsConfig(adiabatic_electrons=True, tau_e=1.0),
    )


def test_w7x_zonal_state_convention_audit_closes_synthetic_phi_state(
    tmp_path: Path,
) -> None:
    mod = load_artifact_tool("build_w7x_zonal_validation_artifacts")
    audit = mod.build_state_audit(
        _w7x_zonal_state_convention_audit_cfg(),
        kx_target=0.07,
        ky_target=0.0,
        Nl=2,
        Nm=2,
    )

    row = audit["row"]
    assert audit["passed"] is True
    assert row["profile_relative_l2"] < 1.0e-4
    assert row["line_helper_vs_manual_rel"] < 1.0e-6
    assert row["mode_helper_vs_manual_rel"] < 1.0e-6
    assert row["line_first_initial_over_init_amp"] < 1.0

    out_png = tmp_path / "state.png"
    mod.write_state_outputs(
        audit,
        out_png=out_png,
        out_csv=out_png.with_suffix(".csv"),
        out_json=out_png.with_suffix(".json"),
        config=Path("synthetic.toml"),
    )

    assert out_png.exists()
    assert out_png.with_suffix(".pdf").exists()
    payload = json.loads(out_png.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["validation_status"] == "state_convention_closed"
    assert payload["gate_index_include"] is False
