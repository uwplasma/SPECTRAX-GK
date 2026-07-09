from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
from dataclasses import dataclass

import matplotlib
from PIL import Image

matplotlib.use("Agg")
import numpy as np

from support.paths import load_artifact_tool
from spectraxgk.diagnostics.modes import save_eigenfunction_reference_bundle
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
    mod = load_artifact_tool("compress_previews")
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
    mod = load_artifact_tool("compress_previews")
    path = tmp_path / "panel.png"
    Image.new("RGBA", (64, 32), (255, 255, 255, 255)).save(path)
    before = path.read_bytes()

    report = mod.compress_png_preview(path, max_width=16, colors=8, dry_run=True)

    assert report["dry_run"] is True
    assert report["saved_bytes"] == 0
    assert path.read_bytes() == before


def test_release_preview_targets_and_compression_use_manifest(tmp_path: Path) -> None:
    mod = load_artifact_tool("compress_previews")
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
    mod = load_artifact_tool("compress_previews")
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


def test_generate_geometry_eik_routes_vmec_and_miller_subcommands(
    tmp_path: Path, monkeypatch
) -> None:
    mod = load_artifact_tool("generate_geometry_eik")

    @dataclass(frozen=True)
    class _GeometryConfig:
        geometry_helper_python: str | None = None
        geometry_helper_repo: str | None = None

    @dataclass(frozen=True)
    class _RuntimeConfig:
        geometry: _GeometryConfig

    cfg = _RuntimeConfig(geometry=_GeometryConfig())
    loaded_configs: list[Path] = []
    calls: list[tuple[str, object, Path | None, bool]] = []

    def _load_runtime(path):
        loaded_configs.append(Path(path))
        return cfg, {"source": str(path)}

    def _vmec(runtime_cfg, *, output_path, force):
        calls.append(("vmec", runtime_cfg, output_path, force))
        return tmp_path / "vmec.eik.nc"

    def _miller(runtime_cfg, *, output_path, force):
        calls.append(("miller", runtime_cfg, output_path, force))
        return tmp_path / "miller.eiknc.nc"

    monkeypatch.setattr(mod, "load_runtime_from_toml", _load_runtime)
    monkeypatch.setattr(mod, "generate_runtime_vmec_eik", _vmec)
    monkeypatch.setattr(mod, "generate_runtime_miller_eik", _miller)

    assert (
        mod.main(
            [
                "vmec",
                "--config",
                str(tmp_path / "vmec.toml"),
                "--out",
                str(tmp_path / "vmec.nc"),
                "--force",
            ]
        )
        == 0
    )
    assert calls[-1] == ("vmec", cfg, tmp_path / "vmec.nc", True)

    assert (
        mod.main(
            [
                "miller",
                "--config",
                str(tmp_path / "miller.toml"),
                "--out",
                str(tmp_path / "miller.nc"),
                "--geometry-helper-python",
                "python3.12",
                "--geometry-helper-repo",
                str(tmp_path / "helper"),
            ]
        )
        == 0
    )
    kind, runtime_cfg, output_path, force = calls[-1]
    assert kind == "miller"
    assert output_path == tmp_path / "miller.nc"
    assert force is False
    assert runtime_cfg.geometry.geometry_helper_python == "python3.12"
    assert runtime_cfg.geometry.geometry_helper_repo == str(tmp_path / "helper")
    assert loaded_configs == [tmp_path / "vmec.toml", tmp_path / "miller.toml"]


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
