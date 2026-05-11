import csv
from pathlib import Path

from tools.make_benchmark_atlas import (
    _atlas_manifest_path,
    _build_convergence_gate_reports,
    _load_manifest,
    _resolve_asset_paths,
)

ROOT = Path(__file__).resolve().parents[1]


def _csv_columns(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        return set(next(reader))


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
        "rho_star,mean_gamma_ratio,mean_omega_ratio\n1.0,1.0,1.0\n", encoding="utf-8"
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
        "rho_star,mean_gamma_ratio,mean_omega_ratio\n1.0,1.0,1.0\n", encoding="utf-8"
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
