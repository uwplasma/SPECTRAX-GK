from pathlib import Path

from tools.make_benchmark_atlas import (
    _atlas_manifest_path,
    _load_manifest,
    _resolve_asset_paths,
)


def test_benchmark_atlas_manifest_resolves_tracked_assets() -> None:
    manifest = _load_manifest(_atlas_manifest_path())
    assets = _resolve_asset_paths(manifest)

    assert set(assets) == {"imported_linear", "extended_linear", "core_linear", "core_nonlinear"}
    assert assets["imported_linear"]["w7x"] == Path("/path/to/SPECTRAX-GK/docs/_static/w7x_linear_t2_scan.csv")
    assert assets["extended_linear"]["miller"] == Path("/path/to/SPECTRAX-GK/docs/_static/kbm_miller_exact_growth_dump.csv")
    assert assets["core_linear"]["cyclone"].name == "cyclone_comparison.png"
    assert assets["core_nonlinear"]["hsx"].name == "hsx_nonlinear_compare_t50_true.png"
