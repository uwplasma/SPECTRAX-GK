"""Regression tests for the GX RHS comparison helpers."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from spectraxgk.benchmarks import (
    KBM_OMEGA_D_SCALE,
    KBM_OMEGA_STAR_SCALE,
    KBM_RHO_STAR,
    KBMBaseCase,
    _build_initial_condition,
    _two_species_params,
    run_kbm_linear,
)
from spectraxgk.config import GridConfig
from spectraxgk.geometry import SAlphaGeometry, sample_flux_tube_geometry
from spectraxgk.grids import build_spectral_grid, select_ky_grid
from spectraxgk.linear import build_linear_cache
from spectraxgk.terms.assembly import assemble_rhs_terms_cached, compute_fields_cached
from spectraxgk.terms.config import TermConfig


def test_manual_linear_contributions_match_assembly_for_multispecies_kbm() -> None:
    tools_dir = Path(__file__).resolve().parents[1] / "tools"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_rhs_terms as mod
    finally:
        sys.path.remove(str(tools_dir))

    cfg = KBMBaseCase(grid=GridConfig(Nx=1, Ny=8, Nz=24, Lx=62.8, Ly=62.8, y0=10.0, ntheta=8, nperiod=2))
    geom = SAlphaGeometry.from_config(cfg.geometry)
    params = _two_species_params(
        cfg.model,
        kpar_scale=float(geom.gradpar()),
        omega_d_scale=KBM_OMEGA_D_SCALE,
        omega_star_scale=KBM_OMEGA_STAR_SCALE,
        rho_star=KBM_RHO_STAR,
        nhermite=6,
    )
    grid_full = build_spectral_grid(cfg.grid)
    ky_idx = int(np.argmin(np.abs(np.asarray(grid_full.ky) - 0.3)))
    grid = select_ky_grid(grid_full, ky_idx)
    cache = build_linear_cache(grid, geom, params, 4, 6)

    G0_single = _build_initial_condition(
        grid,
        geom,
        ky_index=0,
        kx_index=0,
        Nl=4,
        Nm=6,
        init_cfg=cfg.init,
    )
    G = np.zeros((2, 4, 6, grid.ky.size, grid.kx.size, grid.z.size), dtype=np.complex64)
    G[1] = np.asarray(G0_single, dtype=np.complex64)
    G_j = jnp.asarray(G)

    term_cfg = TermConfig(hypercollisions=0.0, end_damping=0.0, bpar=0.0)
    rhs_total, fields_ref, contrib_ref = assemble_rhs_terms_cached(G_j, cache, params, terms=term_cfg)
    fields = compute_fields_cached(G_j, cache, params, terms=term_cfg, use_custom_vjp=False)
    fields_manual, contrib_manual = mod._manual_linear_contributions_from_fields(
        G_j,
        cache,
        params,
        term_cfg,
        phi=np.asarray(fields.phi),
        apar=np.asarray(fields.apar),
        bpar=np.asarray(fields.bpar if fields.bpar is not None else np.zeros_like(fields.phi)),
    )

    assert np.allclose(np.asarray(fields_manual.phi), np.asarray(fields_ref.phi))
    assert np.allclose(np.asarray(fields_manual.apar), np.asarray(fields_ref.apar))
    for key in ("streaming", "mirror", "curvature", "gradb", "diamagnetic", "collisions"):
        assert np.allclose(np.asarray(contrib_manual[key]), np.asarray(contrib_ref[key]))
    contrib_sum = sum(np.asarray(contrib_manual[key]) for key in contrib_manual)
    assert np.allclose(contrib_sum, np.asarray(rhs_total))


def test_compare_gx_rhs_terms_parser_defaults_to_dump_metadata() -> None:
    tools_dir = Path(__file__).resolve().parents[1] / "tools"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_rhs_terms as mod
    finally:
        sys.path.remove(str(tools_dir))

    parser = mod.build_parser()
    args = parser.parse_args(["--gx-dir", "/tmp/gx", "--gx-out", "/tmp/gx.out.nc"])

    assert args.Nl is None
    assert args.Nm is None
    assert args.y0 is None


def test_compare_gx_rhs_terms_parser_accepts_runtime_config() -> None:
    tools_dir = Path(__file__).resolve().parents[1] / "tools"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_rhs_terms as mod
    finally:
        sys.path.remove(str(tools_dir))

    parser = mod.build_parser()
    args = parser.parse_args(
        ["--gx-dir", "/tmp/gx", "--gx-out", "/tmp/gx.out.nc", "--config", "/tmp/runtime.toml"]
    )

    assert args.config == Path("/tmp/runtime.toml")


def test_compare_gx_rhs_terms_parser_accepts_imported_geometry_args() -> None:
    tools_dir = Path(__file__).resolve().parents[1] / "tools"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_rhs_terms as mod
    finally:
        sys.path.remove(str(tools_dir))

    parser = mod.build_parser()
    args = parser.parse_args(
        [
            "--gx-dir",
            "/tmp/gx",
            "--gx-out",
            "/tmp/gx.out.nc",
            "--gx-input",
            "/tmp/gx.in",
            "--geometry-file",
            "/tmp/geom.nc",
        ]
    )

    assert args.gx_input == Path("/tmp/gx.in")
    assert args.geometry_file == Path("/tmp/geom.nc")


def test_compare_gx_rhs_terms_runtime_context_overrides_grid_from_dump(monkeypatch) -> None:
    tools_dir = Path(__file__).resolve().parents[1] / "tools"
    sys.path.insert(0, str(tools_dir))
    try:
        import compare_gx_rhs_terms as mod
    finally:
        sys.path.remove(str(tools_dir))

    cfg = type("Cfg", (), {"grid": type("Grid", (), {"Nx": 8, "Ny": 8, "Nz": 8, "y0": None})()})()
    captured: dict[str, object] = {}

    monkeypatch.setattr(mod, "load_runtime_from_toml", lambda _path: (cfg, None))
    monkeypatch.setattr(
        mod,
        "replace",
        lambda obj, **updates: type("Obj", (), obj.__dict__ | updates)(),
    )

    def _fake_build_runtime_geometry(cfg_use):
        captured["cfg_use"] = cfg_use
        return "geom"

    monkeypatch.setattr(mod, "build_runtime_geometry", _fake_build_runtime_geometry)
    monkeypatch.setattr(mod, "apply_gx_geometry_grid_defaults", lambda _geom, grid: grid)
    grid_obj = type("GridObj", (), {"ky": np.array([0.0, 0.2, -0.2]), "kx": np.array([0.0])})()
    monkeypatch.setattr(mod, "build_spectral_grid", lambda _grid: grid_obj)
    monkeypatch.setattr(mod, "build_runtime_linear_params", lambda *_args, **_kwargs: "params")
    monkeypatch.setattr(
        mod,
        "build_runtime_term_config",
        lambda _cfg: TermConfig(hypercollisions=1.0, end_damping=1.0),
    )

    cfg_use, geom, grid_full, params, term_cfg = mod._build_runtime_compare_context(
        Path("runtime.toml"),
        nx=3,
        ny_full=6,
        nz=5,
        nm=4,
        ky_vals=np.array([0.2, 0.4], dtype=float),
        y0_override=None,
    )

    assert cfg_use.grid.Nx == 3
    assert cfg_use.grid.Ny == 6
    assert cfg_use.grid.Nz == 5
    assert cfg_use.grid.y0 == 5.0
    assert captured["cfg_use"] is cfg_use
    assert geom == "geom"
    assert grid_full is grid_obj
    assert params == "params"
    assert term_cfg.hypercollisions == 0.0
    assert term_cfg.end_damping == 0.0


def test_run_kbm_linear_accepts_vmec_and_desc_eik_benchmark_aliases(tmp_path: Path) -> None:
    netcdf4 = pytest.importorskip("netCDF4")
    Dataset = netcdf4.Dataset

    grid = GridConfig(Nx=1, Ny=8, Nz=24, Lx=62.8, Ly=62.8, y0=10.0, ntheta=8, nperiod=2)
    cfg = KBMBaseCase(grid=grid)
    theta = np.linspace(-3.0 * np.pi, 3.0 * np.pi, grid.Nz + 1)
    analytic = SAlphaGeometry.from_config(cfg.geometry)
    sampled = sample_flux_tube_geometry(analytic, theta)
    path = tmp_path / "geom.eik.nc"
    with Dataset(path, "w") as root:
        root.createDimension("z", theta.size)
        root.createVariable("theta", "f8", ("z",))[:] = theta
        root.createVariable("bmag", "f8", ("z",))[:] = np.asarray(sampled.bmag_profile)
        root.createVariable("gds2", "f8", ("z",))[:] = np.asarray(sampled.gds2_profile)
        root.createVariable("gds21", "f8", ("z",))[:] = np.asarray(sampled.gds21_profile)
        root.createVariable("gds22", "f8", ("z",))[:] = np.asarray(sampled.gds22_profile)
        root.createVariable("cvdrift", "f8", ("z",))[:] = np.asarray(sampled.cv_profile)
        root.createVariable("gbdrift", "f8", ("z",))[:] = np.asarray(sampled.gb_profile)
        root.createVariable("cvdrift0", "f8", ("z",))[:] = np.asarray(sampled.cv0_profile)
        root.createVariable("gbdrift0", "f8", ("z",))[:] = np.asarray(sampled.gb0_profile)
        root.createVariable("jacob", "f8", ("z",))[:] = np.asarray(sampled.jacobian_profile)
        root.createVariable("grho", "f8", ("z",))[:] = np.asarray(sampled.grho_profile)
        root.createVariable("gradpar", "f8", ("z",))[:] = np.full(theta.size, sampled.gradpar_value)
        root.createVariable("q", "f8", ())[:] = sampled.q
        root.createVariable("shat", "f8", ())[:] = sampled.s_hat
        root.createVariable("Rmaj", "f8", ())[:] = sampled.R0
        root.createVariable("kxfac", "f8", ())[:] = sampled.kxfac
        root.createVariable("scale", "f8", ())[:] = sampled.theta_scale
        root.createVariable("nfp", "f8", ())[:] = sampled.nfp
        root.createVariable("alpha", "f8", ())[:] = sampled.alpha

    for model in ("vmec-eik", "desc-eik"):
        cfg_nc = replace(
            cfg,
            geometry=replace(
                cfg.geometry,
                model=model,
                geometry_file=str(path),
            ),
        )
        result = run_kbm_linear(
            ky_target=0.3,
            cfg=cfg_nc,
            Nl=4,
            Nm=6,
            dt=0.01,
            steps=40,
            solver="gx_time",
            sample_stride=2,
        )
        assert np.isfinite(result.gamma)
        assert np.isfinite(result.omega)
