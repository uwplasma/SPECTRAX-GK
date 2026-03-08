"""Geometry helper tests."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spectraxgk.config import GeometryConfig, GridConfig
from spectraxgk.geometry import (
    SAlphaGeometry,
    apply_gx_geometry_grid_defaults,
    build_flux_tube_geometry,
    ensure_flux_tube_geometry_data,
    gx_twist_shift_params,
    load_gx_geometry_netcdf,
    sample_flux_tube_geometry,
)
from spectraxgk.grids import build_spectral_grid


def test_kperp2_matches_s_alpha():
    """k_perp^2 should match the s-alpha formula for kx(theta)."""
    geom = SAlphaGeometry(q=1.4, s_hat=1.0, epsilon=0.0)
    kx0 = jnp.array(0.0)
    ky = jnp.array(1.0)
    theta = jnp.array([0.0, 2.0])
    kperp2 = geom.k_perp2(kx0, ky, theta)
    assert jnp.allclose(kperp2[0], 1.0)
    assert jnp.allclose(kperp2[1], 5.0)


def test_geometry_from_config():
    """Geometry config should map cleanly into the geometry class."""
    cfg = GeometryConfig(q=1.7, s_hat=0.9, epsilon=0.2, R0=3.0, B0=2.0, alpha=0.1)
    geom = SAlphaGeometry.from_config(cfg)
    assert geom.q == 1.7
    assert geom.R0 == 3.0
    assert geom.alpha == 0.1


def test_build_flux_tube_geometry_analytic_from_config():
    cfg = GeometryConfig(q=1.7, s_hat=0.9, epsilon=0.2, R0=3.0, B0=2.0, alpha=0.1)
    geom = build_flux_tube_geometry(cfg)

    assert isinstance(geom, SAlphaGeometry)
    assert geom.q == 1.7


def test_bmag_and_omega_d_shapes():
    """Magnetic field and drift frequency should have consistent shapes."""
    geom = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.1)
    theta = jnp.array([0.0])
    bmag = geom.bmag(theta)
    assert jnp.isclose(bmag[0], 1.0 / (1.0 + geom.epsilon))
    assert jnp.isclose(geom.gradpar(), jnp.abs(1.0 / (geom.q * geom.R0)))

    grid = build_spectral_grid(GridConfig(Nx=4, Ny=4, Nz=8, Lx=6.0, Ly=6.0))
    omega_d = geom.omega_d(grid.kx, grid.ky, grid.z)
    assert omega_d.shape == (grid.ky.size, grid.kx.size, grid.z.size)


def test_metric_and_drift_coeffs_at_midplane():
    """Metric and drift coefficients should reduce cleanly at theta=0."""
    geom = SAlphaGeometry(q=1.4, s_hat=0.7, epsilon=0.0, R0=2.0, alpha=0.2)
    theta = jnp.array([0.0])
    gds2, gds21, gds22 = geom.metric_coeffs(theta)
    assert jnp.isclose(gds2[0], 1.0)
    assert jnp.isclose(gds21[0], 0.0)
    assert jnp.isclose(gds22, geom.s_hat * geom.s_hat)

    cv, gb, cv0, gb0 = geom.drift_coeffs(theta)
    expected = geom.drift_scale * (1.0 / geom.R0)
    assert jnp.isclose(cv[0], expected)
    assert jnp.isclose(gb[0], cv[0])
    assert jnp.isclose(cv0[0], 0.0)
    assert jnp.isclose(gb0[0], 0.0)

    cv_d, gb_d = geom.drift_components(jnp.array([0.0]), jnp.array([1.0]), theta)
    assert cv_d.shape == (1, 1, 1)
    assert gb_d.shape == (1, 1, 1)
    bgrad = geom.bgrad(theta)
    assert jnp.isfinite(bgrad[0])


def test_kx_effective_shear_shift():
    """kx_effective should include the s-alpha shear shift."""
    geom = SAlphaGeometry(q=1.4, s_hat=1.0, epsilon=0.0, alpha=0.5)
    kx0 = jnp.array([0.2])
    ky = jnp.array([0.3])
    theta = jnp.array([1.0])
    kx_eff = geom.kx_effective(kx0, ky, theta)
    shear = geom.s_hat * theta - geom.alpha * jnp.sin(theta)
    assert jnp.isclose(kx_eff[0], kx0[0] - shear[0] * ky[0])


def test_geometry_tree_roundtrip():
    """Geometry pytree should round-trip through flatten/unflatten."""
    geom = SAlphaGeometry(q=1.5, s_hat=0.8, epsilon=0.2, R0=3.0, B0=1.8, alpha=0.1)
    children, aux = geom.tree_flatten()
    geom2 = SAlphaGeometry.tree_unflatten(aux, children)
    assert geom2.q == geom.q
    assert geom2.s_hat == geom.s_hat
    assert geom2.epsilon == geom.epsilon
    assert geom2.R0 == geom.R0
    assert geom2.B0 == geom.B0
    assert geom2.alpha == geom.alpha


def test_sampled_flux_tube_geometry_matches_salpha_profiles():
    """Sampled geometry data should preserve the analytic s-alpha profiles."""
    geom = SAlphaGeometry(q=1.4, s_hat=0.8, epsilon=0.18, R0=2.77778, alpha=0.1)
    theta = jnp.linspace(-jnp.pi, jnp.pi, 17)
    sampled = sample_flux_tube_geometry(geom, theta)

    assert jnp.allclose(sampled.bmag(theta), geom.bmag(theta))
    assert jnp.allclose(sampled.bgrad(theta), geom.bgrad(theta))
    gds2_s, gds21_s, gds22_s = sampled.metric_coeffs(theta)
    gds2_g, gds21_g, gds22_g = geom.metric_coeffs(theta)
    assert jnp.allclose(gds2_s, gds2_g)
    assert jnp.allclose(gds21_s, gds21_g)
    assert jnp.allclose(gds22_s, jnp.full_like(theta, gds22_g))

    kx = jnp.array([0.0, 0.2])
    ky = jnp.array([0.1, 0.3])
    theta_b = theta[None, None, :]
    assert jnp.allclose(
        sampled.k_perp2(kx[None, :, None], ky[:, None, None], theta_b),
        geom.k_perp2(kx[None, :, None], ky[:, None, None], theta_b),
    )


def test_sampled_flux_tube_geometry_tree_roundtrip():
    """Sampled geometry should behave as a pytree for JAX transforms."""

    geom = SAlphaGeometry(q=1.8, s_hat=0.6, epsilon=0.14, R0=2.4, alpha=0.2)
    theta = jnp.linspace(-jnp.pi, jnp.pi, 9)
    sampled = sample_flux_tube_geometry(geom, theta)

    leaves, treedef = jax.tree_util.tree_flatten(sampled)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)

    assert restored.source_model == sampled.source_model
    assert jnp.allclose(restored.theta, sampled.theta)
    assert jnp.allclose(restored.bmag_profile, sampled.bmag_profile)
    assert jnp.allclose(restored.cv_profile, sampled.cv_profile)


def test_ensure_flux_tube_geometry_data_reuses_sampled_input():
    """The geometry contract helper should preserve pre-sampled geometry objects."""

    geom = SAlphaGeometry(q=1.7, s_hat=0.9, epsilon=0.1)
    theta = jnp.linspace(-jnp.pi, jnp.pi, 13)
    sampled = sample_flux_tube_geometry(geom, theta)

    ensured = ensure_flux_tube_geometry_data(sampled, theta)

    assert ensured is sampled


def test_ensure_flux_tube_geometry_data_trims_closed_theta_interval():
    """Imported GX geometry should drop the terminal theta point for solver grids."""

    geom = SAlphaGeometry(q=1.7, s_hat=0.9, epsilon=0.1)
    theta_closed = jnp.linspace(-jnp.pi, jnp.pi, 17)
    theta_solver = theta_closed[:-1]
    sampled = sample_flux_tube_geometry(geom, theta_closed)

    ensured = ensure_flux_tube_geometry_data(sampled, theta_solver)

    assert ensured is not sampled
    assert ensured.theta.shape == theta_solver.shape
    assert jnp.allclose(ensured.theta, theta_solver)
    assert jnp.allclose(ensured.bmag_profile, sampled.bmag_profile[:-1])


def test_load_gx_geometry_netcdf_reads_sampled_contract(tmp_path):
    """GX-style NetCDF geometry output should map into the sampled contract."""

    netcdf4 = pytest.importorskip("netCDF4")
    Dataset = netcdf4.Dataset

    path = tmp_path / "geom.out.nc"
    theta = np.linspace(-np.pi, np.pi, 5)
    with Dataset(path, "w") as root:
        root.createDimension("theta", theta.size)
        grids = root.createGroup("Grids")
        geom = root.createGroup("Geometry")
        grids.createVariable("theta", "f8", ("theta",))[:] = theta
        for name, values in {
            "bmag": np.linspace(1.0, 1.2, theta.size),
            "bgrad": np.linspace(-0.1, 0.1, theta.size),
            "gds2": np.linspace(1.0, 2.0, theta.size),
            "gds21": np.linspace(-0.2, 0.2, theta.size),
            "gds22": np.full(theta.size, 0.8),
            "cvdrift": np.linspace(0.3, 0.5, theta.size),
            "gbdrift": np.linspace(0.3, 0.5, theta.size),
            "cvdrift0": np.linspace(-0.1, 0.1, theta.size),
            "gbdrift0": np.linspace(-0.1, 0.1, theta.size),
            "jacobian": np.linspace(2.0, 3.0, theta.size),
            "grho": np.linspace(1.0, 1.4, theta.size),
        }.items():
            geom.createVariable(name, "f8", ("theta",))[:] = values
        for name, value in {
            "gradpar": 0.4,
            "q": 1.7,
            "shat": 0.6,
            "rmaj": 5.0,
            "aminor": 1.0,
            "kxfac": 1.3,
            "theta_scale": 2.0,
            "nfp": 5.0,
            "alpha": 0.2,
        }.items():
            geom.createVariable(name, "f8", ())[:] = value

    loaded = load_gx_geometry_netcdf(path)

    assert loaded.source_model == "gx-netcdf"
    assert jnp.allclose(loaded.theta, theta)
    assert jnp.allclose(loaded.jacobian_profile, np.linspace(2.0, 3.0, theta.size))
    assert jnp.allclose(loaded.grho_profile, np.linspace(1.0, 1.4, theta.size))
    assert loaded.kxfac == pytest.approx(1.3)
    assert loaded.theta_scale == pytest.approx(2.0)
    assert loaded.nfp == 5
    assert loaded.epsilon == pytest.approx(0.2)


def test_load_gx_geometry_netcdf_reads_root_level_eik_layout(tmp_path):
    """Root-level GX eik.nc geometry should map into the sampled contract."""

    netcdf4 = pytest.importorskip("netCDF4")
    Dataset = netcdf4.Dataset

    path = tmp_path / "geom.eik.nc"
    theta = np.linspace(-np.pi, np.pi, 5)
    bmag = np.linspace(1.0, 1.2, theta.size)
    with Dataset(path, "w") as root:
        root.createDimension("z", theta.size)
        root.createVariable("theta", "f8", ("z",))[:] = theta
        root.createVariable("bmag", "f8", ("z",))[:] = bmag
        root.createVariable("gds2", "f8", ("z",))[:] = np.linspace(1.0, 2.0, theta.size)
        root.createVariable("gds21", "f8", ("z",))[:] = np.linspace(-0.2, 0.2, theta.size)
        root.createVariable("gds22", "f8", ("z",))[:] = np.full(theta.size, 0.8)
        root.createVariable("cvdrift", "f8", ("z",))[:] = np.linspace(0.3, 0.5, theta.size)
        root.createVariable("gbdrift", "f8", ("z",))[:] = np.linspace(0.3, 0.5, theta.size)
        root.createVariable("cvdrift0", "f8", ("z",))[:] = np.linspace(-0.1, 0.1, theta.size)
        root.createVariable("gbdrift0", "f8", ("z",))[:] = np.linspace(-0.1, 0.1, theta.size)
        root.createVariable("jacob", "f8", ("z",))[:] = np.linspace(2.0, 3.0, theta.size)
        root.createVariable("grho", "f8", ("z",))[:] = np.linspace(1.0, 1.4, theta.size)
        root.createVariable("gradpar", "f8", ("z",))[:] = np.full(theta.size, 0.4)
        root.createVariable("q", "f8", ())[:] = 1.7
        root.createVariable("shat", "f8", ())[:] = 0.6
        root.createVariable("Rmaj", "f8", ())[:] = 5.0
        root.createVariable("kxfac", "f8", ())[:] = 1.3
        root.createVariable("scale", "f8", ())[:] = 2.0
        root.createVariable("nfp", "f8", ())[:] = 5.0
        root.createVariable("alpha", "f8", ())[:] = 0.2

    loaded = load_gx_geometry_netcdf(path)

    assert loaded.source_model == "gx-netcdf"
    assert jnp.allclose(loaded.theta, theta)
    assert jnp.allclose(loaded.jacobian_profile, np.linspace(2.0, 3.0, theta.size))
    assert jnp.allclose(loaded.grho_profile, np.linspace(1.0, 1.4, theta.size))
    assert loaded.kxfac == pytest.approx(1.3)
    assert loaded.theta_scale == pytest.approx(2.0)
    assert loaded.nfp == 5
    assert loaded.R0 == pytest.approx(5.0)
    assert np.all(np.isfinite(np.asarray(loaded.bgrad_profile)))


def test_build_flux_tube_geometry_loads_gx_netcdf(tmp_path):
    netcdf4 = pytest.importorskip("netCDF4")
    Dataset = netcdf4.Dataset

    path = tmp_path / "geom.out.nc"
    theta = np.linspace(-np.pi, np.pi, 5)
    with Dataset(path, "w") as root:
        root.createDimension("theta", theta.size)
        grids = root.createGroup("Grids")
        geom = root.createGroup("Geometry")
        grids.createVariable("theta", "f8", ("theta",))[:] = theta
        for name in (
            "bmag",
            "bgrad",
            "gds2",
            "gds21",
            "gds22",
            "cvdrift",
            "gbdrift",
            "cvdrift0",
            "gbdrift0",
            "jacobian",
            "grho",
        ):
            geom.createVariable(name, "f8", ("theta",))[:] = np.ones(theta.size)
        geom.createVariable("gradpar", "f8", ())[:] = 0.4
        geom.createVariable("q", "f8", ())[:] = 1.7
        geom.createVariable("shat", "f8", ())[:] = 0.6
        geom.createVariable("rmaj", "f8", ())[:] = 5.0
        geom.createVariable("aminor", "f8", ())[:] = 1.0

    loaded = build_flux_tube_geometry(GeometryConfig(model="gx-netcdf", geometry_file=str(path)))

    assert loaded.source_model == "gx-netcdf"


def test_apply_gx_geometry_grid_defaults_uses_imported_theta_and_kxfac(tmp_path):
    netcdf4 = pytest.importorskip("netCDF4")
    Dataset = netcdf4.Dataset

    path = tmp_path / "geom.eik.nc"
    theta = np.linspace(-3.0 * np.pi, 3.0 * np.pi, 9)
    with Dataset(path, "w") as root:
        root.createDimension("z", theta.size)
        root.createVariable("theta", "f8", ("z",))[:] = theta
        root.createVariable("bmag", "f8", ("z",))[:] = np.ones(theta.size)
        root.createVariable("gds2", "f8", ("z",))[:] = np.ones(theta.size)
        root.createVariable("gds21", "f8", ("z",))[:] = np.ones(theta.size)
        root.createVariable("gds22", "f8", ("z",))[:] = np.full(theta.size, 0.5)
        root.createVariable("cvdrift", "f8", ("z",))[:] = np.zeros(theta.size)
        root.createVariable("gbdrift", "f8", ("z",))[:] = np.zeros(theta.size)
        root.createVariable("cvdrift0", "f8", ("z",))[:] = np.zeros(theta.size)
        root.createVariable("gbdrift0", "f8", ("z",))[:] = np.zeros(theta.size)
        root.createVariable("jacob", "f8", ("z",))[:] = np.ones(theta.size)
        root.createVariable("grho", "f8", ("z",))[:] = np.ones(theta.size)
        root.createVariable("gradpar", "f8", ("z",))[:] = np.full(theta.size, 0.4)
        root.createVariable("q", "f8", ())[:] = 1.7
        root.createVariable("shat", "f8", ())[:] = 0.5
        root.createVariable("Rmaj", "f8", ())[:] = 5.0
        root.createVariable("kxfac", "f8", ())[:] = 1.7

    geom = load_gx_geometry_netcdf(path)
    grid = GridConfig(Nx=4, Ny=4, Nz=16, Lx=6.28, Ly=6.28, boundary="linked", y0=10.0)
    adjusted = apply_gx_geometry_grid_defaults(geom, grid)
    jtwist, x0 = gx_twist_shift_params(geom, adjusted)

    assert adjusted.Nz == theta.size - 1
    assert adjusted.z_min == pytest.approx(theta[0])
    assert adjusted.z_max == pytest.approx(theta[-1])
    assert adjusted.kxfac == pytest.approx(1.7)
    assert adjusted.jtwist == jtwist
    assert adjusted.Lx == pytest.approx(2.0 * np.pi * x0)
