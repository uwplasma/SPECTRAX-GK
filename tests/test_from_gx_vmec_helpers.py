from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from spectraxgk.from_gx.vmec import (
    _apply_flux_tube_cut,
    _booz_xform_jax_search_paths,
    _equal_arc_remap,
    _import_booz_backend,
    _import_module_with_search_paths,
    dermv,
    generate_vmec_eik_internal,
    internal_vmec_backend_available,
    nperiod_set,
    write_vmec_eik_netcdf,
)


def test_booz_search_paths_include_env_and_src(monkeypatch, tmp_path: Path) -> None:
    checkout = tmp_path / "booz_xform_jax"
    (checkout / "src").mkdir(parents=True)
    monkeypatch.setenv("SPECTRAX_BOOZ_XFORM_JAX_PATH", str(checkout))
    paths = _booz_xform_jax_search_paths()
    assert checkout.resolve() in paths
    assert (checkout / "src").resolve() in paths


def test_import_module_with_search_paths_loads_temp_module(tmp_path: Path) -> None:
    pkg = tmp_path / "mods"
    pkg.mkdir()
    (pkg / "demo_mod.py").write_text("VALUE = 7\n", encoding="utf-8")
    sys.modules.pop("demo_mod", None)
    mod = _import_module_with_search_paths("demo_mod", [pkg])
    assert mod.VALUE == 7
    sys.modules.pop("demo_mod", None)


def test_import_module_with_search_paths_raises_on_missing(tmp_path: Path) -> None:
    with pytest.raises(ImportError):
        _import_module_with_search_paths("missing_demo_mod", [tmp_path / "missing"])


def test_import_booz_backend_falls_back_to_booz_xform(monkeypatch) -> None:
    monkeypatch.setattr("spectraxgk.from_gx.vmec._booz_xform_jax_search_paths", lambda: [])
    monkeypatch.setattr(
        "spectraxgk.from_gx.vmec._import_module_with_search_paths",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ImportError("jax backend missing")),
    )

    marker = SimpleNamespace(name="fallback")

    def _import_module(name: str):
        if name == "booz_xform":
            return marker
        raise ImportError(name)

    monkeypatch.setattr("spectraxgk.from_gx.vmec.importlib.import_module", _import_module)
    assert _import_booz_backend() is marker


def test_internal_vmec_backend_available_uses_backend_probe(monkeypatch) -> None:
    monkeypatch.setattr("spectraxgk.from_gx.vmec._import_booz_backend", lambda: object())
    assert internal_vmec_backend_available() is True
    monkeypatch.setattr(
        "spectraxgk.from_gx.vmec._import_booz_backend",
        lambda: (_ for _ in ()).throw(ImportError("missing")),
    )
    assert internal_vmec_backend_available() is False


def test_nperiod_set_and_dermv(monkeypatch) -> None:
    monkeypatch.setattr(
        "spectraxgk.from_gx.vmec.nperiod_contract",
        lambda values, theta, npol: (values[1:-1], theta[1:-1]),
    )
    values, theta = nperiod_set(np.array([0.0, 1.0, 2.0]), np.array([-2.0, 0.0, 2.0]), 1.0)
    np.testing.assert_allclose(values, [1.0])
    np.testing.assert_allclose(theta, [0.0])

    out = dermv(np.array([0.0, 1.0, 4.0, 9.0]), np.array([0.0, 1.0, 2.0, 3.0]))
    np.testing.assert_allclose(out[1:-1], [2.0, 4.0], atol=1.0e-6)

    with pytest.raises(ValueError):
        nperiod_set(np.array([1.0, 2.0]), np.array([1.0]), 1.0)
    with pytest.raises(ValueError):
        dermv(np.ones((2, 2)), np.ones((2, 2)))
    with pytest.raises(ValueError):
        dermv(np.ones(2), np.ones(3))


def test_apply_flux_tube_cut_none_and_unknown() -> None:
    theta = np.linspace(-np.pi, np.pi, 7)
    base = np.linspace(1.0, 2.0, theta.size)
    geo = SimpleNamespace(
        bmag=base[None, None, :],
        gradpar_theta_b=np.abs(base)[None, None, :],
        cvdrift=base[None, None, :],
        gbdrift=base[None, None, :],
        cvdrift0=base[None, None, :],
        gbdrift0=base[None, None, :],
        gds2=(base + 1.0)[None, None, :],
        gds21=np.linspace(-1.0, 1.0, theta.size)[None, None, :],
        gds22=(base + 2.0)[None, None, :],
        grho=base[None, None, :],
        R_b=(base + 3.0)[None, None, :],
        Z_b=(base - 1.0)[None, None, :],
        grad_x=np.stack([base, base + 1.0, base + 2.0])[:, None, None, :],
        grad_y=np.stack([base + 3.0, base + 4.0, base + 5.0])[:, None, None, :],
        s_hat_input=0.8,
    )

    theta_cut, arrays = _apply_flux_tube_cut(
        theta,
        geo,
        ntheta=theta.size,
        flux_tube_cut="none",
        npol_min=None,
        which_crossing=0,
        y0=1.0,
        x0=1.0,
        jtwist_in=None,
    )
    np.testing.assert_allclose(theta_cut, theta)
    assert arrays["grad_x"].shape == (3, theta.size)
    assert arrays["b_vec"].shape == (3, theta.size)

    with pytest.raises(ValueError):
        _apply_flux_tube_cut(
            theta,
            geo,
            ntheta=theta.size,
            flux_tube_cut="bad",
            npol_min=None,
            which_crossing=0,
            y0=1.0,
            x0=1.0,
            jtwist_in=None,
        )


def test_equal_arc_remap_returns_constant_gradpar() -> None:
    theta = np.linspace(-np.pi, np.pi, 7)
    arrays = {
        "theta_PEST": theta.copy(),
        "bmag": np.linspace(1.0, 2.0, theta.size),
        "gradpar": np.full(theta.size, 2.0),
        "cvdrift": np.linspace(0.0, 1.0, theta.size),
        "gbdrift": np.linspace(1.0, 0.0, theta.size),
        "cvdrift0": np.linspace(0.5, 1.5, theta.size),
        "gbdrift0": np.linspace(1.5, 0.5, theta.size),
        "gds2": np.linspace(2.0, 3.0, theta.size),
        "gds21": np.linspace(-0.5, 0.5, theta.size),
        "gds22": np.linspace(3.0, 4.0, theta.size),
        "grho": np.linspace(0.8, 1.2, theta.size),
        "Rplot": np.linspace(5.0, 6.0, theta.size),
        "Zplot": np.linspace(-1.0, 1.0, theta.size),
        "grad_x": np.vstack([np.ones(theta.size), 2.0 * np.ones(theta.size), 3.0 * np.ones(theta.size)]),
        "grad_y": np.vstack([4.0 * np.ones(theta.size), 5.0 * np.ones(theta.size), 6.0 * np.ones(theta.size)]),
    }

    gradpar_eqarc, out = _equal_arc_remap(theta, arrays, ntheta=9)
    assert np.isfinite(gradpar_eqarc)
    np.testing.assert_allclose(out["gradpar"], np.full(9, gradpar_eqarc))
    assert out["theta"].shape == (9,)
    assert out["grad_x"].shape == (3, 9)
    assert out["b_vec"].shape == (3, 9)
    assert np.isfinite(out["scale"])


def _mock_geo_for_cut(theta: np.ndarray, *, gds21: np.ndarray | None = None, gbdrift0: np.ndarray | None = None) -> SimpleNamespace:
    base = np.linspace(1.0, 2.0, theta.size)
    gds21_arr = gds21 if gds21 is not None else np.linspace(-1.0, 1.0, theta.size)
    gbdrift0_arr = gbdrift0 if gbdrift0 is not None else np.linspace(-1.0, 1.0, theta.size)
    return SimpleNamespace(
        bmag=base[None, None, :],
        gradpar_theta_b=np.abs(base)[None, None, :],
        cvdrift=base[None, None, :],
        gbdrift=base[None, None, :],
        cvdrift0=base[None, None, :],
        gbdrift0=gbdrift0_arr[None, None, :],
        gds2=(base + 1.0)[None, None, :],
        gds21=gds21_arr[None, None, :],
        gds22=np.ones(theta.size)[None, None, :],
        grho=base[None, None, :],
        R_b=(base + 3.0)[None, None, :],
        Z_b=(base - 1.0)[None, None, :],
        grad_x=np.stack([base, base + 1.0, base + 2.0])[:, None, None, :],
        grad_y=np.stack([base + 3.0, base + 4.0, base + 5.0])[:, None, None, :],
        s_hat_input=0.5,
    )


@pytest.mark.parametrize(
    ("flux_tube_cut", "geo", "expected_cut", "jtwist_in"),
    [
        ("gds21", _mock_geo_for_cut(np.linspace(-2.0, 2.0, 9), gds21=np.linspace(-2.0, 2.0, 9) - 0.5), 0.5, None),
        ("gbdrift0", _mock_geo_for_cut(np.linspace(-2.0, 2.0, 9), gbdrift0=np.linspace(-2.0, 2.0, 9) - 0.75), 0.75, None),
        ("aspect", _mock_geo_for_cut(np.linspace(-2.0, 2.0, 9), gds21=np.linspace(-2.0, 2.0, 9)), 1.0, 1),
    ],
)
def test_apply_flux_tube_cut_branch_specific_roots(
    flux_tube_cut: str,
    geo: SimpleNamespace,
    expected_cut: float,
    jtwist_in: int | None,
) -> None:
    theta = np.linspace(-2.0, 2.0, 9)
    theta_cut, arrays = _apply_flux_tube_cut(
        theta,
        geo,
        ntheta=11,
        flux_tube_cut=flux_tube_cut,
        npol_min=None,
        which_crossing=0,
        y0=1.0,
        x0=1.0,
        jtwist_in=jtwist_in,
    )

    assert theta_cut[0] == pytest.approx(-expected_cut, abs=1.0e-6)
    assert theta_cut[-1] == pytest.approx(expected_cut, abs=1.0e-6)
    assert arrays["theta"].shape == (11,)
    assert arrays["grad_x"].shape == (3, 11)


def test_write_vmec_eik_netcdf_writes_expected_variables(tmp_path: Path) -> None:
    netcdf4 = pytest.importorskip("netCDF4")
    path = tmp_path / "geom.eik.nc"
    theta = np.linspace(-np.pi, np.pi, 5)
    bmag = np.linspace(1.0, 2.0, theta.size)
    gradpar = np.full(theta.size, 0.5)
    profiles = {
        "theta": theta,
        "theta_PEST": theta + 0.1,
        "bmag": bmag,
        "gradpar": gradpar,
        "grho": np.full(theta.size, 1.2),
        "gds2": np.linspace(2.0, 3.0, theta.size),
        "gds21": np.linspace(-0.5, 0.5, theta.size),
        "gds22": np.linspace(3.0, 4.0, theta.size),
        "gbdrift": np.linspace(0.1, 0.2, theta.size),
        "gbdrift0": np.linspace(0.2, 0.3, theta.size),
        "cvdrift": np.linspace(0.3, 0.4, theta.size),
        "cvdrift0": np.linspace(0.4, 0.5, theta.size),
        "Rplot": np.linspace(5.0, 6.0, theta.size),
        "Zplot": np.linspace(-1.0, 1.0, theta.size),
        "grad_x": np.vstack([np.ones(theta.size), 2.0 * np.ones(theta.size), 3.0 * np.ones(theta.size)]),
        "grad_y": np.vstack([4.0 * np.ones(theta.size), 5.0 * np.ones(theta.size), 6.0 * np.ones(theta.size)]),
        "b_vec": np.vstack([np.ones(theta.size), np.zeros(theta.size), np.zeros(theta.size)]),
        "dpsidrho": 2.0,
        "kxfac": 1.7,
        "Rmaj": 5.5,
        "q": 1.4,
        "shat": 0.8,
        "scale": 2.0,
        "alpha": 0.25,
        "zeta_center": 0.125,
        "nfp": 5,
    }

    write_vmec_eik_netcdf(path, profiles, request=SimpleNamespace())

    with netcdf4.Dataset(path) as ds:
        np.testing.assert_allclose(ds.variables["theta"][:], theta)
        np.testing.assert_allclose(ds.variables["theta_PEST"][:], theta + 0.1)
        np.testing.assert_allclose(ds.variables["bmag"][:], bmag)
        np.testing.assert_allclose(ds.variables["gradpar"][:], gradpar)
        np.testing.assert_allclose(ds.variables["grad_x"][:, :], profiles["grad_x"])
        assert float(ds.variables["kxfac"].getValue()) == pytest.approx(1.7)
        assert float(ds.variables["Rmaj"].getValue()) == pytest.approx(5.5)
        assert int(ds.variables["nfp"].getValue()) == 5
        expected_jacob = 1.0 / abs((1.0 / abs(profiles["dpsidrho"])) * gradpar[0] * bmag)
        np.testing.assert_allclose(ds.variables["jacob"][:], expected_jacob)


def test_generate_vmec_eik_internal_maps_boundary_and_computes_betaprim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, object] = {}

    theta_out = np.linspace(-np.pi, np.pi, 9)
    arrays_gx = {
        "theta": theta_out,
        "theta_PEST": theta_out,
        "bmag": np.linspace(1.0, 2.0, theta_out.size),
        "gradpar": np.full(theta_out.size, 0.5),
        "cvdrift": np.linspace(0.0, 1.0, theta_out.size),
        "gbdrift": np.linspace(1.0, 2.0, theta_out.size),
        "cvdrift0": np.linspace(2.0, 3.0, theta_out.size),
        "gbdrift0": np.linspace(3.0, 4.0, theta_out.size),
        "gds2": np.linspace(4.0, 5.0, theta_out.size),
        "gds21": np.linspace(-1.0, 1.0, theta_out.size),
        "gds22": np.linspace(5.0, 6.0, theta_out.size),
        "grho": np.full(theta_out.size, 1.1),
        "Rplot": np.linspace(4.0, 6.0, theta_out.size),
        "Zplot": np.linspace(-1.0, 1.0, theta_out.size),
        "grad_x": np.vstack([np.ones(theta_out.size), 2.0 * np.ones(theta_out.size), 3.0 * np.ones(theta_out.size)]),
        "grad_y": np.vstack([4.0 * np.ones(theta_out.size), 5.0 * np.ones(theta_out.size), 6.0 * np.ones(theta_out.size)]),
        "b_vec": np.vstack([np.ones(theta_out.size), np.zeros(theta_out.size), np.zeros(theta_out.size)]),
        "scale": 1.0,
    }

    def _mock_fieldlines(**kwargs):
        calls["fieldlines"] = kwargs
        return SimpleNamespace(
            dpsidrho=2.0,
            iota_input=0.5,
            s_hat_input=0.8,
            nfp=5,
            alpha=0.125,
            zeta_center=0.25,
        )

    def _mock_cut(**kwargs):
        calls["cut"] = kwargs
        return np.linspace(-1.0, 1.0, 9), {"theta_PEST": np.linspace(-1.0, 1.0, 9)}

    def _mock_remap(**kwargs):
        calls["remap"] = kwargs
        return 0.5, arrays_gx

    def _mock_write(path: Path, profiles: dict[str, object], *, request: object) -> None:
        calls["write"] = {"path": path, "profiles": profiles, "request": request}

    monkeypatch.setattr("spectraxgk.from_gx.vmec._vmec_fieldlines", _mock_fieldlines)
    monkeypatch.setattr("spectraxgk.from_gx.vmec._apply_flux_tube_cut", _mock_cut)
    monkeypatch.setattr("spectraxgk.from_gx.vmec._equal_arc_remap", _mock_remap)
    monkeypatch.setattr("spectraxgk.from_gx.vmec.write_vmec_eik_netcdf", _mock_write)

    request = SimpleNamespace(
        vmec_file=str(tmp_path / "wout_test.nc"),
        torflux=0.64,
        beta=0.02,
        alpha=0.1,
        include_shear_variation=True,
        include_pressure_variation=False,
        npol=1.0,
        npol_min=None,
        ntheta=8,
        isaxisym=False,
        boundary="fix aspect",
        which_crossing=None,
        betaprim=None,
        z=(1.0, -1.0),
        dens=(2.0, 1.0),
        temp=(3.0, 0.5),
        tprim=(4.0, 5.0),
        fprim=(6.0, 7.0),
        y0=2.5,
        x0=None,
        jtwist=3,
    )

    out = generate_vmec_eik_internal(output_path=tmp_path / "out.eik.nc", request=request)

    assert out == (tmp_path / "out.eik.nc").resolve()
    assert calls["fieldlines"]["betaprim"] == pytest.approx(-0.02 * (2.0 * 3.0 * (4.0 + 6.0) + 1.0 * 0.5 * (5.0 + 7.0)))
    assert calls["cut"]["flux_tube_cut"] == "aspect"
    assert calls["cut"]["which_crossing"] == -1
    assert calls["cut"]["ntheta"] == 9
    assert calls["cut"]["x0"] == pytest.approx(2.5)
    profiles = calls["write"]["profiles"]
    assert profiles["q"] == pytest.approx(2.0)
    assert profiles["shat"] == pytest.approx(0.8)
    assert profiles["Rmaj"] == pytest.approx(5.0)
    assert profiles["alpha"] == pytest.approx(0.125)
    assert profiles["nfp"] == 5
