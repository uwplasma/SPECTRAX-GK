from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from spectraxgk.from_gx.miller import (
    MillerCoreParams,
    _request_attr,
    _safe_denom,
    build_collocation_surfaces,
    cumulative_trapezoid,
    generate_miller_eik_internal,
    internal_miller_backend_available,
)


def _request() -> SimpleNamespace:
    return SimpleNamespace(
        ntheta=24,
        nperiod=1,
        rhoc=0.5,
        q=1.4,
        s_hat=0.8,
        R0=3.0,
        R_geo=3.0,
        shift=0.0,
        akappa=1.0,
        tri=0.0,
        akappri=0.0,
        tripri=0.0,
        betaprim=0.0,
    )


def test_safe_denom_and_request_attr() -> None:
    assert _safe_denom(0.0) > 0.0
    assert _safe_denom(-0.0) > 0.0
    arr = _safe_denom(np.array([0.0, -1.0e-40, 2.0]))
    assert np.all(np.abs(arr) > 0.0)

    req = SimpleNamespace(q=1.4, s_hat=0.9)
    assert _request_attr(req, "qinp", "q") == 1.4
    assert _request_attr(req, "shat", "s_hat") == 0.9
    with pytest.raises(AttributeError):
        _request_attr(req, "missing_a", "missing_b")


def test_cumulative_trapezoid_supports_1d_and_2d() -> None:
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 2.0])
    np.testing.assert_allclose(cumulative_trapezoid(y, x), [0.0, 0.5, 2.0])

    y2 = np.vstack([y, 2.0 * y])
    out = cumulative_trapezoid(y2, x, axis=1)
    np.testing.assert_allclose(out[0], [0.0, 0.5, 2.0])
    np.testing.assert_allclose(out[1], [0.0, 1.0, 4.0])

    x2 = np.vstack([x, x + 0.5])
    out2 = cumulative_trapezoid(y2, x2, axis=1)
    assert out2.shape == y2.shape

    with pytest.raises(ValueError):
        cumulative_trapezoid(np.array([1.0, 2.0]), np.array([[0.0, 1.0]]))
    with pytest.raises(NotImplementedError):
        cumulative_trapezoid(np.ones((2, 2)), np.ones(2), axis=0)
    with pytest.raises(ValueError):
        cumulative_trapezoid(np.ones((2, 2, 2)), np.ones((2, 2, 2)))


def test_miller_collocation_preserves_signed_shift_derivative() -> None:
    params = MillerCoreParams(
        ntgrid=8,
        nperiod=1,
        rhoc=0.5,
        qinp=1.4,
        shat=0.8,
        rmaj=3.0,
        r_geo=3.0,
        shift=-0.2,
        akappa=1.0,
        tri=0.0,
        akappri=0.0,
        tripri=0.0,
        betaprim=0.0,
        delrho=1.0e-3,
    )

    state = build_collocation_surfaces(params)
    r_midplane = np.asarray(state["r"])[:, 0]
    rho = np.asarray(state["rho"])
    d_rgeom_drho = np.gradient(r_midplane - rho, rho)[1]

    assert d_rgeom_drho == pytest.approx(params.shift, rel=1.0e-6)


def test_generate_miller_eik_internal_requires_request() -> None:
    with pytest.raises(NotImplementedError):
        generate_miller_eik_internal(output_path="/tmp/ignored.nc", request=None)


def test_generate_miller_eik_internal_writes_netcdf(tmp_path: Path) -> None:
    out = generate_miller_eik_internal(output_path=tmp_path / "miller.eiknc.nc", request=_request())
    assert out.exists()

    import netCDF4 as nc

    with nc.Dataset(out) as ds:
        assert "theta" in ds.variables
        assert "bmag" in ds.variables
        assert ds.variables["theta"][:].size > 0
        assert float(ds.variables["q"].getValue()) == pytest.approx(1.4)
        assert float(ds.variables["shat"].getValue()) == pytest.approx(0.8)


def test_internal_miller_backend_available_returns_bool() -> None:
    assert isinstance(internal_miller_backend_available(), bool)
