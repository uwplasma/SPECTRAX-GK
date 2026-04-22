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
    internal_vmec_backend_available,
    nperiod_set,
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
