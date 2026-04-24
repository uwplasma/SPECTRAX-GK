from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import spectraxgk
from spectraxgk.geometry.differentiable import (
    _candidate_paths,
    _find_importable_module,
    discover_differentiable_geometry_backends,
    flux_tube_geometry_from_mapping,
)


def _sample_mapping() -> dict[str, object]:
    theta = np.linspace(-np.pi, np.pi, 8, endpoint=False)
    ones = np.ones_like(theta)
    zeros = np.zeros_like(theta)
    return {
        "theta": theta,
        "gradpar": 0.7 * ones,
        "bmag": 1.0 + 0.05 * np.cos(theta),
        "bgrad": 0.05 * np.sin(theta),
        "gds2": ones,
        "gds21": zeros,
        "gds22": ones,
        "cvdrift": 0.2 * np.cos(theta),
        "gbdrift": 0.2 * np.cos(theta),
        "cvdrift0": zeros,
        "gbdrift0": zeros,
        "jacobian": ones,
        "grho": ones,
        "q": 1.7,
        "s_hat": 0.4,
        "R0": 1.5,
        "nfp": 5,
    }


def test_flux_tube_geometry_from_mapping_builds_solver_contract() -> None:
    assert spectraxgk.flux_tube_geometry_from_mapping is flux_tube_geometry_from_mapping
    geom = flux_tube_geometry_from_mapping(_sample_mapping(), source_model="vmec_jax:test")

    assert geom.source_model == "vmec_jax:test"
    assert geom.theta.shape == (8,)
    assert geom.nfp == 5
    assert geom.gradpar() == pytest.approx(0.7)
    assert np.allclose(np.asarray(geom.bmag(jnp.asarray(geom.theta))), np.asarray(geom.bmag_profile))


def test_flux_tube_geometry_from_mapping_rejects_bad_contracts() -> None:
    bad = _sample_mapping()
    bad.pop("bmag")
    with pytest.raises(ValueError, match="bmag"):
        flux_tube_geometry_from_mapping(bad)

    bad = _sample_mapping()
    bad["gds2"] = np.ones((2, 2))
    with pytest.raises(ValueError, match="gds2"):
        flux_tube_geometry_from_mapping(bad)

    bad = _sample_mapping()
    gradpar = np.asarray(bad["gradpar"]).copy()
    gradpar[-1] = 0.9
    bad["gradpar"] = gradpar
    with pytest.raises(ValueError, match="gradpar"):
        flux_tube_geometry_from_mapping(bad)

    bad = _sample_mapping()
    bad["gds21"] = np.ones(7)
    with pytest.raises(ValueError, match="length"):
        flux_tube_geometry_from_mapping(bad)

    bad = _sample_mapping()
    bmag = np.asarray(bad["bmag"]).copy()
    bmag[2] = np.nan
    bad["bmag"] = bmag
    with pytest.raises(ValueError, match="non-finite"):
        flux_tube_geometry_from_mapping(bad)


def test_flux_tube_geometry_from_mapping_uses_jax_native_defaults_and_shat_alias() -> None:
    data = _sample_mapping()
    data.pop("jacobian")
    data.pop("grho")
    data["shat"] = data.pop("s_hat")

    geom = flux_tube_geometry_from_mapping(data)

    expected_jacobian = 1.0 / np.asarray(data["gradpar"]) / np.asarray(data["bmag"])
    assert np.allclose(np.asarray(geom.jacobian_profile), expected_jacobian)
    assert np.allclose(np.asarray(geom.grho_profile), 1.0)
    assert geom.s_hat == pytest.approx(0.4)


def test_differentiable_backend_path_helpers_handle_missing_modules(tmp_path: Path, monkeypatch) -> None:
    existing = tmp_path / "backend"
    (existing / "src").mkdir(parents=True)
    monkeypatch.setenv("SPECTRAX_VMEC_JAX_PATH", str(existing))

    paths = _candidate_paths(("SPECTRAX_VMEC_JAX_PATH",), (existing, tmp_path / "missing"))

    assert paths == [existing.resolve(), (existing / "src").resolve()]
    assert _find_importable_module("spectraxgk_definitely_missing_backend", paths) is None


def test_discover_differentiable_geometry_backends_reports_optional_apis(tmp_path: Path, monkeypatch) -> None:
    vmec_root = tmp_path / "vmec_jax" / "src" / "vmec_jax"
    booz_root = tmp_path / "booz_xform_jax" / "src" / "booz_xform_jax"
    vmec_root.mkdir(parents=True)
    booz_root.mkdir(parents=True)
    (vmec_root / "__init__.py").write_text("marker = 'vmec'\n", encoding="utf-8")
    (booz_root / "__init__.py").write_text("marker = 'booz'\n", encoding="utf-8")
    (booz_root / "jax_api.py").write_text(
        "def prepare_booz_xform_constants_from_inputs(*args, **kwargs): return None\n"
        "def booz_xform_jax_impl(*args, **kwargs): return None\n",
        encoding="utf-8",
    )
    for name in ("vmec_jax", "booz_xform_jax", "booz_xform_jax.jax_api"):
        sys.modules.pop(name, None)
    monkeypatch.setenv("SPECTRAX_VMEC_JAX_PATH", str(tmp_path / "vmec_jax"))
    monkeypatch.setenv("SPECTRAX_BOOZ_XFORM_JAX_PATH", str(tmp_path / "booz_xform_jax"))

    info = discover_differentiable_geometry_backends()

    assert info["vmec_jax_available"] is True
    assert info["booz_xform_jax_available"] is True
    assert info["booz_xform_jax_api_available"] is True
