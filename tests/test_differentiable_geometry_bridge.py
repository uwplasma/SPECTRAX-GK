from __future__ import annotations

import sys
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import spectraxgk
import spectraxgk.geometry.autodiff_checks as diff_autodiff
import spectraxgk.geometry.backend_discovery as backend_discovery
import spectraxgk.geometry.differentiable as diff_geom
import spectraxgk.geometry.flux_tube_contract as geom_contract
import spectraxgk.geometry.numerics as geom_numerics
import spectraxgk.geometry.sensitivity as geom_sensitivity
from spectraxgk.geometry.differentiable import (
    _array_parity_metrics,
    _boozer_half_mesh_s_grid,
    _candidate_paths,
    _cumulative_trapezoid,
    _find_importable_module,
    _interp_equal_arc_profile,
    _interp_radial,
    _is_traced,
    _periodic_bilinear_sample_2d,
    _radial_derivative_array,
    _radial_derivative_profile,
    _scalar_parity_metrics,
    booz_xform_flux_tube_sensitivity_report,
    booz_xform_spectral_sensitivity_report,
    discover_differentiable_geometry_backends,
    finite_difference_jacobian,
    flux_tube_geometry_from_mapping,
    flux_tube_geometry_from_vmec_boozer_state,
    flux_tube_geometry_observables,
    geometry_inverse_design_report,
    geometry_observable_names,
    geometry_sensitivity_report,
    vmec_jax_boozer_equal_arc_core_profiles_from_state,
    vmec_jax_boozer_flux_tube_sensitivity_report,
    vmec_jax_field_line_tensor_sensitivity_report,
    vmec_jax_flux_tube_array_parity_report,
    vmec_jax_flux_tube_sensitivity_report,
    vmec_jax_metric_tensor_sensitivity_report,
    vmec_boundary_aspect_sensitivity_report,
    vmec_field_line_tensor_observable_names,
    vmec_metric_tensor_observable_names,
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


def test_differentiable_geometry_facade_preserves_split_symbol_identity() -> None:
    """The legacy geometry bridge remains a compatibility facade."""

    assert diff_geom._candidate_paths is backend_discovery._candidate_paths
    assert (
        diff_geom._find_importable_module is backend_discovery._find_importable_module
    )
    assert diff_geom._is_traced is backend_discovery._is_traced
    assert (
        diff_geom.discover_differentiable_geometry_backends
        is backend_discovery.discover_differentiable_geometry_backends
    )
    assert (
        diff_geom.finite_difference_jacobian is diff_autodiff.finite_difference_jacobian
    )
    assert (
        diff_geom.observable_gradient_validation_report
        is diff_autodiff.observable_gradient_validation_report
    )
    assert diff_geom._array_parity_metrics is geom_numerics._array_parity_metrics
    assert diff_geom._scalar_parity_metrics is geom_numerics._scalar_parity_metrics
    assert diff_geom._interp_radial is geom_numerics._interp_radial
    assert (
        diff_geom._interp_equal_arc_profile is geom_numerics._interp_equal_arc_profile
    )
    assert diff_geom._boozer_half_mesh_s_grid is geom_numerics._boozer_half_mesh_s_grid
    assert (
        diff_geom._radial_derivative_profile is geom_numerics._radial_derivative_profile
    )
    assert diff_geom._radial_derivative_array is geom_numerics._radial_derivative_array
    assert diff_geom._cumulative_trapezoid is geom_numerics._cumulative_trapezoid
    assert (
        diff_geom._periodic_bilinear_sample_2d
        is geom_numerics._periodic_bilinear_sample_2d
    )
    assert diff_geom._array is geom_contract._array
    assert diff_geom._scalar is geom_contract._scalar
    assert (
        diff_geom.flux_tube_geometry_from_mapping
        is geom_contract.flux_tube_geometry_from_mapping
    )
    assert (
        diff_geom.flux_tube_geometry_observables
        is geom_contract.flux_tube_geometry_observables
    )
    assert (
        diff_geom.geometry_observable_names is geom_contract.geometry_observable_names
    )
    assert (
        diff_geom.vmec_metric_tensor_observable_names
        is geom_contract.vmec_metric_tensor_observable_names
    )
    assert (
        diff_geom.vmec_field_line_tensor_observable_names
        is geom_contract.vmec_field_line_tensor_observable_names
    )
    assert (
        diff_geom.geometry_sensitivity_report
        is geom_sensitivity.geometry_sensitivity_report
    )
    assert (
        diff_geom.geometry_inverse_design_report
        is geom_sensitivity.geometry_inverse_design_report
    )


def test_flux_tube_geometry_from_mapping_builds_solver_contract() -> None:
    assert spectraxgk.flux_tube_geometry_from_mapping is flux_tube_geometry_from_mapping
    assert (
        spectraxgk.flux_tube_geometry_from_vmec_boozer_state
        is flux_tube_geometry_from_vmec_boozer_state
    )
    assert spectraxgk.geometry_observable_names is geometry_observable_names
    assert spectraxgk.flux_tube_geometry_observables is flux_tube_geometry_observables
    geom = flux_tube_geometry_from_mapping(
        _sample_mapping(), source_model="vmec_jax:test"
    )

    assert geom.source_model == "vmec_jax:test"
    assert geom.theta.shape == (8,)
    assert geom.nfp == 5
    assert geom.gradpar() == pytest.approx(0.7)
    assert np.allclose(
        np.asarray(geom.bmag(jnp.asarray(geom.theta))), np.asarray(geom.bmag_profile)
    )
    observables = np.asarray(flux_tube_geometry_observables(geom))
    assert observables.shape == (len(geometry_observable_names()),)
    assert np.all(np.isfinite(observables))


def test_equal_arc_interpolation_keeps_value_gradients_finite() -> None:
    theta_uniform = jnp.linspace(-jnp.pi, jnp.pi, 9)
    theta_base = jnp.linspace(-jnp.pi, jnp.pi, 9)

    def remapped_mean(scale: jnp.ndarray) -> jnp.ndarray:
        theta_equal_arc = theta_base + 0.05 * scale * jnp.sin(theta_base)
        values = (1.0 + scale) * jnp.cos(theta_base)
        return jnp.mean(
            _interp_equal_arc_profile(theta_uniform, theta_equal_arc, values)
        )

    scale = jnp.asarray(0.2)
    step = jnp.asarray(1.0e-3)
    grad = jax.grad(remapped_mean)(scale)
    fd = (remapped_mean(scale + step) - remapped_mean(scale - step)) / (2.0 * step)

    assert np.isfinite(float(grad))
    assert float(grad) == pytest.approx(float(fd), rel=2.0e-3, abs=2.0e-5)


def test_flux_tube_geometry_from_vmec_boozer_state_wraps_in_memory_bridge(
    monkeypatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_core_profiles(state, static, indata, wout, **kwargs):  # noqa: ANN001, ANN202
        calls.append(
            {
                "state": state,
                "static": static,
                "indata": indata,
                "wout": wout,
                **kwargs,
            }
        )
        return _sample_mapping()

    monkeypatch.setattr(
        diff_geom,
        "vmec_jax_boozer_equal_arc_core_profiles_from_state",
        fake_core_profiles,
    )

    geom = flux_tube_geometry_from_vmec_boozer_state(
        "state",
        "static",
        "indata",
        "wout",
        surface_index=3,
        torflux=0.42,
        alpha=0.25,
        ntheta=16,
        mboz=21,
        nboz=23,
        jit=True,
        surface_stencil_width=5,
        reference_length=1.7,
        reference_b=2.3,
        source_model="unit-test-vmec-boozer",
    )

    assert geom.source_model == "unit-test-vmec-boozer"
    assert geom.theta.shape == (8,)
    assert len(calls) == 1
    assert calls[0]["surface_index"] == 3
    assert calls[0]["torflux"] == 0.42
    assert calls[0]["alpha"] == 0.25
    assert calls[0]["ntheta"] == 16
    assert calls[0]["mboz"] == 21
    assert calls[0]["nboz"] == 23
    assert calls[0]["jit"] is True
    assert calls[0]["surface_stencil_width"] == 5
    assert calls[0]["reference_length"] == 1.7
    assert calls[0]["reference_b"] == 2.3


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
    bad["q"] = np.ones(2)
    with pytest.raises(ValueError, match="q"):
        flux_tube_geometry_from_mapping(bad)

    bad = _sample_mapping()
    bmag = np.asarray(bad["bmag"]).copy()
    bmag[2] = np.nan
    bad["bmag"] = bmag
    with pytest.raises(ValueError, match="non-finite"):
        flux_tube_geometry_from_mapping(bad)

    bad = _sample_mapping()
    bad["q"] = np.nan
    with pytest.raises(ValueError, match="q.*non-finite"):
        flux_tube_geometry_from_mapping(bad)

    bad = _sample_mapping()
    bad["nfp"] = 0
    with pytest.raises(ValueError, match="nfp"):
        flux_tube_geometry_from_mapping(bad)

    bad = _sample_mapping()
    for key, value in list(bad.items()):
        if isinstance(value, np.ndarray):
            bad[key] = value[:0]
    with pytest.raises(ValueError, match="theta"):
        flux_tube_geometry_from_mapping(bad)


def test_flux_tube_geometry_from_mapping_uses_jax_native_defaults_and_shat_alias() -> (
    None
):
    data = _sample_mapping()
    data.pop("jacobian")
    data.pop("grho")
    data["shat"] = data.pop("s_hat")

    geom = flux_tube_geometry_from_mapping(data)

    expected_jacobian = 1.0 / np.asarray(data["gradpar"]) / np.asarray(data["bmag"])
    assert np.allclose(np.asarray(geom.jacobian_profile), expected_jacobian)
    assert np.allclose(np.asarray(geom.grho_profile), 1.0)
    assert geom.s_hat == pytest.approx(0.4)


def test_differentiable_backend_path_helpers_handle_missing_modules(
    tmp_path: Path, monkeypatch
) -> None:
    existing = tmp_path / "backend"
    (existing / "src").mkdir(parents=True)
    monkeypatch.setenv("SPECTRAX_VMEC_JAX_PATH", str(existing))

    paths = _candidate_paths(
        ("SPECTRAX_VMEC_JAX_PATH",), (existing, tmp_path / "missing")
    )

    assert paths == [existing.resolve(), (existing / "src").resolve()]
    assert (
        _find_importable_module("spectraxgk_definitely_missing_backend", paths) is None
    )


def test_differentiable_backend_path_helpers_prefer_configured_checkout(
    tmp_path: Path, monkeypatch
) -> None:
    installed_root = tmp_path / "installed"
    local_root = tmp_path / "local_vmec"
    installed_pkg = installed_root / "vmec_jax"
    local_pkg = local_root / "vmec_jax"
    installed_pkg.mkdir(parents=True)
    local_pkg.mkdir(parents=True)
    (installed_pkg / "__init__.py").write_text(
        "marker = 'installed'\n", encoding="utf-8"
    )
    (local_pkg / "__init__.py").write_text("marker = 'local'\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(installed_root))
    sys.modules.pop("vmec_jax", None)

    installed = __import__("vmec_jax")
    assert installed.marker == "installed"

    module = _find_importable_module("vmec_jax", [local_root])

    assert module is not None
    assert module.marker == "local"
    assert str(local_pkg) in str(module.__file__)


def test_discover_differentiable_geometry_backends_reports_optional_apis(
    tmp_path: Path, monkeypatch
) -> None:
    vmec_root = tmp_path / "vmec_jax" / "src" / "vmec_jax"
    booz_root = tmp_path / "booz_xform_jax" / "src" / "booz_xform_jax"
    vmec_root.mkdir(parents=True)
    booz_root.mkdir(parents=True)
    (vmec_root / "__init__.py").write_text("marker = 'vmec'\n", encoding="utf-8")
    (booz_root / "__init__.py").write_text("marker = 'booz'\n", encoding="utf-8")
    (booz_root / "jax_api.py").write_text(
        "def prepare_booz_xform_constants_from_inputs(*args, **kwargs): return None\n"
        "def booz_xform_from_inputs(*args, **kwargs): return None\n"
        "def booz_xform_jax_impl(*args, **kwargs): return None\n",
        encoding="utf-8",
    )
    for name in ("vmec_jax", "booz_xform_jax", "booz_xform_jax.jax_api"):
        sys.modules.pop(name, None)
    monkeypatch.setenv("SPECTRAX_VMEC_JAX_PATH", str(tmp_path / "vmec_jax"))
    monkeypatch.setenv("SPECTRAX_BOOZ_XFORM_JAX_PATH", str(tmp_path / "booz_xform_jax"))

    info = discover_differentiable_geometry_backends()

    assert info["vmec_jax_available"] is True
    assert info["vmec_jax_boundary_api_available"] is False
    assert info["booz_xform_jax_available"] is True
    assert info["booz_xform_jax_api_available"] is True


def test_vmec_boundary_aspect_sensitivity_report_uses_discovered_jax_api(
    tmp_path: Path, monkeypatch
) -> None:
    vmec_root = tmp_path / "vmec_jax" / "src" / "vmec_jax"
    vmec_root.mkdir(parents=True)
    (vmec_root / "__init__.py").write_text(
        "import jax.numpy as jnp\n"
        "class BoundaryCoeffs:\n"
        "    def __init__(self, R_cos, R_sin, Z_cos, Z_sin):\n"
        "        self.R_cos = R_cos; self.R_sin = R_sin; self.Z_cos = Z_cos; self.Z_sin = Z_sin\n"
        "class _Modes:\n"
        "    K = 2\n"
        "def vmec_mode_table(*args, **kwargs): return _Modes()\n"
        "def make_angle_grid(*args, **kwargs): return object()\n"
        "def build_helical_basis(*args, **kwargs): return object()\n"
        "def boundary_aspect_ratio(boundary, basis):\n"
        "    return 2.0 * boundary.R_cos[1] + 0.5 * boundary.Z_sin[1]\n",
        encoding="utf-8",
    )
    sys.modules.pop("vmec_jax", None)
    monkeypatch.setenv("SPECTRAX_VMEC_JAX_PATH", str(tmp_path / "vmec_jax"))

    report = vmec_boundary_aspect_sensitivity_report(
        jnp.asarray([0.08, 0.2]), fd_step=1.0e-3
    )

    assert report["available"] is True
    assert report["backend_info"]["vmec_jax_boundary_api_available"] is True
    assert float(report["max_abs_ad_fd_error"]) < 2.0e-5
    conditioning = report["conditioning"]
    assert conditioning["jacobian_shape"] == [1, 2]
    assert conditioning["sensitivity_map_rank"] == 1
    assert conditioning["worst_abs_error"]["observable_name"] == "aspect_ratio"
    assert conditioning["finite_difference_step_by_parameter"][0]["parameter_name"] == (
        "ripple"
    )


def test_vmec_boundary_aspect_sensitivity_report_validates_parameter_shape() -> None:
    with pytest.raises(ValueError, match="length-2"):
        vmec_boundary_aspect_sensitivity_report(jnp.ones(3))


def test_booz_xform_spectral_sensitivity_report_is_bounded_when_available() -> None:
    for name in ("booz_xform_jax", "booz_xform_jax.jax_api"):
        sys.modules.pop(name, None)

    report = booz_xform_spectral_sensitivity_report(ripple=0.05, fd_step=2.0e-5)

    assert (
        spectraxgk.booz_xform_spectral_sensitivity_report
        is booz_xform_spectral_sensitivity_report
    )
    assert "available" in report
    if not report["available"]:
        assert report["objective"] is None
        return

    assert float(report["objective"]) > 0.0
    assert float(report["max_abs_ad_fd_error"]) < 1.0e-7
    assert np.asarray(report["bmnc_b"]).shape == (1, 2)


def test_booz_xform_flux_tube_sensitivity_report_is_bounded_when_available() -> None:
    for name in ("booz_xform_jax", "booz_xform_jax.jax_api"):
        sys.modules.pop(name, None)

    report = booz_xform_flux_tube_sensitivity_report(ntheta=32, fd_step=2.0e-5)

    assert (
        spectraxgk.booz_xform_flux_tube_sensitivity_report
        is booz_xform_flux_tube_sensitivity_report
    )
    assert "available" in report
    if not report["available"]:
        assert report["sensitivity"] is None
        return

    sensitivity = report["sensitivity"]
    assert sensitivity["observable_names"] == list(geometry_observable_names())
    assert np.asarray(sensitivity["jacobian_ad"]).shape == (
        len(geometry_observable_names()),
        2,
    )
    assert float(sensitivity["max_abs_ad_fd_error"]) < 2.0e-6
    assert float(sensitivity["max_rel_ad_fd_error"]) < 2.0e-4
    assert sensitivity["conditioning"]["jacobian_shape"] == [
        len(geometry_observable_names()),
        2,
    ]
    assert np.asarray(report["bmnc_b"]).shape == (5,)


def test_vmec_jax_boozer_flux_tube_sensitivity_report_starts_from_real_vmec_state_when_available() -> (
    None
):
    for name in (
        "vmec_jax",
        "vmec_jax.driver",
        "vmec_jax.config",
        "vmec_jax.static",
        "vmec_jax.wout",
        "vmec_jax.booz_input",
        "booz_xform_jax",
        "booz_xform_jax.jax_api",
    ):
        sys.modules.pop(name, None)

    report = vmec_jax_boozer_flux_tube_sensitivity_report(ntheta=16, fd_step=2.0e-5)

    assert (
        spectraxgk.vmec_jax_boozer_flux_tube_sensitivity_report
        is vmec_jax_boozer_flux_tube_sensitivity_report
    )
    assert "available" in report
    if not report["available"]:
        assert report["sensitivity"] is None
        return

    sensitivity = report["sensitivity"]
    assert report["case_name"] == "circular_tokamak"
    assert report["param_names"] == ["delta_Rcos", "delta_Zsin"]
    assert sensitivity["observable_names"] == list(geometry_observable_names())
    assert np.asarray(sensitivity["jacobian_ad"]).shape == (
        len(geometry_observable_names()),
        2,
    )
    assert float(sensitivity["max_abs_ad_fd_error"]) < 2.0e-5
    assert float(sensitivity["max_rel_ad_fd_error"]) < 2.0e-4
    assert sensitivity["conditioning"]["finite_ad_jacobian"] is True
    assert np.asarray(report["bmnc_b"]).shape == (2,)


def test_vmec_jax_flux_tube_sensitivity_report_starts_from_real_vmec_state_when_available() -> (
    None
):
    for name in (
        "vmec_jax",
        "vmec_jax.driver",
        "vmec_jax.config",
        "vmec_jax.static",
        "vmec_jax.wout",
        "vmec_jax.geom",
        "vmec_jax.vmec_bcovar",
        "vmec_jax.field",
    ):
        sys.modules.pop(name, None)

    report = vmec_jax_flux_tube_sensitivity_report(ntheta=12, fd_step=2.0e-6)

    assert (
        spectraxgk.vmec_jax_flux_tube_sensitivity_report
        is vmec_jax_flux_tube_sensitivity_report
    )
    assert "available" in report
    if not report["available"]:
        assert report["sensitivity"] is None
        return

    sensitivity = report["sensitivity"]
    assert report["case_name"] == "nfp4_QH_warm_start"
    assert report["param_names"] == ["delta_Rcos", "delta_Zsin"]
    assert sensitivity["observable_names"] == list(geometry_observable_names())
    assert np.asarray(sensitivity["jacobian_ad"]).shape == (
        len(geometry_observable_names()),
        2,
    )
    assert float(sensitivity["max_abs_ad_fd_error"]) < 1.0e1
    assert float(sensitivity["max_rel_ad_fd_error"]) < 1.0e-3
    assert sensitivity["conditioning"]["finite_fd_jacobian"] is True
    assert int(report["surface_index"]) > 0
    assert float(report["reference_length"]) > 0.0
    assert float(report["reference_b"]) > 0.0


def test_vmec_jax_flux_tube_array_parity_report_tracks_production_gap_when_available() -> (
    None
):
    for name in (
        "vmec_jax",
        "vmec_jax.driver",
        "vmec_jax.config",
        "vmec_jax.static",
        "vmec_jax.wout",
        "vmec_jax.geom",
        "vmec_jax.vmec_bcovar",
        "vmec_jax.field",
        "booz_xform_jax",
        "booz_xform_jax.jax_api",
    ):
        sys.modules.pop(name, None)

    report = vmec_jax_flux_tube_array_parity_report(ntheta=8)

    assert (
        spectraxgk.vmec_jax_flux_tube_array_parity_report
        is vmec_jax_flux_tube_array_parity_report
    )
    assert "available" in report
    if not report["available"]:
        assert "reason" in report or "error" in report
        return

    assert report["case_name"] == "nfp4_QH_warm_start"
    assert report["status"] in {"diagnostic_open", "passed"}
    assert set(report["array_metrics"]) >= {
        "bmag",
        "gds2",
        "gds21",
        "gds22",
        "gbdrift",
        "jacobian",
        "grho",
    }
    assert set(report["scalar_metrics"]) == {"gradpar", "q", "s_hat"}
    assert "equal_arc_core_array_metrics" in report
    assert "equal_arc_metric_array_metrics" in report
    assert "equal_arc_drift_array_metrics" in report
    assert "equal_arc_core_scalar_metrics" in report
    assert np.isfinite(float(report["worst_core_normalized_max_abs"]))
    assert np.isfinite(float(report["worst_scalar_rel"]))
    assert bool(report["array_metrics"]["bmag"]["shape_match"])
    if report["equal_arc_core_array_metrics"]:
        assert (
            spectraxgk.vmec_jax_boozer_equal_arc_core_profiles_from_state
            is vmec_jax_boozer_equal_arc_core_profiles_from_state
        )
        assert set(report["equal_arc_core_array_metrics"]) >= {
            "bmag",
            "bgrad",
            "jacobian",
        }
        assert set(report["equal_arc_metric_array_metrics"]) == {
            "gds2",
            "gds21",
            "gds22",
            "grho",
        }
        assert set(report["equal_arc_drift_array_metrics"]) == {
            "cvdrift",
            "gbdrift",
            "cvdrift0",
            "gbdrift0",
        }
        assert set(report["equal_arc_core_scalar_metrics"]) == {"gradpar", "q", "s_hat"}
        assert np.isfinite(float(report["equal_arc_core_worst_normalized_max_abs"]))
        assert np.isfinite(float(report["equal_arc_core_worst_scalar_rel"]))
        assert np.isfinite(
            float(report["equal_arc_derivative_worst_normalized_max_abs"])
        )
        assert np.isfinite(float(report["equal_arc_metric_worst_normalized_max_abs"]))
        assert np.isfinite(float(report["equal_arc_drift_worst_normalized_max_abs"]))
        assert float(report["equal_arc_core_worst_normalized_max_abs"]) < 5.0e-2
        assert float(report["equal_arc_core_worst_scalar_rel"]) < 5.0e-2
        assert float(report["equal_arc_derivative_worst_normalized_max_abs"]) < 1.0e-1
        assert float(report["equal_arc_metric_worst_normalized_max_abs"]) < 1.2e-1
        assert float(report["equal_arc_drift_worst_normalized_max_abs"]) < 1.2e-1


def test_vmec_jax_flux_tube_array_parity_report_enforces_boozer_resolution_floor() -> (
    None
):
    with pytest.raises(ValueError, match="mboz and nboz"):
        vmec_jax_flux_tube_array_parity_report(mboz=20, nboz=21)


def test_boozer_half_mesh_s_grid_uses_fortran_half_mesh_indices() -> None:
    s_half = _boozer_half_mesh_s_grid(
        jnp.asarray([2, 3, 4]),
        ns_b=3,
        ns_b_full=3,
        dtype=jnp.float64,
    )

    np.testing.assert_allclose(
        np.asarray(s_half),
        np.asarray([1.0 / 6.0, 3.0 / 6.0, 5.0 / 6.0]),
    )
    fallback = _boozer_half_mesh_s_grid(
        None,
        ns_b=3,
        ns_b_full=3,
        dtype=jnp.float64,
    )
    np.testing.assert_allclose(np.asarray(fallback), np.asarray(s_half))


def test_vmec_jax_boozer_equal_arc_core_profiles_supports_surface_stencil(
    monkeypatch,
) -> None:
    vmec_pkg = types.ModuleType("vmec_jax")
    vmec_pkg.__path__ = []  # type: ignore[attr-defined]
    booz_pkg = types.ModuleType("booz_xform_jax")
    booz_pkg.__path__ = []  # type: ignore[attr-defined]
    booz_input = types.ModuleType("vmec_jax.booz_input")
    booz_api = types.ModuleType("booz_xform_jax.jax_api")
    captured: dict[str, list[int] | None] = {}

    def booz_xform_inputs_from_state(*args, **kwargs):
        return types.SimpleNamespace(bmns=None)

    def prepare_booz_xform_constants_from_inputs(*args, **kwargs):
        return object(), object()

    def booz_xform_from_inputs(*, surface_indices=None, **kwargs):
        if surface_indices is None:
            idx = jnp.arange(5, dtype=jnp.int32)
            captured["surface_indices"] = None
        else:
            idx = jnp.asarray(surface_indices, dtype=jnp.int32)
            captured["surface_indices"] = [int(x) for x in np.asarray(idx)]
        s = idx.astype(jnp.float64)
        rows = idx.size
        return {
            "bmnc_b": jnp.stack((1.0 + 0.01 * s, 0.04 + 0.002 * s), axis=1),
            "rmnc_b": jnp.stack((2.0 + 0.02 * s, 0.03 + 0.001 * s), axis=1),
            "zmns_b": jnp.stack((jnp.zeros(rows), 0.12 + 0.003 * s), axis=1),
            "pmns_b": jnp.stack((jnp.zeros(rows), 0.01 + 0.001 * s), axis=1),
            "iota_b": 0.42 + 0.01 * s,
            "buco_b": 0.08 + 0.002 * s,
            "bvco_b": 1.1 + 0.01 * s,
            "ixm_b": jnp.asarray([0, 1], dtype=jnp.int32),
            "ixn_b": jnp.asarray([0, 0], dtype=jnp.int32),
            "ns_b": 5,
            "jlist": idx + 2,
        }

    booz_input.booz_xform_inputs_from_state = booz_xform_inputs_from_state
    booz_api.prepare_booz_xform_constants_from_inputs = (
        prepare_booz_xform_constants_from_inputs
    )
    booz_api.booz_xform_from_inputs = booz_xform_from_inputs
    monkeypatch.setitem(sys.modules, "vmec_jax", vmec_pkg)
    monkeypatch.setitem(sys.modules, "vmec_jax.booz_input", booz_input)
    monkeypatch.setitem(sys.modules, "booz_xform_jax", booz_pkg)
    monkeypatch.setitem(sys.modules, "booz_xform_jax.jax_api", booz_api)
    monkeypatch.setattr(
        diff_geom,
        "discover_differentiable_geometry_backends",
        lambda: {"vmec_jax_available": True, "booz_xform_jax_api_available": True},
    )

    state = types.SimpleNamespace(Rcos=jnp.ones((6, 2), dtype=jnp.float64))
    wout = types.SimpleNamespace(
        signgs=1, Aminor_p=1.0, phi=np.asarray([0.0, -np.pi]), nfp=4
    )
    mapping = vmec_jax_boozer_equal_arc_core_profiles_from_state(
        state,
        static=object(),
        indata=object(),
        wout=wout,
        ntheta=8,
        surface_stencil_width=3,
    )

    assert captured["surface_indices"] == [1, 2, 3]
    assert mapping["surface_stencil_width"] == 3
    assert mapping["boozer_surface_indices"] == [1, 2, 3]
    assert np.all(np.isfinite(np.asarray(mapping["bmag"])))

    with pytest.raises(ValueError, match="surface_stencil_width"):
        vmec_jax_boozer_equal_arc_core_profiles_from_state(
            state,
            static=object(),
            indata=object(),
            wout=wout,
            ntheta=8,
            surface_stencil_width=2,
        )


def test_vmec_jax_metric_tensor_sensitivity_report_checks_real_metric_tensors_when_available() -> (
    None
):
    for name in (
        "vmec_jax",
        "vmec_jax.driver",
        "vmec_jax.config",
        "vmec_jax.static",
        "vmec_jax.wout",
        "vmec_jax.geom",
    ):
        sys.modules.pop(name, None)

    report = vmec_jax_metric_tensor_sensitivity_report(fd_step=2.0e-5)

    assert (
        spectraxgk.vmec_jax_metric_tensor_sensitivity_report
        is vmec_jax_metric_tensor_sensitivity_report
    )
    assert (
        spectraxgk.vmec_metric_tensor_observable_names
        is vmec_metric_tensor_observable_names
    )
    assert "available" in report
    if not report["available"]:
        assert report["sensitivity"] is None
        return

    assert report["case_name"] == "circular_tokamak"
    assert report["param_names"] == ["delta_Rcos", "delta_Zsin"]
    assert report["observable_names"] == list(vmec_metric_tensor_observable_names())
    assert np.asarray(report["jacobian_ad"]).shape == (
        len(vmec_metric_tensor_observable_names()),
        2,
    )
    assert float(report["max_abs_ad_fd_error"]) < 2.0e-5
    assert float(report["max_rel_ad_fd_error"]) < 2.0e-4
    assert report["conditioning"]["jacobian_shape"] == [
        len(vmec_metric_tensor_observable_names()),
        2,
    ]
    assert len(report["metric_grid_shape"]) == 3


def test_vmec_jax_field_line_tensor_sensitivity_report_checks_stellarator_tensors_when_available() -> (
    None
):
    for name in (
        "vmec_jax",
        "vmec_jax.driver",
        "vmec_jax.config",
        "vmec_jax.static",
        "vmec_jax.wout",
        "vmec_jax.geom",
        "vmec_jax.vmec_bcovar",
        "vmec_jax.field",
    ):
        sys.modules.pop(name, None)

    report = vmec_jax_field_line_tensor_sensitivity_report(ntheta=24, fd_step=1.0e-6)

    assert (
        spectraxgk.vmec_jax_field_line_tensor_sensitivity_report
        is vmec_jax_field_line_tensor_sensitivity_report
    )
    assert (
        spectraxgk.vmec_field_line_tensor_observable_names
        is vmec_field_line_tensor_observable_names
    )
    assert "available" in report
    if not report["available"]:
        assert report["sensitivity"] is None
        return

    assert report["case_name"] == "nfp4_QH_warm_start"
    assert report["param_names"] == ["delta_Rcos", "delta_Zsin"]
    assert report["observable_names"] == list(vmec_field_line_tensor_observable_names())
    assert np.asarray(report["jacobian_ad"]).shape == (
        len(vmec_field_line_tensor_observable_names()),
        2,
    )
    assert float(report["max_abs_ad_fd_error"]) < 5.0e-3
    assert float(report["max_rel_ad_fd_error"]) < 5.0e-4
    assert report["conditioning"]["worst_rel_error"]["parameter_name"] in {
        "delta_Rcos",
        "delta_Zsin",
    }
    assert len(report["metric_grid_shape"]) == 3
    assert int(report["metric_grid_shape"][2]) > 1


def _differentiable_mapping(params: jnp.ndarray) -> dict[str, object]:
    theta = jnp.linspace(-jnp.pi, jnp.pi, 16, endpoint=False)
    ripple, elongation = params
    ones = jnp.ones_like(theta)
    shear = 0.35 + 0.1 * elongation
    bmag = 1.0 + ripple * jnp.cos(theta) + 0.03 * jnp.cos(2.0 * theta)
    return {
        "theta": theta,
        "gradpar": (0.8 + 0.02 * ripple) * ones,
        "bmag": bmag,
        "bgrad": -ripple * jnp.sin(theta),
        "gds2": 1.0 + (shear * theta - 0.2 * elongation * jnp.sin(theta)) ** 2,
        "gds21": -shear * (shear * theta - 0.2 * elongation * jnp.sin(theta)),
        "gds22": (shear * shear) * ones,
        "cvdrift": jnp.cos(theta) + 0.1 * elongation * jnp.sin(theta) ** 2,
        "gbdrift": jnp.cos(theta) + 0.1 * elongation * jnp.sin(theta) ** 2,
        "cvdrift0": -shear * jnp.sin(theta),
        "gbdrift0": -shear * jnp.sin(theta),
        "jacobian": 1.0 / ((0.8 + 0.02 * ripple) * bmag),
        "grho": ones,
        "q": 1.4 + 0.05 * elongation,
        "s_hat": shear,
        "epsilon": ripple,
        "R0": 1.0,
        "nfp": 5,
    }


def test_flux_tube_geometry_from_mapping_is_tracer_safe_for_geometry_sensitivities() -> (
    None
):
    x64_enabled = bool(jax.config.jax_enable_x64)
    fd_step = 2.0e-5 if x64_enabled else 1.0e-3
    abs_tol = 5.0e-6 if x64_enabled else 1.0e-3
    rel_tol = 5.0e-4 if x64_enabled else 2.0e-3
    params = jnp.asarray([0.08, 0.4], dtype=jnp.float64 if x64_enabled else jnp.float32)

    report = geometry_sensitivity_report(
        _differentiable_mapping, params, fd_step=fd_step
    )

    assert spectraxgk.geometry_sensitivity_report is geometry_sensitivity_report
    assert report["observable_names"] == list(geometry_observable_names())
    assert np.asarray(report["jacobian_ad"]).shape == (
        len(geometry_observable_names()),
        2,
    )
    assert np.asarray(report["jacobian_fd"]).shape == (
        len(geometry_observable_names()),
        2,
    )
    assert float(report["max_abs_ad_fd_error"]) < abs_tol
    assert float(report["max_rel_ad_fd_error"]) < rel_tol
    conditioning = report["conditioning"]
    assert conditioning["jacobian_shape"] == [len(geometry_observable_names()), 2]
    assert conditioning["finite_ad_jacobian"] is True
    assert conditioning["finite_fd_jacobian"] is True
    assert conditioning["sensitivity_map_rank"] == 2
    assert np.all(np.isfinite(conditioning["jacobian_singular_values"]))
    assert (
        conditioning["worst_abs_error"]["observable_name"]
        in geometry_observable_names()
    )
    assert conditioning["finite_difference_step_by_parameter"][0]["absolute_step"] == (
        pytest.approx(fd_step)
    )


def test_geometry_tracer_detection_recurses_through_containers() -> None:
    """Container tracer detection keeps validation JAX-transform safe."""

    assert bool(jax.jit(lambda x: _is_traced([x]))(jnp.asarray(1.0)))
    assert bool(jax.jit(lambda x: _is_traced({"x": x}))(jnp.asarray(1.0)))


def test_finite_difference_jacobian_matches_closed_form_linear_map() -> None:
    assert spectraxgk.finite_difference_jacobian is finite_difference_jacobian
    x64_enabled = bool(jax.config.jax_enable_x64)
    fd_step = 1.0e-5 if x64_enabled else 1.0e-3
    rtol = 1.0e-10 if x64_enabled else 2.0e-4
    atol = 1.0e-10 if x64_enabled else 2.0e-4

    def fn(params: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray([2.0 * params[0] - params[1], params[0] + 3.0 * params[1]])

    jac = finite_difference_jacobian(fn, jnp.asarray([0.2, -0.5]), step=fd_step)

    np.testing.assert_allclose(
        np.asarray(jac), np.asarray([[2.0, -1.0], [1.0, 3.0]]), rtol=rtol, atol=atol
    )

    with pytest.raises(ValueError, match="one-dimensional"):
        finite_difference_jacobian(fn, jnp.ones((2, 1)))
    with pytest.raises(ValueError, match="step"):
        finite_difference_jacobian(fn, jnp.ones(2), step=0.0)


def test_low_level_radial_and_sampling_helpers_cover_edge_contracts() -> None:
    s_grid = jnp.asarray([0.0, 0.5, 1.0])
    profile = jnp.asarray([1.0, 2.0, 5.0])
    modes = jnp.asarray([[1.0, 10.0], [2.0, 20.0], [5.0, 50.0]])

    assert float(_interp_radial(profile, s_grid, 0.25)) == pytest.approx(1.5)
    assert np.allclose(np.asarray(_interp_radial(modes, s_grid, 0.25)), [1.5, 15.0])
    with pytest.raises(ValueError, match="radial interpolation"):
        _interp_radial(jnp.ones((2, 2, 2)), s_grid[:2], 0.25)

    assert np.allclose(
        np.asarray(_radial_derivative_profile(profile, 0.5)), [2.0, 4.0, 6.0]
    )
    assert np.allclose(
        np.asarray(_radial_derivative_profile(jnp.asarray([3.0]), 0.5)), [0.0]
    )
    with pytest.raises(ValueError, match="one-dimensional"):
        _radial_derivative_profile(jnp.ones((2, 2)), 0.5)

    assert np.allclose(
        np.asarray(_radial_derivative_array(modes, 0.5)),
        [[2.0, 20.0], [4.0, 40.0], [6.0, 60.0]],
    )
    assert np.allclose(
        np.asarray(_radial_derivative_array(jnp.ones((1, 2)), 0.5)), [[0.0, 0.0]]
    )
    with pytest.raises(ValueError, match="two-dimensional"):
        _radial_derivative_array(jnp.ones(2), 0.5)

    cumulative = _cumulative_trapezoid(
        jnp.asarray([0.0, 2.0, 2.0]), jnp.asarray([0.0, 1.0, 3.0])
    )
    assert np.allclose(np.asarray(cumulative), [0.0, 1.0, 5.0])
    assert np.allclose(
        np.asarray(_cumulative_trapezoid(jnp.asarray([7.0]), jnp.asarray([0.0]))), [0.0]
    )
    with pytest.raises(ValueError, match="cumulative trapezoid"):
        _cumulative_trapezoid(jnp.ones((2, 1)), jnp.ones(2))

    grid = jnp.asarray([[0.0, 1.0], [2.0, 3.0]])
    sampled = _periodic_bilinear_sample_2d(
        grid,
        jnp.asarray([0.0, jnp.pi / 2.0]),
        jnp.asarray([0.0, jnp.pi / 2.0]),
    )
    assert sampled.shape == (2,)
    assert np.all(np.isfinite(np.asarray(sampled)))
    with pytest.raises(ValueError, match="two-dimensional"):
        _periodic_bilinear_sample_2d(jnp.ones(2), jnp.ones(2), jnp.ones(2))
    with pytest.raises(ValueError, match="same shape"):
        _periodic_bilinear_sample_2d(grid, jnp.ones(2), jnp.ones(3))
    with pytest.raises(ValueError, match="non-empty"):
        _periodic_bilinear_sample_2d(jnp.ones((0, 2)), jnp.ones(1), jnp.ones(1))


def test_parity_metric_helpers_report_shape_and_error_scales() -> None:
    mismatch = _array_parity_metrics(np.ones((2,)), np.ones((2, 1)))
    assert mismatch["shape_match"] is False
    assert mismatch["candidate_shape"] == [2]

    metrics = _array_parity_metrics(np.asarray([1.0, 3.0]), np.asarray([1.0, 2.0]))
    assert metrics["shape_match"] is True
    assert metrics["max_abs"] == pytest.approx(1.0)
    assert metrics["normalized_max_abs"] == pytest.approx(0.5)
    assert metrics["candidate_max"] == pytest.approx(3.0)

    scalar = _scalar_parity_metrics(3.0, 2.0)
    assert scalar["abs"] == pytest.approx(1.0)
    assert scalar["rel"] == pytest.approx(0.5)


def test_optional_vmec_boundary_report_unavailable_path(monkeypatch) -> None:
    monkeypatch.setattr(
        diff_geom,
        "discover_differentiable_geometry_backends",
        lambda: {"vmec_jax_boundary_api_available": False},
    )

    report = vmec_boundary_aspect_sensitivity_report(jnp.asarray([0.1, 0.2]))

    assert report["available"] is False
    assert report["aspect"] is None
    assert report["grad_ad"] is None


def test_geometry_sensitivity_report_rejects_nondesign_parameter_array() -> None:
    with pytest.raises(ValueError, match="params"):
        geometry_sensitivity_report(_differentiable_mapping, jnp.ones((2, 1)))


def test_geometry_inverse_design_report_recovers_selected_observables() -> None:
    x64_enabled = bool(jax.config.jax_enable_x64)
    dtype = jnp.float64 if x64_enabled else jnp.float32
    fd_step = 2.0e-5 if x64_enabled else 1.0e-3
    rel_tol = 5.0e-4 if x64_enabled else 3.0e-3
    initial = jnp.asarray([0.035, 0.12], dtype=dtype)
    target_params = jnp.asarray([0.085, 0.34], dtype=dtype)
    target_geom = flux_tube_geometry_from_mapping(
        _differentiable_mapping(target_params),
        validate_finite=False,
    )
    target = flux_tube_geometry_observables(target_geom)[jnp.asarray([1, 2])]

    report = geometry_inverse_design_report(
        _differentiable_mapping,
        initial,
        target,
        observable_indices=[1, 2],
        max_steps=6,
        damping=1.0e-8,
        fd_step=fd_step,
    )

    assert spectraxgk.geometry_inverse_design_report is geometry_inverse_design_report
    assert report["observable_names"] == [
        "relative_bmag_ripple",
        "metric_frobenius_rms",
    ]
    assert len(report["history"]) == 7
    assert float(report["final_residual_norm"]) < 1.0e-5
    assert float(report["max_rel_ad_fd_error"]) < rel_tol
    assert report["uq"]["sensitivity_map_rank"] == 2
    assert report["conditioning"]["sensitivity_map_rank"] == 2
    assert report["conditioning"]["worst_rel_error"]["observable_name"] in {
        "relative_bmag_ripple",
        "metric_frobenius_rms",
    }
    covariance = np.asarray(report["uq"]["covariance"])
    assert np.allclose(covariance, covariance.T)
    assert np.all(np.linalg.eigvalsh(covariance) >= -1.0e-16)
    assert float(report["uq"]["jacobian_condition_number"]) < 1.0e5


def test_geometry_inverse_design_report_rejects_invalid_contracts() -> None:
    with pytest.raises(ValueError, match="initial_params"):
        geometry_inverse_design_report(
            _differentiable_mapping,
            jnp.ones((2, 1)),
            jnp.ones(2),
            observable_indices=[1, 2],
        )
    with pytest.raises(ValueError, match="max_steps"):
        geometry_inverse_design_report(
            _differentiable_mapping,
            jnp.ones(2),
            jnp.ones(2),
            observable_indices=[1, 2],
            max_steps=-1,
        )
    with pytest.raises(ValueError, match="damping"):
        geometry_inverse_design_report(
            _differentiable_mapping,
            jnp.ones(2),
            jnp.ones(2),
            observable_indices=[1, 2],
            damping=-1.0,
        )
    with pytest.raises(ValueError, match="observable_indices"):
        geometry_inverse_design_report(
            _differentiable_mapping, jnp.ones(2), jnp.ones(2), observable_indices=[]
        )
    with pytest.raises(ValueError, match="target_observables"):
        geometry_inverse_design_report(
            _differentiable_mapping, jnp.ones(2), jnp.ones(1), observable_indices=[1, 2]
        )
    with pytest.raises(ValueError, match="out-of-range"):
        geometry_inverse_design_report(
            _differentiable_mapping, jnp.ones(2), jnp.ones(1), observable_indices=[99]
        )


def test_geometry_inverse_design_report_defaults_to_all_observables_for_square_problem() -> (
    None
):
    initial = jnp.asarray([0.08, 0.40])
    target = flux_tube_geometry_observables(
        flux_tube_geometry_from_mapping(
            _differentiable_mapping(initial), validate_finite=False
        )
    )

    report = geometry_inverse_design_report(
        _differentiable_mapping,
        initial,
        target,
        max_steps=0,
    )

    assert report["observable_names"] == list(geometry_observable_names())
    assert report["history"][0]["step"] == 0
    assert float(report["history"][0]["residual_norm"]) == pytest.approx(0.0)
