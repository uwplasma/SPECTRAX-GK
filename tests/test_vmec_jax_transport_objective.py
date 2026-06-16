"""Tests for VMEC-JAX to SPECTRAX-GK transport objective plumbing."""

from __future__ import annotations

import sys
from types import ModuleType
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

import spectraxgk
from spectraxgk import (
    StellaratorITGSampleSet,
    VMECJAXSpectraxTransportObjective,
    VMECJAXTransportObjectiveConfig,
    vmec_jax_transport_growth_branch_locality_report_from_states,
    vmec_jax_transport_objective_from_state,
)
from spectraxgk.solver_objective_core import SOLVER_OBJECTIVE_NAMES


def _fake_geometry() -> SimpleNamespace:
    theta = jnp.linspace(-jnp.pi, jnp.pi, 8, endpoint=False)
    return SimpleNamespace(
        theta=theta,
        bmag_profile=1.0 + 0.05 * jnp.cos(theta),
        jacobian_profile=jnp.ones_like(theta),
        gds2_profile=1.2 + 0.1 * jnp.cos(theta),
        gds21_profile=0.05 * jnp.sin(theta),
        gds22_profile=1.0 + 0.08 * jnp.cos(2.0 * theta),
        cv_profile=0.03 * jnp.sin(theta),
        gb_profile=0.04 * jnp.cos(theta),
        cv0_profile=0.02 * jnp.sin(2.0 * theta),
        gb0_profile=0.02 * jnp.cos(2.0 * theta),
    )


def _fake_solver_rows(scale: float = 1.0) -> jnp.ndarray:
    rows = []
    idx = {name: i for i, name in enumerate(SOLVER_OBJECTIVE_NAMES)}
    for gamma in (0.08, 0.10, 0.12, 0.14):
        row = np.zeros(len(SOLVER_OBJECTIVE_NAMES), dtype=float)
        row[idx["gamma"]] = scale * gamma
        row[idx["omega"]] = -0.2
        row[idx["kperp_eff2"]] = 0.42
        row[idx["linear_heat_flux_weight"]] = 1.5
        row[idx["linear_particle_flux_weight"]] = 0.3
        row[idx["mixing_length_heat_flux_proxy"]] = scale * 0.04
        rows.append(row)
    return jnp.asarray(rows)


def test_vmec_jax_transport_objective_reduces_fake_solver_rows(monkeypatch) -> None:
    import spectraxgk.vmec_jax_transport_objective as mod

    calls: list[dict[str, object]] = []
    growth_calls: list[dict[str, object]] = []
    rows = _fake_solver_rows()
    row_counter = {"i": 0}

    def fake_geom(state, static, indata, wout, **kwargs):
        calls.append({"state": state, "static": static, "indata": indata, "wout": wout, **kwargs})
        return _fake_geometry()

    def fake_growth(_geom, **kwargs):
        growth_calls.append(kwargs)
        value = rows[row_counter["i"], SOLVER_OBJECTIVE_NAMES.index("gamma")]
        row_counter["i"] += 1
        return value

    monkeypatch.setattr(mod, "flux_tube_geometry_from_vmec_boozer_state", fake_geom)
    monkeypatch.setattr(mod, "solver_growth_rate_from_geometry", fake_growth)
    samples = StellaratorITGSampleSet(surfaces=(0.5, 0.7), alphas=(0.0,), ky_values=(0.2, 0.4))
    cfg = VMECJAXTransportObjectiveConfig(kind="growth", sample_set=samples, ny=4)

    value = vmec_jax_transport_objective_from_state(
        object(),
        object(),
        object(),
        SimpleNamespace(signgs=1, nfp=2, Aminor_p=1.0, phi=np.asarray([0.0, -np.pi])),
        cfg,
    )

    assert np.isclose(float(value), np.mean([0.08, 0.10, 0.12, 0.14]))
    assert calls[0]["mboz"] == 21
    assert calls[0]["nboz"] == 21
    assert [call["torflux"] for call in calls] == list(samples.surfaces)
    assert [call["selected_ky_index"] for call in growth_calls] == [1, 2, 1, 2]
    assert np.isclose(growth_calls[0]["ly"], 2.0 * np.pi / min(samples.ky_values))
    assert int(growth_calls[0]["ny"]) >= 6


def test_vmec_jax_transport_surface_chunking_matches_unchunked_weighted_mean(monkeypatch) -> None:
    import spectraxgk.vmec_jax_transport_objective as mod

    def fake_geom(*_args, **_kwargs):
        return _fake_geometry()

    rows = _fake_solver_rows()

    def evaluate(*, chunk_size: int) -> float:
        row_counter = {"i": 0}

        def fake_growth(_geom, **_kwargs):
            value = rows[row_counter["i"], SOLVER_OBJECTIVE_NAMES.index("gamma")]
            row_counter["i"] += 1
            return value

        monkeypatch.setattr(mod, "solver_growth_rate_from_geometry", fake_growth)
        samples = StellaratorITGSampleSet(
            surfaces=(0.5, 0.7),
            alphas=(0.0,),
            ky_values=(0.2, 0.4),
            surface_weights=(3.0, 1.0),
        )
        cfg = VMECJAXTransportObjectiveConfig(
            kind="growth",
            sample_set=samples,
            ny=4,
            objective_transform="log1p",
            surface_chunk_size=chunk_size,
        )
        value = vmec_jax_transport_objective_from_state(
            object(),
            object(),
            object(),
            SimpleNamespace(signgs=1, nfp=2, Aminor_p=1.0, phi=np.asarray([0.0, -np.pi])),
            cfg,
        )
        assert row_counter["i"] == 4
        return float(value)

    monkeypatch.setattr(mod, "flux_tube_geometry_from_vmec_boozer_state", fake_geom)

    assert evaluate(chunk_size=1) == pytest.approx(evaluate(chunk_size=0))


def test_vmec_jax_transport_growth_branch_locality_report_accepts_consistent_branch(monkeypatch) -> None:
    import spectraxgk.vmec_jax_transport_objective as mod

    def fake_geom(state, *_args, **kwargs):
        return SimpleNamespace(state=state, theta=jnp.ones(2), kwargs=kwargs)

    def fake_matrix(geom, **_kwargs):
        if geom.state == "base":
            return jnp.diag(jnp.asarray([1.0 + 0.0j, 0.5 + 0.0j]))
        if geom.state == "plus":
            return jnp.diag(jnp.asarray([1.02 + 0.0j, 0.48 + 0.0j]))
        return jnp.diag(jnp.asarray([0.98 + 0.0j, 0.52 + 0.0j]))

    monkeypatch.setattr(mod, "flux_tube_geometry_from_vmec_boozer_state", fake_geom)
    monkeypatch.setattr(mod, "solver_linear_operator_matrix_from_geometry", fake_matrix)
    samples = StellaratorITGSampleSet(surfaces=(0.5,), alphas=(0.0,), ky_values=(0.2,))
    cfg = VMECJAXTransportObjectiveConfig(kind="growth", sample_set=samples)

    report = vmec_jax_transport_growth_branch_locality_report_from_states(
        "base",
        "plus",
        "minus",
        "static",
        "indata",
        object(),
        cfg,
        step=1.0e-2,
    )

    assert (
        spectraxgk.vmec_jax_transport_growth_branch_locality_report_from_states
        is vmec_jax_transport_growth_branch_locality_report_from_states
    )
    assert report["passed"] is True
    assert report["classification"] == "all_samples_dominant_growth_branch_locally_consistent"
    assert report["sample_count"] == 1
    assert report["evaluated_sample_count"] == 1
    assert report["rows"][0]["classification"] == "dominant_branch_locally_consistent"


def test_vmec_jax_transport_growth_branch_locality_report_fails_on_branch_switch(monkeypatch) -> None:
    import spectraxgk.vmec_jax_transport_objective as mod

    def fake_geom(state, *_args, **kwargs):
        return SimpleNamespace(state=state, theta=jnp.ones(2), kwargs=kwargs)

    def fake_matrix(geom, **_kwargs):
        if geom.state == "base":
            return jnp.diag(jnp.asarray([1.0 + 0.0j, 0.8 + 0.0j]))
        if geom.state == "plus":
            return jnp.diag(jnp.asarray([1.02 + 0.0j, 1.05 + 0.0j]))
        return jnp.diag(jnp.asarray([0.98 + 0.0j, 0.65 + 0.0j]))

    monkeypatch.setattr(mod, "flux_tube_geometry_from_vmec_boozer_state", fake_geom)
    monkeypatch.setattr(mod, "solver_linear_operator_matrix_from_geometry", fake_matrix)
    samples = StellaratorITGSampleSet(surfaces=(0.5,), alphas=(0.0,), ky_values=(0.2,))
    cfg = VMECJAXTransportObjectiveConfig(kind="growth", sample_set=samples)

    report = vmec_jax_transport_growth_branch_locality_report_from_states(
        "base",
        "plus",
        "minus",
        "static",
        "indata",
        object(),
        cfg,
        step=1.0e-2,
    )

    assert report["passed"] is False
    assert report["classification"] == "growth_branch_locality_failed_or_incomplete"
    assert report["blockers"] == ["branch_locality_mismatch_or_underisolated"]
    assert report["rows"][0]["classification"] == "dominant_branch_differs_from_nearest_branch"


def test_vmec_jax_transport_objective_nonlinear_proxy_is_positive_and_exported(monkeypatch) -> None:
    import spectraxgk.vmec_jax_transport_objective as mod

    scale = {"value": 1.0}

    def fake_geom(*_args, **_kwargs):
        return _fake_geometry()

    def fake_growth(_geom, **_kwargs):
        return jnp.asarray(0.1 * scale["value"])

    monkeypatch.setattr(mod, "flux_tube_geometry_from_vmec_boozer_state", fake_geom)
    monkeypatch.setattr(mod, "solver_growth_rate_from_geometry", fake_growth)
    samples = StellaratorITGSampleSet(surfaces=(0.5, 0.7), alphas=(0.0,), ky_values=(0.2, 0.4))
    cfg = VMECJAXTransportObjectiveConfig(kind="nonlinear_window_heat_flux", sample_set=samples)

    low = vmec_jax_transport_objective_from_state("state", "static", "indata", object(), cfg)
    scale["value"] = 2.0
    high = vmec_jax_transport_objective_from_state("state", "static", "indata", object(), cfg)

    assert spectraxgk.VMECJAXTransportObjectiveConfig is VMECJAXTransportObjectiveConfig
    assert spectraxgk.VMECJAXSpectraxTransportObjective is VMECJAXSpectraxTransportObjective
    assert float(low) > 0.0
    assert float(high) > float(low)


def test_vmec_jax_transport_objective_transform_scales_large_residuals(monkeypatch) -> None:
    import spectraxgk.vmec_jax_transport_objective as mod

    def fake_geom(*_args, **_kwargs):
        return _fake_geometry()

    def fake_growth(_geom, **_kwargs):
        return jnp.asarray(20.0)

    monkeypatch.setattr(mod, "flux_tube_geometry_from_vmec_boozer_state", fake_geom)
    monkeypatch.setattr(mod, "solver_growth_rate_from_geometry", fake_growth)
    samples = StellaratorITGSampleSet(surfaces=(0.5,), alphas=(0.0,), ky_values=(0.2,))
    raw_cfg = VMECJAXTransportObjectiveConfig(
        kind="nonlinear_window_heat_flux",
        sample_set=samples,
        objective_transform="raw",
    )
    scaled_cfg = VMECJAXTransportObjectiveConfig(
        kind="nonlinear_window_heat_flux",
        sample_set=samples,
        objective_transform="scaled",
        objective_scale=10.0,
    )
    log_cfg = VMECJAXTransportObjectiveConfig(
        kind="nonlinear_window_heat_flux",
        sample_set=samples,
        objective_transform="log1p",
        objective_scale=10.0,
    )

    raw = vmec_jax_transport_objective_from_state("state", "static", "indata", object(), raw_cfg)
    scaled = vmec_jax_transport_objective_from_state("state", "static", "indata", object(), scaled_cfg)
    logged = vmec_jax_transport_objective_from_state("state", "static", "indata", object(), log_cfg)

    assert float(raw) > 1.0
    assert float(scaled) == pytest.approx(float(raw) / 10.0)
    assert float(logged) == pytest.approx(float(jnp.log1p(jnp.abs(scaled))))
    assert float(logged) < float(scaled)


def test_vmec_jax_transport_objective_vmec_callback_builds_reference_wout(monkeypatch) -> None:
    import spectraxgk.vmec_jax_transport_objective as mod

    captured: dict[str, object] = {}

    def fake_eval(state, static, indata, wout_reference, config):
        captured["state"] = state
        captured["static"] = static
        captured["indata"] = indata
        captured["wout"] = wout_reference
        captured["config"] = config
        return jnp.asarray(0.125)

    monkeypatch.setattr(mod, "vmec_jax_transport_objective_from_state", fake_eval)
    objective = VMECJAXSpectraxTransportObjective()
    ctx = SimpleNamespace(static=SimpleNamespace(cfg=SimpleNamespace(nfp=3)), indata="indata", signgs=-1)

    value = objective.J(ctx, "state")

    assert float(value) == 0.125
    assert captured["state"] == "state"
    assert captured["indata"] == "indata"
    assert captured["wout"].nfp == 3
    assert captured["wout"].signgs == -1


def test_vmec_jax_transport_config_rejects_underresolved_boozer_modes() -> None:
    assert VMECJAXTransportObjectiveConfig(kind="growth").gradient_scope == "eigenvalue_growth_ad"
    assert (
        VMECJAXTransportObjectiveConfig(kind="quasilinear_flux").gradient_scope
        == "eigenvalue_growth_ad_with_geometry_transport_weights"
    )
    try:
        VMECJAXTransportObjectiveConfig(mboz=12, nboz=21)
    except ValueError as exc:
        assert "at least 21" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("underresolved Boozer mode count should fail")
    with pytest.raises(ValueError, match="objective_scale"):
        VMECJAXTransportObjectiveConfig(objective_scale=0.0)
    with pytest.raises(ValueError, match="objective transform"):
        VMECJAXTransportObjectiveConfig(objective_transform="bad")  # type: ignore[arg-type]


def test_vmec_jax_transport_objective_pins_imported_backend_paths(monkeypatch, tmp_path) -> None:
    import spectraxgk.vmec_jax_transport_objective as mod

    vmec_root = tmp_path / "vmec_jax_repo"
    vmec_pkg = vmec_root / "vmec_jax"
    vmec_pkg.mkdir(parents=True)
    vmec_file = vmec_pkg / "__init__.py"
    vmec_file.write_text("", encoding="utf-8")

    booz_root = tmp_path / "booz_xform_jax_repo" / "src"
    booz_pkg = booz_root / "booz_xform_jax"
    booz_pkg.mkdir(parents=True)
    booz_file = booz_pkg / "__init__.py"
    booz_file.write_text("", encoding="utf-8")

    vmec_module = ModuleType("vmec_jax")
    vmec_module.__file__ = str(vmec_file)
    booz_module = ModuleType("booz_xform_jax")
    booz_module.__file__ = str(booz_file)
    monkeypatch.setitem(sys.modules, "vmec_jax", vmec_module)
    monkeypatch.setitem(sys.modules, "booz_xform_jax", booz_module)
    monkeypatch.delenv("SPECTRAX_VMEC_JAX_PATH", raising=False)
    monkeypatch.delenv("VMEC_JAX_PATH", raising=False)
    monkeypatch.delenv("SPECTRAX_BOOZ_XFORM_JAX_PATH", raising=False)
    monkeypatch.delenv("BOOZ_XFORM_JAX_PATH", raising=False)

    mod._pin_current_optional_backend_paths()

    assert str(vmec_root) == mod.os.environ["SPECTRAX_VMEC_JAX_PATH"]
    assert str(booz_root) == mod.os.environ["SPECTRAX_BOOZ_XFORM_JAX_PATH"]


def test_module_search_root_handles_paths_and_missing_modules(monkeypatch, tmp_path) -> None:
    import spectraxgk.vmec_jax_transport_objective as mod

    namespace_root = tmp_path / "namespace_backend"
    namespace_root.mkdir()
    namespace_module = ModuleType("namespace_backend")
    namespace_module.__path__ = [str(namespace_root)]

    missing_path_module = ModuleType("missing_path_backend")
    missing_path_module.__path__ = [str(tmp_path / "does_not_exist")]

    no_path_module = ModuleType("no_path_backend")

    monkeypatch.setitem(sys.modules, "namespace_backend", namespace_module)
    monkeypatch.setitem(sys.modules, "missing_path_backend", missing_path_module)
    monkeypatch.setitem(sys.modules, "no_path_backend", no_path_module)

    assert mod._module_search_root("namespace_backend") == namespace_root.resolve(strict=False)
    assert mod._module_search_root("missing_path_backend") is None
    assert mod._module_search_root("no_path_backend") is None
    assert mod._module_search_root("spectraxgk_missing_backend_for_test") is None


def test_pin_current_optional_backend_paths_respects_explicit_environment(monkeypatch) -> None:
    import spectraxgk.vmec_jax_transport_objective as mod

    def unexpected_search(module_name: str):
        raise AssertionError(f"backend search should be skipped for {module_name}")

    monkeypatch.setattr(mod, "_module_search_root", unexpected_search)
    monkeypatch.delenv("SPECTRAX_VMEC_JAX_PATH", raising=False)
    monkeypatch.setenv("VMEC_JAX_PATH", "/explicit/vmec-jax")
    monkeypatch.setenv("SPECTRAX_BOOZ_XFORM_JAX_PATH", "/explicit/booz-xform-jax")
    monkeypatch.delenv("BOOZ_XFORM_JAX_PATH", raising=False)

    mod._pin_current_optional_backend_paths()

    assert "SPECTRAX_VMEC_JAX_PATH" not in mod.os.environ
    assert mod.os.environ["VMEC_JAX_PATH"] == "/explicit/vmec-jax"
    assert mod.os.environ["SPECTRAX_BOOZ_XFORM_JAX_PATH"] == "/explicit/booz-xform-jax"


def test_static_grid_options_maps_integer_ky_multiples_to_solver_grid() -> None:
    import spectraxgk.vmec_jax_transport_objective as mod

    options = mod._static_grid_options_from_ky_values((0.15, 0.45), min_ny=12)

    assert options["ky_base"] == pytest.approx(0.15)
    assert options["ly"] == pytest.approx(2.0 * np.pi / 0.15)
    assert options["ny"] == 12
    assert options["selected_ky_indices"] == (1, 3)


@pytest.mark.parametrize(
    ("ky_values", "message"),
    (
        ((), "finite non-empty vector"),
        ((0.2, np.nan), "finite non-empty vector"),
        ((0.0,), "positive"),
        ((0.2, 0.31), "integer multiples"),
        ((0.2, 0.2), "duplicate selected ky indices"),
    ),
)
def test_static_grid_options_rejects_invalid_ky_values(
    ky_values: tuple[float, ...],
    message: str,
) -> None:
    import spectraxgk.vmec_jax_transport_objective as mod

    with pytest.raises(ValueError, match=message):
        mod._static_grid_options_from_ky_values(ky_values, min_ny=3)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"kind": "invalid"}, "unknown VMEC-JAX transport objective kind"),
        ({"ntheta": 3}, "ntheta must be >= 4"),
        ({"n_laguerre": 0}, "n_laguerre and n_hermite must be positive"),
        ({"ny": 2}, "nx must be positive and ny must be at least 3"),
        ({"nonlinear_csat": 0.0}, "nonlinear_csat must be positive"),
        ({"surface_chunk_size": -1}, "surface_chunk_size must be non-negative"),
        (
            {"sample_set": StellaratorITGSampleSet(reduction="max"), "surface_chunk_size": 1},
            "surface_chunk_size currently supports only mean or weighted_mean reductions",
        ),
    ),
)
def test_vmec_jax_transport_config_rejects_invalid_edges(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        VMECJAXTransportObjectiveConfig(**kwargs)


def test_vmec_jax_transport_config_objective_options_filter_none_values() -> None:
    default_options = VMECJAXTransportObjectiveConfig().objective_options()
    configured_options = VMECJAXTransportObjectiveConfig(
        reference_length=2.5,
        reference_b=0.7,
        validate_finite=False,
    ).objective_options()

    assert "reference_length" not in default_options
    assert "reference_b" not in default_options
    assert configured_options["reference_length"] == 2.5
    assert configured_options["reference_b"] == 0.7
    assert configured_options["validate_finite"] is False


def test_geometry_transport_weights_use_safe_defaults_for_minimal_geometry() -> None:
    import spectraxgk.vmec_jax_transport_objective as mod

    theta = jnp.linspace(-jnp.pi, jnp.pi, 6, endpoint=False)
    kperp, heat_weight, particle_weight = mod._geometry_transport_weights(
        SimpleNamespace(theta=theta),
        selected_ky_index=2,
        ly=5.0,
    )

    assert np.isfinite(float(kperp))
    assert np.isfinite(float(heat_weight))
    assert np.isfinite(float(particle_weight))
    assert float(kperp) > 0.0
    assert float(heat_weight) > 0.0
    assert float(particle_weight) == pytest.approx(0.25 * float(heat_weight))


def test_transport_feature_table_rejects_empty_sample_rows() -> None:
    import spectraxgk.vmec_jax_transport_objective as mod

    config = SimpleNamespace(
        sample_set=SimpleNamespace(surfaces=(), alphas=(0.0,), ky_values=(0.2,)),
        kind="growth",
    )

    with pytest.raises(RuntimeError, match="produced no sample rows"):
        mod._transport_feature_table_from_state(
            "state",
            "static",
            "indata",
            object(),
            config,
            {"selected_ky_indices": (1,), "ny": 4, "ly": 2.0 * np.pi / 0.2},
        )


def test_quasilinear_flux_uses_geometry_transport_weights(monkeypatch) -> None:
    import spectraxgk.vmec_jax_transport_objective as mod

    def fake_geom(*_args, **_kwargs):
        return _fake_geometry()

    def fake_growth(_geom, **_kwargs):
        return jnp.asarray(0.2)

    monkeypatch.setattr(mod, "flux_tube_geometry_from_vmec_boozer_state", fake_geom)
    monkeypatch.setattr(mod, "solver_growth_rate_from_geometry", fake_growth)
    samples = StellaratorITGSampleSet(surfaces=(0.5,), alphas=(0.0,), ky_values=(0.2,))
    cfg = VMECJAXTransportObjectiveConfig(kind="quasilinear_flux", sample_set=samples)

    value = vmec_jax_transport_objective_from_state("state", "static", "indata", object(), cfg)

    assert float(value) > 0.0


def test_vmec_objective_term_prepares_and_prewarms_backend(monkeypatch) -> None:
    import spectraxgk.vmec_jax_transport_objective as mod

    constructed_terms: list[SimpleNamespace] = []
    prewarm_calls: list[dict[str, object]] = []

    def fake_objective_term(name, callback, *, target, weight, metadata, prepare=None):
        term = SimpleNamespace(
            name=name,
            callback=callback,
            target=target,
            weight=weight,
            metadata=metadata,
            prepare=prepare,
        )
        constructed_terms.append(term)
        return term

    def fake_prewarm(static, wout_reference, **kwargs):
        prewarm_calls.append({"static": static, "wout": wout_reference, **kwargs})

    vmec_module = ModuleType("vmec_jax")
    vmec_module.ObjectiveTerm = fake_objective_term
    monkeypatch.setitem(sys.modules, "vmec_jax", vmec_module)
    monkeypatch.setattr(mod, "prewarm_vmec_boozer_equal_arc_cache", fake_prewarm)
    cfg = VMECJAXTransportObjectiveConfig(kind="growth", ntheta=28, mboz=23, nboz=25, surface_chunk_size=1)
    objective = VMECJAXSpectraxTransportObjective(config=cfg, name="transport_test")
    ctx = SimpleNamespace(static=SimpleNamespace(cfg=SimpleNamespace(nfp=7)), indata="indata", signgs=-1)

    term = objective.to_objective_term(target=0.0, residual_weight=2.5)
    prepared = term.prepare(ctx)

    assert len(constructed_terms) == 2
    assert term.name == "transport_test"
    assert term.weight == 2.5
    assert term.metadata["spectraxgk_transport_kind"] == "growth"
    assert term.metadata["gradient_scope"] == "eigenvalue_growth_ad"
    assert term.metadata["mboz"] == 23
    assert term.metadata["nboz"] == 25
    assert term.metadata["ntheta"] == 28
    assert term.metadata["surface_chunk_size"] == 1
    assert prepared.prepare is None
    assert prepared.metadata == term.metadata
    assert len(prewarm_calls) == 1
    assert prewarm_calls[0]["static"] is ctx.static
    assert prewarm_calls[0]["mboz"] == 23
    assert prewarm_calls[0]["nboz"] == 25
    assert prewarm_calls[0]["wout"].signgs == -1
    assert prewarm_calls[0]["wout"].nfp == 7
    assert prewarm_calls[0]["wout"].Aminor_p == 1.0
    assert np.allclose(prewarm_calls[0]["wout"].phi, np.asarray([0.0, -np.pi]))


def test_spectrax_transport_objective_tuple_uses_config_and_wout_reference(monkeypatch) -> None:
    import spectraxgk.vmec_jax_transport_objective as mod

    captured: dict[str, object] = {}
    wout_reference = object()
    config = VMECJAXTransportObjectiveConfig(kind="growth")

    def fake_eval(state, static, indata, wout, cfg):
        captured["state"] = state
        captured["static"] = static
        captured["indata"] = indata
        captured["wout"] = wout
        captured["config"] = cfg
        return jnp.asarray(0.5)

    monkeypatch.setattr(mod, "vmec_jax_transport_objective_from_state", fake_eval)

    callback, target, weight = mod.spectrax_transport_objective_tuple(
        weight=3.0,
        target=1.5,
        config=config,
        wout_reference=wout_reference,
    )
    value = callback(SimpleNamespace(static="static", indata="indata", signgs=1), "state")

    assert target == 1.5
    assert weight == 3.0
    assert float(value) == 0.5
    assert captured == {
        "state": "state",
        "static": "static",
        "indata": "indata",
        "wout": wout_reference,
        "config": config,
    }
