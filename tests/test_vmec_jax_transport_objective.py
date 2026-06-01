"""Tests for VMEC-JAX to SPECTRAX-GK transport objective plumbing."""

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

import spectraxgk
from spectraxgk import (
    StellaratorITGSampleSet,
    VMECJAXSpectraxTransportObjective,
    VMECJAXTransportObjectiveConfig,
    vmec_jax_transport_objective_from_state,
)
from spectraxgk.solver_objective_gradients import SOLVER_OBJECTIVE_NAMES


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
