from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from spectraxgk.nonlinear_parallel import (
    NonlinearDomainDecompositionPlan,
    build_nonlinear_domain_decomposition_plan,
    deterministic_nonlinear_domain_state,
    nonlinear_domain_identity_report,
    nonlinear_domain_parallel_identity_gate,
    nonlinear_domain_transport_window_identity_gate,
    prototype_nonlinear_domain_decomposed_step,
    prototype_nonlinear_domain_serial_step,
)


def test_nonlinear_domain_plan_uses_static_halo_chunks() -> None:
    plan = build_nonlinear_domain_decomposition_plan(
        (7, 3),
        axis=-2,
        num_domains=3,
    )

    assert plan == NonlinearDomainDecompositionPlan(
        state_shape=(7, 3),
        axis=0,
        chunk_sizes=(3, 2, 2),
        halo=1,
    )
    assert plan.num_domains == 3
    assert plan.domain_size == 7
    assert plan.offsets == (0, 3, 5)
    assert plan.chunk_bounds == ((0, 3), (3, 5), (5, 7))
    assert plan.boundary_indices == (0, 2, 3, 4, 5, 6)
    assert plan.decomposition_metadata() == {
        "state_shape": (7, 3),
        "axis": 0,
        "chunk_sizes": (3, 2, 2),
        "halo": 1,
        "num_domains": 3,
        "domain_size": 7,
        "offsets": (0, 3, 5),
        "chunk_bounds": ((0, 3), (3, 5), (5, 7)),
        "boundary_indices": (0, 2, 3, 4, 5, 6),
    }
    assert plan.to_dict() == {
        "state_shape": (7, 3),
        "axis": 0,
        "chunk_sizes": (3, 2, 2),
        "halo": 1,
    }


def test_nonlinear_domain_identity_gate_enables_only_matching_decomposition() -> None:
    state = deterministic_nonlinear_domain_state((6, 4))
    plan = build_nonlinear_domain_decomposition_plan(state.shape, num_domains=2)

    gated_state, report = nonlinear_domain_parallel_identity_gate(
        state,
        plan,
        dt=0.025,
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    serial = prototype_nonlinear_domain_serial_step(state, axis=plan.axis, dt=0.025)
    decomposed = prototype_nonlinear_domain_decomposed_step(state, plan, dt=0.025)

    assert report.identity_passed is True
    assert report.decomposed_path_enabled is True
    assert report.gate_name == "nonlinear_domain_local_stencil_identity"
    assert report.plan_valid is True
    assert report.blocked_reasons == ()
    assert report.max_abs_error <= report.atol
    assert report.max_rel_error <= report.rtol
    assert report.boundary_indices == (0, 2, 3, 5)
    assert report.boundary_max_abs_error <= report.atol
    assert report.boundary_max_rel_error <= report.rtol
    assert "bounded local-stencil prototype" in report.claim_scope
    assert "no production routing or speedup claim" in report.claim_scope
    assert jnp.allclose(decomposed, serial, atol=1.0e-6, rtol=1.0e-6)
    assert jnp.allclose(gated_state, decomposed, atol=1.0e-6, rtol=1.0e-6)


def test_nonlinear_domain_transport_window_gate_tracks_conservation_proxies() -> None:
    state = deterministic_nonlinear_domain_state((6, 4))
    plan = build_nonlinear_domain_decomposition_plan(state.shape, num_domains=2)

    report = nonlinear_domain_transport_window_identity_gate(
        state,
        plan,
        dt=0.025,
        steps=5,
        atol=1.0e-6,
        rtol=1.0e-6,
    )

    assert report.gate_name == "nonlinear_domain_transport_window_identity"
    assert report.identity_passed is True
    assert report.decomposed_path_enabled is True
    assert report.steps == 5
    assert report.plan_valid is True
    assert report.blocked_reasons == ()
    assert report.boundary_indices == (0, 2, 3, 5)
    assert report.max_abs_state_error <= report.atol
    assert report.max_abs_boundary_error <= report.atol
    assert report.mass_trace_max_abs_error <= report.atol
    assert report.mass_trace_max_rel_error <= report.rtol
    assert report.free_energy_trace_max_abs_error <= report.atol
    assert report.free_energy_trace_max_rel_error <= report.rtol
    assert report.flux_proxy_trace_max_abs_error <= report.atol
    assert report.flux_proxy_trace_max_rel_error <= report.rtol
    assert len(report.serial_mass_trace) == report.steps + 1
    assert len(report.decomposed_free_energy_trace) == report.steps + 1
    assert report.serial_mass_drift == pytest.approx(report.decomposed_mass_drift)
    assert report.serial_free_energy_drift == pytest.approx(
        report.decomposed_free_energy_drift
    )
    assert "transport-window identity gate" in report.claim_scope
    assert "no production routing or speedup claim" in report.claim_scope
    assert report.to_dict()["identity_passed"] is True


def test_nonlinear_domain_transport_window_gate_fails_closed_on_invalid_plan() -> None:
    state = deterministic_nonlinear_domain_state((6, 4))
    invalid_plan = NonlinearDomainDecompositionPlan(
        state_shape=state.shape,
        axis=0,
        chunk_sizes=(2, 2),
        halo=1,
    )

    report = nonlinear_domain_transport_window_identity_gate(
        state,
        invalid_plan,
        steps=2,
        atol=1.0e-6,
        rtol=1.0e-6,
    )

    assert report.identity_passed is False
    assert report.decomposed_path_enabled is False
    assert report.plan_valid is False
    assert report.blocked_reasons == ("chunk_sizes_do_not_cover_axis",)
    assert report.max_abs_state_error == float("inf")
    assert report.mass_trace_max_abs_error == float("inf")


def test_nonlinear_domain_identity_report_fails_closed_on_mismatch() -> None:
    state = deterministic_nonlinear_domain_state((5, 3))
    plan = build_nonlinear_domain_decomposition_plan(state.shape, num_domains=2)
    serial = prototype_nonlinear_domain_serial_step(state, axis=plan.axis, dt=0.05)
    perturbed = serial.at[0, 0].add(1.0e-3)

    report = nonlinear_domain_identity_report(
        serial,
        perturbed,
        plan,
        atol=1.0e-6,
        rtol=1.0e-6,
    )

    assert report.identity_passed is False
    assert report.decomposed_path_enabled is False
    assert report.plan_valid is True
    assert report.blocked_reasons == ()
    assert report.max_abs_error > report.atol
    assert report.boundary_max_abs_error > report.atol
    assert report.to_dict()["identity_passed"] is False


def test_nonlinear_domain_identity_report_blocks_invalid_plan() -> None:
    state = deterministic_nonlinear_domain_state((6, 4))
    invalid_plan = NonlinearDomainDecompositionPlan(
        state_shape=state.shape,
        axis=0,
        chunk_sizes=(2, 2),
        halo=1,
    )

    report = nonlinear_domain_identity_report(
        state,
        state,
        invalid_plan,
        atol=1.0e-6,
        rtol=1.0e-6,
    )

    assert report.identity_passed is False
    assert report.decomposed_path_enabled is False
    assert report.plan_valid is False
    assert report.blocked_reasons == ("chunk_sizes_do_not_cover_axis",)


def test_nonlinear_domain_identity_report_blocks_shape_mismatch() -> None:
    state = deterministic_nonlinear_domain_state((6, 4))
    candidate = deterministic_nonlinear_domain_state((5, 4))
    plan = build_nonlinear_domain_decomposition_plan(state.shape, num_domains=2)

    report = nonlinear_domain_identity_report(
        state,
        candidate,
        plan,
        atol=1.0e-6,
        rtol=1.0e-6,
    )

    assert report.identity_passed is False
    assert report.decomposed_path_enabled is False
    assert report.plan_valid is True
    assert report.blocked_reasons == ("decomposed_shape_does_not_match_serial",)
    assert report.max_abs_error == float("inf")
    assert report.max_rel_error == float("inf")


def test_nonlinear_domain_decomposed_step_is_jax_jittable_with_static_plan() -> None:
    state = deterministic_nonlinear_domain_state((6, 4))
    plan = build_nonlinear_domain_decomposition_plan(state.shape, axis=0, num_domains=3)

    serial = prototype_nonlinear_domain_serial_step(state, axis=plan.axis, dt=0.05)
    jit_step = jax.jit(
        lambda item: prototype_nonlinear_domain_decomposed_step(item, plan, dt=0.05)
    )
    decomposed = jit_step(state)

    assert jnp.allclose(decomposed, serial, atol=1.0e-6, rtol=1.0e-6)


def test_nonlinear_domain_plan_rejects_unsupported_or_invalid_domains() -> None:
    with pytest.raises(ValueError, match="num_domains cannot exceed"):
        build_nonlinear_domain_decomposition_plan((2, 4), num_domains=3)
    with pytest.raises(ValueError, match="one-cell halo"):
        build_nonlinear_domain_decomposition_plan((4, 4), halo=2)
    with pytest.raises(ValueError, match="non-empty"):
        build_nonlinear_domain_decomposition_plan((0, 4))


def test_nonlinear_domain_decomposed_step_rejects_manual_invalid_plan() -> None:
    state = deterministic_nonlinear_domain_state((6, 4))
    invalid_plan = NonlinearDomainDecompositionPlan(
        state_shape=state.shape,
        axis=-1,
        chunk_sizes=(3, 3),
        halo=1,
    )

    with pytest.raises(ValueError, match="axis_not_canonical"):
        prototype_nonlinear_domain_decomposed_step(state, invalid_plan)
